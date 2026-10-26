"""Lightning training of independent student segments and the shared graph."""
import json
import random

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from .losses import progressive_batch_loss
from ..data_handler.data_module import KTDataModule
from ..data_handler.data_router import load_model_inputs
from ..data_handler.sequence_dataset import SequenceDataset, segment_sequences
from ..common.paths import compatible_training_config, training_dir
from .utils import auc_score, make_model, resolve_device
from .training import update_structure, validate_schedule
from .question_evaluation import question_predictions


def _random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


class KTLightning(pl.LightningModule):
    def __init__(self, config, metadata, maps, edges, pair_keys):
        super().__init__()
        self.config, self.maps, self.edges = config, maps, edges
        self.pair_keys = pair_keys
        self.model = make_model(metadata, config, edges, pair_keys)
        self.monitor = {"previous_mask": None, "previous_weights": None,
                        "stable_epochs": 0}
        self.best_auc, self.best_epoch = None, None
        self.best_changed = False
        self._labels, self._probabilities = [], []

    def configure_optimizers(self):
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": self.config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=self.config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", threshold_mode="abs", **self.config["lr_scheduler"])
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"}}

    def lr_scheduler_step(self, scheduler, metric):
        super().lr_scheduler_step(scheduler, metric)
        # Lightning steps plateau schedulers after the epoch-end hooks. Save
        # afterwards so resume includes this epoch's scheduler state and LR.
        self.trainer.save_checkpoint(
            training_dir(self.config) / "last.ckpt", weights_only=False)

    def training_step(self, batch, batch_idx):
        loss, _ = progressive_batch_loss(self.model, batch, self.config)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 batch_size=1, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._labels, self._probabilities = [], []

    def validation_step(self, batch, batch_idx):
        labels, probabilities = question_predictions(self.model, batch, self.config["warmup_k"])
        self._labels.extend(labels.cpu().tolist())
        self._probabilities.extend(probabilities.cpu().tolist())

    def on_validation_epoch_end(self):
        if not self._labels:
            raise ValueError("No validation targets")
        auc = auc_score(self._labels, self._probabilities)
        epoch = self.current_epoch + 1
        self.best_changed = False
        if auc is not None and (self.best_auc is None or auc > self.best_auc):
            self.best_auc, self.best_epoch = auc, epoch
            self.best_changed = True
        if auc is not None:
            self.log("val_auc", auc, prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["protocol"] = {"config": self.config, "id_maps": self.maps,
                                  "mapped_edges": self.edges, "pair_keys": self.pair_keys}
        checkpoint["selection"] = {"best_auc": self.best_auc, "best_epoch": self.best_epoch,
                                   "monitor": self.monitor}
        checkpoint["random_state"] = _random_state()

    def on_load_checkpoint(self, checkpoint):
        protocol = checkpoint["protocol"]
        if compatible_training_config(protocol["config"]) != compatible_training_config(self.config):
            raise ValueError("Incompatible Lightning checkpoint: config")
        for name, current in (("id_maps", self.maps), ("mapped_edges", self.edges),
                              ("pair_keys", self.pair_keys)):
            if protocol[name] != current:
                raise ValueError(f"Incompatible Lightning checkpoint: {name}")
        selection = checkpoint["selection"]
        self.best_auc, self.best_epoch = selection["best_auc"], selection["best_epoch"]
        self.monitor = selection["monitor"]
        for key in ("previous_mask", "previous_weights"):
            if self.monitor[key] is not None:
                self.monitor[key] = self.monitor[key].cpu()
        _restore_random_state(checkpoint["random_state"])

    def on_fit_start(self):
        self.model.graph.restore_frozen_parameters()
        self.model.graph.encoder.restore_frozen_parameters()


class EpochCheckpoints(pl.Callback):
    def __init__(self, run_dir):
        self.run_dir = run_dir

    def on_validation_end(self, trainer, pl_module):
        if pl_module.best_changed:
            # Save exactly the graph just evaluated, before pruning/freezing.
            trainer.save_checkpoint(self.run_dir / "best.ckpt", weights_only=False)
        update_structure(pl_module.model, pl_module.config["schedule"],
                         pl_module.monitor, pl_module.current_epoch + 1)


def run_training(config, resume=False):
    validate_schedule(config)
    total = config["max_epochs"]
    device = resolve_device(config)
    handler, metadata, maps, edges, report, pair_keys = load_model_inputs(config)
    report_path = handler.cache_file.with_name("relations.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    train = SequenceDataset(handler.train_records).sequences
    valid = SequenceDataset(handler.val_records).sequences
    train = segment_sequences(train, config["max_sequence_length"], config["warmup_k"])
    valid = segment_sequences(valid, config["max_sequence_length"], config["warmup_k"])
    run_dir = training_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "last.ckpt"
    if resume and not last.exists():
        raise FileNotFoundError(last)
    module = KTLightning(config, metadata, maps, edges, pair_keys)
    callbacks = [EpochCheckpoints(run_dir)]
    early_stopping = config["early_stopping"]
    if early_stopping["enabled"]:
        callbacks.append(EarlyStopping(
            monitor="val_auc", mode="max",
            patience=early_stopping["patience"], min_delta=early_stopping["min_delta"],
            check_on_train_epoch_end=False))
    trainer = pl.Trainer(max_epochs=total,
                         accelerator="gpu" if device.type == "cuda" else "cpu",
                         devices=1, enable_checkpointing=False, logger=False,
                         callbacks=callbacks,
                         default_root_dir=str(run_dir), num_sanity_val_steps=0,
                         reload_dataloaders_every_n_epochs=1)
    trainer.fit(module, datamodule=KTDataModule(train, valid, config),
                ckpt_path=str(last) if resume else None, weights_only=False if resume else None)
    result = {"device": str(device), "train_pairs": len(pair_keys),
              "mapped_relation_edges": len(edges), "relation_report": str(report_path),
              "checkpoint": str(last),
              "best_model": str(run_dir / "best.ckpt") if module.best_epoch else None,
              "best_epoch": module.best_epoch, "best_validation_auc": module.best_auc,
              "graph_frozen": bool(module.model.graph.frozen),
              "static_frozen": bool(module.model.graph.encoder.static_frozen)}
    return result
