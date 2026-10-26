"""Lightning evaluation of the selected model without prediction artifacts."""
from pathlib import Path
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data_handler.sequence_dataset import SequenceDataset, segment_sequences, materialize_sequences
from .common.paths import compatible_training_config, training_dir
from .common.config import get_batch_size
from .data_handler.data_router import load_model_inputs
from .models.utils import auc_score, make_model, resolve_device
from .models.question_evaluation import question_predictions

def _metrics(labels, probabilities, threshold):
    if not labels:
        return {'targets': 0, 'auc': None, 'accuracy': None, 'rmse': None, 'bce': None}
    y = torch.tensor(labels, dtype=torch.float64)
    p = torch.tensor(probabilities, dtype=torch.float64)
    return {'targets': len(labels), 'auc': auc_score(labels, probabilities), 'accuracy': ((p >= threshold) == (y == 1)).double().mean().item(), 'rmse': (p - y).square().mean().sqrt().item(), 'bce': F.binary_cross_entropy(p.clamp(1e-12, 1 - 1e-12), y).item()}

class KTTest(pl.LightningModule):

    def __init__(self, model, warmup_k, threshold, metric_prefix):
        super().__init__()
        self.model = model
        self.warmup_k = warmup_k
        self.threshold = threshold
        self.metric_prefix = metric_prefix
        (self.labels, self.probabilities) = ([], [])
        self.metrics = None

    def test_step(self, batch, batch_idx):
        (labels, probabilities) = question_predictions(self.model, batch, self.warmup_k)
        self.labels.extend(labels.cpu().tolist())
        self.probabilities.extend(probabilities.cpu().tolist())

    def on_test_epoch_end(self):
        self.metrics = _metrics(self.labels, self.probabilities, self.threshold)
        for name in ('auc', 'accuracy', 'rmse'):
            value = self.metrics[name]
            if value is not None:
                self.log(f'{self.metric_prefix}_{name}', value)

def _test_metrics(model, sequences, config, device, metric_prefix):
    loader = DataLoader(sequences, batch_size=get_batch_size(config), shuffle=False, collate_fn=materialize_sequences, num_workers=config['num_workers'])
    module = KTTest(model, config['warmup_k'], config['evaluation']['accuracy_threshold'], metric_prefix)
    trainer = pl.Trainer(accelerator='gpu' if device.type == 'cuda' else 'cpu', devices=1, enable_checkpointing=False, logger=False)
    trainer.test(module, dataloaders=loader)
    return module.metrics

def load_selected_model(config, device, checkpoint=None):
    (handler, metadata, maps, edges, _, pairs) = load_model_inputs(config)
    checkpoint = Path(checkpoint or training_dir(config) / 'best.ckpt')
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    protocol = saved['protocol']
    selection = saved['selection']
    for (key, value) in (('id_maps', maps), ('mapped_edges', edges), ('pair_keys', pairs)):
        if protocol[key] != value:
            raise ValueError(f'Selected model checkpoint mismatch: {key}')
    if compatible_training_config(protocol['config']) != compatible_training_config(config):
        raise ValueError('Selected model training configuration differs')
    model = make_model(metadata, config, edges, pairs).to(device)
    model.load_state_dict({key.removeprefix('model.'): value for (key, value) in saved['state_dict'].items() if key.startswith('model.')})
    model.eval()
    return (handler, metadata, model, selection, checkpoint)

@torch.no_grad()
def evaluate_selected_model(config, checkpoint=None):
    device = resolve_device(config)
    (handler, _, model, saved, checkpoint) = load_selected_model(config, device, checkpoint)
    sequences = SequenceDataset(handler.val_records).sequences
    sequences = segment_sequences(sequences, config['max_sequence_length'], config['warmup_k'])
    metrics = _test_metrics(model, sequences, config, device, 'validation')
    return {'checkpoint': str(checkpoint), 'selected_epoch': saved['best_epoch'], 'selection_metric': 'validation_auc', 'parameters': sum((p.numel() for p in model.parameters())), 'assignment_connections_initial': model.graph.active_mask.numel(), 'active_assignment_connections': int(model.graph.active_mask.sum()), 'graph_frozen': bool(model.graph.frozen), 'metrics': {'students': len({s['student_raw'] for s in sequences}), 'sequences': len(sequences), **metrics}}

@torch.no_grad()
def run_formal_test(config, checkpoint=None):
    device = resolve_device(config)
    (handler, _, model, saved, checkpoint) = load_selected_model(config, device, checkpoint)
    sequences = SequenceDataset(handler.test_records).sequences
    sequences = segment_sequences(sequences, config['max_sequence_length'], config['warmup_k'])
    metrics = _test_metrics(model, sequences, config, device, 'test')
    return {'checkpoint': str(checkpoint), 'selected_epoch': saved['best_epoch'], 'selection_metric': 'validation_auc', 'selection_value': saved['best_auc'], 'test_students': len({s['student_raw'] for s in sequences}), 'test_sequences': len(sequences), 'metrics': metrics, 'parameters': sum((p.numel() for p in model.parameters())), 'ability_nodes': model.graph.n_abilities - model.graph.abilities_per_concept, 'active_assignment_connections': int(model.graph.active_mask.sum())}
