import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler

from ..models.losses import training_batches
from ..common.config import get_batch_size
from .sequence_dataset import materialize_sequences


class LengthBuckets(Sampler):
    def __init__(self, sequences, size, multiplier):
        self.sequences, self.size, self.multiplier = sequences, size, multiplier
        self.indices = {id(sequence): index for index, sequence in enumerate(sequences)}

    def __iter__(self):
        for batch in training_batches(self.sequences, self.size, self.multiplier):
            yield [self.indices[id(sequence)] for sequence in batch]

    def __len__(self):
        return (len(self.sequences) + self.size - 1) // self.size


class KTDataModule(pl.LightningDataModule):
    def __init__(self, train, valid, config):
        super().__init__()
        self.train_set, self.valid_set, self.config = train, valid, config

    def train_dataloader(self):
        config = self.config
        sampler = LengthBuckets(self.train_set, get_batch_size(config),
                                config["length_bucket_multiplier"])
        return DataLoader(self.train_set, batch_sampler=sampler, collate_fn=materialize_sequences,
                          num_workers=config["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=get_batch_size(self.config),
                          shuffle=False, collate_fn=materialize_sequences, num_workers=self.config["num_workers"])
