"""Question-level losses and length buckets for complete student sequences."""
import random

import torch.nn.functional as F

from .question_evaluation import question_forward


def sequence_batches(sequences, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(sequences), batch_size):
        yield sequences[start:start + batch_size]


def training_batches(sequences, batch_size, bucket_multiplier):
    """Shuffle similar-length sequences within buckets, then shuffle whole batches."""
    if not isinstance(bucket_multiplier, int) or bucket_multiplier < 0:
        raise ValueError("length_bucket_multiplier must be a nonnegative integer")
    if bucket_multiplier == 0:
        return list(sequence_batches(sequences, batch_size))
    ordered = sorted(sequences, key=lambda seq: len(seq["interactions"]))
    width = batch_size * bucket_multiplier
    batches = []
    for start in range(0, len(ordered), width):
        bucket = ordered[start:start + width]
        random.shuffle(bucket)
        batches.extend(sequence_batches(bucket, batch_size))
    random.shuffle(batches)
    return batches


def progressive_batch_loss(model, sequences, config):
    if not sequences:
        raise ValueError("Empty training batch")
    labels, _, cache = question_forward(model, sequences, config["warmup_k"])
    loss = F.binary_cross_entropy_with_logits(cache["logits"]["question"], labels, reduction="none")
    total = loss.mean() if labels.numel() else loss.sum() * 0
    return total, {"counts": {"final": labels.numel()},
                   "losses": {"final": total.detach().item()}}
