"""Lazy complete histories and segment views; materialize only a loader batch."""
from torch.utils.data import Dataset

from ..common.component import InteractionRows


def materialize_sequences(sequences):
    return [{**sequence, "interactions": list(sequence["interactions"])} for sequence in sequences]


def segment_sequences(sequences, max_length, warmup_k):
    """Nonoverlapping segments; one real answer's annotations stay together.

    Each segment is an independent model sample. The complete source sequences
    remain in the data-handler cache, including tails without prediction targets.
    """
    if max_length < max(2, warmup_k):
        raise ValueError("max_sequence_length must be at least max(2, warmup_k)")
    result = []
    for sequence in sequences:
        items = sequence["interactions"]
        if isinstance(items, InteractionRows):
            boundaries = items.answer_boundaries()
        else:
            boundaries = [0] + [i for i in range(1, len(items))
                                if items[i].get("order_id", i) != items[i - 1].get("order_id", i - 1)]
            boundaries.append(len(items))
        start, segment_index = 0, 0

        def append(end):
            nonlocal start, segment_index
            if end - start >= max(2, warmup_k):
                result.append({**sequence, "interactions": items[start:end],
                               "segment_index": segment_index,
                               "sequence_start": start, "sequence_end": end})
            start = end
            segment_index += 1

        for left, right in zip(boundaries, boundaries[1:]):
            if left > start and right - start > max_length:
                append(left)
        append(len(items))
    return result


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [sequence.as_sequence() for sequence in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


