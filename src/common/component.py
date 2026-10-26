"""Lightweight views into compact arrays; no per-interaction objects are cached."""
from collections.abc import Sequence

import numpy as np


class RecordView:
    __slots__ = ("store", "index", "event_index", "student")

    def __init__(self, store, index, event_index, student):
        self.store, self.index, self.event_index, self.student = store, index, event_index, student

    @property
    def event(self):
        return self.store.events[self.event_index]

    @property
    def annotation(self):
        return self.store.annotations[self.index]

    @property
    def u_id(self):
        return "U_" + self.store.id_maps["student_index_to_raw"][str(self.student)]

    @property
    def question_index(self):
        return int(self.event["question"])

    @property
    def concept_index(self):
        return int(self.annotation["concept"])

    @property
    def q_id(self):
        return "Q_" + self.store.id_maps["question_index_to_raw"][str(self.question_index)]

    @property
    def kc_id(self):
        return "K_" + self.store.id_maps["concept_index_to_raw"][str(self.concept_index)]

    @property
    def kc_name(self):
        return self.store.names[self.concept_index - 1]

    @property
    def is_correct(self):
        return int(self.event["response"])

    @property
    def order_id(self):
        return self.store.orders[self.event["order"]]

    @property
    def source_row(self):
        return int(self.annotation["source_row"])

    @property
    def att_count(self):
        value = int(self.annotation["attempts"])
        return None if value == -1 else value

    @property
    def hint_count(self):
        value = int(self.annotation["hints"])
        return None if value == -1 else value

    @property
    def original_position(self):
        return self.index - self.store.annotation_bounds(self.student)[0]

    @property
    def source_file(self):
        return self.store.sources[self.event["source_file"]]

    @property
    def source_position(self):
        return int(self.event["source_position"])

    @property
    def timestamp(self):
        return int(self.event["timestamp"])

    def as_interaction(self):
        item = {"question": self.question_index, "question_raw": self.q_id[2:],
                "concepts": [self.concept_index], "concepts_raw": [self.kc_id[2:]],
                "response": self.is_correct, "order_id": self.order_id,
                "original_position": self.original_position, "source_rows": [self.source_row]}
        if self.source_position >= 0:
            item.update(source_file=self.source_file, source_position=self.source_position,
                        timestamp=self.timestamp)
        return item


class RecordRows(Sequence):
    def __init__(self, store, start, end, student=None):
        self.store, self.start, self.end, self.student = store, start, end, student

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, end, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, end, step)]
            return type(self)(self.store, self.start + start, self.start + max(start, end), self.student)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        absolute = self.start + index
        event = int(np.searchsorted(self.store.event_offsets, absolute, side="right") - 1)
        student = self.student or int(np.searchsorted(self.store.student_offsets, event, side="right"))
        return RecordView(self.store, absolute, event, student)

    def __iter__(self):
        offsets = self.store.event_offsets
        first = int(np.searchsorted(offsets, self.start, side="right") - 1)
        for event in range(first, len(offsets) - 1):
            if offsets[event] >= self.end:
                break
            student = self.student or int(np.searchsorted(self.store.student_offsets, event, side="right"))
            for index in range(max(self.start, int(offsets[event])), min(self.end, int(offsets[event + 1]))):
                yield RecordView(self.store, index, event, student)


class InteractionRows(RecordRows):
    def __getitem__(self, index):
        record = super().__getitem__(index)
        return record if isinstance(index, slice) else record.as_interaction()

    def __iter__(self):
        return (record.as_interaction() for record in super().__iter__())

    def answer_boundaries(self):
        offsets = self.store.event_offsets
        start = np.searchsorted(offsets, self.start, side="right")
        end = np.searchsorted(offsets, self.end, side="left")
        return [0, *(offsets[start:end] - self.start).tolist(), len(self)]


class StudentSequence:
    __slots__ = ("store", "student_index")

    def __init__(self, store, student_index):
        self.store, self.student_index = store, int(student_index)

    @property
    def u_id(self):
        return "U_" + self.store.id_maps["student_index_to_raw"][str(self.student_index)]

    @property
    def records(self):
        return RecordRows(self.store, *self.store.annotation_bounds(self.student_index), self.student_index)

    def as_sequence(self):
        return {"student": self.student_index, "student_raw": self.u_id[2:],
                "interactions": InteractionRows(self.store, *self.store.annotation_bounds(self.student_index),
                                                 self.student_index)}
