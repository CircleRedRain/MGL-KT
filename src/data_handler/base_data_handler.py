"""Compact answer cache and random student split of complete interaction sequences."""
from abc import ABC, abstractmethod
from collections import Counter
from types import SimpleNamespace

import numpy as np

from ..common.component import RecordRows, StudentSequence
from ..common.paths import records_cache_dir
from .compact_store import CompactBuilder, CompactStore


def split_students(student_ids, ratios, seed):
    students = sorted(student_ids)
    shuffled = np.random.default_rng(seed).permutation(students)
    train_end = int(len(students) * ratios["train"])
    valid_end = train_end + int(len(students) * ratios["valid"])
    return {**{str(uid): "train" for uid in shuffled[:train_end]},
            **{str(uid): "valid" for uid in shuffled[train_end:valid_end]},
            **{str(uid): "test" for uid in shuffled[valid_end:]}}


class BaseDataHandler(ABC):
    def __init__(self, config, rebuild=False):
        self.config, self.data_config = config, config[config["dataset"]]
        self.cache_dir = records_cache_dir(config)
        self.cache_file = self.cache_dir / "metadata.json"
        if rebuild or not self.cache_file.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.counts = Counter()
            self._builder = CompactBuilder()
            self._filter_student, self._filter_order = None, None
            self._filter_answers, self._filter_pending = 0, []
            self.load_raw_data()
            self._finish_student()
            self.counts["actual_answers"] = len(self._builder.events["question"])
            self.counts["expanded_rows"] = len(self._builder.annotations["concept"])
            self.counts["answers_before_student_filter"] = (
                self.counts["actual_answers"] + self.counts["filtered_short_answers"])
            self.counts["expanded_rows_before_student_filter"] = (
                self.counts["expanded_rows"] + self.counts["filtered_short_annotations"])
            metadata, splits = self.split_data()
            self._builder.write(self.cache_dir, metadata, splits)
            del self._builder
        self.load_cache()

    @abstractmethod
    def load_raw_data(self):
        """Stream ordered answers into the numeric builder."""

    def add_answer(self, *args, **kwargs):
        student, order = args[0], args[5]
        if student != self._filter_student:
            self._finish_student()
            self._filter_student = student
        if order != self._filter_order:
            self._filter_answers += 1
            self._filter_order = order
        if self._filter_answers < self.config["min_student_interactions"]:
            self._filter_pending.append((args, kwargs))
            return
        # Buffer only the initial short prefix. Once eligible, stream the entire
        # original history, including that prefix, into the numeric cache.
        if self._filter_pending:
            for pending_args, pending_kwargs in self._filter_pending:
                self._builder.add_answer(*pending_args, **pending_kwargs)
            self._filter_pending.clear()
        self._builder.add_answer(*args, **kwargs)

    def _finish_student(self):
        if self._filter_student is not None and self._filter_answers < self.config["min_student_interactions"]:
            self.counts["filtered_short_students"] += 1
            self.counts["filtered_short_answers"] += self._filter_answers
            self.counts["filtered_short_annotations"] += sum(len(args[2]) for args, _ in self._filter_pending)
        self._filter_student, self._filter_order = None, None
        self._filter_answers, self._filter_pending = 0, []

    def protocol(self):
        return {"data_handler": "compact_answers", "dataset": self.config["dataset"],
                **{key: self.config[key] for key in ("random_seed", "split_ratios", "warmup_k",
                                                   "min_student_interactions")},
                "data_config": self.data_config}

    def split_data(self):
        builder, config = self._builder, self.config
        event_offsets = np.asarray(builder.event_offsets)
        student_offsets = np.r_[builder.student_offsets, len(builder.events["question"])].astype(np.int64)
        lengths = np.diff(event_offsets[student_offsets])
        eligible = [f"U_{raw}" for raw, index in builder.maps["students"].items()
                    if lengths[index - 1] >= max(2, config["warmup_k"])]
        membership = split_students(eligible, config["split_ratios"], config["random_seed"])
        splits = {"train": [], "valid": [], "test": []}
        for raw, index in builder.maps["students"].items():
            if f"U_{raw}" in membership:
                splits[membership[f"U_{raw}"]].append(index)
        split_counts = {}
        for name, students in splits.items():
            selected = lengths[np.asarray(students, dtype=np.int64) - 1]
            split_counts.update({f"{name}_students": len(students), f"{name}_sequences": len(students),
                f"{name}_answers": int(selected.sum()),
                f"{name}_targets": int((selected - max(1, config["warmup_k"] - 1)).sum())})
        records = len(builder.annotations["concept"])
        self.counts.update(students=len(lengths), records=records, eligible_students=len(membership),
            students_without_targets=len(lengths) - len(membership),
            records_without_targets=records - sum(split_counts[f"{name}_answers"] for name in splits))
        metadata = {"protocol": self.protocol(), "counts": dict(self.counts), "split_counts": split_counts,
                    "n_questions": len(builder.maps["questions"]) + 1,
                    "n_concepts": len(builder.maps["concepts"]) + 1, "max_concepts": 1}
        return metadata, splits

    def load_cache(self):
        self.store = CompactStore(self.cache_dir)
        if self.store.metadata["protocol"] != self.protocol():
            raise ValueError("Cached data protocol differs from config; rerun python prepare.py")
        self.metadata, self.id_maps = self.store.metadata, self.store.id_maps
        self.counts = Counter(self.metadata["counts"])
        self._train_records, self._val_records, self._test_records = (
            [StudentSequence(self.store, student) for student in self.store.splits[name]]
            for name in ("train", "valid", "test"))

    @property
    def records(self):
        return RecordRows(self.store, 0, len(self.store.annotations))

    @property
    def users(self):
        return {f"U_{raw}": SimpleNamespace(record_indices=range(*self.store.annotation_bounds(index)))
                for raw, index in self.id_maps["students"].items()}

    @property
    def train_records(self):
        return self._train_records

    @property
    def val_records(self):
        return self._val_records

    @property
    def test_records(self):
        return self._test_records
