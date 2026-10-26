"""Compact answer/annotation arrays; only read-only mappings are loaded at runtime."""
from array import array
import json

import numpy as np


EVENT_DTYPE = np.dtype([("question", "<i4"), ("response", "i1"), ("order", "<i4"),
                        ("source_file", "<i2"), ("source_position", "<i4"), ("timestamp", "<i8")])
ANNOTATION_DTYPE = np.dtype([("concept", "<i4"), ("source_row", "<i4"),
                             ("attempts", "<i4"), ("hints", "<i4")])


def collect_pairs(questions, concepts, event_offsets, student_offsets, students):
    pairs = set()
    for student in students:
        start, end = student_offsets[student - 1:student + 1]
        offsets = event_offsets[start:end + 1]
        expanded = np.repeat(questions[start:end], np.diff(offsets))
        pairs.update(zip(map(int, expanded), map(int, concepts[offsets[0]:offsets[-1]])))
    return sorted(pairs)


class CompactBuilder:
    """Adapters append ordered answers; repeated annotations share an answer row."""
    def __init__(self):
        self.events = {name: array(code) for name, code in (
            ("question", "i"), ("response", "b"), ("order", "i"),
            ("source_file", "h"), ("source_position", "i"), ("timestamp", "q"))}
        self.annotations = {name: array("i") for name in ANNOTATION_DTYPE.names}
        self.event_offsets = array("q", [0])
        self.student_offsets = array("q")
        self.maps = {name: {} for name in ("students", "questions", "concepts")}
        self.names, self.orders, self.sources = [], {}, {}
        self.previous = None

    def add_answer(self, uid, question, concepts, names, response, order, source_row,
                   attempts=None, hints=None, source_file=None, source_position=-1, timestamp=-1):
        previous = self.previous
        if previous is None or previous[0] != uid:
            if uid in self.maps["students"]:
                raise ValueError(f"Student {uid} appears in noncontiguous source blocks")
            self.maps["students"][uid] = len(self.maps["students"]) + 1
            self.student_offsets.append(len(self.events["question"]))
        q = self.maps["questions"].setdefault(question, len(self.maps["questions"]) + 1)
        same_answer = previous is not None and previous[:2] == (uid, order)
        if same_answer and previous[2:] != (question, response):
            raise ValueError(f"Inconsistent question/answer for student {uid}, order {order}")
        if not same_answer:
            values = (q, response, self.orders.setdefault(order, len(self.orders)),
                      -1 if source_file is None else self.sources.setdefault(source_file, len(self.sources)),
                      source_position, timestamp)
            for name, value in zip(self.events, values):
                self.events[name].append(value)
        for kc, name in zip(concepts, names):
            if kc not in self.maps["concepts"]:
                self.maps["concepts"][kc] = len(self.maps["concepts"]) + 1
                self.names.append(name)
            values = (self.maps["concepts"][kc], source_row,
                      -1 if attempts is None else attempts, -1 if hints is None else hints)
            for field, value in zip(self.annotations, values):
                self.annotations[field].append(value)
        end = len(self.annotations["concept"])
        if same_answer:
            self.event_offsets[-1] = end
        else:
            self.event_offsets.append(end)
        self.previous = (uid, order, question, response)

    def write(self, directory, metadata, splits):
        directory.mkdir(parents=True, exist_ok=True)
        for filename, columns, dtype in (("events", self.events, EVENT_DTYPE),
                                          ("annotations", self.annotations, ANNOTATION_DTYPE)):
            values = np.lib.format.open_memmap(directory / f"{filename}.npy", mode="w+",
                dtype=dtype, shape=(len(next(iter(columns.values()))),))
            for name, column in columns.items():
                values[name] = np.frombuffer(column, dtype=np.dtype(column.typecode))
            values.flush()
            del values
        student_offsets = np.r_[self.student_offsets, len(self.events["question"])].astype(np.int64)
        np.save(directory / "event_offsets.npy", np.asarray(self.event_offsets, dtype=np.int64))
        np.save(directory / "student_offsets.npy", student_offsets)
        for name, students in splits.items():
            np.save(directory / f"{name}_students.npy", np.asarray(students, dtype=np.int32))
        pairs = collect_pairs(np.frombuffer(self.events["question"], dtype=np.int32),
            np.frombuffer(self.annotations["concept"], dtype=np.int32),
            np.asarray(self.event_offsets), student_offsets, splits["train"])
        np.save(directory / "training_pairs.npy", np.asarray(pairs, dtype=np.int32).reshape(-1, 2))
        maps = {**self.maps, "unknown_index": 0,
                **{f"{name}_index_to_raw": {str(v): k for k, v in self.maps[plural].items()}
                   for name, plural in (("student", "students"), ("question", "questions"),
                                         ("concept", "concepts"))}}
        payload = {"metadata": metadata, "id_maps": maps, "concept_names": self.names,
                   "order_ids": list(self.orders), "source_files": list(self.sources)}
        (directory / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class CompactStore:
    def __init__(self, directory):
        self.directory = directory
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.metadata, self.id_maps = payload["metadata"], payload["id_maps"]
        self.names, self.orders, self.sources = (payload[key] for key in (
            "concept_names", "order_ids", "source_files"))
        self.events = np.load(directory / "events.npy", mmap_mode="r")
        self.annotations = np.load(directory / "annotations.npy", mmap_mode="r")
        self.event_offsets = np.load(directory / "event_offsets.npy", mmap_mode="r")
        self.student_offsets = np.load(directory / "student_offsets.npy", mmap_mode="r")
        self.splits = {name: np.load(directory / f"{name}_students.npy", mmap_mode="r")
                       for name in ("train", "valid", "test")}
        self.pair_array = np.load(directory / "training_pairs.npy", mmap_mode="r")

    def __getstate__(self):
        # Spawned DataLoader workers reopen files instead of pickling array contents.
        return {"directory": self.directory}

    def __setstate__(self, state):
        self.__init__(state["directory"])

    def annotation_bounds(self, student):
        start, end = self.student_offsets[student - 1:student + 1]
        return int(self.event_offsets[start]), int(self.event_offsets[end])

    def training_pairs(self):
        return [tuple(map(int, row)) for row in self.pair_array]
