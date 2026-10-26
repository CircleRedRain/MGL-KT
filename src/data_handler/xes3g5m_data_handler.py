"""XES3G5M question sequences, reconstructed before random student splitting."""
import csv
import json
from collections import Counter

from ..common.paths import project_path
from .base_data_handler import BaseDataHandler


class XES3G5MDataHandler(BaseDataHandler):
    def load_raw_data(self):
        with project_path(self.data_config["kcs_data"]).open(
                encoding=self.data_config["encoding"]) as file:
            names = json.load(file)
        answer_positions = Counter()
        last_timestamp = {}
        columns = ("questions", "concepts", "responses", "timestamps")
        for source in self.data_config["records_data"]:
            path = project_path(source)
            with path.open(encoding=self.data_config["encoding"], newline="") as file:
                reader = csv.DictReader(file)
                for line, row in enumerate(reader, 2):
                    self.counts["raw_sequence_rows"] += 1
                    uid = row["uid"]
                    values = {key: row[key].split(",") for key in columns}
                    if "selectmasks" in row:
                        values["selectmasks"] = row["selectmasks"].split(",")
                    if len({len(value) for value in values.values()}) != 1:
                        raise ValueError(f"Sequence field length mismatch at {source}:{line}")
                    for position, question in enumerate(values["questions"]):
                        if question == "-1":
                            if any(value[position] != "-1" for value in values.values()):
                                raise ValueError(f"Inconsistent padding at {source}:{line}, position {position}")
                            self.counts["padding_positions"] += 1
                            continue
                        answer = values["responses"][position]
                        if answer not in ("0", "1"):
                            raise ValueError(f"Nonbinary response at {source}:{line}, position {position}")
                        timestamp = int(values["timestamps"][position])
                        if uid in last_timestamp and timestamp < last_timestamp[uid]:
                            raise ValueError(f"Time order reversed at {source}:{line}, position {position}")
                        last_timestamp[uid] = timestamp
                        kcs = values["concepts"][position].split("_")
                        order = str(answer_positions[uid])
                        for kc in kcs:
                            if kc not in names:
                                raise ValueError(f"Unknown KC {kc!r} at {source}:{line}, position {position}")
                        self.add_answer(uid, question, kcs, [names[kc] for kc in kcs],
                            int(answer), order, line, source_file=source,
                            source_position=position, timestamp=timestamp)
                        answer_positions[uid] += 1
                        self.counts["actual_answers"] += 1
                        self.counts["multi_kc_answers"] += len(kcs) > 1
                        self.counts["max_kcs_per_answer"] = max(self.counts["max_kcs_per_answer"], len(kcs))
        self.counts["source_files"] = len(self.data_config["records_data"])
        self.counts["metadata_kcs"] = len(names)
        self.counts["expanded_rows"] = len(self._builder.annotations["concept"])
