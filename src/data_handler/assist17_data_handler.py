"""ASSIST2017 chronological answers with the supplied skill-name/ID table."""
import csv
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

import pandas as pd

from ..common.paths import project_path
from .base_data_handler import BaseDataHandler


class Assist17DataHandler(BaseDataHandler):
    def _rows(self, skill_ids):
        columns = ["studentId", "action_num", "problemId", "skill", "startTime",
                   "correct", "attemptCount", "hintCount"]
        position = 0
        with pd.read_csv(project_path(self.data_config["records_data"]),
                usecols=columns, dtype=str, keep_default_na=False,
                encoding=self.data_config["encoding"], chunksize=100000) as reader:
            for data in reader:
                data["source_row"] = range(position + 2, position + len(data) + 2)
                position += len(data)
                self.counts["raw_rows"] += len(data)
                for key in columns:
                    data[key] = data[key].str.strip()
                no_skill = data["skill"] == "noskill"
                self.counts["noskill_rows"] += int(no_skill.sum())
                data = data.loc[~no_skill].copy()
                required = ["studentId", "action_num", "problemId", "skill", "startTime", "correct"]
                missing = (data[required] == "").any(axis=1)
                self.counts["missing_required_rows"] += int(missing.sum())
                data = data.loc[~missing].copy()
                unknown = ~data["skill"].isin(skill_ids)
                if unknown.any():
                    row = data.loc[unknown].iloc[0]
                    raise ValueError(f"Unmapped ASSIST2017 skill {row['skill']!r} at source line {row.source_row}")
                scores = pd.to_numeric(data["correct"], errors="raise")
                if (scores > 1).any():
                    raise ValueError("ASSIST2017 correct must not exceed 1")
                self.counts["partial_scores_mapped_to_zero"] += int(((scores < 1) & (scores != 0)).sum())
                data["correct"] = scores.mask(scores < 1, 0).astype("int64")
                data["timestamp"] = pd.to_numeric(data["startTime"], errors="raise").astype("int64") * 1000
                self.counts["rows_with_required_fields"] += len(data)
                for row in data.itertuples(index=False):
                    yield (row.studentId, int(row.timestamp), row.action_num, int(row.source_row),
                           row.problemId, skill_ids[row.skill], row.skill, int(row.correct),
                           int(row.attemptCount) if row.attemptCount else None,
                           int(row.hintCount) if row.hintCount else None)

    def load_raw_data(self):
        # The raw release has skill strings but no numeric skill IDs. The
        # supplied relation file explicitly records the concept IDs.
        with project_path(self.data_config["kc_relation_data"]).open(
                encoding=self.data_config["encoding"], newline="") as file:
            skill_ids = {row["skill_name"].strip(): row["skill_id"].strip()
                         for row in csv.DictReader(file) if row["skill_name"].strip() != "noskill"}
        source = self.data_config["records_data"]
        with TemporaryDirectory(prefix="assist2017_") as folder:
            connection = sqlite3.connect(str(Path(folder) / "answers.sqlite"))
            try:
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("CREATE TABLE answers (student TEXT,time INTEGER,answer TEXT,"
                    "line INTEGER,question TEXT,concept TEXT,name TEXT,label INTEGER,attempts INTEGER,hints INTEGER)")
                connection.executemany("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?)", self._rows(skill_ids))
                cursor = connection.execute("SELECT * FROM answers ORDER BY student,time,CAST(answer AS INTEGER),line")
                for student, timestamp, order, line, question, concept, name, label, attempts, hints in cursor:
                    self.add_answer(student, question, [concept], [name], label, order,
                        line, attempts=attempts, hints=hints, source_file=source,
                        source_position=line - 2, timestamp=timestamp)
            finally:
                connection.close()
        self.counts["actual_answers"] = len(self._builder.events["question"])
        self.counts["expanded_rows"] = len(self._builder.annotations["concept"])
