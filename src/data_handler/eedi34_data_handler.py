"""Eedi Task 3/4 chronological answers with Level-3 concept annotations."""
import ast
import csv
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from ..common.paths import project_path
from .base_data_handler import BaseDataHandler


class Eedi34DataHandler(BaseDataHandler):
    def _rows(self, key, required, check_values=True):
        with project_path(self.data_config[key]).open(
                encoding=self.data_config["encoding"], newline="") as file:
            reader = csv.DictReader(file)
            missing = set(required) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing Eedi {key} columns: {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                if check_values and any(not row[key].strip() for key in required):
                    raise ValueError(f"Missing Eedi {key} value at source line {line}")
                yield line, row

    def load_raw_data(self):
        subjects = {row["SubjectId"]: (row["Name"], int(row["Level"]))
                    for _, row in self._rows("kcs_data", ("SubjectId", "Name", "Level"))}
        question_kcs = {}
        for line, row in self._rows("question_kc_data", ("QuestionId", "SubjectId")):
            raw = ast.literal_eval(row["SubjectId"])
            if not isinstance(raw, list) or any(str(kc) not in subjects for kc in raw):
                raise ValueError(f"Invalid Eedi subjects at question metadata line {line}")
            # Filter by declared level, not list position or the largest ID.
            concepts = list(dict.fromkeys(str(kc) for kc in raw if subjects[str(kc)][1] == 3))
            if not concepts:
                if self.data_config.get("missing_level3") != "exclude":
                    raise ValueError(f"No Level-3 subject at Eedi question metadata line {line}")
                self.counts["questions_without_level3"] += 1
            question_kcs[row["QuestionId"]] = concepts
        self.counts["metadata_questions"] = len(question_kcs)

        def answers():
            for line, row in self._rows("records_data", ("QuestionId", "UserId", "AnswerId", "IsCorrect")):
                self.counts["raw_rows"] += 1
                if row["IsCorrect"] not in ("0", "1"):
                    raise ValueError(f"Nonbinary Eedi IsCorrect at source line {line}")
                if row["QuestionId"] not in question_kcs:
                    raise ValueError(f"Unmapped Eedi question at source line {line}")
                if not question_kcs[row["QuestionId"]]:
                    self.counts["selected_answers_excluded_without_level3"] += 1
                    continue
                # AnswerValue and CorrectAnswer are options, not binary labels.
                yield row["AnswerId"], row["UserId"], row["QuestionId"], int(row["IsCorrect"]), line

        epoch = datetime(1970, 1, 1)

        @lru_cache(maxsize=65536)
        def milliseconds(value):
            delta = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f") - epoch
            return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000

        def dates():
            for line, row in self._rows("answer_metadata", ("AnswerId", "DateAnswered")):
                self.counts["answer_metadata_rows"] += 1
                if not row["AnswerId"].strip():
                    # An unkeyed metadata row cannot refer to any selected answer.
                    self.counts["answer_metadata_without_id"] += 1
                    continue
                answer = row["AnswerId"]
                if not row["DateAnswered"].strip():
                    raise ValueError(f"Missing Eedi answer time for AnswerId {answer} "
                                     f"at metadata source line {line}")
                self.counts["joined_answer_metadata_rows"] += 1
                yield answer, milliseconds(row["DateAnswered"])

        # Input answers are shuffled across students. Join and sort on disk,
        # without retaining all raw CSV dictionaries or student lists in RAM.
        with TemporaryDirectory(prefix=f"{self.config['dataset']}_") as folder:
            connection = sqlite3.connect(str(Path(folder) / "answers.sqlite"))
            try:
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("CREATE TABLE answers (answer TEXT PRIMARY KEY, student TEXT, "
                                   "question TEXT, response INTEGER, line INTEGER)")
                connection.execute("CREATE TABLE dates (answer TEXT PRIMARY KEY, time INTEGER)")
                connection.executemany("INSERT INTO answers VALUES (?,?,?,?,?)", answers())
                connection.executemany("INSERT INTO dates VALUES (?,?)", dates())
                missing = connection.execute("SELECT a.answer, a.line FROM answers a LEFT JOIN dates d "
                    "ON a.answer=d.answer WHERE d.answer IS NULL LIMIT 1").fetchone()
                if missing:
                    raise ValueError(f"Missing Eedi answer time for AnswerId {missing[0]} at line {missing[1]}")
                cursor = connection.execute("SELECT a.answer,a.student,a.question,a.response,a.line,d.time "
                    "FROM answers a JOIN dates d ON a.answer=d.answer "
                    "ORDER BY a.student,d.time,CAST(a.answer AS INTEGER),a.line")
                for answer, student, question, response, line, timestamp in cursor:
                    concepts = question_kcs[question]
                    self.add_answer(student, question, concepts, [subjects[kc][0] for kc in concepts],
                        response, answer, line, source_file=self.data_config["records_data"],
                        source_position=line - 2, timestamp=timestamp)
                    self.counts["actual_answers"] += 1
                    self.counts["multi_kc_answers"] += len(concepts) > 1
                    self.counts["max_kcs_per_answer"] = max(self.counts["max_kcs_per_answer"], len(concepts))
            finally:
                connection.close()
        self.counts["expanded_rows"] = len(self._builder.annotations["concept"])
