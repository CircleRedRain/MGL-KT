"""Parse only the supplied typed, directed knowledge-point relations."""
import csv
import json
from collections import Counter
from pathlib import Path


RELATION_TYPES = ("Successor", "Associative")


def load_relations(path, id_maps, report_path=None, column_names=None, relation_aliases=None,
                   node_key="id"):
    """Return mapped edges and a row-traceable cleaning report.

    Lists are split before validation. Empty positions are errors, never removed
    independently from the target and type lists.
    """
    path = Path(path)
    mapping = id_maps["concepts"]
    if node_key not in ("id", "name"):
        raise ValueError(f"Unknown relation node key: {node_key}")
    node_names = {}
    isolated_rows = []
    seen = {}
    duplicates = []
    self_loops = []
    counts = Counter()
    column_names = column_names or {}
    relation_aliases = relation_aliases or {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"kc_id", "kc_name", "related_kc_id", "relationship"}
        expected = {column_names.get(name, name) for name in required}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing relationship fields: {sorted(expected - set(reader.fieldnames or []))}")
        for line, raw_row in enumerate(reader, 2):
            row = {name: raw_row[column_names.get(name, name)] for name in required}
            counts["rows"] += 1
            source = row["kc_id"].strip()
            targets = row["related_kc_id"].split(";")
            types = row["relationship"].split(";")
            if len(targets) != len(types):
                raise ValueError(f"Relationship list length mismatch at raw line {line}: "
                                 f"{len(targets)} targets versus {len(types)} types")
            if not source:
                raise ValueError(f"Empty kc_id at raw line {line}")
            name = row["kc_name"].strip()
            if node_key == "name" and source in node_names and node_names[source] != name:
                raise ValueError(f"Conflicting name for kc_id {source!r} at raw line {line}")
            node_names[source] = name
            # Both whole fields empty declare no outgoing edges. Empty list
            # positions inside otherwise populated lists remain errors below.
            if not row["related_kc_id"].strip() and not row["relationship"].strip():
                isolated_rows.append({"source_raw": source, "kc_name": name, "source_line": line})
                continue
            for position in range(len(targets)):
                target = targets[position].strip()
                raw_relation = types[position].strip()
                relation = relation_aliases.get(raw_relation, raw_relation)
                if not target or not raw_relation:
                    raise ValueError(f"Empty relationship element at raw line {line}, position {position + 1}")
                if relation not in RELATION_TYPES:
                    raise ValueError(f"Unknown relationship {relation!r} at raw line {line}, position {position + 1}")
                counts[f"raw_{raw_relation}"] += 1
                key = (source, target, relation)
                entry = {"source_raw": source, "target_raw": target,
                         "relation_type": relation, "source_line": line, "position": position + 1}
                if raw_relation != relation:
                    entry["original_relation_type"] = raw_relation
                if source == target:
                    self_loops.append(entry)
                    continue
                if key in seen:
                    duplicates.append({**entry, "first_source_line": seen[key]["source_line"]})
                    continue
                seen[key] = entry
    mapped, unmatched = [], []
    unmatched_nodes = set()
    for entry in seen.values():
        source_key = (node_names.get(entry["source_raw"]) if node_key == "name" else entry["source_raw"])
        target_key = (node_names.get(entry["target_raw"]) if node_key == "name" else entry["target_raw"])
        src = mapping.get(source_key)
        dst = mapping.get(target_key)
        if node_key == "name":
            entry = {**entry, "source_dataset_raw": source_key, "target_dataset_raw": target_key}
        if src is None or dst is None:
            unmatched.append(entry)
            if src is None:
                unmatched_nodes.add(entry["source_raw"])
            if dst is None:
                unmatched_nodes.add(entry["target_raw"])
            continue
        mapped.append({**entry, "source": src, "target": dst})
    if not mapped:
        raise ValueError("All knowledge relations were filtered by ID mapping; check dataset/version")
    report = {"source_path": str(path),
              "counts": {**counts, "deduplicated_nonself_edges": len(seen),
                         "duplicate_edges": len(duplicates), "self_loops": len(self_loops),
                         "mapped_edges": len(mapped), "unmatched_edges": len(unmatched),
                         "mapped_coverage": len(mapped) / len(seen),
                         "rows_without_outgoing_edges": len(isolated_rows)},
              "mapped_type_counts": dict(Counter(e["relation_type"] for e in mapped)),
              "unmatched_nodes": sorted(unmatched_nodes, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)),
              "unmatched_edges": unmatched, "duplicate_log": duplicates,
              "self_loop_log": self_loops, "isolated_rows": isolated_rows, "mapped_edges": mapped}
    if node_key == "name":
        matched_nodes = {raw: {"dataset_raw": name, "index": mapping[name]}
                         for raw, name in node_names.items() if name in mapping}
        report["node_mapping"] = {
            "method": "exact_name", "declared_nodes": len(node_names),
            "matched_nodes": len(matched_nodes),
            "coverage": len(matched_nodes) / len(node_names) if node_names else 0.0,
            "id_to_dataset": matched_nodes,
            "unmatched": {raw: name for raw, name in node_names.items() if name not in mapping}}
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapped, report
