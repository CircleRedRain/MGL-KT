import json
from pathlib import Path

from .eedi34_data_handler import Eedi34DataHandler
from .assist17_data_handler import Assist17DataHandler
from .xes3g5m_data_handler import XES3G5MDataHandler
from .relations import load_relations
from ..common.paths import relation_path


def load_configured_relations(config, id_maps, report_path=None):
    settings = config[config["dataset"]]
    if settings["kc_relation_data"] is None:
        report = {"source_path": None, "status": "not_provided",
                  "counts": {"mapped_edges": 0}, "mapped_edges": [], "mapped_type_counts": {}}
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return [], report
    return load_relations(relation_path(config), id_maps, report_path,
                          column_names=settings.get("relation_columns"),
                          relation_aliases=settings.get("relation_aliases"),
                          node_key=settings.get("relation_node_key", "id"))


class DataRouter:
    handlers = {
        "assist2017": Assist17DataHandler,
        "eedi34": Eedi34DataHandler,
        "xes3g5m": XES3G5MDataHandler,
    }

    def __init__(self, config):
        self.config = config

    def create(self, rebuild=False):
        name = self.config["dataset"]
        if name not in self.handlers:
            raise ValueError(f"Unsupported dataset: {name}")
        return self.handlers[name](self.config, rebuild=rebuild)


def load_model_inputs(config):
    handler = DataRouter(config).create()
    maps = handler.id_maps
    edges, report = load_configured_relations(config, maps)
    pairs = handler.store.training_pairs()
    return handler, handler.metadata, maps, edges, report, pairs
