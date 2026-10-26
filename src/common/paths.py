"""Paths configured per dataset and resolved from the project root."""
from pathlib import Path

from .config import PROJECT_ROOT


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def records_cache_dir(config):
    settings = config[config["dataset"]]
    return project_path(settings["records_cache_dir"].format(**config))


def relation_path(config):
    return project_path(config[config["dataset"]]["kc_relation_data"])


def training_dir(config):
    return (project_path(config["checkpoint_dir"]) / config["dataset"] /
            f"seed{config.get('training_seed', config['random_seed'])}" / "full")


def compatible_training_config(config):
    # The epoch budget may be extended when resuming an unfinished structure
    # search. It does not change the model, losses, data or freeze criteria.
    runtime_keys = {"evaluation", "device", "checkpoint_dir", "num_workers", "max_epochs",
                    "early_stopping"}
    result = {key: value for key, value in config.items() if key not in runtime_keys}
    # Cache representation/locations do not alter the learned model or data protocol.
    for key, value in result.items():
        if isinstance(value, dict):
            result[key] = {name: setting for name, setting in value.items()
                           if name not in {"records_cache_file", "records_cache_dir"}}
    return result
