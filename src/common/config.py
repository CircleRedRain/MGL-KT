"""Project configuration, shared by preparation, training and evaluation."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def get_batch_size(config):
    return config["batch_size"][config["dataset"]]


def load_config(path=None):
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config["warmup_k"] < 1:
        raise ValueError("warmup_k must be positive")
    if config["min_student_interactions"] < 1:
        raise ValueError("min_student_interactions must be positive")
    if get_batch_size(config) < 1 or config["num_workers"] < 0:
        raise ValueError("Invalid batch_size or num_workers")
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
