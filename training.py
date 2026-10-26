"""Train with Lightning and select models by validation AUC."""
import argparse

import torch

from src.common.config import load_config, set_seed
from src.models.lightning_module import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["random_seed"] = args.seed
    if config["device"] == "cpu":
        torch.set_num_threads(args.cpu_threads)
    set_seed(config.get("training_seed", config["random_seed"]))
    torch.set_float32_matmul_precision("high")
    run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
