"""Display Lightning validation or test metrics for the selected model."""
import argparse
import json
from pathlib import Path

import torch

from src.common.config import load_config
from src.evaluation import evaluate_selected_model, run_formal_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/common/config.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--result-file", type=Path, help="Save evaluation metrics as JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["random_seed"] = args.seed
    if config["device"] == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if args.split == "validation":
        result = evaluate_selected_model(config, checkpoint=args.checkpoint)
    else:
        result = run_formal_test(config, checkpoint=args.checkpoint)
    if args.result_file is not None:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
