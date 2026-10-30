"""Train and evaluate one configured task/model using local supplied data."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from config import DEFAULT_TRAINING, DATA_ROOT
from src.data import load_aptos, load_chest_xray14, load_isic_manifest
from src.models import SUPPORTED_ARCHITECTURES
from src.pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a medical image research classifier")
    parser.add_argument("--task", choices=("chest_xray14", "aptos_retinopathy", "isic_skin_lesion"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional ISIC CSV manifest")
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default="densenet121")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAINING.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_TRAINING.learning_rate)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING.seed)
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize backbone without downloaded ImageNet weights")
    parser.add_argument("--augment", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    defaults = {
        "chest_xray14": DATA_ROOT / "raw" / "chest_xray14",
        "aptos_retinopathy": DATA_ROOT / "raw" / "aptos",
        "isic_skin_lesion": DATA_ROOT / "raw" / "isic2016",
    }
    root = args.data_root or defaults[args.task]
    if args.task == "chest_xray14":
        bundle = load_chest_xray14(root)
    elif args.task == "aptos_retinopathy":
        bundle = load_aptos(root)
    else:
        bundle = load_isic_manifest(root, args.manifest)
    config = replace(
        DEFAULT_TRAINING,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_pretrained_weights=not args.no_pretrained,
        augment=args.augment,
    )
    metrics = run_experiment(bundle, args.architecture, config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

