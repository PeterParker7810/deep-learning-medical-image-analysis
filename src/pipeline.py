"""End-to-end training, held-out testing, and results persistence."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_ROOT, MODEL_ROOT, PROJECT_ROOT, RESULTS_ROOT, TrainingConfig
from src.data.common import DatasetBundle
from src.data.splitting import split_bundle
from src.datasets import MedicalImageDataset
from src.evaluation import evaluate_predictions
from src.preprocessing import build_transform
from src.training import train_model, write_training_history
from src.visualizations import create_result_figures


def _evaluate_partition(bundle, records, model, config, device) -> dict[str, Any]:
    dataset = MedicalImageDataset(records, bundle.task_type, build_transform(config.image_size))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    truths, scores = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits) if bundle.task_type == "multilabel" else torch.softmax(logits, dim=1)
            truths.append(targets.numpy())
            scores.append(probabilities.cpu().numpy())
    return evaluate_predictions(np.concatenate(truths), np.concatenate(scores), bundle.classes, bundle.task_type)


def run_experiment(
    bundle: DatasetBundle,
    architecture: str,
    config: TrainingConfig,
    *,
    output_root: str | Path = RESULTS_ROOT,
) -> dict[str, Any]:
    """Split once, train with validation, score the held-out test partition once."""
    output_root = Path(output_root)
    run_dir = output_root / bundle.task / architecture
    checkpoint_path = Path(MODEL_ROOT) / bundle.task / architecture / "model.pt"
    split = split_bundle(
        bundle,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    result = train_model(bundle, split, architecture, config, checkpoint_path)
    test_metrics = _evaluate_partition(bundle, split["test"], result["model"], config, result["device"])
    history_path = run_dir / "history.json"
    write_training_history(result["history"], history_path)
    metrics = {
        "task": bundle.task,
        "architecture": architecture,
        "dataset_records": len(bundle.records),
        "split_counts": {name: len(records) for name, records in split.items()},
        "class_counts": bundle.label_counts(),
        "missing_files": len(bundle.missing_files),
        "seed": config.seed,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.is_relative_to(PROJECT_ROOT) else checkpoint_path.name,
        "best_validation_loss": result["best_validation_loss"],
        "epochs_completed": result["epochs_completed"],
        "metrics": test_metrics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    split_manifest = {}
    for name, records in split.items():
        entries = []
        for record in records:
            try:
                relative_path = record.path.resolve().relative_to(DATA_ROOT.resolve())
            except ValueError:
                relative_path = Path(record.path.name)
            group_hash = hashlib.sha256(record.group_id.encode("utf-8")).hexdigest() if record.group_id else None
            entries.append({"path": str(relative_path), "group_hash": group_hash})
        split_manifest[name] = entries
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    create_result_figures(run_dir)
    return metrics

