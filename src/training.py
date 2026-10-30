"""Reproducible transfer-learning training with validation-based checkpointing."""

from __future__ import annotations

import json
import math
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import TrainingConfig
from src.data.common import DatasetBundle, ImageRecord
from src.datasets import MedicalImageDataset
from src.models import build_model, set_backbone_trainable
from src.preprocessing import build_transform
from src.reproducibility import seed_everything, select_device

LOGGER = logging.getLogger(__name__)


def _class_weights(records: list[ImageRecord], classes: list[str], task_type: str, device: torch.device):
    if task_type == "multilabel":
        targets = np.asarray([record.target for record in records], dtype=np.float32)
        positives = targets.sum(axis=0)
        negatives = len(records) - positives
        ratio = np.divide(negatives, positives, out=np.ones_like(positives), where=positives > 0)
        return torch.tensor(np.clip(ratio, 1.0, 100.0), dtype=torch.float32, device=device)
    targets = np.asarray([int(record.target) for record in records], dtype=int)
    counts = np.bincount(targets, minlength=len(classes)).astype(float)
    present = counts > 0
    weights = np.ones(len(classes), dtype=np.float32)
    if present.any():
        weights[present] = len(targets) / (present.sum() * counts[present])
        weights /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loader(records, task_type, config: TrainingConfig, training: bool) -> DataLoader:
    dataset = MedicalImageDataset(
        records,
        task_type,
        build_transform(config.image_size, training=training, augment=training and config.augment),
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator if training else None,
    )


def _epoch(model, loader, loss_function, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        count = images.shape[0]
        total_loss += float(loss.detach().item()) * count
        total_count += count
    return total_loss / max(total_count, 1)


def train_model(
    bundle: DatasetBundle,
    splits: dict[str, list[ImageRecord]],
    architecture: str,
    config: TrainingConfig,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Train on the train partition, select by validation loss, then save best weights."""
    if not splits.get("train") or not splits.get("validation") or not splits.get("test"):
        raise ValueError("Train, validation, and test partitions must all contain records")
    seed_everything(config.seed)
    device = select_device()
    model = build_model(architecture, len(bundle.classes), pretrained=config.use_pretrained_weights).to(device)
    set_backbone_trainable(model, trainable=config.freeze_backbone_epochs <= 0)
    class_weights = _class_weights(splits["train"], bundle.classes, bundle.task_type, device)
    if bundle.task_type == "multilabel":
        loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    train_loader = _loader(splits["train"], bundle.task_type, config, True)
    validation_loader = _loader(splits["validation"], bundle.task_type, config, False)
    best_loss = math.inf
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        set_backbone_trainable(model, trainable=epoch >= config.freeze_backbone_epochs)
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("No trainable parameters remain")
        training_loss = _epoch(model, train_loader, loss_function, device, optimizer)
        validation_loss = _epoch(model, validation_loader, loss_function, device)
        scheduler.step(validation_loss)
        history.append({"epoch": float(epoch + 1), "train_loss": training_loss, "validation_loss": validation_loss})
        LOGGER.info(
            "Epoch %d/%d - train_loss=%.5f validation_loss=%.5f",
            epoch + 1,
            config.epochs,
            training_loss,
            validation_loss,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "task": bundle.task,
        "task_type": bundle.task_type,
        "classes": bundle.classes,
        "architecture": architecture,
        "image_size": config.image_size,
        "threshold": 0.5,
        "best_validation_loss": best_loss,
        "config": asdict(config),
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
    return {
        "model": model,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "history": history,
    }


def write_training_history(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")

