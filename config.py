"""Project paths and default training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_ROOT = PROJECT_ROOT / "models" / "saved_models"
RESULTS_ROOT = PROJECT_ROOT / "reports" / "generated"


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    use_pretrained_weights: bool = True
    freeze_backbone_epochs: int = 1
    augment: bool = False


DEFAULT_TRAINING = TrainingConfig()

