"""Generate performance figures from saved, actual experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def create_result_figures(run_dir: str | Path) -> list[Path]:
    # Some packaged/headless environments omit Matplotlib's bundled DejaVu
    # font data. Prefer an installed font and fail clearly if none are usable.
    for font_name in ("Arial", "DejaVu Sans", "Liberation Sans"):
        try:
            font_manager.findfont(font_name, fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams["font.family"] = font_name
        # Matplotlib 3.11 builds may omit LastResortHE from mpl-data while
        # still enabling it as a glyph fallback. Our plot labels are ASCII.
        plt.rcParams["font.enable_last_resort"] = False
        break
    else:
        raise RuntimeError("No supported Matplotlib font found; install Arial, DejaVu Sans, or Liberation Sans")
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.json"
    if not metrics_path.is_file() or not history_path.is_file():
        return []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    created: list[Path] = []
    if history:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="Training loss")
        axis.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="Validation loss")
        axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path = run_dir / "training_curve.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    per_class = metrics.get("metrics", {}).get("per_class", {})
    if per_class:
        names = list(per_class)
        series = {
            key: [per_class[name].get(key) for name in names]
            for key in ("precision", "recall_sensitivity", "f1")
        }
        figure, axis = plt.subplots(figsize=(max(8, len(names) * 0.65), 4.8))
        x = np.arange(len(names))
        width = 0.25
        for offset, (label, values) in enumerate(series.items()):
            axis.bar(x + (offset - 1) * width, [value or 0 for value in values], width, label=label.replace("_", " "))
        axis.set_xticks(x, names, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title("Held-out per-class metrics")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        path = run_dir / "class_metrics.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)

    matrix = metrics.get("metrics", {}).get("confusion_matrix")
    if matrix is not None:
        labels = metrics.get("metrics", {}).get("class_order", [])
        values = np.asarray(matrix)
        figure, axis = plt.subplots(figsize=(max(6, len(labels) * 0.6), max(5, len(labels) * 0.55)))
        image = axis.imshow(values, cmap="Blues")
        threshold = float(values.max()) / 2 if values.size else 0
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(column, row, f"{int(values[row, column])}", ha="center", va="center", color="white" if values[row, column] > threshold else "#17324D", fontweight="bold")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set(xlabel="Predicted", ylabel="Actual", title="Held-out confusion matrix")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = run_dir / "confusion_matrix.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(path)
    return created

