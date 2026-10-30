"""Checkpoint loading and image inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.models import build_model
from src.preprocessing import build_transform, open_rgb_image
from src.reproducibility import select_device


def load_checkpoint(path: str | Path, device: torch.device | None = None):
    device = device or select_device()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    required = {"state_dict", "task", "task_type", "classes", "architecture", "image_size"}
    if not required.issubset(payload):
        raise ValueError(f"Checkpoint metadata is incomplete: missing {sorted(required - payload.keys())}")
    model = build_model(payload["architecture"], len(payload["classes"]), pretrained=False).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, device


def predict_image(image: Image.Image | bytes | str, model, checkpoint: dict[str, Any], device=None) -> dict[str, Any]:
    device = device or select_device()
    rgb = open_rgb_image(image)
    transform = build_transform(int(checkpoint["image_size"]))
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if checkpoint["task_type"] == "multilabel":
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            threshold = float(checkpoint.get("threshold", 0.5))
            predicted = [name for name, value in zip(checkpoint["classes"], probabilities) if value >= threshold]
        else:
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            predicted = checkpoint["classes"][int(np.argmax(probabilities))]
    return {
        "task": checkpoint["task"],
        "task_type": checkpoint["task_type"],
        "classes": list(checkpoint["classes"]),
        "probabilities": [float(value) for value in probabilities],
        "prediction": predicted,
        "threshold": checkpoint.get("threshold", 0.5),
        "image": rgb,
        "tensor": tensor,
    }

