"""PyTorch dataset wrapper shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.common import ImageRecord, TaskType
from src.preprocessing import open_rgb_image


class MedicalImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], task_type: TaskType, transform: Callable | None = None):
        self.records = records
        self.task_type = task_type
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = open_rgb_image(str(record.path))
        tensor = self.transform(image) if self.transform else image
        if self.task_type == "multilabel":
            target = torch.tensor(record.target, dtype=torch.float32)
        else:
            target = torch.tensor(int(record.target), dtype=torch.long)
        return tensor, target

