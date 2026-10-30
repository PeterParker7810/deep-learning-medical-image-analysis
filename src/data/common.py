"""Shared records, image indexing, and data-quality audit helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

TaskType = Literal["multiclass", "multilabel"]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    target: int | tuple[float, ...]
    group_id: str | None = None
    license_id: str | None = None
    split_name: str | None = None


@dataclass
class DatasetBundle:
    task: str
    task_type: TaskType
    classes: list[str]
    records: list[ImageRecord]
    missing_files: list[str] = field(default_factory=list)
    duplicate_paths: list[str] = field(default_factory=list)
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)

    def label_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        if self.task_type == "multiclass":
            for record in self.records:
                counts[self.classes[int(record.target)]] += 1
        else:
            for record in self.records:
                vector = record.target
                assert isinstance(vector, tuple)
                for name, value in zip(self.classes, vector):
                    counts[name] += int(value > 0)
        return {name: counts[name] for name in self.classes}


@dataclass
class ImageAudit:
    checked: int
    readable: int
    unreadable: list[str]
    duplicate_hashes: list[tuple[str, str]]
    dimensions: dict[str, int]


def index_images(root: Path) -> dict[str, Path]:
    """Index image basenames recursively, rejecting ambiguous duplicate names."""
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            if path.name in index:
                duplicates.append(path.name)
            else:
                index[path.name] = path
    if duplicates:
        raise ValueError(f"Duplicate image basenames found under {root}: {sorted(set(duplicates))[:10]}")
    return index


def audit_images(paths: list[Path], *, hash_images: bool = False) -> ImageAudit:
    """Check image readability, dimensions, and optionally byte-identical files."""
    import hashlib

    unreadable: list[str] = []
    dimensions: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    duplicate_hashes: list[tuple[str, str]] = []
    readable = 0
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("invalid image dimensions")
                dimensions[f"{width}x{height}"] += 1
            readable += 1
            if hash_images:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in hashes:
                    duplicate_hashes.append((hashes[digest], str(path)))
                else:
                    hashes[digest] = str(path)
        except (OSError, ValueError, UnidentifiedImageError):
            unreadable.append(str(path))
    return ImageAudit(len(paths), readable, unreadable, duplicate_hashes, dict(dimensions))

