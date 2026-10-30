"""Adapter for the NIH ChestX-ray14 metadata and image archive."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from src.data.common import DatasetBundle, ImageRecord, index_images


def _find_metadata(root: Path) -> Path:
    candidates = ("Data_Entry_2017_v2020.csv", "Data_Entry_2017.csv")
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Expected one of {candidates} in {root}")


def load_chest_xray14(root: str | Path) -> DatasetBundle:
    """Load NIH image-level labels as a 14-class multi-label task.

    `No Finding` rows are represented by all-zero pathology targets. Patient IDs
    are used for grouped splitting when present; otherwise they are parsed from
    standard NIH image names and finally left unset if neither is available.
    """
    root = Path(root)
    metadata_path = _find_metadata(root)
    image_index = index_images(root)
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    required = {"Image Index", "Finding Labels"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{metadata_path} must contain columns {sorted(required)}")

    labels = sorted({
        label.strip()
        for row in rows
        for label in (row.get("Finding Labels") or "").split("|")
        if label.strip() and label.strip().lower() != "no finding"
    })
    if not labels:
        raise ValueError("No disease labels were found in the Finding Labels column")
    records: list[ImageRecord] = []
    missing: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = (row.get("Image Index") or "").strip()
        if not name:
            missing.append("<empty Image Index>")
            continue
        path = image_index.get(name)
        if path is None:
            missing.append(name)
            continue
        if str(path.resolve()) in seen:
            continue
        seen.add(str(path.resolve()))
        present = {label.strip() for label in (row.get("Finding Labels") or "").split("|") if label.strip()}
        target = tuple(float(label in present) for label in labels)
        patient_id = (row.get("Patient ID") or "").strip()
        if not patient_id:
            match = re.match(r"patient(\d+)", name, flags=re.IGNORECASE)
            patient_id = match.group(1) if match else ""
        records.append(ImageRecord(path=path, target=target, group_id=patient_id or None))
    if not records:
        raise ValueError(f"No metadata rows resolved to image files under {root}")
    return DatasetBundle(
        task="chest_xray14",
        task_type="multilabel",
        classes=labels,
        records=records,
        missing_files=missing,
        metadata={"metadata_file": metadata_path.name, "source": "NIH ChestX-ray14"},
    )

