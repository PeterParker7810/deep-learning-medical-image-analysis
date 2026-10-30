"""Adapter for the Kaggle APTOS diabetic-retinopathy training files."""

from __future__ import annotations

import csv
from pathlib import Path

from src.data.common import DatasetBundle, ImageRecord, index_images

APTOS_GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def load_aptos(root: str | Path) -> DatasetBundle:
    """Load labeled APTOS images from `train.csv` and an image directory.

    The adapter intentionally requires the actual metadata and images. It does
    not infer the release from the directory or include competition test files.
    """
    root = Path(root)
    csv_path = root / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected {csv_path}")
    image_index = index_images(root)
    stem_index = {path.stem: path for path in image_index.values()}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"id_code", "diagnosis"}.issubset(rows[0]):
        raise ValueError("APTOS train.csv must contain id_code and diagnosis columns")
    records: list[ImageRecord] = []
    missing: list[str] = []
    for row in rows:
        image_id = (row.get("id_code") or "").strip()
        try:
            target = int(row["diagnosis"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid APTOS diagnosis for {image_id or '<empty id>'}") from exc
        if target < 0 or target >= len(APTOS_GRADES):
            raise ValueError(f"APTOS diagnosis must be between 0 and 4; got {target}")
        path = image_index.get(image_id) or image_index.get(Path(image_id).name) or stem_index.get(Path(image_id).stem)
        if path is None:
            missing.append(image_id)
            continue
        records.append(ImageRecord(path=path, target=target))
    if not records:
        raise ValueError(f"No APTOS records resolved to images under {root}")
    return DatasetBundle(
        task="aptos_retinopathy",
        task_type="multiclass",
        classes=APTOS_GRADES,
        records=records,
        missing_files=missing,
        metadata={"metadata_file": csv_path.name, "source": "Kaggle APTOS Retinopathy"},
    )

