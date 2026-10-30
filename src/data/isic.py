"""Manifest-driven adapter for a selected ISIC Archive/Challenge subset."""

from __future__ import annotations

import csv
from pathlib import Path

from src.data.common import DatasetBundle, ImageRecord, index_images


def load_isic_manifest(root: str | Path, manifest: str | Path | None = None) -> DatasetBundle:
    """Load an explicit ISIC manifest with `image_path` and `diagnosis` columns.

    A per-image `license` column is required because Archive image licenses can
    differ. Optional `patient_id`, `lesion_id`, and `split` columns are retained.
    Class names come from the supplied manifest; no challenge is selected here.
    """
    root = Path(root)
    manifest_path = Path(manifest) if manifest else root / "metadata.csv"
    if manifest is not None and not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Expected an ISIC label manifest at {manifest_path}")
    image_index = index_images(root)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"image_path", "diagnosis", "license"}.issubset(rows[0]):
        raise ValueError("ISIC manifest must contain image_path, diagnosis, and license columns")
    labels = sorted({(row.get("diagnosis") or "").strip() for row in rows if (row.get("diagnosis") or "").strip()})
    if len(labels) < 2:
        raise ValueError("ISIC manifest must provide at least two non-empty diagnosis classes")
    label_to_id = {name: index for index, name in enumerate(labels)}
    records: list[ImageRecord] = []
    missing: list[str] = []
    license_counts: dict[str, int] = {}
    for row in rows:
        label = (row.get("diagnosis") or "").strip()
        relative = (row.get("image_path") or "").strip()
        license_id = (row.get("license") or "").strip()
        if not label or not relative or not license_id:
            missing.append(relative or "<empty image_path, diagnosis, or license>")
            continue
        candidate = Path(relative)
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.is_file():
            path = image_index.get(candidate.name)
        if path is None or not Path(path).is_file():
            missing.append(relative)
            continue
        group = (row.get("patient_id") or row.get("lesion_id") or "").strip()
        license_counts[license_id] = license_counts.get(license_id, 0) + 1
        split_name = (row.get("split") or "").strip().lower() or None
        if split_name and split_name not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split value {split_name!r}; use train, validation, or test")
        records.append(ImageRecord(path=Path(path), target=label_to_id[label], group_id=group or None, license_id=license_id, split_name=split_name))
    if not records:
        raise ValueError(f"No ISIC records resolved to images under {root}")
    return DatasetBundle(
        task="isic_skin_lesion",
        task_type="multiclass",
        classes=labels,
        records=records,
        missing_files=missing,
        metadata={"metadata_file": manifest_path.name, "source": "ISIC Archive subset specified by local manifest", "license_distribution": license_counts},
    )

