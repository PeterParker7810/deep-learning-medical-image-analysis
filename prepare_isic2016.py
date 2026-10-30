"""Extract and manifest the official ISIC 2016 Task 3 CC0 challenge data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from src.data.common import audit_images


def safe_extract(archive: Path, root: Path) -> list[Path]:
    root_resolved = root.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe path in challenge archive: {member.filename}")
            target = (root / Path(*name.parts)).resolve()
            if root_resolved not in target.parents and target != root_resolved:
                raise ValueError(f"Archive path escapes data root: {member.filename}")
        zf.extractall(root)
        extracted = [root / Path(*PurePosixPath(info.filename).parts) for info in zf.infolist() if not info.is_dir()]
    return extracted


def read_labels(path: Path, split: str) -> list[dict[str, str]]:
    records = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.reader(stream), start=1):
            if len(row) != 2:
                raise ValueError(f"Expected two columns in {path.name}, row {row_number}")
            image_id, raw_label = (value.strip() for value in row)
            if split == "train":
                label = raw_label.lower()
                if label not in {"benign", "malignant"}:
                    raise ValueError(f"Unknown ISIC 2016 training label {raw_label!r}")
            else:
                value = float(raw_label)
                if value not in {0.0, 1.0}:
                    raise ValueError(f"Unknown ISIC 2016 test label {raw_label!r}")
                label = "malignant" if value == 1 else "benign"
            records.append({"image_id": image_id, "diagnosis": label, "split": split})
    return records


def prepare(root: Path) -> dict:
    train_zip = root / "ISBI2016_ISIC_Part3_Training_Data.zip"
    test_zip = root / "ISBI2016_ISIC_Part3_Test_Data.zip"
    train_csv = root / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
    test_csv = root / "ISBI2016_ISIC_Part3_Test_GroundTruth.csv"
    if not all(path.is_file() for path in (train_zip, test_zip, train_csv, test_csv)):
        raise FileNotFoundError("Download all four official ISIC 2016 Task 3 train/test image and ground-truth files first")
    train_records = read_labels(train_csv, "train")
    test_records = read_labels(test_csv, "test")
    train_image_paths = safe_extract(train_zip, root)
    test_image_paths = safe_extract(test_zip, root)
    image_index = {path.stem: path for path in train_image_paths + test_image_paths if path.suffix.lower() in {".jpg", ".jpeg"}}
    all_records = train_records + test_records
    ids = [row["image_id"] for row in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image identifiers appear in ISIC 2016 train/test label files")
    missing = [row["image_id"] for row in all_records if row["image_id"] not in image_index]
    if missing:
        raise ValueError(f"Ground-truth rows have no matching image: {missing[:10]}")
    split_index = {path.stem: ("train" if path in train_image_paths else "test") for path in train_image_paths + test_image_paths if path.suffix.lower() in {".jpg", ".jpeg"}}
    if set(split_index) != set(ids):
        raise ValueError(f"Label/image mismatch: {len(set(split_index)-set(ids))} unlabeled images; {len(set(ids)-set(split_index))} missing images")
    # Drop redundant training copies, especially any exact duplicate of an
    # official test image, to prevent train/test image leakage.
    hashes: dict[str, list[dict[str, str]]] = {}
    for row in all_records:
        digest = hashlib.sha256(image_index[row["image_id"]].read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(row)
    excluded = []
    retained = []
    for duplicate_group in hashes.values():
        test_copies = [row for row in duplicate_group if row["split"] == "test"]
        train_copies = [row for row in duplicate_group if row["split"] == "train"]
        if test_copies and train_copies:
            excluded.extend(train_copies)
            retained.extend(test_copies)
        elif train_copies:
            retained.append(train_copies[0])
            excluded.extend(train_copies[1:])
        else:
            retained.extend(test_copies)
    manifest = root / "metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_path", "diagnosis", "license", "split"))
        writer.writeheader()
        for record in sorted(retained, key=lambda row: (row["split"], row["image_id"])):
            image_path = image_index[record["image_id"]]
            writer.writerow({"image_path": str(image_path.relative_to(root)), "diagnosis": record["diagnosis"], "license": "CC0", "split": record["split"]})
    image_paths = [image_index[row["image_id"]] for row in retained]
    audit = audit_images(image_paths, hash_images=True)
    summary = {
        "dataset": "ISIC 2016 Challenge Task 3",
        "source": "https://challenge.isic-archive.com/data/",
        "source_files": [path.name for path in (train_zip, train_csv, test_zip, test_csv)],
        "license": "CC0 (as listed for the 2016 Challenge data)",
        "source_records": len(all_records),
        "records": len(retained),
        "partition_counts": dict(Counter(row["split"] for row in retained)),
        "class_counts": dict(Counter(row["diagnosis"] for row in retained)),
        "partition_class_counts": {partition: dict(Counter(r["diagnosis"] for r in retained if r["split"] == partition)) for partition in ("train", "test")},
        "excluded_duplicate_training_records": [row["image_id"] for row in excluded],
        "missing_image_references": missing,
        "unlabeled_images": [],
        "readable_images": audit.readable,
        "unreadable_images": [Path(path).name for path in audit.unreadable],
        "duplicate_image_hashes": [[Path(a).name, Path(b).name] for a, b in audit.duplicate_hashes],
        "image_dimension_counts": audit.dimensions,
        "patient_or_lesion_ids": "not provided in the downloaded Task 3 labels; official train/test challenge partition retained",
        "manifest": manifest.name,
    }
    report = root / "dataset_audit.json"
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/isic2016"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root), indent=2))
