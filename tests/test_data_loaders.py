from __future__ import annotations

import csv

from src.data.aptos import load_aptos
from src.data.chest_xray14 import load_chest_xray14
from src.data.isic import load_isic_manifest


def test_chest_xray14_parses_multilabel_and_patient_group(tmp_path, write_image):
    image_root = tmp_path / "images"
    write_image(image_root / "patient00001_000.png")
    write_image(image_root / "patient00002_001.png")
    metadata = tmp_path / "Data_Entry_2017.csv"
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Image Index", "Finding Labels", "Patient ID"])
        writer.writeheader()
        writer.writerow({"Image Index": "patient00001_000.png", "Finding Labels": "Pneumonia|Effusion", "Patient ID": "1"})
        writer.writerow({"Image Index": "patient00002_001.png", "Finding Labels": "No Finding", "Patient ID": "2"})
    bundle = load_chest_xray14(tmp_path)
    assert bundle.task_type == "multilabel"
    assert bundle.classes == ["Effusion", "Pneumonia"]
    assert bundle.records[0].target == (1.0, 1.0)
    assert bundle.records[1].target == (0.0, 0.0)
    assert bundle.records[0].group_id == "1"


def test_aptos_resolves_id_code_without_extension(tmp_path, write_image):
    write_image(tmp_path / "train_images" / "case-1.png")
    with (tmp_path / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id_code", "diagnosis"])
        writer.writeheader()
        writer.writerow({"id_code": "case-1", "diagnosis": "2"})
    bundle = load_aptos(tmp_path)
    assert bundle.records[0].target == 2
    assert bundle.classes[2] == "Moderate"


def test_isic_requires_explicit_image_license(tmp_path, write_image):
    write_image(tmp_path / "lesion.jpg")
    with (tmp_path / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "diagnosis", "lesion_id", "license"])
        writer.writeheader()
        writer.writerow({"image_path": "lesion.jpg", "diagnosis": "nevus", "lesion_id": "L1", "license": "CC-BY-NC"})
        writer.writerow({"image_path": "lesion.jpg", "diagnosis": "melanoma", "lesion_id": "L2", "license": "CC-BY"})
    bundle = load_isic_manifest(tmp_path)
    assert len(bundle.records) == 2
    assert bundle.records[0].group_id == "L1"
    assert bundle.metadata["license_distribution"] == {"CC-BY-NC": 1, "CC-BY": 1}

