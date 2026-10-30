"""Inspect the supplied dataset files and write a factual data-quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DATA_ROOT, RESULTS_ROOT
from src.data import audit_images, load_aptos, load_chest_xray14, load_isic_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a local image dataset")
    parser.add_argument("--task", required=True, choices=("chest_xray14", "aptos_retinopathy", "isic_skin_lesion"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hash-images", action="store_true", help="Compute SHA-256 hashes to identify byte-identical images")
    args = parser.parse_args()
    defaults = {
        "chest_xray14": DATA_ROOT / "raw" / "chest_xray14",
        "aptos_retinopathy": DATA_ROOT / "raw" / "aptos",
        "isic_skin_lesion": DATA_ROOT / "raw" / "isic2016",
    }
    root = args.data_root or defaults[args.task]
    if args.task == "chest_xray14":
        bundle = load_chest_xray14(root)
    elif args.task == "aptos_retinopathy":
        bundle = load_aptos(root)
    else:
        bundle = load_isic_manifest(root, args.manifest)
    image_audit = audit_images([record.path for record in bundle.records], hash_images=args.hash_images)
    report = {
        "task": bundle.task,
        "source": bundle.metadata.get("source"),
        "metadata_file": bundle.metadata.get("metadata_file"),
        "license_distribution": bundle.metadata.get("license_distribution"),
        "records_resolved": len(bundle.records),
        "classes": bundle.classes,
        "label_counts": bundle.label_counts(),
        "missing_image_references": bundle.missing_files,
        "image_audit": {
            "checked": image_audit.checked,
            "readable": image_audit.readable,
            "unreadable": image_audit.unreadable,
            "duplicate_file_hashes": image_audit.duplicate_hashes,
            "dimension_distribution": image_audit.dimensions,
        },
    }
    output = Path(RESULTS_ROOT) / bundle.task / "dataset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(output)}, indent=2))


if __name__ == "__main__":
    main()

