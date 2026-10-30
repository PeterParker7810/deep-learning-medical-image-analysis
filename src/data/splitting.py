"""Deterministic train/validation/test splitting with group isolation."""

from __future__ import annotations

import random
from collections import defaultdict

from src.data.common import DatasetBundle, ImageRecord


def _validate_fractions(train: float, validation: float, test: float) -> None:
    if min(train, validation, test) <= 0 or abs(train + validation + test - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test fractions must be positive and sum to 1")


def split_bundle(
    bundle: DatasetBundle,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    """Split records without group leakage when every record has a group ID.

    Ungrouped multiclass data is stratified by class. Ungrouped multilabel data
    is deterministically shuffled and should be checked for per-label coverage.
    """
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    records = list(bundle.records)
    if len(records) < 3:
        raise ValueError("At least three records are required for three-way splitting")
    fractions = (train_fraction, validation_fraction, test_fraction)
    names = ("train", "validation", "test")
    rng = random.Random(seed)

    fixed_splits = {record.split_name for record in records if record.split_name}
    if fixed_splits:
        if any(record.split_name is None for record in records):
            raise ValueError("If fixed split labels are used, every record must have one")
        train_pool = [record for record in records if record.split_name == "train"]
        validation = [record for record in records if record.split_name == "validation"]
        test = [record for record in records if record.split_name == "test"]
        if validation:
            if not train_pool or not test:
                raise ValueError("Predefined splits must include train and test records")
            return {"train": train_pool, "validation": validation, "test": test}
        if not train_pool or not test:
            raise ValueError("Predefined split metadata must include train and test records")
        # Preserve the source's independent test partition and carve validation
        # only from the official training partition.
        val_fraction = validation_fraction / (train_fraction + validation_fraction)
        train_out: list[ImageRecord] = []
        validation_out: list[ImageRecord] = []
        if all(record.group_id for record in train_pool):
            by_group: dict[str, list[ImageRecord]] = defaultdict(list)
            for record in train_pool:
                by_group[str(record.group_id)].append(record)
            group_items = list(by_group.items())
            rng.shuffle(group_items)
            target = len(train_pool) * val_fraction
            for _, group_records in sorted(group_items, key=lambda item: len(item[1]), reverse=True):
                if abs((len(validation_out) + len(group_records)) - target) < abs(len(validation_out) - target):
                    validation_out.extend(group_records)
                else:
                    train_out.extend(group_records)
        elif bundle.task_type == "multiclass":
            by_class: dict[int, list[ImageRecord]] = defaultdict(list)
            for record in train_pool:
                by_class[int(record.target)].append(record)
            for class_records in by_class.values():
                rng.shuffle(class_records)
                n_validation = max(1, round(len(class_records) * val_fraction)) if len(class_records) > 1 else 0
                validation_out.extend(class_records[:n_validation])
                train_out.extend(class_records[n_validation:])
        else:
            rng.shuffle(train_pool)
            n_validation = max(1, round(len(train_pool) * val_fraction))
            validation_out, train_out = train_pool[:n_validation], train_pool[n_validation:]
        if not train_out or not validation_out:
            raise ValueError("Predefined training data is too small to create train and validation partitions")
        rng.shuffle(train_out)
        rng.shuffle(validation_out)
        return {"train": train_out, "validation": validation_out, "test": test}

    has_groups = all(record.group_id is not None and str(record.group_id).strip() for record in records)
    if has_groups:
        groups: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            groups[str(record.group_id)].append(record)
        if len(groups) < 3:
            raise ValueError("At least three distinct patient/lesion groups are required")
        group_items = list(groups.items())
        rng.shuffle(group_items)
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        target_sizes = [len(records) * fraction for fraction in fractions]
        buckets: list[list[ImageRecord]] = [[], [], []]
        for _, group_records in group_items:
            options = []
            for index in range(3):
                projected = [len(bucket) for bucket in buckets]
                projected[index] += len(group_records)
                size_cost = sum(((projected[j] - target_sizes[j]) / max(target_sizes[j], 1)) ** 2 for j in range(3))
                options.append((size_cost, len(buckets[index]), rng.random(), index))
            chosen = min(options)[-1]
            buckets[chosen].extend(group_records)
        result = dict(zip(names, buckets))
        all_groups = [{str(r.group_id) for r in result[name]} for name in names]
        if any(all_groups[i] & all_groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("Group leakage detected after split")
        return result

    if bundle.task_type == "multiclass":
        by_class: dict[int, list[ImageRecord]] = defaultdict(list)
        for record in records:
            by_class[int(record.target)].append(record)
        buckets = [[], [], []]
        for class_records in by_class.values():
            rng.shuffle(class_records)
            n = len(class_records)
            counts = [int(n * fraction) for fraction in fractions]
            for index in range(n - sum(counts)):
                counts[index % 3] += 1
            offset = 0
            for index, count in enumerate(counts):
                buckets[index].extend(class_records[offset : offset + count])
                offset += count
        for bucket in buckets:
            rng.shuffle(bucket)
        return dict(zip(names, buckets))

    rng.shuffle(records)
    n_train = max(1, int(len(records) * train_fraction))
    n_validation = max(1, int(len(records) * validation_fraction))
    if n_train + n_validation >= len(records):
        n_train = len(records) - 2
        n_validation = 1
    return {
        "train": records[:n_train],
        "validation": records[n_train : n_train + n_validation],
        "test": records[n_train + n_validation :],
    }

