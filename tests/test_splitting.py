from src.data.common import DatasetBundle, ImageRecord
from src.data.splitting import split_bundle


def test_group_split_has_no_leakage_and_is_reproducible(tmp_path):
    records = [
        ImageRecord(tmp_path / f"{group}_{index}.png", index % 2, group_id=group)
        for group in range(15)
        for index in range(2)
    ]
    bundle = DatasetBundle("fixture", "multiclass", ["a", "b"], records)
    first = split_bundle(bundle, seed=13)
    second = split_bundle(bundle, seed=13)
    ids = [{record.group_id for record in first[name]} for name in ("train", "validation", "test")]
    assert all(not (ids[i] & ids[j]) for i in range(3) for j in range(i + 1, 3))
    assert {name: [record.path.name for record in values] for name, values in first.items()} == {
        name: [record.path.name for record in values] for name, values in second.items()
    }
    assert all(first[name] for name in ("train", "validation", "test"))


def test_ungrouped_multiclass_stratifies(tmp_path):
    records = [ImageRecord(tmp_path / f"{label}_{index}.png", label) for label in range(2) for index in range(20)]
    bundle = DatasetBundle("fixture", "multiclass", ["a", "b"], records)
    split = split_bundle(bundle, seed=3)
    assert all(split.values())
    for values in split.values():
        assert {record.target for record in values} == {0, 1}


def test_predefined_test_partition_is_preserved(tmp_path):
    records = []
    for label in range(2):
        records.extend(ImageRecord(tmp_path / f"train_{label}_{i}.jpg", label, split_name="train") for i in range(6))
        records.extend(ImageRecord(tmp_path / f"test_{label}_{i}.jpg", label, split_name="test") for i in range(3))
    bundle = DatasetBundle("isic_fixture", "multiclass", ["benign", "malignant"], records)
    splits = split_bundle(bundle, seed=19)
    assert all(record.split_name == "test" for record in splits["test"])
    assert len(splits["test"]) == 6
    assert {record.target for record in splits["train"]} == {0, 1}
    assert {record.target for record in splits["validation"]} == {0, 1}
    assert not ({record.path for record in splits["test"]} & {record.path for record in splits["train"]})

