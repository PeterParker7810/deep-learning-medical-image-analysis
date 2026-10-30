"""Mechanics-only integration test; fixtures are not medical data or results."""

from dataclasses import replace

from config import DEFAULT_TRAINING
from src.data.common import DatasetBundle, ImageRecord
from src.data.splitting import split_bundle
from src.training import train_model


def test_one_epoch_training_saves_reloadable_checkpoint(tmp_path, write_image):
    records = []
    for label in range(2):
        for index in range(6):
            image = write_image(tmp_path / "images" / f"{label}-{index}.png", color=(40 + label * 120, 60, 100))
            records.append(ImageRecord(image, label))
    bundle = DatasetBundle("fixture", "multiclass", ["class-a", "class-b"], records)
    splits = split_bundle(bundle, train_fraction=0.5, validation_fraction=0.25, test_fraction=0.25, seed=9)
    config = replace(DEFAULT_TRAINING, image_size=32, batch_size=4, epochs=1, patience=1, freeze_backbone_epochs=0, use_pretrained_weights=False)
    result = train_model(bundle, splits, "resnet18", config, tmp_path / "model.pt")
    assert result["checkpoint_path"] == str(tmp_path / "model.pt")
    assert (tmp_path / "model.pt").is_file()
    assert result["epochs_completed"] == 1

