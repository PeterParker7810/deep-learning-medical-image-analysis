import torch
from PIL import Image

from src.models import build_model
from src.prediction import load_checkpoint, predict_image


def test_model_initializes_with_expected_output_shape():
    model = build_model("resnet18", 3, pretrained=False)
    output = model(torch.zeros(2, 3, 64, 64))
    assert tuple(output.shape) == (2, 3)


def test_checkpoint_load_and_inference_return_real_model_probabilities(tmp_path):
    model = build_model("resnet18", 2, pretrained=False)
    checkpoint = {
        "state_dict": model.state_dict(),
        "task": "fixture",
        "task_type": "multiclass",
        "classes": ["class-a", "class-b"],
        "architecture": "resnet18",
        "image_size": 64,
        "threshold": 0.5,
    }
    path = tmp_path / "fixture.pt"
    torch.save(checkpoint, path)
    loaded_model, metadata, device = load_checkpoint(path, torch.device("cpu"))
    result = predict_image(Image.new("RGB", (64, 64), "gray"), loaded_model, metadata, device)
    assert len(result["probabilities"]) == 2
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-6
    assert result["prediction"] in metadata["classes"]

