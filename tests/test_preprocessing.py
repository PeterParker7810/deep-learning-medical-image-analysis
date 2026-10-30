from PIL import Image

from src.preprocessing import build_transform, open_rgb_image


def test_grayscale_is_converted_to_normalized_rgb_tensor():
    image = Image.new("L", (40, 30), color=127)
    tensor = build_transform(32)(open_rgb_image(image))
    assert tuple(tensor.shape) == (3, 32, 32)


def test_invalid_image_is_rejected():
    try:
        open_rgb_image(b"not an image")
    except ValueError as exc:
        assert "not a readable" in str(exc)
    else:
        raise AssertionError("Invalid image bytes should be rejected")

