"""Shared image validation and model transforms."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def open_rgb_image(source: str | bytes | BytesIO | Image.Image) -> Image.Image:
    """Open a supported raster image and return an independent RGB image."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        else:
            input_stream = BytesIO(source) if isinstance(source, bytes) else source
            with Image.open(input_stream) as opened:
                opened.load()
                image = opened.copy()
        if image.width < 16 or image.height < 16:
            raise ValueError("Image must be at least 16 by 16 pixels")
        return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("The uploaded file is not a readable supported image") from exc


def build_transform(image_size: int = 224, *, training: bool = False, augment: bool = False):
    """Build the same resize and ImageNet normalization for train and inference."""
    steps = [transforms.Resize((image_size, image_size))]
    if training and augment:
        steps.extend([transforms.RandomRotation(5), transforms.RandomHorizontalFlip(p=0.5)])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(steps)

