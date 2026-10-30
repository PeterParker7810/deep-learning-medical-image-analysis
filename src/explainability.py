"""Optional Grad-CAM visualization for classifier attention inspection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def _last_conv_layer(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    if architecture.startswith("resnet"):
        return model.layer4[-1]
    if architecture == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"Grad-CAM is unsupported for architecture {architecture!r}")


def grad_cam(model: torch.nn.Module, image_tensor: torch.Tensor, class_index: int, architecture: str) -> torch.Tensor:
    """Return a normalized 2D attention map; this is not a lesion segmentation."""
    layer = _last_conv_layer(model, architecture)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    def capture_activation(_module, _inputs, output):
        activations.append(output)
        if output.requires_grad:
            output.register_hook(lambda gradient: gradients.append(gradient))

    handle_forward = layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        scores = model(image_tensor)
        if class_index < 0 or class_index >= scores.shape[1]:
            raise ValueError("class_index is outside model output range")
        scores[:, class_index].sum().backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture model activations")
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        maximum = cam.max()
        return cam / maximum if maximum > 0 else cam
    finally:
        handle_forward.remove()


def overlay_heatmap(image: Image.Image, cam: torch.Tensor, alpha: float = 0.35) -> Image.Image:
    import numpy as np
    import matplotlib

    rgb = image.convert("RGB")
    heat = to_pil_image(cam.detach().cpu()).resize(rgb.size)
    heat_array = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    base = np.asarray(rgb).astype(float) / 255.0
    blended = ((1 - alpha) * base + alpha * heat_array).clip(0, 1)
    return Image.fromarray((blended * 255).astype("uint8"))

