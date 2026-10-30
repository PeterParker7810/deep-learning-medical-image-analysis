"""Transfer-learning model registry based on architectures named in the report."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models

Architecture = Literal["resnet18", "resnet50", "densenet121"]
SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("resnet18", "resnet50", "densenet121")


def build_model(architecture: str, num_outputs: int, *, pretrained: bool = True) -> nn.Module:
    """Create a classifier with logits for multiclass or multilabel objectives."""
    if num_outputs < 1:
        raise ValueError("num_outputs must be at least one")
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        network = models.resnet18(weights=weights)
        network.fc = nn.Linear(network.fc.in_features, num_outputs)
    elif architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        network = models.resnet50(weights=weights)
        network.fc = nn.Linear(network.fc.in_features, num_outputs)
    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        network = models.densenet121(weights=weights)
        network.classifier = nn.Linear(network.classifier.in_features, num_outputs)
    else:
        raise ValueError(f"Unsupported architecture {architecture!r}; choose from {SUPPORTED_ARCHITECTURES}")
    return network


def classifier_parameters(model: nn.Module):
    """Return the final classifier parameters for a frozen-backbone warm-up."""
    classifier = getattr(model, "fc", None) or getattr(model, "classifier", None)
    if classifier is None:
        raise ValueError("Model has no supported fc/classifier output layer")
    return list(classifier.parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the feature extractor while keeping the head trainable."""
    head_parameters = {id(parameter) for parameter in classifier_parameters(model)}
    for parameter in model.parameters():
        parameter.requires_grad = trainable or id(parameter) in head_parameters

