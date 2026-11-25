"""Lesion segmentation net (shared by train/export): torchvision LR-ASPP MobileNetV3-Large, 3 classes.

Classes: 0 background (incl. other leaves), 1 healthy tissue of the main leaf, 2 lesion.
Input: float32 NCHW RGB in [0, 1] (the app's Preprocess.toNchw); ImageNet normalisation happens inside the model.
Output: logits (1, 3, H, W) at input resolution.
"""
import torch
import torchvision

CLASSES = ["background", "leaf", "lesion"]


class LesionNet(torch.nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.net = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=None, weights_backbone=w, num_classes=len(CLASSES))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)["out"]
