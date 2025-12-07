"""
Loads pretrained ResNet-50 and EfficientNet-B0 and replaces their classification heads.
"""

import torch.nn as nn
import torchvision.models as tvm

def resnet50(num_classes):
    """
    Initialize a ResNet-50 model pretrained on ImageNet and adapts
    it to the number of classes in our waste classification dataset.
    """
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

    # Replace the final fully connected layer (classifier)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def efficientnet_b0(num_classes):
    """
    Initializes an EfficientNet-B0 model pretrained on ImageNet and
    replaces the classifier head for custom number of classes.
    """
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Access last layer in classifier sequence and replace this final layer so the model outputs our class number
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m
