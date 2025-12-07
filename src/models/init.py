"""
Returns the correct model given a model name string.
"""

from .baseline import BaselineCNN
from .transfer import resnet50, efficientnet_b0

def get_model(name, num_classes):
    name = name.lower()
    if name == "baseline":
        return BaselineCNN(num_classes)
    elif name == "resnet50":
        return resnet50(num_classes)
    elif name == "efficientnet_b0":
        return efficientnet_b0(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
