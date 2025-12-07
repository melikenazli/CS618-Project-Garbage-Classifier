"""
Defines all data transformations (augmentation, normalization) and constructs datasets + dataloaders.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Data Augmentation
def train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),    # Resize images to img_size × img_size
        transforms.RandomHorizontalFlip(0.5),       # 50% chance to flip the image horizontally
        transforms.RandomRotation(15),              # Random rotation between -15° and +15°
        transforms.ColorJitter(0.2,0.2,0.2,0.02),   # Random brightness, contrast, saturation, hue
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # Normalize using ImageNet mean/std
    ])

def eval_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# Load datasets
def get_datasets(split_dir, img_size):
    train_ds = datasets.ImageFolder(split_dir / "train", transform=train_transforms(img_size))
    val_ds   = datasets.ImageFolder(split_dir / "val",   transform=eval_transforms(img_size))
    test_ds  = datasets.ImageFolder(split_dir / "test",  transform=eval_transforms(img_size))
    return train_ds, val_ds, test_ds

# Converts PyTorch’s internal class dictionary into an ordered list of class labels 
# so the model’s outputs correspond correctly to class names
def get_class_names(train_ds):
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]

# PyTorch DataLoaders
def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
