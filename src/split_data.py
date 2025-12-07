import os
import random
import shutil


data_path = "../data/raw"           # Path to the original dataset
output_path = "../data/split"       # Output directory where train/val/test folders will be created

split_ratio = (0.7, 0.15, 0.15)     # Train/validation/test split ratio (70% / 15% / 15%)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
seed = 42

random.seed(seed)

# Create output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_path, split), exist_ok=True)

# List class folders
class_names = [d for d in os.listdir(data_path)
               if os.path.isdir(os.path.join(data_path, d))]
print(f"Found classes: {class_names}")


# Split data for each class
for cls in class_names:
    cls_path = os.path.join(data_path, cls)

    # Select only valid image files
    files = [f for f in os.listdir(cls_path)
             if os.path.splitext(f)[-1].lower() in image_extensions]

    random.shuffle(files)

    # Number of samples assigned to each split
    n_total = len(files)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)
    n_test = n_total - n_train - n_val      # to ensure all images are used

    # Partition files according to computed split points
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    # Create class subfolders
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

    # Copy files
    for f in train_files:
        shutil.copy2(os.path.join(cls_path, f),
                     os.path.join(output_path, "train", cls, f))
    for f in val_files:
        shutil.copy2(os.path.join(cls_path, f),
                     os.path.join(output_path, "val", cls, f))
    for f in test_files:
        shutil.copy2(os.path.join(cls_path, f),
                     os.path.join(output_path, "test", cls, f))


print("\nDataset successfully split into train/val/test!")