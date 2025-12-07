"""
Ensemble of EfficientNet-B0 and Baseline CNN
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import SPLIT_DIR, RESULTS_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED
from src.utils import set_seed, plot_confusion
from src.data import get_datasets, get_class_names, make_loaders
from src.models.init import get_model


# Runs a model over the test loader and collects raw logits
@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)

        # Store predictions and true labels
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset and class information
    train_ds, val_ds, test_ds = get_datasets(SPLIT_DIR, IMG_SIZE)
    class_names = get_class_names(train_ds)

    # Only test loader needed for evaluation
    _, _, test_loader = make_loaders(train_ds, val_ds, test_ds, BATCH_SIZE, NUM_WORKERS)

    # Load models
    effnet = get_model("efficientnet_b0", len(class_names)).to(device)
    baseline = get_model("baseline", len(class_names)).to(device)

    # Paths to best checkpoints
    ckpt_eff = RESULTS_DIR / "best_efficientnet_b0_phase2.pth"
    ckpt_baseline = RESULTS_DIR / "best_baseline.pth"

    # Ensure checkpoints exist before loading
    assert ckpt_eff.exists(), f"Missing EfficientNet ckpt: {ckpt_eff}"
    assert ckpt_baseline.exists(), f"Missing Baseline ckpt: {ckpt_baseline}"

    # Load trained weights
    effnet.load_state_dict(torch.load(ckpt_eff, map_location=device))
    baseline.load_state_dict(torch.load(ckpt_baseline, map_location=device))

    # Collect logits from each model on the test set
    print("Collecting EfficientNet logits...")
    eff_logits, y_true_1 = get_logits(effnet, test_loader, device)
    print("Collecting Baseline logits...")
    baseline_logits, y_true_2 = get_logits(baseline, test_loader, device)

    # Make sure models saw same test images
    assert np.array_equal(y_true_1, y_true_2)
    y_true = y_true_1

    # Weighted ensemble of logits
    # EfficientNet 90%, Baseline 10%
    ensemble_logits = 0.9 * eff_logits + 0.1 * baseline_logits

    # Pick the class with highest probability
    y_pred = ensemble_logits.argmax(axis=1)

    print("\nENSEMBLE TEST RESULTS:\n")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )

    (RESULTS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "reports" / "report_efficientNet_baseline_ensemble.txt", "w") as f:
        f.write(report)
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_true, y_pred)
    out_path = RESULTS_DIR / "cm_ensemble_e_b.png"
    plot_confusion(cm, class_names, out_path)
    print(f"Saved ensemble confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
