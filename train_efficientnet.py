"""
Garbage Classification EfficientNet-B0 Trainer
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from src.config import (SPLIT_DIR, RESULTS_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED)
from src.utils import set_seed, ensure_dir, plot_confusion
from src.data import (get_datasets, get_class_names, make_loaders)
from src.models.init import get_model


@torch.no_grad()  
def evaluate(model, loader, criterion, device):
    model.eval()

    total, correct, running_loss = 0, 0, 0.0
    ys, ps = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)

        # Accumulate *sum* of losses: loss.item() is mean over batch by default
        running_loss += loss.item() * x.size(0)

        # predicted class = index of max logit
        pred = logits.argmax(1)

        correct += (pred == y).sum().item()
        total   += y.size(0)

        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    acc      = correct      / max(total, 1)
    return avg_loss, acc, np.concatenate(ys), np.concatenate(ps)


def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, epochs, model_name):
    best_val_acc = 0.0
    best_path = Path(RESULTS_DIR) / f"best_{model_name}.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}: train_acc={correct/total:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  Saved new best model: {best_path}")


def main():
    set_seed(SEED)

    # Ensure output folders exist when saving results
    ensure_dir(RESULTS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load datasets
    train_ds, val_ds, test_ds = get_datasets(SPLIT_DIR, IMG_SIZE)

    class_names = get_class_names(train_ds)

    # Get DataLoaders
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds, BATCH_SIZE, NUM_WORKERS)

    # Get EfficientNet
    model = get_model("efficientnet_b0", num_classes=len(class_names)).to(device)

    # Phase 1 – Freeze backbone, train only classifier
    for p in model.features.parameters():
        p.requires_grad = False     # freeze feature extractor

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=1e-3, weight_decay=1e-4       # only classifier is trained
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\nPhase 1:")
    train_model(model, train_loader, val_loader, optimizer, criterion,
                device, epochs=5, model_name="efficientnet_b0_phase1")

    # Phase 2 – Unfreeze last 2–3 blocks and fine-tune
    # Freeze everything again first
    for p in model.features.parameters():
        p.requires_grad = False

    # Unfreeze last few blocks
    for idx in [-1, -2, -3]:
        for p in model.features[idx].parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )

    print("\nPhase 2:")
    train_model(model, train_loader, val_loader, optimizer, criterion,
                device, epochs=25, model_name="efficientnet_b0_phase2")


    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(Path(RESULTS_DIR)/"best_efficientnet_b0_phase2.pth"))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Save Results
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    (RESULTS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "reports" / "report_efficientnet_b0.txt", "w") as f:
        f.write(report)
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, RESULTS_DIR / "cm_efficientnet_b0.png")
    print(f"Saved confusion matrix to {RESULTS_DIR / 'cm_efficientnet_b0.png'}")


if __name__ == "__main__":
    main()
