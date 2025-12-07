"""
Garbage Classification Baseline and ResNet-50 Trainer
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from src.config import (SPLIT_DIR, RESULTS_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, SEED, MODEL_NAME)
from src.utils import set_seed, ensure_dir, plot_confusion
from src.data import (get_datasets, get_class_names, make_loaders)
from src.models.init import get_model       # "baseline", "resnet50"


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

    # Get model
    model     = get_model(MODEL_NAME, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Track best validation accuracy and save that checkpoint
    best_val_acc = 0.0
    best_path    = RESULTS_DIR / f"best_{MODEL_NAME}.pth"   # For each model trained their best result is saved

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        t0 = time.time()        # To measure epoch duration

        for x, y in train_loader:
            # 1) Move batch to device
            x, y = x.to(device), y.to(device)

            # 2) Reset gradients from previous step
            optimizer.zero_grad()

            # 3) Forward pass
            logits = model(x)

            # 4) Compute loss
            loss = criterion(logits, y)

            # 5) Backpropagate -> accumulate gradients
            loss.backward()

            # 6) Parameter update
            optimizer.step()

            running_loss += loss.item() * x.size(0)       # sum of batch losses
            pred = logits.argmax(1)                       # predicted class ids
            correct += (pred == y).sum().item()           # correct in batch
            total += y.size(0)                            # samples in batch

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        # Epoch timing and progress
        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d} | {dt:5.1f}s  "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # Save the best model performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best (val_acc={best_val_acc:.4f}) = {best_path}")

    # Final testing
    # Load the best checkpoint before testing
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Report Results
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    (RESULTS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "reports" / f"report_{MODEL_NAME}.txt", "w") as f:
        f.write(report)
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, RESULTS_DIR / f"cm_{MODEL_NAME}.png")
    print(f"Saved confusion matrix to {RESULTS_DIR / f'cm_{MODEL_NAME}.png'}")


if __name__ == "__main__":
    main()
