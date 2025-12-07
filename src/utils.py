import random, numpy as np, torch
import matplotlib.pyplot as plt

# Ensures full reproducibility of experiments
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Creates a directory if it does not already exist
def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

# Visualize and save confusion matrix as an image
def plot_confusion(cm, classes, out_path):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha='right')
    plt.yticks(ticks, classes)
    plt.tight_layout()
    plt.ylabel('True'); plt.xlabel('Predicted')
    fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
