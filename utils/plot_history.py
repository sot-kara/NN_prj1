import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_history(filepath,save_dir=None):

    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    name = filepath.stem

    if save_dir is None:
        
        save_dir = Path(f"results/{name}/figures")
        save_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # LOSS
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} - Loss vs Epoch")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_dir / f"{name}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ==========================================
    # ACCURACY
    # ==========================================

    plt.figure(figsize=(8, 5))

    plt.plot(df["epoch"], df["train_accuracy"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} - Accuracy vs Epoch")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(save_dir / f"{name}_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved in: {save_dir}")

if __name__ == "__main__":
    plot_history(sys.argv[1])