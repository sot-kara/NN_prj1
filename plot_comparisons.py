from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path(__file__).resolve().parent / "results"

OUTPUT_DIR = RESULTS_DIR / "comparison_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The six architectures to compare
MODELS = {
    "MLP-Small": "mlp_small",
    "MLP-Medium": "mlp_mid",
    "MLP-Large": "mlp_large",

    "CNN-Small": "cnn_small",
    "CNN-Medium": "cnn_mid",
    "CNN-Large": "cnn_large",
}

# Learning rate experiments (Medium-sized architectures only)
LR_EXPERIMENTS = {
    "MLP-Medium": {
        "Small LR": "mlp_mid_small_lr",
        "Medium LR": "mlp_mid",
        "Large LR": "mlp_mid_big_lr",
    },

    "CNN-Medium": {
        "Small LR": "cnn_mid_small_lr",
        "Medium LR": "cnn_mid",
        "Large LR": "cnn_mid_big_lr",
    },
}

# Dropout experiments
DROPOUT_EXPERIMENTS = {
    "Baseline (No Dropout)": "mlp_mid",
    "Dropout 0.3": "mlp_mid_dropout_0.3",
    "Dropout 0.5": "mlp_mid_dropout_0.5",
    "Dropout 0.7": "mlp_mid_dropout_0.7",
}


# PCA experiments (MLP-mid only)
PCA_EXPERIMENTS = {
    "MLP-Medium (Baseline)": "mlp_mid",
    "PCA 100": "mlp_mid_pca_100",
    "PCA 256": "mlp_mid_pca_256",
    "PCA 1024": "mlp_mid_pca_1024",
}

OPTIMIZED_CNN = {
    "CNN Optimized": "cnn_optimized",
}

MAX_DROPOUT_EPOCHS = 50

MAX_LR_EPOCHS = 50

# Baseline test accuracies.
# Replace None with your measured accuracy values.
# You can use either fractions (0.17) or percentages (17.0).

BASELINE_ACCURACY = {
    "KNN": 0.1755,
    "Class Centroid": 0.1004,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_accuracy(values):
    """
    Convert accuracy from [0, 1] to [0, 100]
    when necessary.
    """

    values = np.asarray(values, dtype=float)

    if np.nanmax(values) <= 1.0:
        values = values * 100

    return values


def load_history(experiment_name):
    """
    Load per-epoch training history.
    """

    history_path = (
        RESULTS_DIR
        / experiment_name
        / "history"
        / f"{experiment_name}.csv"
    )

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}"
        )

    df = pd.read_csv(history_path)

    required_columns = ["epoch", "val_accuracy"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing column '{column}' in {history_path}"
            )

    return df


def load_test_accuracy(experiment_name):
    """
    Read test accuracy from the experiment summary CSV.
    """

    csv_path = (
        RESULTS_DIR
        / experiment_name
        / "csv"
        / "experiments.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Experiment summary not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Find test_accuracy column (case-insensitive)
    accuracy_column = next(
        (
            col for col in df.columns
            if col.lower() == "test_accuracy"
        ),
        None
    )

    if accuracy_column is None:
        raise ValueError(
            f"No test_accuracy column in {csv_path}"
        )

    if df.empty:
        raise ValueError(
            f"No experiment results in {csv_path}"
        )

    # Use the latest experiment entry
    accuracy = float(df.iloc[-1][accuracy_column])

    return float(normalize_accuracy([accuracy])[0])


# ============================================================
# FIGURE 1: VALIDATION ACCURACY VS EPOCH
# ============================================================

def plot_accuracy_history():

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "Small": "tab:blue",
        "Medium": "tab:orange",
        "Large": "tab:green",
    }

    styles = {
        "MLP": {
            "linestyle": "-",
            "marker": "o",
            "markersize": 5,
            "linewidth": 2.5,
        },
        "CNN": {
            "linestyle": "--",
            "marker": "^",
            "markersize": 6,
            "linewidth": 2.5,
        },
    }

    for label, experiment_name in MODELS.items():

        history = load_history(experiment_name)

        epochs = history["epoch"].to_numpy()

        accuracy = normalize_accuracy(
            history["val_accuracy"].to_numpy()
        )

        family, size = label.split("-")

        ax.plot(
            epochs,
            accuracy,
            label=label,
            color=colors[size],
            **styles[family],
            alpha=0.9,
        )

    ax.set_title(
        "Validation Accuracy vs Epoch",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=13)

    ax.set_ylim(0, 100)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5
    )

    ax.legend(
        fontsize=11,
        loc="best",
        ncol=2,
        frameon=True,
        fancybox=True
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "validation_accuracy.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 2: TEST ACCURACY COMPARISON
# ============================================================

def plot_test_accuracy():

    results = {}

    # Load MLP and CNN results
    for label, experiment_name in MODELS.items():

        accuracy = load_test_accuracy(experiment_name)

        results[label] = accuracy

    # Add KNN and Class Centroid
    for label, accuracy in BASELINE_ACCURACY.items():

        if accuracy is None:
            print(
                f"WARNING: Missing test accuracy for {label}. "
                "Skipping it."
            )
            continue

        results[label] = float(
            normalize_accuracy([accuracy])[0]
        )

    if not results:
        raise ValueError("No test accuracy results found.")

    labels = list(results.keys())
    values = list(results.values())

    # Colors by model family
    colors = []

    for label in labels:

        if label.startswith("MLP"):
            colors.append("tab:blue")

        elif label.startswith("CNN"):
            colors.append("tab:orange")

        elif label == "KNN":
            colors.append("tab:green")

        else:
            colors.append("tab:red")

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.7
    )

    # Display accuracy above each bar
    ax.bar_label(
        bars,
        labels=[f"{v:.2f}%" for v in values],
        padding=4,
        fontsize=10
    )

    ax.set_title(
        "Test Accuracy Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)

    ax.set_ylim(
        0,
        max(values) * 1.15
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)

    plt.xticks(rotation=0)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "test_accuracy_comparison.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)

# ============================================================
# FIGURE 3: TRAINING TIME COMPARISON
# ============================================================

def load_training_time(experiment_name):

    # First, try experiments.csv
    csv_path = (
        RESULTS_DIR
        / experiment_name
        / "csv"
        / "experiments.csv"
    )

    if csv_path.exists():

        df = pd.read_csv(csv_path)

        candidates = [
            "training_time",
            "train_time",
            "total_training_time",
            "training_time_seconds",
            "total_time",
        ]

        column = next(
            (c for c in df.columns if c.lower() in candidates),
            None
        )

        if column is not None and not df.empty:
            return float(df.iloc[-1][column])

    # Otherwise, try calculating it from per-epoch history
    history_path = (
        RESULTS_DIR
        / experiment_name
        / "history"
        / f"{experiment_name}.csv"
    )

    if history_path.exists():

        df = pd.read_csv(history_path)

        candidates = [
            "epoch_time",
            "training_time",
            "train_time",
            "time",
        ]

        column = next(
            (c for c in df.columns if c.lower() in candidates),
            None
        )

        if column is not None:
            return float(df[column].sum())

    raise ValueError(
        f"Could not find training time for {experiment_name}"
    )


def plot_training_time():

    results = {}

    for label, experiment_name in MODELS.items():

        results[label] = load_training_time(experiment_name)

    labels = list(results.keys())
    values = list(results.values())

    colors = []

    for label in labels:

        if label.startswith("MLP"):
            colors.append("tab:blue")
        else:
            colors.append("tab:orange")

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.7
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.2f}s" for v in values],
        padding=4,
        fontsize=10
    )

    ax.set_title(
        "Training Time Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)

    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "training_time_comparison.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 4: LEARNING RATE CONVERGENCE COMPARISON
# ============================================================

def plot_lr_comparison():

    colors = {
        "Small LR": "tab:blue",
        "Medium LR": "tab:orange",
        "Large LR": "tab:green",
    }

    for model_label, experiments in LR_EXPERIMENTS.items():

        fig, ax = plt.subplots(figsize=(10, 6))

        for lr_label, experiment_name in experiments.items():

            history = load_history(experiment_name)

            # Keep only the first 50 epochs
            history = (
                history.sort_values("epoch")
                .head(MAX_LR_EPOCHS)
            )

            epochs = history["epoch"].to_numpy()

            accuracy = normalize_accuracy(
                history["val_accuracy"].to_numpy()
            )

            ax.plot(
                epochs,
                accuracy,
                label=lr_label,
                color=colors[lr_label],
                linewidth=2.5,
                alpha=0.9,
            )

        ax.set_title(
            f"{model_label} - Learning Rate Comparison",
            fontsize=16,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Validation Accuracy (%)", fontsize=13)

        ax.set_xlim(1, MAX_LR_EPOCHS)
        ax.set_ylim(0, 100)

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.8,
            alpha=0.5
        )

        ax.legend(
            fontsize=11,
            loc="best",
            frameon=True,
            fancybox=True
        )

        fig.tight_layout()

        # Save figure
        filename = (
            f"{model_label.lower().replace('-', '_')}"
            "_lr_comparison.png"
        )

        output_path = OUTPUT_DIR / filename

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved: {output_path}")

        plt.show()
        plt.close(fig)

# ============================================================
# FIGURE 5: DROPOUT - TRAINING VS VALIDATION ACCURACY
# ============================================================

def plot_dropout_comparison():

    baseline_name = "mlp_mid"

    dropout_experiments = {
        "Dropout 0.3": "mlp_mid_dropout_0.3",
        "Dropout 0.5": "mlp_mid_dropout_0.5",
        "Dropout 0.7": "mlp_mid_dropout_0.7",
    }

    dropout_colors = {
        "Baseline": "black",
        "Dropout": "tab:blue",
    }

    for dropout_label, experiment_name in dropout_experiments.items():

        fig, ax = plt.subplots(figsize=(11, 6))

        # Load baseline history
        baseline = load_history(baseline_name)

        baseline = (
            baseline.sort_values("epoch")
            .head(MAX_DROPOUT_EPOCHS)
        )

        # Load dropout history
        dropout = load_history(experiment_name)

        dropout = (
            dropout.sort_values("epoch")
            .head(MAX_DROPOUT_EPOCHS)
        )

        # Normalize accuracy values
        baseline_train = normalize_accuracy(
            baseline["train_accuracy"].to_numpy()
        )

        baseline_val = normalize_accuracy(
            baseline["val_accuracy"].to_numpy()
        )

        dropout_train = normalize_accuracy(
            dropout["train_accuracy"].to_numpy()
        )

        dropout_val = normalize_accuracy(
            dropout["val_accuracy"].to_numpy()
        )

        # Baseline curves
        ax.plot(
            baseline["epoch"],
            baseline_train,
            color=dropout_colors["Baseline"],
            linestyle="-",
            linewidth=2.3,
            label="Baseline - Training",
        )

        ax.plot(
            baseline["epoch"],
            baseline_val,
            color=dropout_colors["Baseline"],
            linestyle="--",
            linewidth=2.3,
            label="Baseline - Validation",
        )

        # Dropout curves
        ax.plot(
            dropout["epoch"],
            dropout_train,
            color=dropout_colors["Dropout"],
            linestyle="-",
            linewidth=2.3,
            label=f"{dropout_label} - Training",
        )

        ax.plot(
            dropout["epoch"],
            dropout_val,
            color=dropout_colors["Dropout"],
            linestyle="--",
            linewidth=2.3,
            label=f"{dropout_label} - Validation",
        )

        ax.set_title(
            f"MLP-Medium - {dropout_label} vs Baseline",
            fontsize=16,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)

        ax.set_xlim(1, MAX_DROPOUT_EPOCHS)
        ax.set_ylim(0, 100)

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.8,
            alpha=0.5
        )

        ax.legend(
            fontsize=10,
            loc="best",
            frameon=True,
            fancybox=True
        )

        fig.tight_layout()

        filename = (
            f"dropout_{dropout_label.split()[-1]}_comparison.png"
        )

        output_path = OUTPUT_DIR / filename

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved: {output_path}")

        plt.show()
        plt.close(fig)


# ============================================================
# FIGURE 6: PCA - VALIDATION ACCURACY VS EPOCH
# ============================================================

def plot_pca_validation_accuracy():

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {
        "MLP-Medium (Baseline)": "black",
        "PCA 100": "tab:blue",
        "PCA 256": "tab:orange",
        "PCA 1024": "tab:green",
    }

    for label, experiment_name in PCA_EXPERIMENTS.items():

        history = load_history(experiment_name)
        history = history.sort_values("epoch")

        accuracy = normalize_accuracy(
            history["val_accuracy"].to_numpy()
        )

        ax.plot(
            history["epoch"],
            accuracy,
            label=label,
            color=colors[label],
            linewidth=2.5,
            alpha=0.9,
        )

    ax.set_title(
        "MLP-Medium - PCA Validation Accuracy Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=13)

    ax.set_ylim(0, 100)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5
    )

    ax.legend(
        fontsize=11,
        loc="best",
        frameon=True,
        fancybox=True
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "pca_validation_accuracy.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 7: PCA - TRAINING TIME COMPARISON
# ============================================================

def plot_pca_training_time():

    results = {}

    for label, experiment_name in PCA_EXPERIMENTS.items():

        results[label] = load_training_time(experiment_name)

    labels = list(results.keys())
    values = list(results.values())

    colors = [
        "black",
        "tab:blue",
        "tab:orange",
        "tab:green",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.7
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.2f}s" for v in values],
        padding=4,
        fontsize=10
    )

    ax.set_title(
        "MLP-Medium - PCA Training Time Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "pca_training_time.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 8: PCA - BEST CHECKPOINT TEST ACCURACY
# ============================================================

def plot_pca_test_accuracy():

    results = {}

    for label, experiment_name in PCA_EXPERIMENTS.items():

        results[label] = load_test_accuracy(experiment_name)

    labels = list(results.keys())
    values = list(results.values())

    colors = [
        "black",
        "tab:blue",
        "tab:orange",
        "tab:green",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.7
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.2f}%" for v in values],
        padding=4,
        fontsize=10
    )

    ax.set_title(
        "MLP-Medium - PCA Test Accuracy Comparison",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "pca_test_accuracy.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 9: OPTIMIZED CNN - TRAINING VS VALIDATION ACCURACY
# ============================================================

def plot_optimized_cnn_accuracy():

    experiment_name = OPTIMIZED_CNN["CNN Optimized"]

    history = load_history(experiment_name)
    history = history.sort_values("epoch")

    epochs = history["epoch"].to_numpy()

    train_accuracy = normalize_accuracy(
        history["train_accuracy"].to_numpy()
    )

    val_accuracy = normalize_accuracy(
        history["val_accuracy"].to_numpy()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy",
        color="tab:blue",
        linestyle="-",
        linewidth=2.5
    )

    ax.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy",
        color="tab:orange",
        linestyle="--",
        linewidth=2.5
    )

    ax.set_title(
        "Optimized CNN - Training vs Validation Accuracy",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)

    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 100)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        alpha=0.5
    )

    ax.legend(
        fontsize=11,
        loc="best",
        frameon=True,
        fancybox=True
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "cnn_optimized_accuracy.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nGenerating comparison plots...\n")

    plot_accuracy_history()

    plot_test_accuracy()

    plot_training_time()

    plot_lr_comparison()

    plot_dropout_comparison()

    plot_pca_validation_accuracy()
    
    plot_pca_training_time()
    
    plot_pca_test_accuracy()

    plot_optimized_cnn_accuracy()

    print("\nAll plots generated successfully!")