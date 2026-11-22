import pandas as pd
import matplotlib.pyplot as plt
import os


# Configuration
CSV_FILE = "eval_results.csv"
OUTPUT_DIR = "figures"


def plot_comparison(df, metric, title, ylabel, filename):
    """Generate a bar plot comparing models for a given metric."""

    fig, ax = plt.subplots(figsize=(9, 6))

    # Group results by model and calculate mean and standard deviation
    grouped = df.groupby("Model")[metric].agg(["mean", "std"])
    grouped["std"] = grouped["std"].fillna(0)

    models = grouped.index
    means = grouped["mean"]
    stds = grouped["std"]

    # Create bars with error bars
    bars = ax.bar(
        models,
        means,
        yerr=stds,
        capsize=6,
        width=0.5
    )

    # Add exact value labels above the bars
    labels = [
        f"{value:.4f}" if metric == "Accuracy" else f"{value:.4f} s"
        for value in means
    ]

    ax.bar_label(
        bars,
        labels=labels,
        padding=8,
        fontsize=10
    )

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Model", fontsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Leave space above the bars for labels and error bars
    ax.margins(y=0.15)

    fig.tight_layout()

    # Save figure
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)

    print(f"Saved: {filepath}")

def plot():

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load CSV
    df = pd.read_csv(CSV_FILE)

    # Convert PCA column to boolean if necessary
    if df["PCA_Enabled"].dtype == object:
        df["PCA_Enabled"] = (
            df["PCA_Enabled"].astype(str).str.lower() == "true"
        )

    # Separate PCA ON and PCA OFF experiments
    df_pca_on = df[df["PCA_Enabled"] == True]
    df_pca_off = df[df["PCA_Enabled"] == False]

    # Prediction time - PCA OFF
    if not df_pca_off.empty:
        plot_comparison(
            df_pca_off,
            "Prediction_Time_s",
            "Prediction Time (Raw Data)",
            "Time (seconds)",
            "prediction_time_pca_off.png"
        )

    # Prediction time - PCA ON
    if not df_pca_on.empty:
        plot_comparison(
            df_pca_on,
            "Prediction_Time_s",
            "Prediction Time (PCA ON)",
            "Time (seconds)",
            "prediction_time_pca_on.png"
        )

    # Accuracy - PCA OFF
    if not df_pca_off.empty:
        plot_comparison(
            df_pca_off,
            "Accuracy",
            "Classification Accuracy (RAW Data)",
            "Accuracy",
            "accuracy_pca_off.png"
        )

    # Accuracy - PCA ON
    if not df_pca_on.empty:
        plot_comparison(
            df_pca_on,
            "Accuracy",
            "Classification Accuracy (PCA ON)",
            "Accuracy",
            "accuracy_pca_on.png"
        )

plot()