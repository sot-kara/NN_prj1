import pandas as pd
import matplotlib.pyplot as plt
import os


# Configuration
CSV_FILE = "eval_results.csv"
OUTPUT_DIR = "figures"


def plot_comparison(df, metric, title, ylabel, filename):
    """Generate a bar plot comparing models for a given metric."""

    plt.figure(figsize=(9, 6))

    # Group results by model and calculate mean and standard deviation
    grouped = df.groupby("Model")[metric].agg(["mean", "std"])

    # Replace NaN standard deviation (single experiment) with zero
    grouped["std"] = grouped["std"].fillna(0)

    models = grouped.index
    means = grouped["mean"]
    stds = grouped["std"]

    plt.bar(
        models,
        means,
        yerr=stds,
        capsize=6,
        width=0.5
    )

    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Model", fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

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