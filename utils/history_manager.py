import csv
from pathlib import Path


class HistoryManager:

    def __init__(self, results_dir="results/history"):

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_history(self, experiment_name, history):

        filepath = self.results_dir / f"{experiment_name}.csv"

        fields = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy"
        ]

        num_epochs = len(history["train_loss"])

        with open(filepath, "w", newline="") as file:

            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for epoch in range(num_epochs):

                writer.writerow({
                    "epoch": epoch + 1,
                    "train_loss": history["train_loss"][epoch],
                    "train_accuracy": history["train_accuracy"][epoch],
                    "val_loss": history["val_loss"][epoch],
                    "val_accuracy": history["val_accuracy"][epoch]
                })

        print(f"Training history saved to {filepath}")