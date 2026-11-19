import csv
from pathlib import Path


class ResultsManager:

    def __init__(self, filepath="results/csv/experiments.csv"):

        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self.fields = [
            "experiment_name",
            "model_type",
            "epochs",
            "batch_size",
            "learning_rate",
            "best_epoch",
            "best_val_accuracy",
            "test_accuracy",
            "test_loss",
            "total_training_time",
            "average_epoch_time",
            "checkpoint"
        ]

    def save_result(self, result):

        file_exists = self.filepath.exists()

        with open(self.filepath, "a", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.fields
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(result)

        print(f"Results saved to {self.filepath}")