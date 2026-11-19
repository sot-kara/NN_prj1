# experiments/experiment_runner.py

import time
import cupy as cp

from pathlib import Path
from copy import deepcopy

from models.numpy_NN import MLP
from models.numpy_CNN import CNN

from utils.cupy_trainer import CupyTrainer
from utils.model_saver import ModelSaver
from utils.results_manager import ResultsManager


class ExperimentRunner:

    def __init__(self, dataset, results_dir="results"):

        self.dataset = dataset

        self.results_dir = Path(results_dir)

        self.checkpoint_dir = self.results_dir / "models"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_manager = ResultsManager(
            self.results_dir / "csv" / "experiments.csv"
        )


    # ==========================================
    # MODEL CREATION
    # ==========================================

    def create_model(self, config):

        model_type = config["model_type"]

        if model_type == "mlp":

            model = MLP(
                input_size=config["input_size"],
                layer_configs=deepcopy(config["layer_configs"]),
                learning_rate=config["learning_rate"]
            )

        elif model_type == "cnn":

            model = CNN(
                input_shape=config["input_shape"],
                layer_configs=deepcopy(config["layer_configs"]),
                learning_rate=config["learning_rate"]
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model


    # ==========================================
    # TEST EVALUATION
    # ==========================================

    def evaluate_test_set(self, model, test_data, batch_size=128):

        X_test, y_test = test_data

        total_loss = 0.0
        total_correct = 0
        total_samples = len(X_test)

        for start in range(0, total_samples, batch_size):

            end = min(start + batch_size, total_samples)

            X_batch = X_test[start:end]
            y_batch = y_test[start:end]

            probabilities = model.forward(
                X_batch,
                training=False
            )

            batch_size_actual = len(y_batch)

            # Cross-entropy loss
            correct_class_probs = probabilities[
                cp.arange(batch_size_actual),
                y_batch
            ]

            batch_loss = -cp.log(
                correct_class_probs + 1e-12
            ).sum()

            total_loss += float(batch_loss.item())

            # Accuracy
            predictions = cp.argmax(probabilities, axis=1)

            total_correct += int(
                cp.sum(predictions == y_batch).item()
            )

        test_loss = total_loss / total_samples
        test_accuracy = total_correct / total_samples

        return test_loss, test_accuracy


    # ==========================================
    # SINGLE EXPERIMENT
    # ==========================================

    def run_experiment(self, config):

        name = config["name"]

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {name}")
        print("=" * 60)

        # Create the appropriate form of the dataset based on the model type
        self.active_dataset = self.prepare_dataset(config["model_type"])

        # Create model
        model = self.create_model(config)

        # Create checkpoint path
        checkpoint_path = self.checkpoint_dir / f"{name}.npz"

        # Create trainer
        trainer = CupyTrainer(
            model=model,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            seed=42,
            model_saver=ModelSaver(),
            checkpoint_path=checkpoint_path
        )

        # Training, validation, and test data
        train_data = self.active_dataset["train"]
        val_data = self.active_dataset["val"]
        test_data = self.active_dataset["test"]

        # Train
        cp.cuda.runtime.deviceSynchronize()
        start_time = time.perf_counter()

        history = trainer.fit(
            train_data,
            val_data
        )

        cp.cuda.runtime.deviceSynchronize()
        total_training_time = time.perf_counter() - start_time

        # Find best epoch
        best_val_accuracy = max(history["val_accuracy"])

        best_epoch = (
            history["val_accuracy"].index(best_val_accuracy) + 1
        )

        # Restore best model
        trainer.load_best_model()

        # Evaluate on test set
        test_loss, test_accuracy = self.evaluate_test_set(
            model,
            test_data,
            batch_size=config["batch_size"]
        )

        average_epoch_time = (
            sum(history["epoch_time"]) / len(history["epoch_time"])
        )

        # Prepare results
        result = {
            "experiment_name": name,
            "model_type": config["model_type"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "total_training_time": total_training_time,
            "average_epoch_time": average_epoch_time,
            "checkpoint": str(checkpoint_path)
        }

        # Save CSV
        self.results_manager.save_result(result)

        print("\nEXPERIMENT SUMMARY")
        print(f"Best Epoch:       {best_epoch}")
        print(f"Best Val Acc:     {best_val_accuracy:.4f}")
        print(f"Test Accuracy:    {test_accuracy:.4f}")
        print(f"Test Loss:        {test_loss:.4f}")
        print(f"Training Time:    {total_training_time:.2f}s")

        return {
            "result": result,
            "history": history
        }


    # ==========================================
    # MULTIPLE EXPERIMENTS
    # ==========================================

    def run_all(self, experiments):

        results = []

        for config in experiments:

            experiment_output = self.run_experiment(config)

            results.append(experiment_output)

        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETED")
        print("=" * 60)

        return results
    
    # =========================================
    # DATASET PREPARATION
    # =========================================

    def prepare_dataset(self, model_type):

        prepared = {}

        for split in ["train", "val", "test"]:

            X, y = self.dataset[split]

            X = cp.asarray(X, dtype=cp.float32) / 255.0
            y = cp.asarray(y, dtype=cp.int64)

            if model_type == "mlp":
                X = X.reshape(X.shape[0], -1)

            elif model_type == "cnn":
                X = X.transpose(0, 3, 1, 2)
                X = cp.ascontiguousarray(X)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            prepared[split] = (X, y)

        return prepared