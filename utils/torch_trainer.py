import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.results_manager import ResultsManager

from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class TorchTrainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        epochs=100,
        batch_size=256,
        seed=42,
        device=None,
        history_manager=None,
        experiment_name=None,
        checkpoint_path=None,
        save_best=True,
        results_manager=None,
        model_type="MLP_PyTorch",
        learning_rate=None,
        train_transform=None,
        eval_transform=None,
        scheduler=None
    ):

        self.model = model
        self.optimizer = optimizer

        self.criterion = (
            criterion
            if criterion is not None
            else torch.nn.CrossEntropyLoss()
        )

        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model.to(self.device)

        self.history_manager = history_manager
        self.experiment_name = experiment_name

        self.save_best = save_best

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        if self.save_best and self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be provided when save_best=True"
            )

        if self.save_best:
            self.checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_time": [],
            "learning_rate": []
        }

        self.best_val_accuracy = 0.0
        self.best_epoch = 0

        self.total_training_time = 0.0

        self.results_manager = results_manager
        self.model_type = model_type
        self.learning_rate = learning_rate

        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.scheduler = scheduler

    # ==========================================
    # GPU SYNCHRONIZATION
    # ==========================================

    def synchronize(self):

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ==========================================
    # DATA LOADER
    # ==========================================
    def create_loader(self, X, y, shuffle=False, transform=None):
        dataset = ImageDataset(
            X,
            y,
            transform=transform
        )

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

    # ==========================================
    # EVALUATION
    # ==========================================

    @torch.no_grad()
    def evaluate(self, X, y, transform=None):

        self.model.eval()

        if transform is None:
            transform = self.eval_transform

        loader = self.create_loader(X, y, transform=transform, shuffle=False)

        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                self.device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                self.device,
                non_blocking=True
            )

            outputs = self.model(X_batch)

            loss = self.criterion(outputs, y_batch)

            batch_size = y_batch.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += batch_size

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    # ==========================================
    # TRAINING EPOCH
    # ==========================================

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0.0
        total = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(
                self.device,
                non_blocking=True
            )

            y_batch = y_batch.to(
                self.device,
                non_blocking=True
            )

            # Clear previous gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward propagation
            outputs = self.model(X_batch)

            # Loss
            loss = self.criterion(outputs, y_batch)

            # Backpropagation
            loss.backward()

            # Update parameters
            self.optimizer.step()

            batch_size = y_batch.size(0)

            total_loss += loss.item() * batch_size
            total += batch_size

        return total_loss / total

    # ==========================================
    # CHECKPOINT SAVING
    # ==========================================

    def save_best_model(self, epoch, val_accuracy):

        checkpoint = {
            "epoch": epoch,
            "best_val_accuracy": val_accuracy,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(
            checkpoint,
            self.checkpoint_path
        )

        print(
            f"New best model saved with validation accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

    # ==========================================
    # FULL TRAINING
    # ==========================================

    def fit(self, train_data, val_data):

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Set random seeds
        torch.manual_seed(self.seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        # Create training loader
        train_loader = self.create_loader(
            X_train,
            y_train,
            shuffle=True,
            transform=self.train_transform
        )

        print("\nStarting PyTorch training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")

        self.synchronize()
        total_start = time.perf_counter()

        for epoch in range(self.epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.synchronize()
            start_time = time.perf_counter()

            # Training
            train_loss = self.train_epoch(train_loader)

            # Training evaluation
            train_eval_loss, train_accuracy = self.evaluate(
                X_train,
                y_train
            )

            # Validation evaluation
            val_loss, val_accuracy = self.evaluate(
                X_val,
                y_val
            )

            self.synchronize()
            epoch_time = time.perf_counter() - start_time

            if self.scheduler is not None:
                self.scheduler.step()
            # Save best model
            if (
                self.save_best
                and val_accuracy > self.best_val_accuracy
            ):

                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch + 1

                self.save_best_model(
                    epoch=epoch + 1,
                    val_accuracy=val_accuracy
                )

            # Store metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)

            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            self.history["epoch_time"].append(epoch_time)
            self.history["learning_rate"].append(current_lr)

            # Save history
            if (
                self.history_manager is not None
                and self.experiment_name is not None
            ):

                self.history_manager.save_history(
                    self.experiment_name,
                    self.history
                )

            # Print results
            print(
                f"Epoch [{epoch+1:03d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy*100:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )

        self.synchronize()

        self.total_training_time = (
            time.perf_counter() - total_start
        )

        print("\nTraining completed.")

        print(
            f"Best Validation Accuracy: "
            f"{self.best_val_accuracy*100:.2f}% "
            f"(Epoch {self.best_epoch})"
        )

        print(
            f"Total Training Time: "
            f"{self.total_training_time:.2f}s"
        )
        

        return self.history

    # ==========================================
    # LOAD BEST MODEL
    # ==========================================

    def load_best_model(self):

        if (
            self.checkpoint_path is None
            or not self.checkpoint_path.exists()
        ):
            raise FileNotFoundError(
                f"No checkpoint found at {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.best_epoch = checkpoint["epoch"]

        self.best_val_accuracy = checkpoint[
            "best_val_accuracy"
        ]

        print(
            f"Best model loaded from epoch {self.best_epoch}"
        )

        print(
            f"Best validation accuracy: "
            f"{self.best_val_accuracy * 100:.2f}%"
        )

        return {
            "best_epoch": self.best_epoch,
            "best_val_accuracy": self.best_val_accuracy
        }
    
    def save_results(self, test_loss, test_accuracy):

        if self.results_manager is None:
            return

        average_epoch_time = (
            sum(self.history["epoch_time"])
            / len(self.history["epoch_time"])
        )

        result = {
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "best_epoch": self.best_epoch,
            "best_val_accuracy": self.best_val_accuracy,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "total_training_time": self.total_training_time,
            "average_epoch_time": average_epoch_time,
            "checkpoint": str(self.checkpoint_path)
        }

        self.results_manager.save_result(result)