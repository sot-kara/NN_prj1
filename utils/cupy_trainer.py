import time
import cupy as cp
from .model_saver import ModelSaver
from pathlib import Path


class CupyTrainer:

    def __init__(
        self,
        model,
        epochs=50,
        batch_size=128,
        seed=42,
        model_saver=None,
        checkpoint_path=None,
        save_best = True
    ):

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.model_saver = model_saver
        self.save_best = save_best

        self.checkpoint_path = (Path(checkpoint_path) 
                                if checkpoint_path else None
                                )

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_time": []
        }

        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    # ==========================================
    # LOSS
    # ==========================================

    @staticmethod
    def calculate_loss(outputs, y):
        # Calculate the cross-entropy loss
        m = y.shape[0]

        loss = -cp.log(
            cp.clip(
                outputs[cp.arange(m), y],
                1e-12,
                1.0
            )
        ).mean()

        return loss

    # ==========================================
    # EVALUATION
    # ==========================================

    def evaluate(self, X, y):

        n_samples = X.shape[0]

        total_loss = 0.0
        correct = 0

        # Evaluation does not require gradients
        for start in range(0, n_samples, self.batch_size):

            end = min(start + self.batch_size, n_samples)

            X_batch = X[start:end]
            y_batch = y[start:end]

            # Forward pass with training disabled, so dropout is not applied
            outputs = self.model.forward(
                X_batch,
                training=False
            )

            # Loss
            loss = self.calculate_loss(outputs, y_batch)

            total_loss += float(loss.item()) * len(y_batch)

            # Accuracy
            predictions = cp.argmax(outputs, axis=1)

            correct += int(
                cp.sum(predictions == y_batch).item()
            )

        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples

        return avg_loss, accuracy

    # ==========================================
    # TRAINING EPOCH
    # ==========================================

    def train_epoch(self, X_train, y_train):

        n_samples = X_train.shape[0]

        # Shuffle training data
        permutation = cp.random.permutation(n_samples)

        epoch_loss = 0.0

        for start in range(0, n_samples, self.batch_size):

            end = min(start + self.batch_size, n_samples)

            indices = permutation[start:end]

            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # Forward propagation
            outputs = self.model.forward(
                X_batch,
                training=True
            )

            # Calculate loss
            loss = self.calculate_loss(outputs, y_batch)

            epoch_loss += float(loss.item()) * len(y_batch)

            # Backpropagation
            gradients = self.model.backward(y_batch)

            # Update parameters
            self.model.update_parameters(gradients)

        epoch_loss /= n_samples

        return epoch_loss

    # ==========================================
    # FULL TRAINING
    # ==========================================

    def fit(self, train_data, val_data):

        X_train, y_train = train_data
        X_val, y_val = val_data

        # Set random seed
        cp.random.seed(self.seed)

        print("\nStarting CuPy training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")

        for epoch in range(self.epochs):

            # Synchronize before timing
            cp.cuda.Stream.null.synchronize()

            start_time = time.perf_counter()

            # Training
            train_loss = self.train_epoch(
                X_train,
                y_train
            )

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

            # Synchronize before stopping timer
            cp.cuda.Stream.null.synchronize()

            epoch_time = time.perf_counter() - start_time

            # Save best model
            if self.save_best:

                saved = self.model_saver.save_best_model(
                    model=self.model,
                    filepath=self.checkpoint_path,
                    val_accuracy=val_accuracy,
                    epoch=epoch + 1,
                    metadata={
                        "model_type": self.model.__class__.__name__,
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "learning_rate": getattr(
                            self.model.optimizer,
                            "lr",
                            None
                        )
                    }
                )

                if saved:
                    print(f"New best model saved with validation accuracy: {val_accuracy*100:.2f}%")

            # Store metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)

            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            self.history["epoch_time"].append(epoch_time)

            # Track best model
            if val_accuracy > self.best_val_accuracy:

                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch + 1

            # Print results
            print(
                f"Epoch [{epoch+1:03d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy*100:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )

        print("\nTraining completed.")

        print(
            f"Best Validation Accuracy: "
            f"{self.best_val_accuracy*100:.2f}% "
            f"(Epoch {self.best_epoch})"
        )

        return self.history
    
    def load_best_model(self):

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {self.checkpoint_path}"
            )

        metadata = self.model_saver.load_model(
            model=self.model,
            filepath=self.checkpoint_path
        )

        print(
            f"Best model loaded from epoch "
            f"{metadata.get('best_epoch', 'Unknown')}"
        )

        print(
            f"Best validation accuracy: "
            f"{metadata.get('best_val_accuracy', 'Unknown')}"
        )

        return metadata