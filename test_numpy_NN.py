import torch
import numpy as np
import cupy as cp
from torchvision import datasets, transforms
from models.numpy_NN import MLP 

def train_and_evaluate_numpy_nn(
    X_train, y_train, X_test, y_test,
    layers, activation='relu',
    learning_rate=0.01, seed=42,
    epochs=100, batch_size=128
):
    model = MLP(
        layers,
        activation=activation,
        learning_rate=learning_rate,
        seed=seed
    )

    n_samples = X_train.shape[0]

    for epoch in range(epochs):

        # Shuffle training data
        permutation = cp.random.permutation(n_samples)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0.0

        # Mini-batch training
        for start in range(0, n_samples, batch_size):

            end = min(start + batch_size, n_samples)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            outputs = model.forward(X_batch)

            # Cross-entropy loss
            m = y_batch.shape[0]
            loss = -cp.log(
                cp.clip(
                    outputs[cp.arange(m), y_batch],
                    1e-12, 1.0
                )
            ).mean()

            epoch_loss += loss.item() * m

            # Backward pass
            grad_weights, grad_biases = model.backward(y_batch)

            # Update parameters
            model.update_parameters(grad_weights, grad_biases)

        epoch_loss /= n_samples

        # Evaluate after completing the epoch
        train_acc = model.evaluate(X_train, y_train)
        test_acc = model.evaluate(X_test, y_test)

        print(
            f"Epoch [{epoch + 1}/{epochs}]: "
            f"Loss={epoch_loss:.4f}, "
            f"Train Acc={train_acc.item():.4f}, "
            f"Test Acc={test_acc.item():.4f}"
        )

    accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

    return model



# Load CIFAR-100 dataset
transform = transforms.ToTensor()
train_data = datasets.CIFAR100(root="data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR100(root="data", train=False, transform=transform, download=True)

# Prepare training and testing data
X_train = torch.stack([image for image, label in train_data]).numpy().reshape(len(train_data), -1)
y_train = torch.tensor([label for image, label in train_data]).numpy()

X_test = torch.stack([image for image, label in test_data]).numpy().reshape(len(test_data), -1)
y_test = torch.tensor([label for image, label in test_data]).numpy()


print("X_train shape:", X_train.shape)
print("X_train dtype:", X_train.dtype)
print("X_train min:", X_train.min())
print("X_train max:", X_train.max())
print("X_train mean:", X_train.mean())
print("X_train std:", X_train.std())

print("y_train shape:", y_train.shape)
print("y_train dtype:", y_train.dtype)
print("y_train min:", y_train.min())
print("y_train max:", y_train.max())

# Transfer data to GPU using CuPy
X_train = cp.asarray(X_train, dtype=cp.float32)
y_train = cp.asarray(y_train, dtype=cp.int64)

X_test = cp.asarray(X_test, dtype=cp.float32)
y_test = cp.asarray(y_test, dtype=cp.int64)

# Define the architecture of the neural network
layers = [X_train.shape[1], 1024, 512, 256, 100]  # Input layer size, hidden layers sizes, output layer size

# Train and evaluate the NumpyNN model
train_and_evaluate_numpy_nn(X_train, y_train, X_test, y_test, layers, activation='relu', learning_rate=0.09, seed=42, epochs=100)