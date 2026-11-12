from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from models.nn_obj.adam import Adam
import cupy as cp
from torchvision import datasets, transforms
from models.numpy_NN import MLP 

def train_and_evaluate_numpy_nn(
    X_train, y_train, X_test, y_test,
    layer_configs,
    learning_rate=0.01,
    epochs=100, batch_size=128,
    optimizer=None
):
    model = MLP(
        input_size=X_train.shape[1],
        layer_configs=layer_configs,
        learning_rate=learning_rate,
        optimizer=optimizer
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

            # Backward pass to get the weight and bias gradients
            gradients = model.backward(y_batch)

            # Update parameters using the optimizer at the end of each mini-batch
            model.update_parameters(gradients)

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



#Load CIFAR-100 dataset
transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)



# Prepare training and testing data
X_train = torch.stack([image for image, label in train_data]).numpy().reshape(len(train_data), -1)
y_train = torch.tensor([label for image, label in train_data]).numpy()

X_test = torch.stack([image for image, label in test_data]).numpy().reshape(len(test_data), -1)
y_test = torch.tensor([label for image, label in test_data]).numpy()

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train).astype(np.float32)
# X_test = scaler.transform(X_test).astype(np.float32)

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
layer_configs = [
    {"units": 512, "activation": "relu", "dropout": 0.2},
    {"units": 256, "activation": "relu", "dropout": 0.2},
    {"units": 10, "activation": "softmax", "dropout": 0.0}
]

# Train and evaluate the NumpyNN model
model = train_and_evaluate_numpy_nn(X_train, y_train, X_test, y_test, layer_configs, learning_rate=0.001, epochs=50, batch_size=128, optimizer=Adam(learning_rate=0.001))

# Percentages of accuracy for each class
for class_idx in range(10):
    class_mask = (y_test == class_idx)
    # get the name of the class from CIFAR-10 dataset
    class_name = datasets.CIFAR10(root="data", train=False, download=False).classes[class_idx]
    class_accuracy = cp.mean((model.forward(X_test[class_mask]).argmax(axis=1) == y_test[class_mask]).astype(cp.float32))
    print(f"Class {class_idx} ({class_name}): Accuracy = {class_accuracy.item() * 100:.2f}%")