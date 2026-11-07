import torch
import numpy as np
from torchvision import datasets, transforms
from models.numpy_NN import MLP 

def train_and_evaluate_numpy_nn(X_train, y_train, X_test, y_test, layers, activation='relu', learning_rate=0.01, seed=42, epochs=10):
    # Initialize the NumpyNN model
    model = MLP(layers, activation=activation, learning_rate=learning_rate, seed=seed)

    # Train the model
    for epoch in range(epochs):
        # Forward pass
        outputs = model.forward(X_train)
        
        # Compute loss (cross-entropy)
        m = y_train.shape[0]
        log_likelihood = -np.log(outputs[range(m), y_train])
        loss = np.sum(log_likelihood) / m

        # Backward pass
        grad_weights, grad_biases = model.backward(y_train)
        model.update_parameters(grad_weights, grad_biases)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")



# Load CIFAR-100 dataset
transform = transforms.ToTensor()
train_data = datasets.CIFAR100(root="data", train=True, transform=transform, download=True)
test_data = datasets.CIFAR100(root="data", train=False, transform=transform, download=True)

# Prepare training and testing data
X_train = torch.stack([image for image, label in train_data]).numpy().reshape(len(train_data), -1)
y_train = torch.tensor([label for image, label in train_data]).numpy()

X_test = torch.stack([image for image, label in test_data]).numpy().reshape(len(test_data), -1)
y_test = torch.tensor([label for image, label in test_data]).numpy()

# Normalize the data
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Define the architecture of the neural network
layers = [X_train.shape[1], 100, 128, 100]  # Input layer size, hidden layers sizes, output layer size

# Train and evaluate the NumpyNN model
train_and_evaluate_numpy_nn(X_train, y_train, X_test, y_test, layers, activation='relu', learning_rate=0.01, seed=42, epochs=10)