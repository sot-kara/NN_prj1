import numpy as np
import torch
from torchvision import datasets, transforms


class KNN:
    def __init__(self, k=1, batch_size=100):
        if k < 1:
            raise ValueError("k must be at least 1")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.k = k
        self.batch_size = batch_size

        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        '''
        Save training data
        '''
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train)

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array")

        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array")

        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")

        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")

        if self.k > len(X_train):
            raise ValueError("k cannot exceed the number of training samples")

        self.X_train = X_train
        self.y_train = y_train

        return self

    def predict(self, X_test):
        """
        predict each X_test sample's class using k nearest neighbour
        """
        if self.X_train is None:
            raise RuntimeError("Call fit() before predict()")

        X_test = np.asarray(X_test, dtype=np.float32)

        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array")

        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError("X_test and X_train must have the same number of features")

        predictions = []

        
        train_norms = np.sum(self.X_train ** 2, axis=1)

        for start in range(0, len(X_test), self.batch_size):
            X_batch = X_test[start:start + self.batch_size]

            
            test_norms = np.sum(X_batch ** 2, axis=1)

            # Euclidean distance calculation using:
            # |x-y|^2 = |x|^2 + |y|^2- 2*x*y
            distances = (
                test_norms[:, None]
                + train_norms[None, :]
                - 2 * (X_batch @ self.X_train.T)
            )

            
            np.maximum(distances, 0, out=distances)

            
            nearest_indices = np.argpartition(
                distances, kth=self.k - 1, axis=1
            )[:, :self.k]

            
            for indices in nearest_indices:
                labels = self.y_train[indices]
                classes, counts = np.unique(labels, return_counts=True)

                
                prediction = classes[np.argmax(counts)]
                predictions.append(prediction)

        return np.asarray(predictions)

    def score(self, X_test, y_test):
        y_test = np.asarray(y_test)
        predictions = self.predict(X_test) # run predict to get predictions for X_test

        if len(y_test) != len(predictions):
            raise ValueError("X_test and y_test must have the same length")

        # return the accuracy of the model on the test data, the mean number of correct predictions
        return np.mean(predictions == y_test)


# Main
model = KNN(k=3);

transform = transforms.ToTensor()

train_data = datasets.CIFAR100(
    root="data",
    train=True,
    transform=transform,
    download=False
)

test_data = datasets.CIFAR100(
    root="data",
    train=False,
    transform=transform,
    download=False
)

X_train = torch.stack([image for image, label in train_data])
y_train = torch.tensor([label for image, label in train_data])

X_test = torch.stack([image for image, label in test_data])
y_test = torch.tensor([label for image, label in test_data])


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)[:100, :]
y_test = y_test[:100]

model.fit(X_train.numpy(), y_train.numpy())


print("Accuracy:", model.score(X_test.numpy(), y_test.numpy()))