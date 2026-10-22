import numpy as np


class ClassCentroid:
    def __init__(self):
        self.centroids = None
        self.classes = None

    def fit(self, X_train, y_train):
        """
        Calculate centroid for each class
        """
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

        self.classes = np.unique(y_train)

        self.centroids = np.stack([
            X_train[y_train == c].mean(axis=0)
            for c in self.classes
        ])

        return self

    def predict(self, X_test):
        """
        predict class by getting the minimum class centroid distance
        """
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict()")

        X_test = np.asarray(X_test, dtype=np.float32)

        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array")

        if X_test.shape[1] != self.centroids.shape[1]:
            raise ValueError("X_test and X_train must have the same number of features")
        print("Starting class centroid prediction...")
        distances = (
            np.sum(X_test ** 2, axis=1, keepdims=True)
            + np.sum(self.centroids ** 2, axis=1)[None, :]
            - 2 * (X_test @ self.centroids.T)
        )

        # clamp to 0 to avoid negative values 
        np.maximum(distances, 0, out=distances)

        # Index of nearest centroid
        nearest_indices = np.argmin(distances, axis=1)
        print("Class centroid finished processing all samples.")
        print(f"Processed {len(X_test)} samples in total.")
        return self.classes[nearest_indices]

    def score(self, X_test, y_test):
        '''
        Return accuracy
        '''
        y_test = np.asarray(y_test)
        predictions = self.predict(X_test)

        if len(y_test) != len(predictions):
            raise ValueError("X_test and y_test must have the same length")

        return np.mean(predictions == y_test)
