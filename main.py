import csv
import os
import time

import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np

from  models.knn import KNN
from models.class_centroid import ClassCentroid

# global constants
CIFAR_PATH = "data"
CIFAR_PCA_PATH = "cifar100_pca.npz"
TEST_DATA = 10000 # Number of test samples to use for evaluation
USE_PCA = False  # Set to True to use PCA-transformed data, False to use raw data
CSV_RES = "eval_results.csv"



def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train, evaluate, and time a classification model."""

    print(f"\nEvaluating {model_name}...")

    # Measure training time
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # Measure prediction time
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    prediction_time = time.perf_counter() - start

    # Measure accuracy calculation time
    start = time.perf_counter()
    accuracy = np.mean(np.asarray(y_pred) == np.asarray(y_test))
    evaluation_time = time.perf_counter() - start

    # Store results
    results = {
        "Model": model_name,
        "Accuracy": float(accuracy),
        "Training_Time_s": training_time,
        "Prediction_Time_s": prediction_time,
        "Evaluation_Time_s": evaluation_time,
        "Train_Samples": len(X_train),
        "Test_Samples": len(X_test),
        "PCA_Enabled": USE_PCA,
    }

    print(f"Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Training time:   {training_time:.4f} seconds")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print(f"Evaluation time: {evaluation_time:.6f} seconds")

    return results

def main():
    # Main
    if(USE_PCA == False):
        transform = transforms.ToTensor()
        train_data = datasets.CIFAR100(
            root=CIFAR_PATH,
            train=True,
            transform=transform,
            download=False
        )

        test_data = datasets.CIFAR100(
            root=CIFAR_PATH,
            train=False,
            transform=transform,
            download=False
        )

        # Load the data

        X_train = torch.stack([image for image, label in train_data])
        y_train = torch.tensor([label for image, label in train_data])

        X_test = torch.stack([image for image, label in test_data])
        y_test = torch.tensor([label for image, label in test_data])
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)[:TEST_DATA, :]

    else:
        data = np.load(CIFAR_PCA_PATH)
        X_train = data["X_train"]
        X_test = data["X_test"][:TEST_DATA, :]
        y_train = data["y_train"]
        y_test = data["y_test"][:TEST_DATA]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Features:         {X_train.shape[1]}")
    print(f"PCA enabled:      {USE_PCA}")

    # Initialize models
    knn1 = KNN(k=1, batch_size=100)
    knn3 = KNN(k=3, batch_size=100)
    centroid = ClassCentroid()

    # Evaluate both models
    results = []

    results.append(
        evaluate_model(
            knn1, "KNN1",
            X_train, y_train,
            X_test, y_test,
        )
    )
    
    results.append(
        evaluate_model(
            knn3, "KNN3",
            X_train, y_train,
            X_test, y_test,
        )
    )

    results.append(
        evaluate_model(
            centroid, "Class Centroid",
            X_train, y_train,
            X_test, y_test,
        )
    )

    file_exists = os.path.isfile(CSV_RES)
    file_is_empty = not file_exists or os.path.getsize(CSV_RES) == 0

    with open(CSV_RES, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())

        # Write headers only when creating a new file or when the file is empty
        if file_is_empty:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nResults appended to {CSV_RES}")

if __name__ == "__main__":
    main()