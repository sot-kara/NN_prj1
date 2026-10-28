import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np

from  models.knn import KNN
from models.class_centroid import ClassCentroid

# global constants
TEST_DATA = 10000
USE_PCA = True

def main():
    # Main
    model = KNN(k=3);
    model_centr = ClassCentroid();
    transform = transforms.ToTensor()
    if(USE_PCA == False):
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

        # Load the data

        X_train = torch.stack([image for image, label in train_data])
        y_train = torch.tensor([label for image, label in train_data])

        X_test = torch.stack([image for image, label in test_data])
        y_test = torch.tensor([label for image, label in test_data])
        
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)[:TEST_DATA, :]

    else:
        data = np.load("cifar100_pca.npz")
        X_train = data["X_train"]
        X_test = data["X_test"][:TEST_DATA, :]
        y_train = data["y_train"]
        y_test = data["y_test"][:TEST_DATA]

    model.fit(X_train, y_train)
    model_centr.fit(X_train, y_train)

    print("Accuracy (KNN):", model.score(X_test, y_test))
    print("Accuracy (Class Centroid):", model_centr.score(X_test, y_test))

if __name__ == "__main__":
    main()