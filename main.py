import torch
from torchvision import datasets, transforms

from  models.knn import KNN
from models.class_centroid import ClassCentroid

# global constants
TEST_DATA = 10000

def main():
    # Main
    model = KNN(k=3);
    model_centr = ClassCentroid();
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

    # Load the data

    X_train = torch.stack([image for image, label in train_data])
    y_train = torch.tensor([label for image, label in train_data])

    X_test = torch.stack([image for image, label in test_data])
    y_test = torch.tensor([label for image, label in test_data])


    # Convert RGB images to grayscale by averaging the color channels.
    X_train = X_train.mean(dim=1)
    X_test = X_test.mean(dim=1)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)[:TEST_DATA, :]
    
    y_test = y_test[:TEST_DATA]
    

    model.fit(X_train.numpy(), y_train.numpy())
    model_centr.fit(X_train.numpy(), y_train.numpy())

    print("Accuracy (KNN):", model.score(X_test.numpy(), y_test.numpy()))
    print("Accuracy (Class Centroid):", model_centr.score(X_test.numpy(), y_test.numpy()))

if __name__ == "__main__":
    main()