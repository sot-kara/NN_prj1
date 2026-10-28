import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

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
X_test = X_test.reshape(X_test.shape[0], -1)


X_train_np = X_train.numpy().astype(np.float32)
X_test_np = X_test.numpy().astype(np.float32)

y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

# Apply PCA to reduce dimensionality
pca = PCA(
    n_components=100,
    svd_solver="randomized",
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_np)
X_test_pca = pca.transform(X_test_np)

print("Training shape:", X_train_pca.shape)
print("Test shape:", X_test_pca.shape)
print("Explained variance:", pca.explained_variance_ratio_.sum())

# Save the PCA-transformed data to a compressed .npz file
np.savez_compressed(
    "cifar100_pca.npz",
    X_train=X_train_pca,
    X_test=X_test_pca,
    y_train=y_train_np,
    y_test=y_test_np
)
