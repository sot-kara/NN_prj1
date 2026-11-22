import numpy as np
import cupy as cp
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from sklearn.model_selection import train_test_split


# ==========================================
# CONFIGURATION
# ==========================================

DATA_ROOT = "data"

RANDOM_SEED = 42

NUM_CLASSES = 100

VAL_SIZE = 0.10


# ==========================================
# LOAD CIFAR-100
# ==========================================

def load_cifar100(
    root=DATA_ROOT,
    val_size=VAL_SIZE,
    random_state=RANDOM_SEED,
    download=False
):

    print("Loading CIFAR-100...")

    # Load original datasets
    train_dataset = datasets.CIFAR100(
        root=root,
        train=True,
        download=download
    )

    test_dataset = datasets.CIFAR100(
        root=root,
        train=False,
        download=download
    )

    # Original data (uint8, NHWC)
    X = train_dataset.data

    y = np.array(train_dataset.targets, dtype=np.int64)

    X_test = test_dataset.data
    y_test = np.array(test_dataset.targets, dtype=np.int64)

    # Stratified train / validation split
    train_indices, val_indices = train_test_split(
        np.arange(len(X)),
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )

    # Split datasets
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    print("\nDataset loaded successfully.")

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    print(f"Number of classes:  {NUM_CLASSES}")

    print("\nInput shape:", X_train.shape)
    print("Input dtype:", X_train.dtype)

    return {

        "X_train": X_train,
        "y_train": y_train,

        "X_val": X_val,
        "y_val": y_val,

        "X_test": X_test,
        "y_test": y_test,

        "class_names": train_dataset.classes

    }


# ==========================================
# CUPY BATCH CONVERSION
# ==========================================

def load_cupy_dataset(data):
    # Load dataset into CuPy arrays in the GPU

    return {
        "train": (
            cp.asarray(data["X_train"], dtype=cp.float32),
            cp.asarray(data["y_train"], dtype=cp.int64)
        ),

        "val": (
            cp.asarray(data["X_val"], dtype=cp.float32),
            cp.asarray(data["y_val"], dtype=cp.int64)
        ),

        "test": (
            cp.asarray(data["X_test"], dtype=cp.float32),
            cp.asarray(data["y_test"], dtype=cp.int64)
        )
    }


def to_cupy_batch(X, y, model_type):

    import cupy as cp

    # Convert images to float32 [0, 1]
    X = cp.asarray(X, dtype=cp.float32) / 255.0

    # Convert NHWC -> NCHW for CNN
    if model_type == "CNN":

        X = cp.transpose(X, (0, 3, 1, 2))

        X = cp.ascontiguousarray(X)

    # Flatten images for MLP
    elif model_type == "MLP":

        X = X.reshape(X.shape[0], -1)

    else:

        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    y = cp.asarray(y, dtype=cp.int64)

    return X, y


# ==========================================
# PYTORCH DATASET
# ==========================================

class CIFAR100TorchDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        # NHWC -> CHW
        image = torch.from_numpy(
            self.X[index]
        ).permute(2, 0, 1)

        # uint8 -> float32 [0, 1]
        image = image.float() / 255.0

        label = torch.tensor(
            self.y[index],
            dtype=torch.long
        )

        return image, label


# ==========================================
# PYTORCH DATALOADERS
# ==========================================

def create_torch_dataloaders(
    data,
    batch_size=128,
    num_workers=0,
    seed=RANDOM_SEED
):

    # Create datasets
    train_dataset = CIFAR100TorchDataset(
        data["X_train"],
        data["y_train"]
    )

    val_dataset = CIFAR100TorchDataset(
        data["X_val"],
        data["y_val"]
    )

    test_dataset = CIFAR100TorchDataset(
        data["X_test"],
        data["y_test"]
    )

    # Reproducible shuffling
    generator = torch.Generator()

    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader