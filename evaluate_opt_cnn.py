from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

from models.pytorch_CNN import PytorchCNN
from utils.data_loader import load_cifar100


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path("results/cnn_optimized/models/cnn_optimized.pth")

NUM_CLASSES_TO_SHOW = 10
NUM_EXAMPLES = 3
BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# CIFAR-100 normalization
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading CIFAR-100...")

data = load_cifar100()

X_test = (
    data["X_test"]
    .transpose(0, 3, 1, 2)
    .astype(np.float32) / 255.0
)

y_test = data["y_test"]

print(f"Test samples: {len(X_test)}")


# Load class names
cifar_dataset = datasets.CIFAR100(
    root="data",
    train=False,
    download=False
)

class_names = cifar_dataset.classes


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading optimized CNN...")

model = PytorchCNN(
    num_classes=100,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Support both checkpoint dictionaries and raw state dictionaries
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print(f"Model loaded on: {DEVICE}")


# ============================================================
# PREPROCESSING
# ============================================================

X_test_tensor = torch.from_numpy(X_test)

y_test_tensor = torch.as_tensor(
    y_test,
    dtype=torch.long
)

normalization = transforms.Normalize(
    mean=MEAN,
    std=STD
)

X_test_normalized = torch.stack([
    normalization(image)
    for image in X_test_tensor
])

test_dataset = TensorDataset(
    X_test_normalized,
    y_test_tensor
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\nEvaluating model...")

all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

predictions = np.array(all_predictions)

correct = predictions == y_test

overall_accuracy = np.mean(correct) * 100

print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")


# ============================================================
# PER-CLASS ACCURACY - FIRST 10 CLASSES
# ============================================================

class_accuracies = []
class_counts = []

print("\nPer-class accuracy:")

for class_id in range(NUM_CLASSES_TO_SHOW):

    # Samples belonging to the current class
    mask = y_test == class_id

    class_predictions = predictions[mask]
    class_labels = y_test[mask]

    accuracy = np.mean(
        class_predictions == class_labels
    ) * 100

    class_accuracies.append(accuracy)
    class_counts.append(np.sum(mask))

    print(
        f"{class_id:2d} - {class_names[class_id]:20s}: "
        f"{accuracy:6.2f}% "
        f"({np.sum(class_predictions == class_labels)}/"
        f"{np.sum(mask)})"
    )


# ============================================================
# FIGURE 1: PER-CLASS ACCURACY
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(
    class_names[:NUM_CLASSES_TO_SHOW],
    class_accuracies,
    color="tab:blue",
    edgecolor="black",
    linewidth=0.7
)

ax.bar_label(
    bars,
    labels=[f"{value:.1f}%" for value in class_accuracies],
    padding=3,
    fontsize=10
)

ax.set_title(
    "Optimized CNN - Accuracy per Class",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)

ax.set_ylim(
    0,
    max(class_accuracies) * 1.2
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.set_axisbelow(True)

plt.xticks(rotation=35, ha="right")

fig.tight_layout()

plt.show()
plt.close(fig)


# ============================================================
# FIND CORRECT AND INCORRECT EXAMPLES
# ============================================================

# Restrict examples to the first 10 classes
first_10_mask = y_test < NUM_CLASSES_TO_SHOW

correct_indices = np.where(
    correct & first_10_mask
)[0]

incorrect_indices = np.where(
    (~correct) & first_10_mask
)[0]

# Select up to 3 examples of each category
correct_examples = correct_indices[:NUM_EXAMPLES]
incorrect_examples = incorrect_indices[:NUM_EXAMPLES]


# ============================================================
# FIGURE 2: CORRECT CLASSIFICATIONS
# ============================================================

def plot_examples(indices, title, correct_classification):

    if len(indices) == 0:
        print(f"No examples available for: {title}")
        return

    fig, axes = plt.subplots(
        1,
        len(indices),
        figsize=(5 * len(indices), 5)
    )

    if len(indices) == 1:
        axes = [axes]

    for ax, index in zip(axes, indices):

        # Original image, before normalization
        image = X_test[index].transpose(1, 2, 0)

        true_label = int(y_test[index])
        predicted_label = int(predictions[index])

        ax.imshow(image)

        ax.set_title(
            f"True: {class_names[true_label]}\n"
            f"Predicted: {class_names[predicted_label]}",
            fontsize=11
        )

        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold"
    )

    fig.tight_layout()

    plt.show()
    plt.close(fig)


# Correct classifications
plot_examples(
    correct_examples,
    "Optimized CNN - Correct Classifications",
    True
)


# Incorrect classifications
plot_examples(
    incorrect_examples,
    "Optimized CNN - Incorrect Classifications",
    False
)


print("\nEvaluation completed successfully!")