import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# MLP MODEL
# ==========================================

class MLP(nn.Module):

    def __init__(self, input_size=3072, num_classes=10):

        super().__init__()

        self.network = nn.Sequential(

            # Hidden layer 1
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # Flatten images: (batch, 3, 32, 32) -> (batch, 3072)
        x = x.view(x.size(0), -1)

        return self.network(x)


# ==========================================
# TRAINING
# ==========================================

def train(model, train_loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        predicted = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


# ==========================================
# EVALUATION
# ==========================================

def evaluate(model, test_loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    return epoch_loss, accuracy


# ==========================================
# MAIN
# ==========================================

def main():

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Dataset preprocessing
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10
    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transform
    )

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 50

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model
    model = MLP().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Training
    for epoch in range(epochs):

        train_loss, train_acc = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

main()