import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, num_classes=100):
        super(CNN, self).__init__()
        # Conv 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.batch_norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.batch_norm2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Conv 2
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.batch_norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.batch_norm4 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Dense layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):

        # Conv 1
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool1(x)

        # Conv 2
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Dense 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output
        x = self.fc2(x)

        return x
    

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

def evaluate_train(model, train_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

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

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
    ])

    # Load CIFAR-100
    train_dataset = datasets.CIFAR100(
        root="data",
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root="data",
        train=False,
        download=False,
        transform=test_transform
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
    model = CNN().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6
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
        
        #Evaluate on training set with dropout disabled
        train_acc_eval = evaluate_train(
           model,
           train_loader,
           device
        )   
        
        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"Learning Rate: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc (Eval): {train_acc_eval:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )

main()

    
