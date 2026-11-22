import torch
import numpy as np

from torchvision import transforms

from models.pytorch_CNN import PytorchCNN
from utils.torch_trainer import TorchTrainer
from utils.history_manager import HistoryManager
from utils.results_manager import ResultsManager

from utils.data_loader import load_cifar100


# ==========================================
# CONFIGURATION
# ==========================================

EXPERIMENT_NAME = "cnn_optimized"

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001

SEED = 42


# ==========================================
# RANDOM SEED
# ==========================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# DATA
# ==========================================

data = load_cifar100()

X_train = data["X_train"].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
X_val = data["X_val"].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
X_test = data["X_test"].transpose(0, 3, 1, 2).astype(np.float32) / 255.0

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


# ==========================================
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])

eval_transform = transforms.Normalize(
    mean=(0.5071, 0.4867, 0.4408),
    std=(0.2675, 0.2565, 0.2761)
)


# ==========================================
# MODEL
# ==========================================

model = PytorchCNN(
    num_classes=100,
    dropout=0.3
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ==========================================
# SCHEDULER
# ==========================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)


# ==========================================
# MANAGERS
# ==========================================

history_manager = HistoryManager(results_dir=f"results/{EXPERIMENT_NAME}/history")

results_manager = ResultsManager(filepath=f"results/{EXPERIMENT_NAME}/csv/experiments.csv")


# ==========================================
# TRAINER
# ==========================================

trainer = TorchTrainer(
    model=model,
    optimizer=optimizer,

    criterion=torch.nn.CrossEntropyLoss(),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    seed=SEED,

    device="cuda" if torch.cuda.is_available() else "cpu",

    history_manager=history_manager,
    results_manager=results_manager,

    experiment_name=EXPERIMENT_NAME,
    model_type="CNN_Optimized",

    learning_rate=LEARNING_RATE,

    checkpoint_path=f"results/{EXPERIMENT_NAME}/models/{EXPERIMENT_NAME}.pth",
    save_best=True,

    train_transform=train_transform,
    eval_transform=eval_transform,

    scheduler=scheduler
)


# ==========================================
# TRAINING
# ==========================================

trainer.fit(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

trainer.load_best_model()


# ==========================================
# TEST EVALUATION
# ==========================================

test_loss, test_accuracy = trainer.evaluate(
    X_test,
    y_test
)


# ==========================================
# SAVE RESULTS
# ==========================================

trainer.save_results(
    test_loss=test_loss,
    test_accuracy=test_accuracy
)


print("\n================================")
print("FINAL RESULTS")
print("================================")

print(f"Best Epoch: {trainer.best_epoch}")
print(f"Best Validation Accuracy: {trainer.best_val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Training Time: {trainer.total_training_time:.2f}s")