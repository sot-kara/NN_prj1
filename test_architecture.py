from pathlib import Path

from utils.experiment_runner import ExperimentRunner
from utils.plot_history import plot_history
import numpy as np

from utils.data_loader import load_cifar100, load_cupy_dataset



# ==========================================
# CONFIGURATION
# ==========================================

RESULTS_DIR = Path("results")

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001

SEED = 42

NUM_CLASSES = 100

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 20
TEST_PER_CLASS = 20


# ==========================================
# EXPERIMENT CONFIGURATIONS
# ==========================================

experiments = [

    {
        "name": "mlp_small",
        "model_type": "mlp",
        "layer_configs": [
            {"units": 128, "activation": "relu"},
            {"units": 100, "activation": "softmax"}
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_history": True,
        "input_size": 32 * 32 * 3,  # CIFAR-100 images are 32x32 with 3 channels
    },

    {
        "name": "mlp_mid",
        "model_type": "mlp",
        "layer_configs": [
            {"units": 512, "activation": "relu"},
            {"units": 256, "activation": "relu"},
            {"units": 128, "activation": "relu"},
            {"units": 100, "activation": "softmax"}
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_history": True,
        "input_size": 32 * 32 * 3,  # CIFAR-100 images are 32x32 with 3 channels
    },
    {
        "name": "mlp_large",
        "model_type": "mlp",
        "layer_configs": [
            {"units": 1024, "activation": "relu"},
            {"units": 512, "activation": "relu"},
            {"units": 256, "activation": "relu"},
            {"units": 128, "activation": "relu"},
            {"units": 100, "activation": "softmax"}
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_history": True,
        "input_size": 32 * 32 * 3,  # CIFAR-100 images are 32x32 with 3 channels
    },

    {
        "name": "cnn_small",
        "model_type": "cnn",
        "layer_configs": [
            {"type": "conv", "num_filters": 8, "filter_size": 3,
             "stride": 1, "padding": 1},
            {"type": "maxpool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 100, "activation": "softmax"}
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_history": True,
        "input_shape": (3, 32, 32),  # CIFAR-100 images are 32x32 with 3 channels

    },

    {
        "name": "cnn_mid",
        "model_type": "cnn",
        "layer_configs": [
            {"type": "conv", "num_filters": 16, "filter_size": 3,
             "stride": 1, "padding": 1},
            {"type": "maxpool", "pool_size": 2, "stride": 2},
            {"type": "conv", "num_filters": 32, "filter_size": 3,
             "stride": 1, "padding": 1},
            {"type": "maxpool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 100, "activation": "softmax"}
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_history": True,
        "input_shape": (3, 32, 32),  # CIFAR-100 images are 32x32 with 3 channels
    },
        {
        "name": "cnn_large",
        "model_type": "cnn",
        "layer_configs": [
            {"type": "conv", "num_filters": 32, "filter_size": 3,
             "stride": 1, "padding": 1},
             {"type": "conv", "num_filters": 32, "filter_size": 3,
             "stride": 1, "padding": 1},
            {"type": "maxpool", "pool_size": 2, "stride": 2},
            {"type": "conv", "num_filters": 64, "filter_size": 3,
             "stride": 1, "padding": 1},
            {"type": "maxpool", "pool_size": 2, "stride": 2},
            {"type": "flatten"},
            {"type": "dense", "units": 256, "activation": "relu"},
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 100, "activation": "softmax"}
        ],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "save_history": True,
        "input_shape": (3, 32, 32),  # CIFAR-100 images are 32x32 with 3 channels
    }
]


# ==========================================
# MAIN
# ==========================================

def main():

    # Load dataset
    print("Loading CIFAR-100...")
    data = load_cifar100()


    # Transfer data to GPU
    data = load_cupy_dataset(data)

    print("\nStarting experiments...\n")

    results = []

    for experiment in experiments:

        name = experiment["name"]

        print("\n" + "=" * 60)
        print(f"RUNNING EXPERIMENT: {name}")
        print("=" * 60)

        # Each experiment gets its own results directory
        experiment_dir = RESULTS_DIR / name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        FIGURES_DIR = experiment_dir / "figures"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Create runner for this experiment
        runner = ExperimentRunner(
            dataset=data,
            results_dir=str(experiment_dir)
        )

        # Train model
        result = runner.run_all([experiment])

        results.extend(result)

        # Plot history
        history_file = (
            experiment_dir /
            "history" /
            f"{name}.csv"
        )

        if not history_file.exists():
            raise FileNotFoundError(
                f"History file not found: {history_file}"
            )

        print(f"\nGenerating plots for {name}...")

        plot_history(
            history_file,
            save_dir=FIGURES_DIR
        )

        print(f"Experiment {name} completed!")

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 60)

    print(f"\nFigures saved in: {FIGURES_DIR}")

    print("\nExperiment summary:")

    for item in results:
        result = item["result"]

        print(
            f"{result['experiment_name']:15s} | "
            f"Best Val Acc: {result['best_val_accuracy']:.4f} | "
            f"Test Acc: {result['test_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()