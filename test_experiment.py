from utils.data_loader import load_cifar100, load_cupy_dataset
from utils.experiment_runner import ExperimentRunner

import numpy as np


def create_subset(data, num_classes=10,
                  train_per_class=100,
                  val_per_class=20,
                  test_per_class=20,
                  seed=42):

    rng = np.random.default_rng(seed)

    classes = np.arange(num_classes)

    def select(X, y, samples_per_class):

        X_selected = []
        y_selected = []

        for new_label, original_label in enumerate(classes):

            indices = np.where(y == original_label)[0]
            chosen = rng.choice(indices, samples_per_class, replace=False)

            X_selected.append(X[chosen])
            y_selected.append(
                np.full(samples_per_class, new_label, dtype=np.int64)
            )

        X_selected = np.concatenate(X_selected)
        y_selected = np.concatenate(y_selected)

        permutation = rng.permutation(len(y_selected))

        return X_selected[permutation], y_selected[permutation]

    X_train, y_train = select(data["X_train"], data["y_train"], train_per_class)
    X_val, y_val = select(data["X_val"], data["y_val"], val_per_class)
    X_test, y_test = select(data["X_test"], data["y_test"], test_per_class)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }

def main():

    print("Loading CIFAR-100...")
    data = load_cifar100()

    print("Creating subset...")
    data = create_subset(data)

    print("Transferring data to GPU...")
    data = load_cupy_dataset(data)

    experiments = [

        {
            "name": "mini_mlp",
            "model_type": "mlp",
            "input_size": 3072,
            "layer_configs": [
                {"units": 64, "activation": "relu"},
                {"units": 10, "activation": "softmax"}
            ],
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        },

        {
            "name": "mini_cnn",
            "model_type": "cnn",
            "input_shape": (3, 32, 32),
            "layer_configs": [
                {
                    "type": "conv",
                    "num_filters": 8,
                    "filter_size": 3,
                    "stride": 1,
                    "padding": 1
                },
                {
                    "type": "maxpool",
                    "pool_size": 2,
                    "stride": 2
                },
                {"type": "flatten"},
                {
                    "type": "dense",
                    "units": 32,
                    "activation": "relu"
                },
                {
                    "type": "dense",
                    "units": 10,
                    "activation": "softmax"
                }
            ],
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    ]

    runner = ExperimentRunner(data)

    results = runner.run_all(experiments)

    print("\n========== FINAL RESULTS ==========")

    for experiment in results:

        result = experiment["result"]

        print(f"\n{result['experiment_name']}")
        print(f"Best validation accuracy: {result['best_val_accuracy']:.4f}")
        print(f"Test accuracy:            {result['test_accuracy']:.4f}")
        print(f"Test loss:                {result['test_loss']:.4f}")
        print(f"Training time:            {result['total_training_time']:.2f}s")


if __name__ == "__main__":
    main()