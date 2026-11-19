import json
import numpy as np
import cupy as cp

from pathlib import Path


class ModelSaver:

    def __init__(self):
        self.best_val_accuracy = -float("inf")
        self.best_epoch = None

    # ---------------------------------------
    # Save model
    # ---------------------------------------

    def save_model(self, model, filepath, metadata=None):

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        parameters = {}

        # Extract parameters from every layer
        for i, layer in enumerate(model.layers):

            if hasattr(layer, "filters"):
                # Convolutional layer
                parameters[f"layer_{i}_filters"] = cp.asnumpy(
                    layer.filters
                )

                parameters[f"layer_{i}_biases"] = cp.asnumpy(
                    layer.biases
                )

            elif hasattr(layer, "weights"):
                # Dense layer
                parameters[f"layer_{i}_weights"] = cp.asnumpy(
                    layer.weights
                )

                parameters[f"layer_{i}_biases"] = cp.asnumpy(
                    layer.biases
                )

        # Save numerical parameters
        np.savez_compressed(filepath, **parameters)

        # Save metadata separately
        metadata = metadata or {}

        metadata_path = filepath.with_suffix(".json")

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # ---------------------------------------
    # Load model
    # ---------------------------------------

    def load_model(self, model, filepath):

        filepath = Path(filepath)

        # Load numerical parameters
        data = np.load(filepath, allow_pickle=False)

        for i, layer in enumerate(model.layers):

            if hasattr(layer, "filters"):

                filters_key = f"layer_{i}_filters"
                biases_key = f"layer_{i}_biases"

                if layer.filters.shape != data[filters_key].shape:
                    raise ValueError(
                        f"Shape mismatch in convolutional layer {i} filters."
                    )

                layer.filters = cp.asarray(data[filters_key])
                layer.biases = cp.asarray(data[biases_key])

            elif hasattr(layer, "weights"):

                weights_key = f"layer_{i}_weights"
                biases_key = f"layer_{i}_biases"

                if layer.weights.shape != data[weights_key].shape:
                    raise ValueError(
                        f"Shape mismatch in dense layer {i} weights."
                    )

                layer.weights = cp.asarray(data[weights_key])
                layer.biases = cp.asarray(data[biases_key])

        data.close()

        # Load metadata if available
        metadata_path = filepath.with_suffix(".json")

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return metadata

    # ---------------------------------------
    # Save best model
    # ---------------------------------------

    def save_best_model(
        self,
        model,
        filepath,
        val_accuracy,
        epoch,
        metadata=None
    ):

        val_accuracy = float(val_accuracy)

        if val_accuracy <= self.best_val_accuracy:
            return False

        self.best_val_accuracy = val_accuracy
        self.best_epoch = epoch

        metadata = metadata or {}

        metadata.update({
            "best_epoch": epoch,
            "best_val_accuracy": val_accuracy
        })

        self.save_model(
            model=model,
            filepath=filepath,
            metadata=metadata
        )

        return True