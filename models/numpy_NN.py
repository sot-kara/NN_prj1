import cupy as cp

from .nn_obj.layer import Layer
from .nn_obj.adam import Adam


class MLP:

    def __init__(
        self,
        input_size,
        layer_configs,
        optimizer=None,
        learning_rate=0.01
    ):

        self.layers = []

        # Default optimizer
        self.optimizer = (
            optimizer if optimizer is not None
            else Adam(learning_rate=learning_rate)
        )

        # Construct network
        for i, config in enumerate(layer_configs):

            output_size = config["units"]

            activation = config.get("activation", "relu")

            dropout = config.get("dropout", 0.0)

            # Output layer must not use dropout
            if i == len(layer_configs) - 1:
                dropout = 0.0

                if activation != "softmax":
                    raise ValueError(
                        "Output layer must use softmax."
                    )

            layer = Layer(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
                dropout=dropout
            )

            self.layers.append(layer)

            input_size = output_size

    # -------------------------
    # Forward propagation
    # -------------------------

    def forward(self, X, training=True):

        A = X

        for layer in self.layers:
            # Forward pass through each layer, A is used as input for the next layer
            A = layer.forward(A, training)

        return A

    # -------------------------
    # Backpropagation
    # -------------------------

    def backward(self, y):

        batch_size = y.shape[0]

        # Softmax + cross-entropy gradient
        probabilities = self.layers[-1].a

        dZ = probabilities.copy()

        dZ[cp.arange(batch_size), y] -= 1

        dZ /= batch_size

        gradients = []

        # Traverse layers backwards
        for i in reversed(range(len(self.layers))):

            layer = self.layers[i]

            # Output layer already has its gradient
            is_output_layer = (i == len(self.layers) - 1)

            dA_prev, dW, db = layer.backward(
                dZ,
                output_gradient=is_output_layer
            )

            gradients.insert(0, (dW, db))

            dZ = dA_prev

        return gradients

    # -------------------------
    # Parameter updates
    # -------------------------

    def update_parameters(self, gradients):

        parameters = []
        grads = []

        for layer, (dW, db) in zip(self.layers, gradients):

            parameters.extend([layer.weights, layer.biases])
            grads.extend([dW, db])

        self.optimizer.update(parameters, grads)

    # -------------------------
    # Training step
    # -------------------------

    def train_step(self, X, y):

        self.forward(X, training=True)

        gradients = self.backward(y)

        self.update_parameters(gradients)

    # -------------------------
    # Prediction
    # -------------------------

    def predict(self, X):

        probabilities = self.forward(X, training=False)

        return cp.argmax(probabilities, axis=1)

    # -------------------------
    # Evaluation
    # -------------------------

    def evaluate(self, X, y):

        predictions = self.predict(X)

        return cp.mean(predictions == y)