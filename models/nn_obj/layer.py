import cupy as cp
from .activations import ACTIVATIONS


class Layer:

    def __init__(
        self,
        input_size,
        output_size,
        activation="relu",
        dropout=0.0
    ):

        self.actv_name = activation
        self.actv_func = ACTIVATIONS[activation]

        self.dropout = dropout

        # Weight initialization
        if activation == "relu":
            std = cp.sqrt(2.0 / input_size)
        else:
            std = cp.sqrt(1.0 / input_size)
        
        # Use He initialization for ReLU and Xavier initialization for other activations
        self.weights = (
            cp.random.randn(input_size, output_size) * std
        )

        self.biases = cp.zeros((1, output_size))

        # Forward propagation cache
        self.input = None
        self.z = None
        self.a = None
        self.dropout_mask = None

    def forward(self, X, training=True):

        self.input = X

        # Linear transformation
        self.z = X @ self.weights + self.biases

        # Activation
        self.a = self.actv_func.forward(self.z)

        # Dropout
        if training and self.dropout > 0:

            keep_probability = 1.0 - self.dropout

            self.dropout_mask = (
                cp.random.rand(*self.a.shape) < keep_probability
            ).astype(self.a.dtype)

            # Inverted dropout
            output = (
                self.a * self.dropout_mask
            ) / keep_probability

        else:

            self.dropout_mask = None
            output = self.a

        return output

    def backward(self, dA, output_gradient=False):

        # Backpropagate through dropout
        if self.dropout_mask is not None:

            keep_probability = 1.0 - self.dropout

            dA = (
                dA * self.dropout_mask
            ) / keep_probability

        # Backpropagate through activation
        if output_gradient:

            # Softmax + cross-entropy:
            # Gradient is already calculated with respect to Z.
            dZ = dA

        else:

            dZ = (
                dA * self.actv_func.backward(self.z, self.a)
            )

        # Calculate weight and bias gradients
        dW = self.input.T @ dZ
        db = cp.sum(dZ, axis=0, keepdims=True)

        # Propagate gradient to previous layer
        dA_prev = dZ @ self.weights.T

        return dA_prev, dW, db