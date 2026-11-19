from .nn_obj.convolution import Conv
from .nn_obj.cnn_help_layers import MaxPool, Flatten
from .nn_obj.layer import Layer
from .nn_obj.adam import Adam

import cupy as cp

class CNN:
    
    def __init__(self, 
                 input_shape, 
                 layer_configs,
                 optimizer=None, 
                 learning_rate=0.01):
        
        self.layers = []
        self.optimizer = optimizer if optimizer is not None else Adam(learning_rate=learning_rate)

        # Construct network
        for i, config in enumerate(layer_configs):
            layer_type = config["type"]
            
            if layer_type == "conv":
                num_filters = config["num_filters"]
                filter_size = config["filter_size"]
                stride = config.get("stride", 1)
                padding = config.get("padding", 0)
                
                layer = Conv(input_shape=input_shape, 
                             num_filters=num_filters, 
                             filter_size=filter_size, 
                             stride=stride, 
                             padding=padding)
                
                # Update input shape for the next layer
                out_height = (input_shape[1] - filter_size + 2 * padding) // stride + 1
                out_width = (input_shape[2] - filter_size + 2 * padding) // stride + 1
                input_shape = (num_filters, out_height, out_width)
            
            elif layer_type == "maxpool":
                pool_size = config["pool_size"]
                stride = config.get("stride", pool_size)
                
                layer = MaxPool(pool_size=pool_size, stride=stride)
                
                # Update input shape for the next layer
                out_height = (input_shape[1] - pool_size) // stride + 1
                out_width = (input_shape[2] - pool_size) // stride + 1
                input_shape = (input_shape[0], out_height, out_width)
            
            elif layer_type == "flatten":
                layer = Flatten()
                
                # Update input shape for the next layer
                input_shape = (input_shape[0] * input_shape[1] * input_shape[2],)
            
            elif layer_type == "dense":
                output_size = config["units"]
                activation = config.get("activation", "relu")
                dropout = config.get("dropout", 0.0)

                # The input in a dense layer would have been flattened, and be in batches of shape (batch_size, input_size)
                layer = Layer(input_size=input_shape[1] if len(input_shape) > 1 else input_shape[0], 
                              output_size=output_size, 
                              activation=activation,
                              dropout=dropout)
                
                # Update input shape for the next layer
                input_shape = (output_size,)
            
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            self.layers.append(layer)

    def forward(self, X, training=True):
        for layer in self.layers:
            if isinstance(layer, Layer):
                X = layer.forward(X, training=training)
            else:
                X = layer.forward(X)
        return X
    
    def backward(self, y):
        batch_size = y.shape[0]

        probabilities = self.layers[-1].a

        dZ = probabilities.copy()
        dZ[cp.arange(batch_size), y] -= 1
        dZ /= batch_size

        gradients = [None] * len(self.layers)

        for i in range(len(self.layers) - 1, -1, -1):

            layer = self.layers[i]

            if isinstance(layer, Layer):
                # Backpropagation for dense layer
                dA_prev, dW, db = layer.backward(
                    dZ,
                    output_gradient=(i == len(self.layers) - 1)
                )
                gradients[i] = (dW, db)

            elif isinstance(layer, Conv):
                # Backpropagation for convolutional layer
                dA_prev, dW, db = layer.backward(dZ)
                gradients[i] = (dW, db)

            else:
                # For other layers (e.g., MaxPool, Flatten), just backpropagate the gradient
                dA_prev = layer.backward(dZ)

            dZ = dA_prev

        return gradients

    def update_parameters(self, gradients):

        parameters = []
        grads = []

        for layer, gradient in zip(self.layers, gradients):

            if gradient is None:
                continue

            dW, db = gradient

            if isinstance(layer, Conv):
                parameters.extend([layer.filters, layer.biases])
            elif isinstance(layer, Layer):
                parameters.extend([layer.weights, layer.biases])

            grads.extend([dW, db])

        self.optimizer.update(parameters, grads)
    
    def train_step(self, X, y):

        self.forward(X, training=True)

        gradients = self.backward(y)

        self.update_parameters(gradients)

    def predict(self, X, batch_size=128):
        predictions = []

        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])

            X_batch = cp.asarray(X[start:end])

            probabilities = self.forward(X_batch, training=False)
            batch_predictions = cp.argmax(probabilities, axis=1)

            predictions.append(batch_predictions)

        return cp.concatenate(predictions)
    
    def evaluate(self, X, y, batch_size=128):
        n_samples = X.shape[0]
        correct = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            X_batch = cp.asarray(X[start:end])
            y_batch = cp.asarray(y[start:end])

            probabilities = self.forward(X_batch, training=False)
            predictions = cp.argmax(probabilities, axis=1)

            correct += int(cp.sum(predictions == y_batch).item())

        accuracy = correct / n_samples
        return accuracy