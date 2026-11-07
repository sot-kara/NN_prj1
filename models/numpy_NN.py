import numpy as np

class MLP:
    def __init__(self,layers, activation='relu', learning_rate=0.01, seed=42):
        '''
    Initialize the MLP.
        :param layers: List of integers representing the number of neurons in each layer.
        :param activation: The activation function to use.
        :param learning_rate: The learning rate for the optimizer.
        :param seed: The seed for the random number generator.
        '''
        
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            weight = rng.normal(0, 1, (layers[i], layers[i + 1])) # Initialize weights with a normal distribution
            bias = np.zeros((1, layers[i + 1])) # Initialize biases to zero
            self.weights.append(weight)
            self.biases.append(bias)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        '''
        Perform a forward pass through the network.
        :param X: Input data.
        :return: Output of the network.
        '''
        self.z_values = [] # Store the linear combinations (z values) for backpropagation
        self.activations = [X]
        for i in range(len(self.weights)):
            # Compute the linear combination of inputs and weights, add bias
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                # For all layers except the last, use ReLU activation
                a = self.relu(z)
            else:
                # For the last layer, use softmax activation
                a = self.softmax(z)
            self.activations.append(a) # Store the activation for backpropagation
        return self.activations[-1] # Return the output of the last layer

    def backward(self, y):
            
            """
            Calculate gradients and perform backpropagation.

            y: integer class labels
            e.g. y = [1 2 1 10 9 ...] for a batch of size N

            Assumes softmax output and cross-entropy loss.
            """

            batch_size = y.shape[0]

            # Gradient of cross-entropy loss with respect to
            # the output logits, after combining with softmax.
            dZ = self.activations[-1].copy()

            dZ[np.arange(batch_size), y] -= 1
            dZ /= batch_size

            grad_weights = [None] * len(self.weights)
            grad_biases = [None] * len(self.biases)

            # Propagate gradients backward through the network
            for i in reversed(range(len(self.weights))):
                A_prev = self.activations[i] 
                W = self.weights[i]

                # Gradient of loss with respect to weights and biases for current layer
                grad_weights[i] = A_prev.T @ dZ
                grad_biases[i] = np.sum(
                    dZ, axis=0, keepdims=True
                )

                if i > 0:
                    dA_prev = dZ @ W.T

                    # Derivative of ReLU
                    dZ = dA_prev * (self.z_values[i - 1] > 0)

            return grad_weights, grad_biases
    
    def update_parameters(self, grad_weights, grad_biases):
        '''
        Update the weights and biases using the computed gradients.
        :param grad_weights: Gradients of the weights.
        :param grad_biases: Gradients of the biases.
        '''
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def predict(self, X):
        '''
        Predict the class labels for the input data.
        :param X: Input data.
        :return: Predicted class labels.
        '''
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, y):
        '''
        Evaluate the model's accuracy on the given data.
        :param X: Input data.
        :param y: True class labels.
        :return: Accuracy of the model.
        '''
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
