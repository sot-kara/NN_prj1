import cupy as cp

class Linear:
    
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def backward(z, a):
        return cp.ones_like(z)

class ReLU:

    @staticmethod
    def forward(z):
        return cp.maximum(0, z)

    @staticmethod
    def backward(z, a):
        return (z > 0).astype(z.dtype)


class Sigmoid:

    @staticmethod
    def forward(z):
        return 1.0 / (1.0 + cp.exp(-z))

    @staticmethod
    def backward(z, a):
        return a * (1.0 - a)


class Tanh:

    @staticmethod
    def forward(z):
        return cp.tanh(z)

    @staticmethod
    def backward(z, a):
        return 1.0 - a**2


class Softmax:

    @staticmethod
    def forward(z):
        exp_z = cp.exp(z - cp.max(z, axis=1, keepdims=True))
        return exp_z / cp.sum(exp_z, axis=1, keepdims=True)
    
ACTIVATIONS = {
    "linear": Linear,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax
}