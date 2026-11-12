import cupy as cp

class Adam:

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.m = None
        self.v = None

    def update(self, parameters, gradients):

        if self.m is None:

            self.m = [cp.zeros_like(p) for p in parameters]
            self.v = [cp.zeros_like(p) for p in parameters]

        self.t += 1

        for i, (p, g) in enumerate(zip(parameters, gradients)):

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1 - self.beta1) * g
            )

            self.v[i] = (
                self.beta2 * self.v[i]
                + (1 - self.beta2) * g**2
            )

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Parameter update
            p -= self.lr * m_hat / (
                cp.sqrt(v_hat) + self.epsilon
            )