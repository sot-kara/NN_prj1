import cupy as cp


class MaxPool:

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.argmax = None  # To store the indices of the maximum values during forward pass to use in backward pass
        self.input_shape = None  # To store the input shape for backward pass
        
    def forward(self, X):
        N, C, H, W = X.shape

        self.input_shape = X.shape  # Store the input shape for backward pass

        # Calculate output dimensions
        out_height = (H - self.pool_size) // self.stride + 1
        out_width = (W - self.pool_size) // self.stride + 1

        # Initialize output and argmax arrays
        output = cp.zeros((N, C, out_height, out_width), dtype=cp.float32)
        self.argmax = cp.zeros((N, C, out_height, out_width), dtype=cp.int32)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # Extract the pooling region
                pool_region = X[:, :, h_start:h_end, w_start:w_end]

                # Find the maximum value and its index in the pooling region
                max_values = cp.max(pool_region, axis=(2, 3))
                argmax_indices = cp.argmax(pool_region.reshape(N, C, -1), axis=2)

                output[:, :, i, j] = max_values
                self.argmax[:, :, i, j] = argmax_indices

        return output

    def backward(self, dA):
        N, C, H, W = dA.shape
        dX = cp.zeros(self.input_shape, dtype=cp.float32)

        for i in range(H):
            for j in range(W):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                # Create a mask for the max values
                mask = cp.zeros((N, C, self.pool_size * self.pool_size), dtype=cp.float32)
                mask[cp.arange(N)[:, None], cp.arange(C), self.argmax[:, :, i, j]] = 1

                # Reshape the mask to match the pooling region shape
                mask = mask.reshape(N, C, self.pool_size, self.pool_size)

                # Distribute the gradients to the max locations
                dX[:, :, h_start:h_end, w_start:w_end] += mask * dA[:, :, i:i + 1, j:j + 1]

        return dX
    
class Flatten:

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dA):
        return dA.reshape(self.input_shape)
    
