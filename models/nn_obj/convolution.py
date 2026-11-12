import cupy as cp

class Conv:
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding=0):
        self.input_shape = input_shape  # (channels, height, width)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        # Initialize filters and biases
        self.filters = cp.random.randn(num_filters, input_shape[0], filter_size, filter_size) * 0.1
        self.biases = cp.zeros((num_filters, 1))

    def forward(self, X):
        # Perform forward pass for convolutional layer
        outputs = []
        for i in range(self.num_filters):
            # Convolve each filter with the input
            
            # Calculate the output dimensions
            out_height = (self.input_shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1
            out_width = (self.input_shape[2] - self.filter_size + 2 * self.padding) // self.stride + 1

            # Initialize the output feature map
            output = cp.zeros((out_height, out_width))

            # Perform convolution operation
            for h in range(out_height):
                for w in range(out_width):
                    h_start = h * self.stride
                    h_end = h_start + self.filter_size
                    w_start = w * self.stride
                    w_end = w_start + self.filter_size
                    
                    if w_end > self.input_shape[2] or h_end > self.input_shape[1]:
                        continue  # Skip if the filter goes out of bounds
                    
                    # Extract the region to apply the filter to from the input
                    region = X[:, h_start:h_end, w_start:w_end]

                    # Perform element-wise multiplication and sum
                    output[h, w] = cp.sum(region * self.filters[i]) + self.biases[i]

            outputs.append(output)

        return cp.array(outputs)

                    

        
        
    def backward(self, d_out):
        # Implement the backward pass for convolutional layer
        pass



X = cp.random.rand(3, 32, 32)

K = cp.random.randn(3, 3, 3)

conv = Conv(input_shape=X.shape, num_filters=1, filter_size=3, stride=1, padding=1)

output = conv.forward(X)

print("Input shape:", X.shape)
print("Output shape:", output.shape)