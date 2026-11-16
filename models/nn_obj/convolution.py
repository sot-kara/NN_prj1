import cupy as cp
from .activations import ReLU

class Conv:
    def __init__(self, input_shape, num_filters, filter_size, stride=1, padding=0):
        self.input_shape = input_shape  # (channels, height, width)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.input = None  # To store the input during forward pass to use in backward pass
        self.input_pad = None  # To store the padded input during forward pass to use in backward pass
        self.feat_maps = None  # To store the output feature maps after forward pass to use in backward pass

        # Initialize filters and biases
        # Use He initialization for filters
        self.filters = cp.random.randn(num_filters, input_shape[0], filter_size, filter_size) * cp.sqrt(2.0 / (input_shape[0] * filter_size * filter_size))
        self.biases = cp.zeros(num_filters)

    def forward(self, X):
        # Perform forward pass for convolutional layer
        outputs = [] # List to store the output feature maps of size (num_filters, out_height, out_width)

        self.input = X  # Store the input for use in backward pass
        print ("Input shape:", X.shape)
        # Pad the input if padding is specified
        if self.padding > 0:
            self.input_pad = cp.pad(X, 
                       ((0, 0), 
                        (self.padding, self.padding), 
                        (self.padding, self.padding)),
                        mode='constant')
        else:
            self.input_pad = X

        for i in range(self.num_filters):
            # Convolve each filter with the input
            
            # Calculate the output dimensions
            out_height = (self.input_shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1
            out_width = (self.input_shape[2] - self.filter_size + 2 * self.padding) // self.stride + 1

            # Initialize the current output feature map
            output = cp.zeros((out_height, out_width))

            for kh in range(self.filter_size):
                for kw in range(self.filter_size):

                    region = self.input_pad[
                        :,
                        kh:kh + out_height * self.stride:self.stride,
                        kw:kw + out_width * self.stride:self.stride
                    ]

                    output += cp.sum(
                        region * self.filters[i, :, kh, kw][:, None, None],
                        axis=0
                    )

            output += self.biases[i]
            # Apply activation function (ReLU)
            output = ReLU.forward(output)

            outputs.append(output)

        self.feat_maps = cp.array(outputs)
        return self.feat_maps

                    
        
        
    def backward(self, dA):
        # Perform  backward pass for convolutional layer
        # dA is the gradient of the loss with respect to the output of this layer (shape: (num_filters, out_height, out_width))

        # Gradient of the loss with respect to the pre-activation output 
        dZ = ReLU.backward(dA, self.feat_maps)  
       
        # Gradient with respect to weights
        dw = cp.zeros_like(self.filters)
        
        # Gradient with respect to biases
        db = cp.zeros_like(self.biases)

        # Gradient with respect to input
        dX_pad = cp.zeros_like(self.input_pad)  # Initialize dX_pad with the same shape as padded input
        dX = cp.zeros_like(self.input)  # Initialize dX with the same shape as input

        for i in range(self.num_filters):

            for h in range(self.feat_maps.shape[1]):
                for w in range(self.feat_maps.shape[2]):
                    # Iterate over each element in the output feature map of current filter
                    
                    h_start = h * self.stride
                    h_end = h_start + self.filter_size
                    w_start = w * self.stride
                    w_end = w_start + self.filter_size

                    # Extract the region from the input that corresponds to the current output element for the current filter
                    region = self.input_pad[:, h_start:h_end, w_start:w_end]

                    # Update gradients for weights and biases
                    dw[i] += region * dZ[i, h, w]
                    db[i] += dZ[i, h, w]

                    # Update gradient for input to propagate to the previous layer
                    dX_pad[:, h_start:h_end, w_start:w_end] += self.filters[i] * dZ[i, h, w]
        
        # Remove padding from dX
        if self.padding > 0:
            dX = dX_pad[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad
        
        return dX, dw, db