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

        # Pad the input if padding is specified
        if self.padding > 0:
            self.input_pad = cp.pad(X, 
                       ((0, 0),
                        (0, 0),
                        (self.padding, self.padding), 
                        (self.padding, self.padding)),
                        mode='constant')
        else:
            self.input_pad = X

        for i in range(self.num_filters):
            # Convolve each filter with the input
            
            # Calculate the output dimensions
            n_batches = self.input_pad.shape[0]
            out_height = (self.input_shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1
            out_width = (self.input_shape[2] - self.filter_size + 2 * self.padding) // self.stride + 1

            # Initialize the current output feature map
            output = cp.zeros((n_batches, out_height, out_width),dtype=cp.float32)

            for kh in range(self.filter_size):
                for kw in range(self.filter_size):

                    region = self.input_pad[
                        :,
                        :,
                        kh:kh + out_height * self.stride:self.stride,
                        kw:kw + out_width * self.stride:self.stride
                    ]

                    output += cp.sum(
                        region * self.filters[i, :, kh, kw][None,:, None, None],
                        axis=1
                    )

            output += self.biases[i]
            # Apply activation function (ReLU)
            output = ReLU.forward(output)

            outputs.append(output)

        self.feat_maps = cp.stack(outputs,axis=1)  # Stack the outputs along the channel dimension to get shape (batch_size, num_filters, out_height, out_width)
        return self.feat_maps

                    
        
        
    def backward(self, dA):

        # Gradient with respect to pre-activation
        dZ = dA * ReLU.backward(self.feat_maps, self.feat_maps)

        # Gradients
        dw = cp.zeros_like(self.filters)
        db = cp.zeros_like(self.biases)

        dX_pad = cp.zeros_like(self.input_pad)
        dX = cp.zeros_like(self.input)

        out_height = self.feat_maps.shape[2]
        out_width = self.feat_maps.shape[3]

        for i in range(self.num_filters):

            for kh in range(self.filter_size):
                for kw in range(self.filter_size):

                    h_start = kh
                    h_end = kh + out_height * self.stride

                    w_start = kw
                    w_end = kw + out_width * self.stride

                    # Extract corresponding input values
                    region = self.input_pad[
                        :,
                        :,
                        h_start:h_end:self.stride,
                        w_start:w_end:self.stride
                    ]

                    # Gradient with respect to weights
                    dw[i, :, kh, kw] += cp.einsum(
                        'nhw,nchw->c',
                        dZ[:, i],
                        region
                    )

                    # Gradient with respect to biases
                    db[i] += cp.sum(dZ[:, i])

                    # Gradient with respect to input
                    dX_pad[
                        :,
                        :,
                        h_start:h_end:self.stride,
                        w_start:w_end:self.stride
                    ] += (
                        dZ[:, i][:, None, :, :]
                        * self.filters[i, :, kh, kw][None, :, None, None]
                    )

        # Remove padding
        if self.padding > 0:
            dX = dX_pad[
                :,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding
            ]
        else:
            dX = dX_pad
            
        return dX, dw, db