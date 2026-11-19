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
        self.input = X

        if self.padding > 0:
            self.input_pad = cp.pad(
                X,
                ((0, 0), (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.input_pad = X

        N = X.shape[0]

        out_height = (
            self.input_pad.shape[2] - self.filter_size
        ) // self.stride + 1

        out_width = (
            self.input_pad.shape[3] - self.filter_size
        ) // self.stride + 1

        # All filters are calculated simultaneously
        output = cp.zeros(
            (N, self.num_filters, out_height, out_width),
            dtype=cp.float32
        )

        for kh in range(self.filter_size):
            for kw in range(self.filter_size):

                region = self.input_pad[
                    :,
                    :,
                    kh:kh + out_height * self.stride:self.stride,
                    kw:kw + out_width * self.stride:self.stride
                ]

                # region:  (N, C, H, W)
                # weights: (F, C)
                # result:  (N, F, H, W)

                output += cp.einsum(
                    'nchw,fc->nfhw',
                    region,
                    self.filters[:, :, kh, kw]
                )

        output += self.biases[None, :, None, None]

        # Store pre-activation for backpropagation
        self.z = output

        self.feat_maps = cp.maximum(0, output)

        return self.feat_maps


    def backward(self, dA):

        dZ = dA * ReLU.backward(self.feat_maps, self.feat_maps)

        dw = cp.zeros_like(self.filters)
        db = cp.sum(dZ, axis=(0, 2, 3))

        dX_pad = cp.zeros_like(self.input_pad)

        out_height = dZ.shape[2]
        out_width = dZ.shape[3]

        for kh in range(self.filter_size):
            for kw in range(self.filter_size):

                region = self.input_pad[
                    :,
                    :,
                    kh:kh + out_height * self.stride:self.stride,
                    kw:kw + out_width * self.stride:self.stride
                ]

                weights = self.filters[:, :, kh, kw]

                # Gradient with respect to filters
                dw[:, :, kh, kw] = cp.einsum(
                    'nfhw,nchw->fc',
                    dZ,
                    region
                )

                # Gradient with respect to input
                dX_pad[
                    :,
                    :,
                    kh:kh + out_height * self.stride:self.stride,
                    kw:kw + out_width * self.stride:self.stride
                ] += cp.einsum(
                    'nfhw,fc->nchw',
                    dZ,
                    weights
                )

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