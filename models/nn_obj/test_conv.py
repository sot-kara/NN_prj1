import cupy as cp
import torch
import torchvision
import torchvision.transforms as transforms
from convolution import Conv
from activations import ReLU



# --------------------------------------------------
# 1. Load CIFAR-100
# --------------------------------------------------

dataset = torchvision.datasets.CIFAR100(
    root='data',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

# Select one image
image, label = dataset[0]

# Shape: (3, 32, 32)
X = cp.asarray(image.numpy(), dtype=cp.float32)

print("Input shape:", X.shape)
print("Class label:", label)


# --------------------------------------------------
# 2. Initialize convolutional layer
# --------------------------------------------------

conv = Conv(
    input_shape=(3, 32, 32),
    num_filters=8,
    filter_size=3,
    stride=1,
    padding=1
)


# --------------------------------------------------
# 3. Forward pass
# --------------------------------------------------

A = conv.forward(X)

print("\nForward pass:")
print("Output shape:", A.shape)

print("Expected shape: (8, 32, 32)")

print("Output min:", float(cp.min(A).get()))
print("Output max:", float(cp.max(A).get()))


# --------------------------------------------------
# 4. Backward pass
# --------------------------------------------------

# Simulate gradient coming from the next layer
dA = cp.random.randn(*A.shape).astype(cp.float32)

dX, dw, db = conv.backward(dA)

print("\nBackward pass:")

print("dX shape:", dX.shape)
print("dw shape:", dw.shape)
print("db shape:", db.shape)


# --------------------------------------------------
# 5. Check shapes
# --------------------------------------------------

assert A.shape == (8, 32, 32)
assert dX.shape == (3, 32, 32)
assert dw.shape == (8, 3, 3, 3)
assert db.shape == (8,)

print("\nAll shapes are correct!")


# --------------------------------------------------
# 6. Check numerical stability
# --------------------------------------------------

assert cp.all(cp.isfinite(A))
assert cp.all(cp.isfinite(dX))
assert cp.all(cp.isfinite(dw))
assert cp.all(cp.isfinite(db))

print("No NaN or Inf values detected!")

print("\nConvolution test completed successfully!")