import struct

import numpy as np
import torch
from torchvision import datasets, transforms

# generates train and test data (the mnist_samples ar in a random order) to use with the Desktop_Application

# output files
output_file_train = "mnist_train_all_random.bin"
output_file_test = "mnist_test_all_random.bin"

# MNIST dataset with ToTensor ([0.0, 1.0])
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

# Random permutation of training data
indices_train = torch.randperm(len(mnist_train))
with open(output_file_train, "wb") as f:
    for idx in indices_train:
        img_tensor, label = mnist_train[idx]
        flat = img_tensor.squeeze().flatten().numpy().astype(np.float32)
        f.write(struct.pack("<784f", *flat))
        f.write(struct.pack("B", label))

# Random permutation of test data
indices_test = torch.randperm(len(mnist_test))
with open(output_file_test, "wb") as f:
    for idx in indices_test:
        img_tensor, label = mnist_test[idx]
        flat = img_tensor.squeeze().flatten().numpy().astype(np.float32)
        f.write(struct.pack("<784f", *flat))
        f.write(struct.pack("B", label))
