import struct
import torch
from torchvision import datasets, transforms
import numpy as np

# generates train and test data (the mnist_samples ar in a random order) to use with the Desktop_Application

# Ausgabe-Dateien
output_file_train = "mnist_train_all_random.bin"
output_file_test = "mnist_test_all_random.bin"

# MNIST Dataset mit ToTensor ([0.0, 1.0])
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='.', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='.', train=False, download=True, transform=transform)

# Zufällige Permutation für Trainingsdaten
indices_train = torch.randperm(len(mnist_train))
with open(output_file_train, "wb") as f:
    for idx in indices_train:
        img_tensor, label = mnist_train[idx]
        flat = img_tensor.squeeze().flatten().numpy().astype(np.float32)
        f.write(struct.pack('<784f', *flat))
        f.write(struct.pack('B', label))

# Zufällige Permutation für Testdaten
indices_test = torch.randperm(len(mnist_test))
with open(output_file_test, "wb") as f:
    for idx in indices_test:
        img_tensor, label = mnist_test[idx]
        flat = img_tensor.squeeze().flatten().numpy().astype(np.float32)
        f.write(struct.pack('<784f', *flat))
        f.write(struct.pack('B', label))
