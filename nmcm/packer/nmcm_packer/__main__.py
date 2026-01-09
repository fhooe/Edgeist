import struct
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import datasets, transforms

if __name__ == "__main__":
    # generates train and test data (the mnist_samples ar in a random order) to use with the desktop_application
    parser = ArgumentParser(description="Packs the MNIST-dataset into the required format")

    parser.add_argument("--mnist_dir", type=str, help="directory where to download MNIST-data to", default="data")
    parser.add_argument(
        "--out_train", type=str, help="file to save the packed train data to", default="mnist_train_all_random.bin"
    )
    parser.add_argument(
        "--out_test", type=str, help="file to save the packed test data to", default="mnist_test_all_random.bin"
    )

    args = parser.parse_args()

    # MNIST dataset with ToTensor ([0.0, 1.0])
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root=args.mnist_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=args.mnist_dir, train=False, download=True, transform=transform)

    # Random permutation of training data
    indices_train = torch.randperm(len(mnist_train))
    with open(args.out_train, "wb") as f:
        for idx in indices_train:
            img_tensor, label = mnist_train[idx]
            flat = img_tensor.squeeze().flatten().numpy().astype(np.float32)
            f.write(struct.pack("<784f", *flat))
            f.write(struct.pack("B", label))

    # Random permutation of test data
    indices_test = torch.randperm(len(mnist_test))
    with open(args.out_test, "wb") as f:
        for idx in indices_test:
            img_tensor, label = mnist_test[idx]
            flat = img_tensor.squeeze().flatten().numpy().astype(np.float32)
            f.write(struct.pack("<784f", *flat))
            f.write(struct.pack("B", label))
