import time

import torch
import torch.nn as nn
import torch.optim as optim
from nmcm_common.nn import SimpleNN
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    num_epochs = 5

    startTime = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to("cpu"), labels.to("cpu")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}]; Loss: {running_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to("cpu"), labels.to("cpu")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test-dataset: {100 * correct / total}%")

    endTime = time.time()
    duration = endTime - startTime

    print(f"Execution time: {duration:.6f}s for {num_epochs} epochs")

    print(summary(model, input_size=(1, 28, 28)))
    torch.save(model, "mnist_model.pth")
