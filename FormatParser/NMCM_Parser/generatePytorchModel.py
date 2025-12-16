import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

import time

import torch
import torch.nn as nn

 class SimpleNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_p: float = 0.10):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
                                  
        self.fc = nn.Linear(392, num_classes)

        # softmax (for inference if needed)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        # x: (N,1,28,28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        logits = self.fc(x)

        if return_probs:
            return self.softmax(logits)
        return logits


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
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

        print(f'Epoche [{epoch+1}/{num_epochs}], Verlust: {running_loss / len(train_loader)}')

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

        print(f'Accuracy auf dem Testdatensatz: {100 * correct / total}%')

    endTime = time.time()
    duration = endTime - startTime

    print(f"Programmlaufzeit: {duration:.6f} Sekunden für  {num_epochs} epochen")

    print(summary(model, input_size=(1, 28, 28)))
    torch.save(model, './mnist_model.pth')
