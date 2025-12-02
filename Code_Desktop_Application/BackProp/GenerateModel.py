import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()

        # Eingabe: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)     # → (4, 28, 28)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)                             # → (4, 14, 14)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)     # → (8, 14, 14)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)                             # → (8, 7, 7)

        # AdaptiveAvgPool2d: Zielgröße (4, 4)
        self.adaptive_pool2d = nn.AdaptiveAvgPool2d((4, 4))        # → (8, 4, 4)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 4 * 4, 64)                         # 128 → 64
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        # Dropout zur Regularisierung
        self.dropout = nn.Dropout(p=0.5)

        # AdaptiveAvgPool1d: Zielgröße = 20
        self.adaptive_pool1d = nn.AdaptiveAvgPool1d(20)            # → (64 → 20)

        self.fc2 = nn.Linear(20, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.adaptive_pool2d(x)                                # → (8, 4, 4)

        x = self.flatten(x)                                        # → (8*4*4) = 128
        x = self.fc1(x)                                            # → 64
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.adaptive_pool1d(x.unsqueeze(0)).squeeze(0)        # → 20

        x = self.fc2(x)
        x = self.softmax(x)
        return x


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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

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

    print(summary(model, input_size=(1, 28, 28)))
    torch.save(model, './mnist_model.pth')
