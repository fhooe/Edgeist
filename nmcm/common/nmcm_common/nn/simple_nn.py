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
