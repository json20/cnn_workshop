#!/usr/bin/env python


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Model_Google_1(nn.Module):
    """
    A modified LeNet-style CNN for MNIST digit classification.

    Architecture:
    - Input: (1 x 28 x 28)
    - Convolutional blocks with ReLU, BatchNorm, and MaxPooling
    - Fully connected layers with Dropout and BatchNorm
    - Output: 10 log-probabilities for digit classes (0-9)
    """

    def __init__(self, channels=1):
        super(Model_Google_1, self).__init__()

        self.convnet = nn.Sequential(
            # First Conv Block
            nn.Conv2d(channels, 32, kernel_size=3),  # Output: (32 x 26 x 26)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3),  # Output: (32 x 24 x 24)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),  # Output: (32 x 12 x 12)
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3),  # Output: (64 x 10 x 10)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),  # Output: (64 x 5 x 5)
            nn.Dropout(0.2),
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3),  # Output: (128 x 3 x 3)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),  # Output: (128 x 1 x 1)
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),  # 10 classes for digits 0-9
        )

    def forward(self, img):
        x = self.convnet(img)
        x = x.view(img.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Log probabilities


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)  # Negative log likelihood loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    print(f"\nTrain Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"\nTest Set: Avg Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Prepare MNIST data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean/std
    ]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model and optimizer
model = Model_Google_1().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, 6):  # 5 epochs
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# Save model
torch.save(model.state_dict(), "mnist_model_google1.pth")
print("Model saved to mnist_model_google1.pth")
