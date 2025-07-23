import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Input: (1, 32, 32)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Output: (6, 28, 28)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Output: (6, 14, 14)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Output: (16, 10, 10)

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # Output: (16, 5, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool1(x)
        x = self.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


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
        transforms.Pad(2),  # Pad 2 pixels on each side (28x28 -> 32x32)
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean/std
    ]
)

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model and optimizer
model = LeNet5().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1, 6):  # 5 epochs
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# Save model
torch.save(model.state_dict(), "mnist_model_lenet5.pth")
print("Model saved to mnist_model_lenet5.pth")
