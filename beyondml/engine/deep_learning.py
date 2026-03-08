"""
Beginner-friendly Deep Learning Engine using PyTorch.
Includes a simple CNN architecture and a trainer wrapper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List

class SimpleCNN(nn.Module):
    """A standard CNN for image classification beginners."""
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()
        # Assuming 32x32 input, after two 2x2 pools, it's 8x8
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8 if in_channels == 3 else 64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

class SimpleMLP(nn.Module):
    """A standard Multi-Layer Perceptron for tabular data."""
    def __init__(self, input_size: int, num_classes: int = 2):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class DeepLearningTrainer:
    """Trainer wrapper for the Deep Learning Engine."""
    def __init__(self, model: nn.Module, lr: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.history = {"train_loss": [], "test_acc": []}

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(loader.dataset)
        self.history["train_loss"].append(epoch_loss)
        return epoch_loss

    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate accuracy on a dataset."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        self.history["test_acc"].append(acc)
        return acc

    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 5):
        """Train for multiple epochs and log results."""
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            acc = self.evaluate(test_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.2f}%")
        
        return self.history
