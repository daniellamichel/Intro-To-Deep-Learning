import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

train_loader = DataLoader(datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()), batch_size=64)

class FiveLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, epochs, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item()
    return correct / len(test_loader.dataset)

epochs_list = [1, 5, 10, 15, 20, 25, 30]
accuracies = []

for epochs in epochs_list:
    model = FiveLayerNet()
    train_model(model, train_loader, epochs)
    acc = evaluate_model(model, test_loader)
    accuracies.append(acc)
    print(f"Epochs: {epochs}, Accuracy: {acc:.4f}")

# epochs vs accuracy
plt.plot(epochs_list, accuracies, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epochs vs. Accuracy')
plt.grid()
plt.show()

best_epoch = epochs_list[accuracies.index(max(accuracies))]
print(f"The best epoch parameter is: {best_epoch} with accuracy {max(accuracies):.4f}")