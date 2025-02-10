import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_loader = DataLoader(
    datasets.FashionMNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    ),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    datasets.FashionMNIST(
        'data',
        train=False,
        transform=transforms.ToTensor()
    ),
    batch_size=64
)

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

def train_model(model, train_loader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Different optimizers in a loop
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad
}

epochs = 5  

for opt_name, opt_class in optimizers.items():
    model = FiveLayerNet()
    if opt_name == "SGD":
        optimizer = opt_class(model.parameters(), lr=0.01)
    else:
        optimizer = opt_class(model.parameters(), lr=0.001)

    train_model(model, train_loader, optimizer, epochs=epochs)

    acc = evaluate_model(model, test_loader)
    print(f"Optimizer: {opt_name}, Accuracy: {acc:.4f}")
