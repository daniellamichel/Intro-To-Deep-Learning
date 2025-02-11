🧠 FashionMNIST Classification with a Five-Layer Neural Network

📌 Overview

This project implements a deep learning model using PyTorch to classify images from the FashionMNIST dataset. It explores different training epochs and optimizers to determine their impact on accuracy.

📂 Project Structure

train_loader and test_loader are created using torchvision.datasets.FashionMNIST.

FiveLayerNet is a five-layer fully connected neural network.

train_model() function trains the model using cross-entropy loss.

evaluate_model() function tests the trained model.

Epoch-based analysis: Trains and evaluates the model across different epochs.

Optimizer comparison: Tests different optimization algorithms (SGD, Adam, RMSprop, Adagrad).

Accuracy vs. Epochs Graph: Plots the accuracy trend over increasing epochs.

🛠️ Technologies Used

Python

PyTorch (Deep Learning Framework)

Torchvision (Dataset and Transformations)

Matplotlib (Data Visualization)

🏗️ Model Architecture

The model consists of five fully connected layers with ReLU activation:

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

📊 Results

The optimal epoch was identified based on accuracy trends.

Different optimizers (SGD, Adam, RMSprop, Adagrad) were evaluated for model performance.

The project visualizes the accuracy vs. epochs trend using Matplotlib.