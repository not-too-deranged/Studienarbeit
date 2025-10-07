import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision import datasets, models
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

class PretrainedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(PretrainedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def load_resnet18(num_classes=10):
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001
    num_classes = 10

    # Data transformations
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    # Initialize model, loss function, optimizer, and metrics

    model = PretrainedCNN.load_resnet18(num_classes=num_classes)  # Uncomment to use ResNet18
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_metric = torchmetrics.Accuracy().to('cpu')

    # TensorBoard writer
    writer = SummaryWriter('runs/cifar10_experiment')

