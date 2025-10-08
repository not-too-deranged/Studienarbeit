import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from datetime import datetime


#to open tensorboard put in command line:
#tensorboard --logdir runs

class PretrainedCNN(nn.Module):
    '''A simple CNN model for CIFAR-10 classification with an option to load a pretrained ResNet18 model.'''
    def __init__(self, num_classes=10): # variables should be defined outside of class
        super(PretrainedCNN, self).__init__() 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # input channels, output channels, kernel size, padding TODO: talk about values
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # adjust depending on input image size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) #could later be changed to leaky relu or gelu
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def load_resnet18(num_classes=10):
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model


if __name__ == "__main__":
    # Hyperparameters to be tuned
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    num_classes = 10

    # Data transformations needed for pretrained models
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset to test pretrained model

    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)  

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    # Initialize model, loss function, optimizer, and metrics

    model = PretrainedCNN.load_resnet18(num_classes=num_classes)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) #TODO find out if this is correct

    # TensorBoard writer
    writer = SummaryWriter('runs/cifar10_pretrained/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        val_accuracy = 0.0
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy += accuracy_metric(predicted, labels).item()

            avg_loss = running_loss / len(test_loader)
            val_accuracy /= len(test_loader)
            writer.add_scalar('Validation Loss', avg_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            running_loss = 0.0
            val_accuracy = 0.0
    writer.close()
    



