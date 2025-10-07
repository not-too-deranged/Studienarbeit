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
import ssl

#to open tensorboard put in command line:
#tensorboard --logdir runs

# Workaround for SSL certificate issues on some systems
# "certificate verify failed: unable to get local issuer certificate"
# Use certifi's CA bundle as a safe fallback when available.
try:
    import certifi
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    # urllib and other stdlib modules use ssl._create_default_https_context to
    # build HTTPS contexts. Overriding it ensures downloads validate using
    # certifi's CA bundle when available.
    ssl._create_default_https_context = lambda: _ssl_ctx
except Exception:
    # If certifi isn't installed or something fails, leave the default
    # context unchanged. The recommended fix then is to install certifi or
    # run the system-specific certificate installer (e.g. the
    # "Install Certificates.command" that ships with some Python installers
    # on macOS).
    pass

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
    num_epochs = 5
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
    writer = SummaryWriter('runs/cifar10_experiment')

    # Evaluation loop
    model.eval()
    test_loss = 0.0
    accuracy_metric.reset()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, labels)
            test_loss /= len(test_loader.dataset)
            test_accuracy = accuracy_metric.compute().item()
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
            writer.add_scalar('Test/Loss', test_loss, 0)
            writer.add_scalar('Test/Accuracy', test_accuracy, 0)
    writer.close()






