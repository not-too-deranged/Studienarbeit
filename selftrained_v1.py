from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

class SelfTrainedCNN(nn.Module):
    '''A simple CNN model for CIFAR-10 classification.'''
    def __init__(self, num_classes=10): # variables should be defined outside of class
        super(SelfTrainedCNN, self).__init__() 
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
    
if __name__ == "__main__":
    # Hyperparameters to be tuned
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Data transformations
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, optimizer, and metrics
    model = SelfTrainedCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    # TensorBoard writer
    writer = SummaryWriter('runs/cifar10_selftrained/' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Training Loss', avg_loss, epoch)

        # Validation loop
        model.eval()
        val_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy += accuracy_metric(predicted, labels).item()

        val_accuracy /= len(test_loader)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    writer.close()

