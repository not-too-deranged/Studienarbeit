import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm

#Used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b7(weights=None)   # same architecture
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 100)  # same output size
model.load_state_dict(torch.load("./models/efficientnet_cifar100.pth"))
model = model.to(device) #push model to GPU

# ============================================================
# 9. EVALUATION (ON TEST SET)
# ============================================================
model.eval()
correct = 0
total = 0
batch_size = 64

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
val_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, transform=transform_test, download=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on CIFAR-100: {accuracy:.2f}%")
