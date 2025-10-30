import torch
import torch.nn as nn
import torch.optim as optim
import lightning
import torchmetrics
from lightning import Trainer, LightningModule
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import tensorboard
from tqdm import tqdm

class EfficientNetLightning(LightningModule):
    """
    LightningModule wrapper for EfficientNetV2_L fine-tuning on CIFAR-100.
    Handles training, validation, and optimization automatically.
    """

    def __init__(self, learning_rate=1e-3, weight_decay=0.0, dropout_rate=0.0):
        super().__init__()

        # Save hyperparameters (for logging and checkpointing)
        self.save_hyperparameters()

        # Load pretrained EfficientNet
        self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

        # Replace the classifier layer for CIFAR-100 (100 classes)
        in_features = self.model.classifier[1].in_features
        self.dropout = nn.Dropout(dropout_rate)
        self.model.classifier[1] = nn.Linear(in_features, 100)

        # Freeze feature extractor layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Define loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)

    def forward(self, x):
        """Forward pass (used for inference and validation)."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Defines a single training iteration."""
        images, labels = batch
        outputs = self(images)

        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        loss=nn.functional.cross_entropy(outputs, labels)

        self.log('loss/train',loss, prog_bar=True)
        self.log('acc/train', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        """Defines a single validation iteration."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        loss = nn.functional.cross_entropy(outputs, labels)

        self.log('loss/val', loss, prog_bar=True)
        self.log('acc/val', acc)

        return loss

    def test_step(self, batch, batch_idx):
        """Defines a single validation iteration."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)

        loss = nn.functional.cross_entropy(outputs, labels)

        self.log('loss/tst', loss, prog_bar=True)
        self.log('acc/tst', acc)

        return loss

    def configure_optimizers(self):
        """Set up the optimizer (and optionally scheduler)."""
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate)
        return optimizer

writer = SummaryWriter('tensorboard_logs')

#Used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
batch_size = 64
initial_lr = 1e-3
num_epochs = 50
patience = 5     # stop if val loss doesnt improve for 5 epochs

"""
CIFAR-100 consists of 60,000 32x32 color images in 100 classes,
with 600 images per class. There are 50,000 training and 10,000 test images.
"""
transform_train = transforms.Compose([
    transforms.Resize(224),  # EfficientNet expects at least 224x224 inputs
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    ),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Download the CIFAR-100 dataset
train_val_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, transform=transform_train, download=True
)

train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [0.6, 0.4])

test_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, transform=transform_test, download=True
)

# DataLoaders allow batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# ============================================================
# 4. LOAD PRETRAINED EFFICIENTNET
# ============================================================

# We'll use 'efficientnet_b7' - high accuracy but heavy.

model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)


# ============================================================
# 5. MODIFY THE FINAL CLASSIFICATION LAYER
# ============================================================

# ============================================================
# 6. FREEZE EARLY LAYERS
# ============================================================
"""
We can 'freeze' early layers so their weights don't change during training.
This helps retain the pretrained ImageNet features and reduces training time.
"""

for param in model.features.parameters():
    param.requires_grad = False  # Freeze all feature extractor parameters


# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================
early_stop_callback = EarlyStopping(
    monitor="loss/val", mode="min", patience=patience, verbose=True
)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best_model",
    save_top_k=1,
    monitor="loss/val",
    mode="min"
)

#tensorboard logger
filename: str = f'b={batch_size}_l_r={initial_lr}'
tb_logger = loggers.TensorBoardLogger(save_dir="lightning_logs/" +
                                             filename)


# ============================================================
# INITIAL TRAINER FOR LR FINDER
# ============================================================
# We use a shorter trainer here so LR finder doesn't train for long.
trainer = Trainer(
    accelerator="auto",
    max_epochs=3,  # only a few for lr finder
    logger=tb_logger,
)

# Initialize the model
model = EfficientNetLightning(learning_rate=initial_lr)

# ============================================================
# 1?? AUTOMATIC LEARNING RATE FINDING
# ============================================================
print("\n?? Running learning rate finder...")
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, min_lr=0.0001, max_lr=0.1)

# Plot results (optional, if youre running in a notebook or environment with display)
fig = lr_finder.plot(suggest=True)
fig.show()

# Get suggested LR and update model
new_lr = lr_finder.suggestion()
print(f"? Suggested learning rate: {new_lr:.2e}")
model.hparams.learning_rate = new_lr

# ============================================================
# 2?? FULL TRAINING WITH EARLY STOPPING + CHECKPOINTING
# ============================================================
trainer = Trainer(
    max_epochs=num_epochs,
    accelerator="auto",
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=tb_logger,
)

print("\n?? Starting training with tuned learning rate...")
trainer.fit(model, train_loader, val_loader)

torch.save(model.state_dict(), "./models/efficientnet_cifar100.pth")

# ============================================================
# 3?? TEST BEST MODEL
# ============================================================
#print("\n?? Testing best saved model on test data...")
#trainer.test(model, dataloaders=test_loader)


"""
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # Reset gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

    # --- VALIDATE ---
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total

    print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
          f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.2f}%")

    # --- EARLY STOPPING CHECK ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "./models/efficientnet_cifar100.pth")  # save best version
        print("Validation improved, model saved!")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epochs.")
        if epochs_without_improvement >= patience:
            print("\nEarly stopping triggered, training stopped.")
            break

print("\nTraining complete!")
"""
# ============================================================
# 9. EVALUATION (ON TEST SET)
# ============================================================
"""
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on CIFAR-100: {accuracy:.2f}%")

torch.save(model.state_dict(), "./models/efficientnet_cifar100.pth")
print("\nModel saved as 'efficientnet_cifar100.pth'")
"""


"""
To load the model:
model = models.efficientnet_b7(weights=None)   # same architecture
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 100)  # same output size
model.load_state_dict(torch.load("efficientnet_cifar100.pth"))
model.eval()  # set to evaluation mode
"""
