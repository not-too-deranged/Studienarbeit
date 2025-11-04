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

torch.set_float32_matmul_precision('high')

class EfficientNetLightning(LightningModule):
    """
    LightningModule wrapper for EfficientNetV2_L fine-tuning on CIFAR-100.
    Handles training, validation, and optimization automatically.
    """

    def __init__(self, learning_rate=1e-3, weight_decay=1e-4, dropout_rate=0.2):
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
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=100)

    def forward(self, x):
        """Forward pass (used for inference and validation)."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Defines a single training iteration."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)

        self.log('loss/train',loss, prog_bar=True)
        self.log('acc/train', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Defines a single validation iteration."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)

        self.log('loss/val', loss, prog_bar=True)
        self.log('acc/val', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Defines a single validation iteration."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)

        self.log('loss/test', loss, prog_bar=True)
        self.log('acc/test', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Set up the optimizer (and optionally scheduler)."""
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val"
            }
        }

# ============================================================
# HYPERPARAMETERS AND DEVICE
# ============================================================

#Used device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
batch_size = 64
initial_lr = 1e-3
num_epochs = 50
patience = 5     # early stopping patience

# ============================================================
# DATA PREPARATION
# ============================================================

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

# Split train/val properly (80/20 split)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

test_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, transform=transform_test, download=True
)

# DataLoaders allow batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ============================================================
# LOGGING AND CALLBACKS
# ============================================================

#Lightning handles TensorBoard automatically; no manual SummaryWriter needed
run_name = f"b{batch_size}_lr{initial_lr:.0e}"
tb_logger = loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)

#Early stopping monitors validation loss
early_stop_callback = EarlyStopping(
    monitor="loss/val", mode="min", patience=patience, verbose=True
)

#Save best checkpoint by validation loss
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="best_model",
    save_top_k=1,
    monitor="loss/val",
    mode="min"
)


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
# AUTOMATIC LEARNING RATE FINDING
# ============================================================
print("\n Running learning rate finder...")
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, min_lr=1e-5, max_lr=1e-1)

#Automatically select the suggested LR
new_lr = lr_finder.suggestion()
print(f"Suggested learning rate: {new_lr:.2e}")
model.hparams.learning_rate = new_lr

# ============================================================
# FULL TRAINING WITH EARLY STOPPING + CHECKPOINTING
# ============================================================
trainer = Trainer(
    max_epochs=num_epochs,
    accelerator="auto",
    callbacks=[early_stop_callback, checkpoint_callback],
    logger=tb_logger,
    log_every_n_steps=10,
)

print("\n Starting training with tuned learning rate...")
trainer.fit(model, train_loader, val_loader)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(model.state_dict(), "./models/efficientnet_cifar100.pth")
print("? Model saved to './models/efficientnet_cifar100.pth'")

# ============================================================
# TEST BEST MODEL
# ============================================================

#print("\n Testing best saved model on test data...")
#trainer.test(model, dataloaders=test_loader)
