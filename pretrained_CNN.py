import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning import LightningModule
from torchvision import models

from util_files import model_options


class EfficientNetLightning(LightningModule):
    """
    LightningModule wrapper for EfficientNetV2_L.
    Handles training, validation, and optimization automatically.
    """

    def __init__(self, learning_rate, weight_decay, dropout_rate, unfreeze_layers, num_classes):
        super().__init__()

        # Save hyperparameters (for logging and checkpointing)
        self.save_hyperparameters()

        # Load pretrained EfficientNet
        self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)

        # Replace the classifier layer for CIFAR-100 (100 classes)
        in_features = self.model.classifier[1].in_features
        self.dropout = nn.Dropout(dropout_rate)
        #use dropout
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        #weight initialization
        nn.init.xavier_uniform_(self.model.classifier[1].weight)
        nn.init.zeros_(self.model.classifier[1].bias)

        # Freeze feature extractor layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Unfreeze last N layers from the back:
        if unfreeze_layers > 0:
            layers = list(self.model.features.children())
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Define loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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

        self.log('loss/train', loss, prog_bar=True)
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
        """Defines a single test iteration."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)

        self.log('loss/test', loss, prog_bar=True)
        self.log('acc/test', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Set up the optimizer (and optionally scheduler)."""

        # Split parameters into two groups
        backbone_params = []
        classifier_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        #lower learning rate for backbone
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.hparams.learning_rate * 0.1},
            {"params": classifier_params, "lr": self.hparams.learning_rate}
        ], weight_decay=self.hparams.weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=model_options.PATIENCE)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val"
            }
        }
