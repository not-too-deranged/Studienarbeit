import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning import LightningModule
import math

import model_options


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=0.25, fused=True):
        super(MBConv, self).__init__()
        self.fused = fused
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.stride = stride

        # Expansion
        if expand_ratio != 1:
            if fused:
                self.expand_conv = nn.Conv2d(in_channels, in_channels * expand_ratio, 3, stride=stride, padding=1, groups=in_channels, bias=False)
            else:
                self.expand_conv = nn.Conv2d(in_channels, in_channels * expand_ratio, 1, bias=False)
        else:
            self.expand_conv = None

        # Depthwise
        if fused:
            self.depthwise_conv = None
        else:
            self.depthwise_conv = nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels * expand_ratio, bias=False)

        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * expand_ratio * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels * expand_ratio, se_channels, 1),
                nn.SiLU(),
                nn.Conv2d(se_channels, in_channels * expand_ratio, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Project
        self.project_conv = nn.Conv2d(in_channels * expand_ratio, out_channels, 1, bias=False)

        # Batch norms and activation
        self.bn1 = nn.BatchNorm2d(in_channels * expand_ratio)
        if not fused:
            self.bn2 = nn.BatchNorm2d(in_channels * expand_ratio)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        # Skip connection
        self.use_skip = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x

        # Expansion
        if self.expand_conv is not None:
            x = self.act(self.bn1(self.expand_conv(x)))
        else:
            x = self.act(self.bn1(x))

        # Depthwise
        if self.depthwise_conv is not None:
            x = self.act(self.bn2(self.depthwise_conv(x)))

        # SE
        if self.se is not None:
            se_weight = self.se(x)
            x = x * se_weight

        # Project
        x = self.bn3(self.project_conv(x))

        if self.use_skip:
            x = x + identity

        return x


class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=model_options.NUM_CLASSES, dropout_rate=model_options.DROPOUT_RATE, config=None):
        super(EfficientNetV2, self).__init__()
        

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=model_options.STEM_STRIDE, padding=model_options.PADDING, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Stages
        self.stages = nn.ModuleList()
        in_channels = 32
        for stage_config in config:
            blocks, in_c, out_c, exp_r, k, s, se_r, fused = stage_config
            stage = []
            for i in range(blocks):
                stride = s if i == 0 else 1
                stage.append(MBConv(in_channels, out_c, exp_r, k, stride, se_r, fused))
                in_channels = out_c
            self.stages.append(nn.Sequential(*stage))

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x


class EfficientNetLightning(LightningModule):
    """
    LightningModule wrapper for custom EfficientNetV2_L built from scratch.
    Handles training, validation, and optimization automatically.
    Allows manual adjustment of layers via config.
    """

    def __init__(self, learning_rate=model_options.INITIAL_LR, weight_decay=model_options.WEIGHT_DECAY,
                 dropout_rate=model_options.DROPOUT_RATE, config=None):
        super().__init__()

        # Save hyperparameters (for logging and checkpointing)
        self.save_hyperparameters()

        # Build EfficientNetV2_L from scratch
        self.model = EfficientNetV2(num_classes=100, dropout_rate=dropout_rate, config=config)

        # Freeze feature extractor layers (stem and stages)
        for param in self.model.stem.parameters():
            param.requires_grad = False
        for stage in self.model.stages:
            for param in stage.parameters():
                param.requires_grad = False


        # Define loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_options.NUM_CLASSES)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_options.NUM_CLASSES)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=model_options.NUM_CLASSES)

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
            if "head" in name and "6" in name:  # Assuming head is the classifier
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        # Lower learning rate for backbone
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
