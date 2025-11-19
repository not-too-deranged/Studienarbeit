import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning import LightningModule
import math
from typing import Optional

import model_options


class DropPath(nn.Module):
    """DropPath (stochastic depth) per sample (when applied in main path).
    From: https://github.com/rwightman/pytorch-image-models (short implementation)
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob


    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1, 1) to broadcast
        mask = x.new_empty((x.size(0),) + (1,) * (x.dim() - 1)).bernoulli_(keep_prob)
        x = x.div(keep_prob) * mask
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel: int, se_ratio: float = 0.25):
        super().__init__()
        reduced_ch = max(1, int(in_channel * se_ratio))
        self.reduce = nn.Conv2d(in_channel, reduced_ch, 1)
        self.act1 = nn.SiLU()
        self.expand = nn.Conv2d(reduced_ch, in_channel, 1)
        self.gate = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        h = self.pool(x)
        h = self.reduce(h)
        h = self.act1(h)
        h = self.expand(h)
        h = self.gate(h)
        return x * h

class UnfusedMBConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, expand_ratio: int, kernel_size: int, stride: int,
        se_ratio: float = 0.25, drop_path: float = 0.0):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_channel == out_channel)
        mid_ch = in_channel * expand_ratio
        self.has_expand = expand_ratio != 1
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


        layers = []
        # Expand
        if self.has_expand:
            layers.append(nn.Conv2d(in_channel, mid_ch, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(mid_ch))
            layers.append(nn.SiLU())


        # Depthwise conv
        layers.append(nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
        padding=kernel_size // 2, groups=mid_ch, bias=False))
        layers.append(nn.BatchNorm2d(mid_ch))
        layers.append(nn.SiLU())


        self.dw_seq = nn.Sequential(*layers)

        # SE (only typically for non-fused blocks)
        self.has_se = se_ratio is not None and se_ratio > 0
        if self.has_se:
            self.se = SqueezeExcitation(mid_ch, se_ratio=se_ratio)
        else:
            self.se = nn.Identity()


        # Project
        self.project_conv = nn.Conv2d(mid_ch, out_channel, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channel)


    def forward(self, x):
        identity = x
        out = x
        if self.has_expand:
            out = self.dw_seq(out)
        else:
            out = self.dw_seq(out)


        out = self.se(out)
        out = self.project_conv(out)
        out = self.project_bn(out)


        if self.use_res_connect:
            out = self.drop_path(out) + identity
        return out
        

class FusedMBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int, kernel_size: int, stride: int,
        se_ratio: Optional[float] = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


        # If expand_ratio == 1, fused block still uses a kxk conv from in_ch -> out_ch? In V2 implementations,
        # fused blocks often expand then project, so we implement expand to mid_ch then project.
        # Use a kxk conv from in_ch -> mid_ch
        self.fused_conv = nn.Conv2d(in_ch, mid_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.fused_bn = nn.BatchNorm2d(mid_ch)
        self.act = nn.SiLU()


        # SE (rarely used in fused blocks for V2, but allowed)
        self.has_se = se_ratio is not None and se_ratio > 0.0
        if self.has_se:
            self.se = SqueezeExcitation(mid_ch, se_ratio=se_ratio)
        else:
            self.se = nn.Identity()


        # Project
        self.project_conv = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)


    def forward(self, x):
        identity = x
        out = self.fused_conv(x)
        out = self.fused_bn(out)
        out = self.act(out)


        out = self.se(out)


        out = self.project_conv(out)
        out = self.project_bn(out)


        if self.use_res_connect:
            out = self.drop_path(out) + identity
        return out


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
            blocks, in_ch, out_channels, expand_ratio, kernel_size, stride, se_ratio, fused = stage_config
            stage = []
            for i in range(blocks):
                stride = stride if i == 0 else 1
                if fused:
                    stage.append(FusedMBConv(in_channels, out_channels, expand_ratio, kernel_size, stride, 0.0, drop_path=0.0))
                else:
                    stage.append(UnfusedMBConv(in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio, drop_path=0.0))
                in_channels = out_channels  
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
        self.model = EfficientNetV2(num_classes=model_options.NUM_CLASSES, dropout_rate=dropout_rate, config=config)


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

        # Lower learning rate for backbone
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=model_options.PATIENCE)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/val"
            }
        }
