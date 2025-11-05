import optuna
from optuna.integration import PyTorchLightningPruningCallback
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
from torchvision import models
import multiprocessing
from c1_pretrained_CNN import EfficientNetLightning
import model_options

class Hparams:

    def __init__(self, batch_size = model_options.BATCH_SIZE, initial_lr = model_options.INITIAL_LR, num_epochs = model_options.NUM_EPOCHS,
              max_epochs_lr_finder = model_options.MAX_EPOCHS_LR_FINDER, patience = model_options.PATIENCE,
              num_workers = model_options.NUM_WORKERS, padding = model_options.PADDING, input_size = model_options.INPUT_SIZE,
              logging_steps = model_options.LOGGING_STEPS, log_dir = model_options.LOG_DIR):
        self.BATCH_SIZE = batch_size
        self.INITIAL_LR = initial_lr
        self.NUM_EPOCHS = num_epochs
        self.MAX_EPOCHS_LR_FINDER = max_epochs_lr_finder
        self.PATIENCE = patience
        self.NUM_WORKERS = num_workers
        self.PADDING = padding
        self.INPUT_SIZE = input_size
        self.LOGGING_STEPS = logging_steps
        self.LOG_DIR = log_dir


def prepare_data(hparams):
    """
        CIFAR-100 consists of 60,000 32x32 color images in 100 classes,
        with 600 images per class. There are 50,000 training and 10,000 test images.
        """
    transform_train = transforms.Compose([
        transforms.Resize(hparams.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(hparams.INPUT_SIZE, padding=hparams.PADDING),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]  # ImageNet std
        ),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(hparams.INPUT_SIZE),
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
    # Set NUM_WORKERS=0 for compatibility with freeze support (e.g., PyInstaller executables)

    train_loader = DataLoader(train_dataset, batch_size=hparams.BATCH_SIZE, shuffle=True,
                              num_workers=hparams.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False, num_workers=hparams.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=hparams.BATCH_SIZE, shuffle=False,
                             num_workers=hparams.NUM_WORKERS)

    return train_loader, val_loader, test_loader


def main(hparams):
    train_loader, val_loader, test_loader = prepare_data(hparams)

    # ============================================================
    # LOGGING AND CALLBACKS
    # ============================================================

    # Lightning handles TensorBoard automatically; no manual SummaryWriter needed
    run_name = f"b{hparams.BATCH_SIZE}_lr{hparams.INITIAL_LR:.0e}"
    tb_logger = loggers.TensorBoardLogger(save_dir=f"{hparams.LOG_DIR}", name=run_name)

    # Early stopping monitors validation loss
    early_stop_callback = EarlyStopping(
        monitor="loss/val", mode="min", patience=hparams.PATIENCE, verbose=True
    )

    # Save best checkpoint by validation loss
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
    
    # Only a few epochs for LR finder
    trainer = Trainer(
        accelerator="auto",
        max_epochs=hparams.MAX_EPOCHS_LR_FINDER,
        logger=tb_logger,
    )

    # Initialize the model
    model = EfficientNetLightning()

    # ============================================================
    # AUTOMATIC LEARNING RATE FINDING
    # ============================================================

    print("\nRunning learning rate finder...")
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, min_lr=1e-5, max_lr=1e-1)

    # Plot the results (optional)
    new_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {new_lr:.2e}")
    model.hparams.learning_rate = new_lr

    # ============================================================
    # FULL TRAINING WITH EARLY STOPPING + CHECKPOINTING
    # ============================================================

    trainer = Trainer(
        max_epochs=hparams.NUM_EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=hparams.LOGGING_STEPS,
    )

    print("\nStarting training with tuned learning rate...")
    trainer.fit(model, train_loader, val_loader)

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================

    torch.save(model.state_dict(), "./models/efficientnet_cifar100.pth")
    print("Model saved to './models/efficientnet_cifar100.pth'")

    # ============================================================
    # TEST BEST MODEL
    # ============================================================

    print("\nTesting best saved model on test data...")
    trainer.test(model, dataloaders=test_loader)


def objective(trial):
    hparams = Hparams()

    # Hyperparameters to optimize
    unfreeze_layers = trial.suggest_int("unfreeze_layers", 0, 8)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    model = EfficientNetLightning(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        unfreeze_layers=unfreeze_layers
    )

    train_loader, val_loader, _ = prepare_data(hparams)

    tb_logger = loggers.TensorBoardLogger(save_dir="lightning_logs_optuna", name=f"trial_{trial.number}")

    trainer = Trainer(
        max_epochs=5,  # keep short for tuning
        accelerator="auto",
        logger=tb_logger,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="acc/val"),
            EarlyStopping(monitor="loss/val", patience=3, mode="min")
        ],
        enable_progress_bar=True
    )

    trainer.fit(model, train_loader, val_loader)

    # Return final validation accuracy for Optuna optimization
    val_acc = trainer.callback_metrics.get("acc/val")
    return val_acc.item() if val_acc is not None else 0.0

def run_optuna_study(n_trials=10):
    study = optuna.create_study(direction="maximize", study_name="efficientnet_cifar100_unfreeze")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (val_acc): {trial.value:.4f}")
    print(f"  Params: {trial.params}")

    return trial


if __name__ == '__main__':

    torch.set_float32_matmul_precision('high')
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_trial = run_optuna_study(n_trials = 10)
    best_params = best_trial.params
    print(f"the best parameters are: {best_params}")

    #hparams = Hparams()
    #main(hparams)
