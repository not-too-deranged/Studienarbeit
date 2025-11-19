import json
import random
import time

import optuna
from codecarbon import EmissionsTracker
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

import FilteredPlaces365
from c1_selftrained_CNN import EfficientNetLightning
import model_options
from compute_cost_logger import ComputeCostLogger


class Hparams:

    def __init__(self, batch_size = model_options.BATCH_SIZE, initial_lr = model_options.INITIAL_LR, num_epochs = model_options.NUM_EPOCHS,
              max_epochs_lr_finder = model_options.MAX_EPOCHS_LR_FINDER, patience = model_options.PATIENCE,
              num_workers = model_options.NUM_WORKERS, padding = model_options.PADDING, input_size = model_options.INPUT_SIZE,
              logging_steps = model_options.LOGGING_STEPS, log_dir = model_options.LOG_DIR, dropout_rate = model_options.DROPOUT_RATE,
              unfreeze_layers = model_options.UNFREEZE_LAYERS, weight_decay = model_options.WEIGHT_DECAY, stages = model_options.STAGES):
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
        self.DROPOUT_RATE = dropout_rate
        self.UNFREEZE_LAYERS = unfreeze_layers
        self.WEIGHT_DECAY = weight_decay
        self.STAGES = stages


def prepare_data(hparams):

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

    # ============================================================
    # Download datasets
    # ============================================================


    # Download the CIFAR-100 dataset
    #"""
    train_val_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, transform=transform_train, download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, transform=transform_test, download=True
    )
    #"""


    # Download Places365 dataset for third test case
    """
    ALL_CLASSES = list(range(365))
    random.seed(1234)
    SELECTED_150 = sorted(random.sample(ALL_CLASSES, 150))

    train_dataset = torchvision.datasets.Places365(
        root="./data", split="train-standard", transform=transform_train,
        download=True, small=True
    )
    #dataset is reduced in size by only using 150 of the 365 classes
    train_dataset = FilteredPlaces365.start_filter(train_dataset, SELECTED_150)

    print("loaded train")

    #places 365 has a val dataset
    val_dataset = torchvision.datasets.Places365(
        root="./data", split="val", transform=transform_train, download=True, small=True
    )
    # dataset is reduced in size by only using 150 of the 365 classes
    val_dataset = FilteredPlaces365.start_filter(val_dataset, SELECTED_150)
    print("loaded val")

    test_dataset = torchvision.datasets.Places365(
    root="./data", split="test", transform=transform_test, download=True, small=True
    )
    # dataset is reduced in size by only using 150 of the 365 classes
    test_dataset = FilteredPlaces365.start_filter(test_dataset, SELECTED_150)
    print("loaded test")
    #"""


    # load cat dataset: https://github.com/Aml-Hassan-Abd-El-hamid/datasets

    """
    
    train_val_dataset = torchvision.datasets.ImageFolder(
        root="./data/cat-dataset/train", transform=transform_train
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root="./data/cat-dataset/test", transform=transform_test
    )
    #"""

    # Split train/val properly (80/20 split)
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])



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

    # Initialize the model
    model = EfficientNetLightning(learning_rate = hparams.INITIAL_LR, weight_decay=hparams.WEIGHT_DECAY,
                                  dropout_rate=hparams.DROPOUT_RATE, config=hparams.STAGES)

    # ============================================================
    # FULL TRAINING WITH EARLY STOPPING + CHECKPOINTING
    # ============================================================

    trainer = Trainer(
        max_epochs=hparams.NUM_EPOCHS,
        accelerator="auto",
        callbacks=[ComputeCostLogger(), early_stop_callback, checkpoint_callback],
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
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    config = [
    trial.suggest_int("stage1_repeats", 1, 4),
    trial.suggest_int("stage2_repeats", 2, 6),
    trial.suggest_int("stage3_repeats", 2, 6),
    trial.suggest_int("stage4_repeats", 3, 8),
    trial.suggest_int("stage5_repeats", 4, 10),
    trial.suggest_int("stage6_repeats", 6, 16),
    trial.suggest_int("stage7_repeats", 8, 18)
]


    stages = [
    [config[0], 32, 32, 1, 3, 1, 0.25, True],
    [config[1], 32, 64, 4, 3, 2, 0.25, True],
    [config[2], 64, 96, 4, 3, 2, 0.25, True],
    [config[3], 96, 192, 4, 3, 2, 0.25, False],
    [config[4], 192, 224, 6, 3, 1, 0.25, False],
    [config[5], 224, 384, 6, 3, 2, 0.25, False],
    [config[6], 384, 640, 6, 3, 1, 0.25, False]
]



    model = EfficientNetLightning(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        config=stages
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
    tracker = EmissionsTracker(
        project_name="optuna_tuning",
        output_dir="./emission_logs_c1_selftrained",
        measure_power_secs=15
    )
    #tracker.start()
    start_time = time.time()

    study = optuna.create_study(direction="maximize", study_name="efficientnet_cifar100_unfreeze")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (val_acc): {trial.value:.4f}")
    print(f"  Params: {trial.params}")

    total_emissions = tracker.stop()
    duration = time.time() - start_time
    print("\n=== OPTUNA STUDY COST SUMMARY ===")
    print(f"Total time: {duration / 60:.2f} min")

    return trial


if __name__ == '__main__':

    torch.set_float32_matmul_precision('high')
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    best_trial = run_optuna_study(n_trials = 20)
    best_params = best_trial.params

    print(f"the best parameters are: {best_params}")
    with open("best_parameters.json", "w") as f:
        json.dump(best_params, f)

    stages = [
        [best_params["stage1_repeats"], 32, 32, 1, 3, 1, 0.25, True],
        [best_params["stage2_repeats"], 32, 64, 4, 3, 2, 0.25, True],
        [best_params["stage3_repeats"], 64, 96, 4, 3, 2, 0.25, True],
        [best_params["stage4_repeats"], 96, 192, 4, 3, 2, 0.25, False],
        [best_params["stage5_repeats"], 192, 224, 6, 3, 1, 0.25, False],
        [best_params["stage6_repeats"], 224, 384, 6, 3, 2, 0.25, False],
        [best_params["stag7_repeats"], 384, 640, 6, 3, 1, 0.25, False]
    ]
    #"""

    #sanity check of the sanity check
    best_params = {'learning_rate': 0.001572222062812689, 'dropout_rate': 0.2106854604258504, 'weight_decay': 0.09620862174008252}
    hparams = Hparams(dropout_rate=best_params["dropout_rate"], initial_lr=best_params["learning_rate"],
                      weight_decay=best_params["weight_decay"])

    #hparams = Hparams(dropout_rate=best_params["dropout_rate"], initial_lr=best_params["learning_rate"],
    #                  stages=stages, weight_decay=best_params["weight_decay"])
    main(hparams)
