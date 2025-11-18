import json
import multiprocessing
import time

import optuna
import torch
import torchvision
import torchvision.transforms as transforms
from codecarbon import EmissionsTracker
from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

import model_options
from c1_pretrained_CNN import EfficientNetLightning
from compute_cost_logger import ComputeCostLogger


class Hparams:

    def __init__(self, batch_size = model_options.BATCH_SIZE, initial_lr = model_options.INITIAL_LR, num_epochs = model_options.NUM_EPOCHS,
              max_epochs_lr_finder = model_options.MAX_EPOCHS_LR_FINDER, patience = model_options.PATIENCE,
              num_workers = model_options.NUM_WORKERS, padding = model_options.PADDING, input_size = model_options.INPUT_SIZE,
              logging_steps = model_options.LOGGING_STEPS, log_dir = model_options.LOG_DIR, dropout_rate = model_options.DROPOUT_RATE,
              unfreeze_layers = model_options.UNFREEZE_LAYERS, weight_decay = model_options.WEIGHT_DECAY,
                 log_dir_optuna = model_options.LOG_DIR_OPTUNA, checkpoint_dir = model_options.CHECKPOINT_DIR,
                 modelname = model_options.MODELNAME, runname = model_options.RUNNAME):
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
        self.LOG_DIR_OPTUNA = log_dir_optuna
        self.CHECKPOINT_DIR = checkpoint_dir
        self.MODELNAME = modelname
        self.RUNNAME = runname


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
    """
    train_val_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, transform=transform_train, download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, transform=transform_test, download=True
    )
    """


    # Download Places365 dataset for third test case
    #"""
    print("loading data")
    train_dataset = torchvision.datasets.Places365(
        root="./data", split="train-standard", transform=transform_train, download=True, small=True
    )
    print("loaded train")

    #places 365 has a val dataset
    val_dataset = torchvision.datasets.Places365(
        root="./data", split="val", transform=transform_train, download=True, small=True
    )
    print("loaded val")

    test_dataset = torchvision.datasets.Places365(
    root="./data", split="test", transform=transform_test, download=True, small=True
    )
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
    #train_size = int(0.8 * len(train_val_dataset))
    #val_size = len(train_val_dataset) - train_size
    #train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])



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
        dirpath=f"./{hparams.CHECKPOINT_DIR}",
        filename="best_model",
        save_top_k=1,
        monitor="loss/val",
        mode="min"
    )

    # Initialize the model
    model = EfficientNetLightning(learning_rate = hparams.INITIAL_LR, weight_decay=hparams.WEIGHT_DECAY,
                                  dropout_rate=hparams.DROPOUT_RATE, unfreeze_layers=hparams.UNFREEZE_LAYERS)

    # ============================================================
    # FULL TRAINING WITH EARLY STOPPING + CHECKPOINTING
    # ============================================================

    trainer = Trainer(
        max_epochs=hparams.NUM_EPOCHS,
        accelerator="auto",
        callbacks=[ComputeCostLogger(output_dir="emission_logs_c3_pretrained"), early_stop_callback, checkpoint_callback],
        logger=tb_logger,
        log_every_n_steps=hparams.LOGGING_STEPS,
    )

    print("\nStarting training with tuned learning rate...")
    trainer.fit(model, train_loader, val_loader)

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================

    torch.save(model.state_dict(), f"./models/{hparams.MODELNAME}.pth")
    print(f"Model saved to './models/{hparams.MODELNAME}.pth'")

    # ============================================================
    # TEST BEST MODEL
    # ============================================================

    print("\nTesting best saved model on test data...")
    trainer.test(model, dataloaders=test_loader)


def objective(trial):
    hparams = Hparams(log_dir_optuna="lightning_logs_optuna_c3_pretrained")

    # Hyperparameters to optimize
    unfreeze_layers = trial.suggest_int("unfreeze_layers", 0, 8)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

    model = EfficientNetLightning(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        unfreeze_layers=unfreeze_layers,
        weight_decay=weight_decay
    )

    train_loader, val_loader, _ = prepare_data(hparams)

    tb_logger = loggers.TensorBoardLogger(save_dir=f"{model_options.LOG_DIR_OPTUNA}", name=f"trial_{trial.number}")

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
    """
    tracker = EmissionsTracker(
        project_name="optuna_tuning",
        output_dir="./emission_logs_c3_pretrained",
        measure_power_secs=15
    )
    tracker.start()

    """

    start_time = time.time()

    study = optuna.create_study(direction="maximize", study_name="efficientnet_places_unfreeze")
    study.optimize(objective, n_trials=n_trials)

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (val_acc): {trial.value:.4f}")
    print(f"  Params: {trial.params}")

    total_emissions = tracker.stop()
    duration = time.time() - start_time
    print("\n=== OPTUNA STUDY COST SUMMARY ===")
    print(f"Total time: {duration / 60:.2f} min")
    print(f"Total energy: {tracker.final_emissions_data.energy_consumed:.3f} kWh")
    print(f"Total CO2: {total_emissions:.6f} kg")

    return trial


if __name__ == '__main__':

    torch.set_float32_matmul_precision('high')
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_trial = run_optuna_study(n_trials = 40)
    best_params = best_trial.params
    print(f"the best parameters are: {best_params}")
    with open("best_parameters.json", "w") as f:
        json.dump(best_params, f)

    hparams = Hparams(dropout_rate=best_params["dropout_rate"], initial_lr=best_params["learning_rate"],
                      unfreeze_layers=best_params["unfreeze_layers"], weight_decay=best_params["weight_decay"],
                      checkpoint_dir = "c3_pretrained_checkpoints", modelname = "efficientnet_cats_pretrained",
                      runname = "c3_pretrained")
    main(hparams)
