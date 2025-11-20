import json
import multiprocessing
import time

import optuna
import torch
import torchvision
import torchvision.transforms as transforms
from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

from util_files import model_options
from pytorch_test_CNN import EfficientNetLightning
from compute_cost_logger import ComputeCostLogger

from util_files.load_data import prepare_data


class Hparams:

    def __init__(self, batch_size=model_options.BATCH_SIZE, initial_lr=model_options.INITIAL_LR,
                 num_epochs=model_options.NUM_EPOCHS,
                 max_epochs_lr_finder=model_options.MAX_EPOCHS_LR_FINDER, patience=model_options.PATIENCE,
                 num_workers=model_options.NUM_WORKERS, padding=model_options.PADDING,
                 input_size=model_options.INPUT_SIZE,
                 logging_steps=model_options.LOGGING_STEPS, log_dir=model_options.LOG_DIR,
                 dropout_rate=model_options.DROPOUT_RATE,
                 unfreeze_layers=model_options.UNFREEZE_LAYERS, weight_decay=model_options.WEIGHT_DECAY,
                 log_dir_optuna=model_options.LOG_DIR_OPTUNA, checkpoint_dir=model_options.CHECKPOINT_DIR,
                 modelname=model_options.MODELNAME, runname=model_options.RUNNAME, num_layers=model_options.NUM_LAYERS):
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
        self.NUM_LAYERS = num_layers


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
    model = EfficientNetLightning(learning_rate=hparams.INITIAL_LR, weight_decay=hparams.WEIGHT_DECAY,
                                  dropout_rate=hparams.DROPOUT_RATE)

    # ============================================================
    # FULL TRAINING WITH EARLY STOPPING + CHECKPOINTING
    # ============================================================

    trainer = Trainer(
        max_epochs=hparams.NUM_EPOCHS,
        accelerator="auto",
        callbacks=[ComputeCostLogger(output_dir="emission_logs_c3_pretrained"), early_stop_callback,
                   checkpoint_callback],
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
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

    num_layers = [
        trial.suggest_int("stage1_repeats", 1, 4),
        trial.suggest_int("stage2_repeats", 2, 6),
        trial.suggest_int("stage3_repeats", 2, 6),
        trial.suggest_int("stage4_repeats", 3, 8),
        trial.suggest_int("stage5_repeats", 4, 10),
        trial.suggest_int("stage6_repeats", 6, 16),
        trial.suggest_int("stage7_repeats", 8, 18)
    ]

    model = EfficientNetLightning(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        num_layers=num_layers
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

    duration = time.time() - start_time
    print(f" Time: {duration}")

    return trial


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_trial = run_optuna_study(n_trials=10)
    best_params = best_trial.params
    print(f"the best parameters are: {best_params}")
    with open("best_parameters_sanity_check_c1.json", "w") as f:
        json.dump(best_params, f)

    num_layers = [
        best_params["stage1_repeats"],
        best_params["stage2_repeats"],
        best_params["stage3_repeats"],
        best_params["stage4_repeats"],
        best_params["stage5_repeats"],
        best_params["stage6_repeats"],
        best_params["stag7_repeats"]
    ]

    hparams = Hparams(dropout_rate=best_params["dropout_rate"], initial_lr=best_params["learning_rate"],
                      weight_decay=best_params["weight_decay"], num_layers=num_layers,
                      checkpoint_dir="c1_selftrained_sanity_check_checkpoints", modelname="efficientnet_cifar_sanity_selftrained",
                      runname="c1_pretrained")
    main(hparams)
    #TODO cross validate that self built version works similar