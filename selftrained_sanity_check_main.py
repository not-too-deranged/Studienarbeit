import json
import multiprocessing
import time

import optuna
import torch
import torchvision
from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader

from selftrained_sanity_check_CNN import EfficientNetLightning

from util_files.load_data import prepare_data
from util_files.model_options import Hparams


def main(hparams):
    train_loader, val_loader, test_loader = prepare_data(hparams)

    # ============================================================
    # LOGGING AND CALLBACKS
    # ============================================================

    # Lightning handles TensorBoard automatically; no manual SummaryWriter needed
    run_name = hparams.RUNNAME
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
    model = EfficientNetLightning(learning_rate=hparams.LEARNING_RATE, weight_decay=hparams.WEIGHT_DECAY,
                                  dropout_rate=hparams.DROPOUT_RATE, num_classes=hparams.NUM_CLASSES)

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

    torch.save(model.state_dict(), f"./models/{hparams.MODELNAME}.pth")
    print(f"Model saved to './models/{hparams.MODELNAME}.pth'")

    # ============================================================
    # TEST BEST MODEL
    # ============================================================

    print("\nTesting best saved model on test data...")
    trainer.test(model, dataloaders=test_loader)


def objective(trial, hparams):

    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)

    model = EfficientNetLightning(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        num_classes=hparams.NUM_CLASSES
    )

    train_loader, val_loader, _ = prepare_data(hparams)

    tb_logger = loggers.TensorBoardLogger(save_dir=f"{hparams.LOG_DIR_OPTUNA}", name=f"trial_{trial.number}")

    trainer = Trainer(
        max_epochs=hparams.NUM_EPOCHS_OPTUNA,  # keep short for tuning
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


def run_optuna_study(n_trials, hparams):
    """
    tracker = EmissionsTracker(
        project_name="optuna_tuning",
        output_dir="./emission_logs_c3_pretrained",
        measure_power_secs=15
    )
    tracker.start()

    """

    start_time = time.time()

    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: objective(trial, hparams)

    study = optuna.create_study(direction="maximize", study_name=hparams.STUDY_NAME)
    study.optimize(func, n_trials=n_trials)

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

    used_dataset = "CIFAR-100"
    study_type = "selftrained_sanity_check"
    hparams = Hparams(used_dataset=used_dataset)

    best_trial = run_optuna_study(n_trials=40, hparams=hparams)
    best_params = best_trial.params
    print(f"the best parameters are: {best_params}")
    with open("best_parameters_sanity_check_c1.json", "w") as f:
        json.dump(best_params, f)

    hparams = Hparams(dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"],
                      weight_decay=best_params["weight_decay"])
    main(hparams)
