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

import selftrained_CNN
import train_main

from util_files.load_data import prepare_data
from util_files.model_options import Hparams


def objective_selftrained(trial, hparams):

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

    model = selftrained_CNN.EfficientNetLightning(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        num_layers=num_layers,
        num_classes=hparams.NUM_CLASSES
    )

    return train_main.optuna_train_objective(hparams, trial, model)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    used_dataset = "CIFAR-100"
    study_type = "selftrained"
    hparams = Hparams(used_dataset=used_dataset, study_type=study_type)

    best_trial = train_main.run_optuna_study(hparams=hparams)
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

    hparams = Hparams(dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"],
                      weight_decay=best_params["weight_decay"], num_layers=num_layers, study_type=study_type,
                      used_dataset=used_dataset)
    train_main.main(hparams)
