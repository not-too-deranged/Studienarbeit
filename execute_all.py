import json
import multiprocessing

import torch
import torchvision.models

import train_main
import selftrained
from selftrained_CNN import EfficientNetLightning
from util_files.model_options import Hparams

"""
File containing the main loop_which starts all trainings
"""


def main_loop(used_dataset, study_type):
    hparams = Hparams(used_dataset=used_dataset, study_type=study_type)

    best_trial = train_main.run_optuna_study(hparams=hparams)
    best_params = best_trial.params
    print(f"the best parameters are: {best_params}")
    with open(f"./best_parameters/best_parameters_{hparams.MODELNAME}.json", "w") as f:
        json.dump(best_params, f)

    match hparams.STUDY_TYPE:
        case "pretrained":
            hparams = Hparams(dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"],
                              unfreeze_layers=best_params["unfreeze_layers"], weight_decay=best_params["weight_decay"],
                              study_type=study_type, used_dataset=used_dataset)
        case "selftrained":
            num_layers = [
                best_params["stage1_repeats"],
                best_params["stage2_repeats"],
                best_params["stage3_repeats"],
                best_params["stage4_repeats"],
                best_params["stage5_repeats"],
                best_params["stage6_repeats"],
                best_params["stage7_repeats"]
            ]

            hparams = Hparams(dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"],
                              weight_decay=best_params["weight_decay"], num_layers=num_layers, study_type=study_type,
                              used_dataset=used_dataset)
        case "selftrained_sanity_check":
            hparams = Hparams(dropout_rate=best_params["dropout_rate"], learning_rate=best_params["learning_rate"],
                              weight_decay=best_params["weight_decay"], study_type=study_type,
                              used_dataset=used_dataset)
        case _:
            raise ValueError(f"Unrecognized study type: {study_type}")

    train_main.main(hparams)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    multiprocessing.freeze_support()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    used_datasets = ["CIFAR-100", "cats", "places365"]
    study_types = ["selftrained", "pretrained", "selftrained"]
    study_sanity_check = "selftrained_sanity_check" # its own category because it goes last for every dataset besides the first


    #main_loop(used_datasets[0], study_sanity_check)

    for used_dataset in used_datasets:
        for study_type in study_types:
            main_loop(used_dataset, study_type)

    for used_dataset in used_datasets[1:]:
        main_loop(used_dataset, study_sanity_check)
