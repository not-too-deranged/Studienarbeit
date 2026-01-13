import json
import os
import time

import optuna
import torch
import torchvision.models
from lightning import Trainer
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback

import selftrained
import pretrained_CNN
import pretrained_main
import selftrained_CNN
import selftrained_main
import selftrained_sanity_check_CNN
import selftrained_sanity_check_main
from confusion_matrix import plot_confusion_matrix
from util_files.load_data import prepare_data

"""
File containing the program parts, that every training has in common
"""


def main(hparams):
    train_loader, val_loader, test_loader = prepare_data(hparams)

    # ============================================================
    # LOGGING AND CALLBACKS
    # ============================================================

    # Lightning handles TensorBoard automatically; no manual SummaryWriter needed
    run_name = hparams.RUNNAME
    tb_logger = loggers.TensorBoardLogger(save_dir=f"{hparams.LOG_DIR}", name=run_name)
    tb_logger.log_hyperparams(step = hparams.NUM_EPOCHS)



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
    match hparams.STUDY_TYPE:
        case "pretrained":
            model = pretrained_CNN.EfficientNetLightning(learning_rate = hparams.LEARNING_RATE,
                                          weight_decay=hparams.WEIGHT_DECAY,
                                          dropout_rate=hparams.DROPOUT_RATE, unfreeze_layers=hparams.UNFREEZE_LAYERS,
                                          num_classes=hparams.NUM_CLASSES)
        case "selftrained":
            model = selftrained_CNN.EfficientNetLightning(learning_rate=hparams.LEARNING_RATE, weight_decay=hparams.WEIGHT_DECAY,
                                          dropout_rate=hparams.DROPOUT_RATE, num_layers=hparams.NUM_LAYERS,
                                          num_classes=hparams.NUM_CLASSES)
        case "selftrained_sanity_check":
            model = selftrained_sanity_check_CNN.EfficientNetLightning(learning_rate=hparams.LEARNING_RATE, weight_decay=hparams.WEIGHT_DECAY,
                                          dropout_rate=hparams.DROPOUT_RATE, num_classes=hparams.NUM_CLASSES)
        case _:
            raise ValueError(f"Unknown study type: {hparams.STUDY_TYPE}")

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
    test_results = trainer.test(model, dataloaders=test_loader)

    # Collect predictions and labels for confusion matrix
    model.eval()
    correct_labels = []
    predict_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            correct_labels.extend(labels.cpu().numpy().tolist())
            predict_labels.extend(preds.cpu().numpy().tolist())

    # Convert to strings for the confusion matrix function
    correct_labels = [str(x) for x in correct_labels]
    predict_labels = [str(x) for x in predict_labels]

    # Plot confusion matrix
    cm_summary = plot_confusion_matrix(
        correct_labels=correct_labels,
        predict_labels=predict_labels,
        labels=[str(i) for i in range(hparams.NUM_CLASSES)],
        title="Confusion Matrix",
        tensor_name="Confusion_Matrix",
        normalize=False
    )
    tb_logger.experiment.add_summary(cm_summary, global_step=0)

    #tb_logger.experiment.add_figure(
    #    tag="Confusion_Matrix",
    #    figure= cm_summary,
    #    global_step=0
    #)

    # Convert to a single dict (if there's only one test loader)
    if isinstance(test_results, list) and len(test_results) == 1:
        test_results = test_results[0]

    # Optionally: include hyperparams or other metadata
    to_save = {
        "model_name": hparams.MODELNAME,
        "hyperparameters": {k: getattr(hparams, k) for k in dir(hparams) if not k.startswith("_")},
        "test_results": test_results,
    }

    # File path
    results_file = "./test_results.json"

    # Append to JSON file
    if os.path.exists(results_file):
        # Load existing
        with open(results_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(to_save)

    with open(results_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Test results appended to '{results_file}'")


def optuna_train_objective(hparams, trial, model):
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


def callback_time_over(study, trial):
    start_time = study.user_attrs["start_time"]
    max_time = study.user_attrs["max_time"]

    if time.time() - start_time > max_time:
        study.stop()


def run_optuna_study(hparams):
    match hparams.STUDY_TYPE:
        case "pretrained":
            objective = pretrained_main.objective_pretrained
        case "selftrained":
            objective = selftrained_main.objective_selftrained
        case "selftrained_sanity_check":
            objective = selftrained_sanity_check_main.objective_selftrained_sanity_check
        case _:
            raise ValueError(f"Unknown study type: {hparams.STUDY_TYPE}")

    start_time = time.time()

    # Wrap the objective inside a lambda and call objective inside it
    func = lambda trial: objective(trial, hparams)

    study = optuna.create_study(direction="maximize", study_name=hparams.STUDY_NAME)

    study.set_user_attr("start_time", start_time)
    study.set_user_attr("max_time", hparams.OPTUNA_MAX_TIME)

    study.optimize(func, n_trials=hparams.N_TRIALS_OPTUNA, callbacks=[callback_time_over])

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (val_acc): {trial.value:.4f}")
    print(f"  Params: {trial.params}")

    duration = time.time() - start_time
    print("\n=== OPTUNA STUDY FINISHED ===")
    print(f"Total time: {duration / 60:.2f} min")

    return trial
