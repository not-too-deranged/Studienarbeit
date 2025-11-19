    # ============================================================
    # HYPERPARAMETERS AND DEVICE
    # ============================================================

BATCH_SIZE  = 64
INITIAL_LR = 1e-3
NUM_EPOCHS = 50
MAX_EPOCHS_LR_FINDER = 3
PATIENCE = 5  # early stopping patience
NUM_WORKERS = 4
PADDING = 4  # for random crop
INPUT_SIZE = 224  # EfficientNet minimum input size
LOGGING_STEPS = 10
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2
NUM_CLASSES = 100
STEM_STRIDE = 2  # stride for the initial conv layer
LOG_DIR = "lightning_logs_c2_selftrained"
LOG_DIR_OPTUNA = "lightning_logs_optuna_c2_selftrained"
UNFREEZE_LAYERS = 0
CHECKPOINT_DIR = "c2_selftrained_checkpoints"
MODELNAME = "efficientnet_places_pretrained"
RUNNAME = "c2_selftrained"
STAGES = [
    [2, 32, 32, 1, 3, 1, 0.25, True],
    [4, 32, 64, 4, 3, 2, 0.25, True],
    [4, 64, 96, 4, 3, 2, 0.25, True],
    [6, 96, 192, 4, 3, 2, 0.25, False],
    [9, 192, 224, 6, 3, 1, 0.25, False],
    [15, 224, 384, 6, 3, 2, 0.25, False]
]


#tensorboard --logdir /path/to/lightning_logs