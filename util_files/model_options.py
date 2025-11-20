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
LOG_DIR = "lightning_logs_c1_selftrained_v2_l_rebuilt"
LOG_DIR_OPTUNA = "lightning_logs_optuna_c1_selftrained_v2_l_rebuilt"
UNFREEZE_LAYERS = 0
CHECKPOINT_DIR = "c1_selftrained_checkpoints_v2_l_rebuilt"
MODELNAME = "efficientnet_cifar-selftrained_v2_l_rebuilt"
RUNNAME = "c1_selftrained_v2_l_rebuilt"
STAGES = [
    [4, 32, 32, 1, 3, 1, 0., True],
    [7, 32, 64, 4, 3, 2, 0., True],
    [7, 64, 96, 4, 3, 2, 0., True],
    [10, 96, 192, 4, 3, 2, 0.25, False],
    [19, 192, 224, 6, 3, 1, 0.25, False],
    [25, 224, 384, 6, 3, 2, 0.25, False],
    [7, 384, 640, 6, 3, 1, 0.25, False]
]
NUM_LAYERS = [4, 7, 7, 10, 19, 25, 7]


#tensorboard --logdir /path/to/lightning_logs