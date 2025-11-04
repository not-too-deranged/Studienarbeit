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
