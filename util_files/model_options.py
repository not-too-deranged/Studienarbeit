from datetime import datetime

USED_DATASET = "default_val"
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
UNFREEZE_LAYERS = 0
WEIGHT_DECAY = 1e-4
STUDY_TYPE = "default_val"
PATIENCE = 6  # early stopping patience
NUM_LAYERS = [4, 7, 7, 10, 19, 25, 7]
BATCH_SIZE  = 64
NUM_WORKERS = 4
PADDING = 4  # for random crop
INPUT_SIZE = 224  # EfficientNet minimum input size
NUM_EPOCHS = 250
NUM_EPOCHS_OPTUNA = 5
LOGGING_STEPS = 10
N_TRIALS_OPTUNA = 50
OPTUNA_MAX_TIME = 43200 #12 hours

class Hparams:
    def __init__(self, used_dataset=USED_DATASET, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE,
                      unfreeze_layers=UNFREEZE_LAYERS, weight_decay=WEIGHT_DECAY, study_type=STUDY_TYPE,
                      timestamp=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), num_epochs=NUM_EPOCHS, logging_steps=LOGGING_STEPS, patience=PATIENCE,
                      num_layers=NUM_LAYERS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, padding=PADDING,
                      input_size=INPUT_SIZE, num_epochs_optuna=NUM_EPOCHS_OPTUNA, n_trials_optuna=N_TRIALS_OPTUNA,
                      optuna_max_time = OPTUNA_MAX_TIME):

        self.USED_DATASET = used_dataset
        self.DROPOUT_RATE = dropout_rate
        self.LEARNING_RATE = learning_rate
        self.UNFREEZE_LAYERS = unfreeze_layers
        self.WEIGHT_DECAY = weight_decay
        self.STUDY_TYPE = study_type
        self.CHECKPOINT_DIR = f"./checkpoints/{used_dataset}_{study_type}_checkpoints_{timestamp}"
        self.MODELNAME = f"{used_dataset}_{study_type}_{timestamp}"
        self.RUNNAME = f"{used_dataset}_{study_type}_runname_{timestamp}"
        self.LOG_DIR = f"./lightning_logs_per_epoch/lightning_logs_{used_dataset}_{study_type}_logs_{timestamp}"
        self.LOG_DIR_OPTUNA = f"./lightning_logs_optuna/lightning_logs_optuna_{used_dataset}_{study_type}_logs_{timestamp}"
        self.STUDY_NAME = f"{used_dataset}_{study_type}_study_{timestamp}"
        self.NUM_EPOCHS = num_epochs
        self.LOGGING_STEPS = logging_steps
        self.PATIENCE = patience
        self.NUM_LAYERS = num_layers
        self.BATCH_SIZE = batch_size
        self.NUM_WORKERS = num_workers
        self.PADDING = padding
        self.INPUT_SIZE = input_size
        self.NUM_EPOCHS_OPTUNA = num_epochs_optuna
        self.N_TRIALS_OPTUNA = n_trials_optuna
        self.OPTUNA_MAX_TIME = optuna_max_time

        match used_dataset:
            case "CIFAR-100":
                self.NUM_CLASSES = 100
            case "cats":
                self.NUM_CLASSES = 35
            case "places365":
                self.NUM_CLASSES = 150 #since we only use a subset of the dataset
            case _:
                raise ValueError(f"Unrecognized dataset: {used_dataset}")
