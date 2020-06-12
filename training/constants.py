from enum import unique, Enum


@unique
class Metric(Enum):
    LOSS = 'loss'
    ACCURACY = 'accuracy'
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'


# Name of the directory where the artifacts of a training run will be stored
_TRAINING_RUNS_DIR = 'training_runs'

# Name of the directory where a model checkpoint will be stored
_MODEL_CHECKPOINT_DIR = 'model'

# Name of the file that stores the training history of a training run
_HISTORY_FILE_NAME = 'history.pkl'
