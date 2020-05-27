from enum import Enum, unique


@unique
class DatasetName(Enum):
    """Enumeration of the different dataset names.

    Attributes:
        MSASL: The source `MS-ASL` dataset used for pre-training a model.
        SIGNUM: The target `SIGNUM` dataset used for fine-tuning a model.
    """
    MSASL = 'msasl'
    SIGNUM = 'signum'


@unique
class DatasetType(Enum):
    """Enumeration of the different dataset types.

    Attributes:
        TRAIN: The name of the dataset used for training a model.
        VALIDATION: The name of the dataset used for validating a model.
        TEST: The name of the dataset used for evaluating a model.
    """
    TRAIN = 'train'
    VALIDATION = 'val'
    TEST = 'test'


# Number of time steps / frames a dataset example consists of
_N_TIME_STEPS = 20

# Maximum number of dataset examples a single `TFRecord` file contains
_TF_RECORD_SHARD_SIZE = 512

# The width and height of an input frame / image
_FRAME_SIZE = (224, 224)
