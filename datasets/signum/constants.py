from enum import Enum, unique


@unique
class TrainDatasetSize(Enum):
    """Enumeration of the different `SIGNUM` training dataset sizes.

    Attributes:
        SMALL: This dataset consists of a total of 1,800 training examples of 4 different signers.
        MEDIUM: This dataset consists of a total of 3,600 training examples of 8 different signers.
        LARGE: This dataset consists of a total of 7,200 training examples of 16 different signers.
    """
    SMALL = {'n_examples': 1800, 'signers': 4}
    MEDIUM = {'n_examples': 3600, 'signers': 8}
    LARGE = {'n_examples': 7200, 'signers': 16}


# Number of classes
N_CLASSES = 450

# Root directory of the `SIGNUM` dataset
_SIGNUM_DIR = 'D:/datasets/signum'

# Directory of the downloaded `SIGNUM` images
_SIGNUM_IMAGES_DIR = f'{_SIGNUM_DIR}/images'

# Number of frames of a single dataset example
_N_FRAMES_PER_EXAMPLE = 80

# Directory of the `SIGNUM` train, validation and test `TFRecord` files
_SIGNUM_TF_RECORDS_DIR = f'{_SIGNUM_DIR}/tf_records'
