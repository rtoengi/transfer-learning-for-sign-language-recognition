# List of the train, validation and test dataset names
DATASET_NAMES = ['train', 'val', 'test']

# Number of time steps / frames a dataset example consists of
_N_TIME_STEPS = 20

# Maximum number of dataset examples a single `TFRecord` file contains
_TF_RECORD_SHARD_SIZE = 512

# The width and height of an input frame / image
_FRAME_SIZE = (224, 224)
