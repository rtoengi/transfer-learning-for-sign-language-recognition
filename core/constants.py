from enum import unique, Enum


@unique
class Metric(Enum):
    """Enumeration of the different metrics.

    Attributes:
        LOSS: The loss metric evaluated on the train dataset.
        ACCURACY: The accuracy metric evaluated on the train dataset.
        VAL_LOSS: The loss metric evaluated on the validation dataset.
        VAL_ACCURACY: The accuracy metric evaluated on the validation dataset.
    """
    LOSS = 'loss'
    ACCURACY = 'accuracy'
    VAL_LOSS = 'val_loss'
    VAL_ACCURACY = 'val_accuracy'
