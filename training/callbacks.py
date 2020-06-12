from pathlib import Path

from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from training.constants import Metric, _MODEL_CHECKPOINT_DIR


class ThresholdStopping(Callback):
    """Callback to stop training when a given metric exceeds or falls below a given threshold.

    The metric can be either Metric.ACCURACY or Metric.LOSS. If the metric is Metric.ACCURACY, then training stops when
    the model's accuracy exceeds the threshold. If the metric is Metric.LOSS, then training stops when the model's
    loss falls below the threshold.
    """

    def __init__(self, metric: Metric, threshold):
        """Initializes this callback.

        Arguments:
            metric: A Metric value to check for the threshold.
            threshold: A float value the metric is checked for.
        """
        super().__init__()
        self._metric = metric
        self._threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        """Called at the end of a training epoch.

        Stops training when a metric exceeds or falls below a threshold.

        Arguments:
            epoch: The index of the training epoch.
            logs: A dictionary of the metric results for this training epoch.
        """
        value = logs.get(self._metric.value)
        if self._metric == Metric.ACCURACY:
            if value > self._threshold:
                self.model.stop_training = True
        elif self._metric == Metric.LOSS:
            if value < self._threshold:
                self.model.stop_training = True


def early_stopping(metric: Metric, patience):
    """Returns an EarlyStopping callback.

    Arguments:
        metric: A Metric value the model is evaluated against.
        patience: The number of epochs with no improvement after which the training will be stopped.

    Returns:
        An EarlyStopping callback.
    """
    return EarlyStopping(monitor=metric.value, patience=patience, verbose=1)


def model_checkpoint(metric: Metric, path: Path):
    """Returns a ModelCheckpoint callback.

    Arguments:
        metric: A Metric value the model is evaluated against.
        path: A Path object pointing to the directory where the model checkpoint will be stored.

    Returns:
        A ModelCheckpoint callback.
    """
    filepath = str(path / _MODEL_CHECKPOINT_DIR)
    return ModelCheckpoint(filepath, monitor=metric.value, verbose=1, save_best_only=True)
