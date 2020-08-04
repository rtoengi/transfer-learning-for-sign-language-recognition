from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model

from baseline_model.constants import TRAINING_RUNS
from datasets.constants import DatasetName, DatasetType
from datasets.tf_record_utils import tf_record_dataset, transform_for_prediction
from plotting.plot import feature_maps_plot
from training.utils import model_path, load_model


def _dataset():
    """Returns the `SIGNUM` dataset example for the word `dance`.

    The dataset example corresponding to the label with index 420 and the signer with index 1 shall be returned.

    Returns:
        The dataset example for the word `dance`.
    """
    dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.TRAIN)
    dataset = dataset.batch(1)
    dataset = dataset.map(transform_for_prediction)
    dataset = dataset.unbatch()
    dataset = dataset.filter(lambda frames, label, signer: tf.math.equal(label, 420) and tf.math.equal(signer, 1))
    dataset = dataset.batch(1)
    return dataset.take(1)


def _model():
    path = model_path(Path(), TRAINING_RUNS[_TRAINING_RUNS_INDEX])
    base_model = load_model(path)
    return Model(inputs=base_model.inputs, outputs=[base_model.layers[i].output for i in _LAYER_INDICES])


_TRAINING_RUNS_INDEX = 2
_LAYER_INDICES = [7, 90, 236]


def plot_feature_maps():
    dataset = _dataset()
    model = _model()
    feature_maps = model.predict(dataset)
    for i, layer_index in enumerate(_LAYER_INDICES):
        path = Path() / 'feature_maps' / f'layer_{layer_index:03d}'
        Path(path).mkdir(parents=True, exist_ok=True)
        feature_maps_plot(feature_maps[i], path)


if __name__ == '__main__':
    plot_feature_maps()
