from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from base_model.inflated_3d_inception_v3 import Inflated3DInceptionV3, load_inflated_imagenet_weights
from datasets.constants import DatasetName, DatasetType
from datasets.msasl.constants import N_CLASSES
from datasets.tf_record_utils import tf_record_dataset, transform_for_prediction
from plotting.plot import feature_maps_plot


def _dataset():
    """Returns the `MS-ASL` dataset example for the word `dance`.

    The dataset example corresponding to the label with index 84 and the signer with index 347 shall be returned.

    Returns:
        The dataset example for the word `dance`.
    """
    dataset = tf_record_dataset(DatasetName.MSASL, DatasetType.TRAIN)
    dataset = dataset.batch(1)
    dataset = dataset.map(transform_for_prediction)
    dataset = dataset.unbatch()
    dataset = dataset.filter(lambda frames, label, signer: tf.math.equal(label, 84) and tf.math.equal(signer, 347))
    dataset = dataset.batch(1)
    return dataset.take(1)


def _model():
    base_model = Inflated3DInceptionV3(classes=N_CLASSES)
    load_inflated_imagenet_weights(base_model)
    base_model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy')
    return Model(inputs=base_model.inputs, outputs=[base_model.layers[i].output for i in _LAYER_INDICES])


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
