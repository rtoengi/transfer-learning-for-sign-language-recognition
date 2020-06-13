from pathlib import Path

from tensorflow.keras.optimizers import SGD

from base_model.inflated_3d_inception_v3 import Inflated3DInceptionV3, load_inflated_imagenet_weights
from datasets.constants import DatasetName, DatasetType
from datasets.msasl.constants import N_CLASSES
from datasets.tf_record_utils import tf_record_dataset, transform_for_msasl_model
from training.callbacks import ThresholdStopping
from training.constants import Metric
from training.utils import create_training_runs_dir, save_model, save_history


def _train_dataset():
    train_dataset = tf_record_dataset(DatasetName.MSASL, DatasetType.TRAIN)
    train_dataset = train_dataset.shuffle(2048)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.map(transform_for_msasl_model)
    return train_dataset.prefetch(2)


def _model():
    model = Inflated3DInceptionV3(classes=N_CLASSES)
    load_inflated_imagenet_weights(model)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(path: Path):
    train_dataset = _train_dataset()
    ts = ThresholdStopping(Metric.ACCURACY, 0.95)
    model = _model()
    history = model.fit(train_dataset, epochs=100, callbacks=[ts])
    save_model(path, model)
    return history.history


if __name__ == '__main__':
    path = create_training_runs_dir(Path())
    history = train(path)
    save_history(history, path)
