from pathlib import Path

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import pretrained_model as pretrained_model_package
from core.constants import Metric
from core.utils import package_path
from datasets.constants import DatasetName, DatasetType
from datasets.signum.constants import N_CLASSES
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model
from training.callbacks import model_checkpoint
from training.utils import create_training_runs_dir, model_path, load_model, save_history


def _train_dataset():
    train_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.TRAIN)
    train_dataset = train_dataset.shuffle(2048)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.map(transform_for_signum_model)
    return train_dataset.prefetch(2)


def _validation_dataset():
    validation_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.VALIDATION)
    validation_dataset = validation_dataset.batch(32)
    validation_dataset = validation_dataset.map(transform_for_signum_model)
    return validation_dataset.prefetch(2)


def _align_to_target_task(pretrained_model: Model):
    x = pretrained_model.layers[-2].output
    x = Dense(N_CLASSES, activation='softmax', name='predictions')(x)
    return Model(pretrained_model.input, x)


def _model():
    pretrained_model_path = model_path(package_path(pretrained_model_package), PRETRAINED_MODEL_TRAINING_RUN)
    pretrained_model = load_model(pretrained_model_path)
    model = _align_to_target_task(pretrained_model)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(path: Path):
    train_dataset = _train_dataset()
    validation_dataset = _validation_dataset()
    mc = model_checkpoint(Metric.VAL_LOSS, path)
    model = _model()
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=40, callbacks=[mc])
    return history.history


PRETRAINED_MODEL_TRAINING_RUN = '20200612_235400'

if __name__ == '__main__':
    path = create_training_runs_dir(Path())
    history = train(path)
    save_history(history, path)
