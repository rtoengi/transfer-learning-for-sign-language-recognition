from datetime import datetime
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from base_model.inflated_3d_inception_v3 import Inflated3DInceptionV3, load_inflated_imagenet_weights
from datasets.constants import DatasetName, DatasetType
from datasets.signum.constants import N_CLASSES
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model


def _create_training_run_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'training_runs/{timestamp}'
    Path(path).mkdir(parents=True)
    return path


def _train_dataset():
    train_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.TRAIN)
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.map(transform_for_signum_model)
    return train_dataset.prefetch(2)


def _validation_dataset():
    validation_dataset = tf_record_dataset(DatasetName.SIGNUM, DatasetType.VALIDATION)
    validation_dataset = validation_dataset.batch(32)
    validation_dataset = validation_dataset.map(transform_for_signum_model)
    return validation_dataset.prefetch(2)


def _model():
    model = Inflated3DInceptionV3(classes=N_CLASSES)
    load_inflated_imagenet_weights(model)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def _callbacks(path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    filepath = path + '/model@epoch-{epoch:02d}'
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    return [early_stopping, model_checkpoint]


def train(path):
    train_dataset = _train_dataset()
    validation_dataset = _validation_dataset()
    callbacks = _callbacks(path)
    model = _model()
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=callbacks)
    return history.history


def _save_history(history, path):
    np.save(f'{path}/history', history)


if __name__ == '__main__':
    path = _create_training_run_dir()
    history = train(path)
    _save_history(history, path)
