from pathlib import Path

from tensorflow.keras.optimizers import SGD

from base_model.inflated_3d_inception_v3 import Inflated3DInceptionV3, load_inflated_imagenet_weights
from datasets.constants import DatasetName, DatasetType
from datasets.signum.constants import N_CLASSES
from datasets.tf_record_utils import tf_record_dataset, transform_for_signum_model
from training.callbacks import early_stopping, model_checkpoint
from training.utils import save_history, create_training_runs_dir


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


def _model():
    model = Inflated3DInceptionV3(classes=N_CLASSES)
    load_inflated_imagenet_weights(model)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(path: Path):
    train_dataset = _train_dataset()
    validation_dataset = _validation_dataset()
    es = early_stopping('val_loss', 15)
    mc = model_checkpoint('val_loss', path)
    model = _model()
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=100, callbacks=[es, mc])
    return history.history


if __name__ == '__main__':
    path = create_training_runs_dir(Path())
    history = train(path)
    save_history(history, path)
