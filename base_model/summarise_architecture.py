from tensorflow.keras.utils import plot_model

from base_model.inflated_3d_inception_v3 import Inflated3DInceptionV3


def summarise_architecture():
    model = Inflated3DInceptionV3()
    model.summary(line_length=120, positions=[.4, .65, .75, 1.])
    plot_model(model, 'model_architecture.png', show_shapes=True)


if __name__ == '__main__':
    summarise_architecture()
