from base_model.inflated_3d_inception_v3 import Inflated3DInceptionV3


def display_conv_layers():
    """Displays the convolutional layers of the inflated 3D Inception-v3 network."""
    model = Inflated3DInceptionV3()
    for i, layer in enumerate(model.layers):
        if 'conv' not in layer.name:
            continue
        print(i, layer.name, layer.output.shape)


if __name__ == '__main__':
    display_conv_layers()
