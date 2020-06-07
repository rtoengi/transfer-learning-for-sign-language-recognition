# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np

from tensorflow.keras.applications import InceptionV3
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training

_CHANNEL_AXIS = 4


def Inflated3DInceptionV3(input_shape=(20, 224, 224, 3), classes=1000):
    inputs = layers.Input(shape=input_shape)

    x = _conv3d_bn(inputs, 32, 3, 3, 3, strides=(1, 2, 2), padding='valid')
    x = _conv3d_bn(x, 32, 3, 3, 3, padding='valid')
    x = _conv3d_bn(x, 64, 3, 3, 3)
    x = layers.MaxPooling3D((3, 3, 3), strides=(1, 2, 2))(x)

    x = _conv3d_bn(x, 80, 1, 1, 1, padding='valid')
    x = _conv3d_bn(x, 192, 3, 3, 3, padding='valid')
    x = layers.MaxPooling3D((3, 3, 3), strides=(1, 2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = _conv3d_bn(x, 64, 1, 1, 1)

    branch5x5 = _conv3d_bn(x, 48, 1, 1, 1)
    branch5x5 = _conv3d_bn(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = _conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_pool = _conv3d_bn(branch_pool, 32, 1, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=_CHANNEL_AXIS, name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = _conv3d_bn(x, 64, 1, 1, 1)

    branch5x5 = _conv3d_bn(x, 48, 1, 1, 1)
    branch5x5 = _conv3d_bn(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = _conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_pool = _conv3d_bn(branch_pool, 64, 1, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=_CHANNEL_AXIS, name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = _conv3d_bn(x, 64, 1, 1, 1)

    branch5x5 = _conv3d_bn(x, 48, 1, 1, 1)
    branch5x5 = _conv3d_bn(branch5x5, 64, 5, 5, 5)

    branch3x3dbl = _conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)

    branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_pool = _conv3d_bn(branch_pool, 64, 1, 1, 1)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=_CHANNEL_AXIS, name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = _conv3d_bn(x, 384, 3, 3, 3, strides=(1, 2, 2), padding='valid')

    branch3x3dbl = _conv3d_bn(x, 64, 1, 1, 1)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3)
    branch3x3dbl = _conv3d_bn(branch3x3dbl, 96, 3, 3, 3, strides=(1, 2, 2), padding='valid')

    branch_pool = layers.MaxPooling3D((3, 3, 3), strides=(1, 2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=_CHANNEL_AXIS, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = _conv3d_bn(x, 192, 1, 1, 1)

    branch7x7 = _conv3d_bn(x, 128, 1, 1, 1)
    branch7x7 = _conv3d_bn(branch7x7, 128, 1, 1, 7)
    branch7x7 = _conv3d_bn(branch7x7, 192, 1, 7, 1)

    branch7x7dbl = _conv3d_bn(x, 128, 1, 1, 1)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 128, 1, 7, 1)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 128, 1, 1, 7)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 128, 1, 7, 1)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 192, 1, 1, 7)

    branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_pool = _conv3d_bn(branch_pool, 192, 1, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=_CHANNEL_AXIS, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = _conv3d_bn(x, 192, 1, 1, 1)

        branch7x7 = _conv3d_bn(x, 160, 1, 1, 1)
        branch7x7 = _conv3d_bn(branch7x7, 160, 1, 1, 7)
        branch7x7 = _conv3d_bn(branch7x7, 192, 1, 7, 1)

        branch7x7dbl = _conv3d_bn(x, 160, 1, 1, 1)
        branch7x7dbl = _conv3d_bn(branch7x7dbl, 160, 1, 7, 1)
        branch7x7dbl = _conv3d_bn(branch7x7dbl, 160, 1, 1, 7)
        branch7x7dbl = _conv3d_bn(branch7x7dbl, 160, 1, 7, 1)
        branch7x7dbl = _conv3d_bn(branch7x7dbl, 192, 1, 1, 7)

        branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        branch_pool = _conv3d_bn(branch_pool, 192, 1, 1, 1)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=_CHANNEL_AXIS,
                               name=f'mixed{5 + i}')

    # mixed 7: 17 x 17 x 768
    branch1x1 = _conv3d_bn(x, 192, 1, 1, 1)

    branch7x7 = _conv3d_bn(x, 192, 1, 1, 1)
    branch7x7 = _conv3d_bn(branch7x7, 192, 1, 1, 7)
    branch7x7 = _conv3d_bn(branch7x7, 192, 1, 7, 1)

    branch7x7dbl = _conv3d_bn(x, 192, 1, 1, 1)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 192, 1, 7, 1)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 192, 1, 1, 7)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 192, 1, 7, 1)
    branch7x7dbl = _conv3d_bn(branch7x7dbl, 192, 1, 1, 7)

    branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_pool = _conv3d_bn(branch_pool, 192, 1, 1, 1)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=_CHANNEL_AXIS, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = _conv3d_bn(x, 192, 1, 1, 1)
    branch3x3 = _conv3d_bn(branch3x3, 320, 3, 3, 3, strides=(1, 2, 2), padding='valid')

    branch7x7x3 = _conv3d_bn(x, 192, 1, 1, 1)
    branch7x7x3 = _conv3d_bn(branch7x7x3, 192, 1, 1, 7)
    branch7x7x3 = _conv3d_bn(branch7x7x3, 192, 1, 7, 1)
    branch7x7x3 = _conv3d_bn(branch7x7x3, 192, 3, 3, 3, strides=(1, 2, 2), padding='valid')

    branch_pool = layers.MaxPooling3D((3, 3, 3), strides=(1, 2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=_CHANNEL_AXIS, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = _conv3d_bn(x, 320, 1, 1, 1)

        branch3x3 = _conv3d_bn(x, 384, 1, 1, 1)
        branch3x3_1 = _conv3d_bn(branch3x3, 384, 1, 1, 3)
        branch3x3_2 = _conv3d_bn(branch3x3, 384, 1, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=_CHANNEL_AXIS, name=f'mixed9_{i}')

        branch3x3dbl = _conv3d_bn(x, 448, 1, 1, 1)
        branch3x3dbl = _conv3d_bn(branch3x3dbl, 384, 3, 3, 3)
        branch3x3dbl_1 = _conv3d_bn(branch3x3dbl, 384, 1, 1, 3)
        branch3x3dbl_2 = _conv3d_bn(branch3x3dbl, 384, 1, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=_CHANNEL_AXIS)

        branch_pool = layers.AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        branch_pool = _conv3d_bn(branch_pool, 192, 1, 1, 1)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=_CHANNEL_AXIS,
                               name=f'mixed{9 + i}')

    # Classification block
    x = layers.GlobalAveragePooling3D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    model = training.Model(inputs, x, name='inflated_3d_inception_v3')
    return model


def _conv3d_bn(x, filters, num_step, num_row, num_col, padding='same', strides=(1, 1, 1), name=None):
    """Utility function to apply convolution and batch normalization.

    Arguments:
        x: The input tensor.
        filters: The filters in `Conv3D`.
        num_step: The depth of the convolution kernel.
        num_row: The height of the convolution kernel.
        num_col: The width of the convolution kernel.
        padding: The padding mode in `Conv3D`.
        strides: The strides in `Conv3D`.
        name: The name of the ops; will become `name + '_conv'` for the convolution and `name + '_bn'` for the batch
              norm layer.

    Returns:
        The output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = layers.Conv3D(filters, (num_step, num_row, num_col), strides=strides, padding=padding, use_bias=False,
                      name=conv_name)(x)
    x = layers.BatchNormalization(axis=_CHANNEL_AXIS, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)

    return x


def load_inflated_imagenet_weights(target_model):
    source_model = InceptionV3(include_top=True, weights='imagenet')
    for i, source_layer in enumerate(source_model.layers):
        target_layer = target_model.layers[i]
        if 'conv' in source_layer.name:
            _copy_conv_weights(source_layer, target_layer)
        elif 'batch_normalization' in source_layer.name:
            _copy_batch_normalization_weights(source_layer, target_layer)


def _copy_conv_weights(source_layer, target_layer):
    target_shape = target_layer.get_weights()[0].shape
    depth = target_shape[0]
    weights = np.copy(source_layer.get_weights()[0]) / depth
    reps = np.ones(len(target_shape), dtype=int)
    reps[0] = depth
    weights = np.tile(weights, reps)
    target_layer.set_weights([weights])


def _copy_batch_normalization_weights(source_layer, target_layer):
    target_layer.set_weights(source_layer.get_weights().copy())
