from keras.layers import concatenate
from keras.layers import Conv3D, MaxPooling3D, Cropping3D
from keras_contrib.layers import Deconvolution3D
from keras.layers.advanced_activations import PReLU
import keras.initializers
import numpy as np
import warnings
from keras import backend as K


class CNNspecs:
    """configuration of upscaling network and its parameters"""
    def __init__(self, model_type='UNet', n_levels=3, n_convs=3, n_fmaps=dict(start=64, mult=2), kernel_size=(3, 3, 3),
                 pool_size=(2, 2, 2), sc=4, d=280, m=3, s=64):
        self.model_type = model_type.lower()
        if self.model_type == 'unet' or self.model_type == 'u-net':
            self.n_levels, self.n_convs, self.n_fmaps, self.kernel_size, self.pool_size = \
                specify_unet(n_levels=n_levels, n_convs=n_convs, n_fmaps=n_fmaps, kernel_size=kernel_size,
                             pool_size=pool_size)
        elif self.model_type == 'fsrcnn' or self.model_type == 'sparsecoding':
            self.d = d
            self.m = m
            self.s = s
        if sc:
            self.sc = sc


def specify_unet(n_levels, n_convs, n_fmaps, kernel_size, pool_size):
    """converts a basic definition of hyperparameters into the full specification required for upscaling_unet()"""

    if isinstance(n_convs, int):  # same on every level
        n_convs = [n_convs] * n_levels

    if isinstance(n_fmaps, int):  # same on every level, for every conv
        n_fmaps_val = n_fmaps
        n_fmaps = []
        for n_c in n_convs:
            n_fmaps.append([n_fmaps_val]*n_c)
    elif isinstance(n_fmaps, list):
        assert len(n_fmaps) == n_levels
        if isinstance(n_fmaps[0], int): # one value per level -> same for every conv on that level
            for k, (n_f, n_c) in enumerate(zip(n_fmaps, n_convs)):
                n_fmaps[k] = [n_f] * n_c
    elif isinstance(n_fmaps, dict):
        n_f_curr = n_fmaps['start']
        n_f_mult = n_fmaps['mult']
        n_fmaps = []
        for n_c in n_convs:
            n_fmaps.append([n_f_curr]*n_c)
            n_f_curr *= n_f_mult

    if isinstance(kernel_size, tuple):
        kernel_size_val = kernel_size
        kernel_size = []
        for n_c in n_convs:
            kernel_size.append([kernel_size_val]*n_c)
    elif isinstance(kernel_size, list):
        assert len(kernel_size) == n_levels
        if isinstance(kernel_size[0], tuple):
            for k, (ks, n_c) in enumerate(zip(kernel_size, n_convs)):
                kernel_size[k] = [ks] * n_c
                kernel_size = [kernel_size] * n_levels

    if isinstance(pool_size, tuple):
        pool_size = [pool_size] * (n_levels - 1)

    assert len(n_convs) == n_levels
    assert len(n_fmaps) == n_levels
    assert len(kernel_size) == n_levels
    assert len(pool_size) == n_levels - 1

    for l in range(n_levels):
        len(n_fmaps[l]) == n_convs[l]
        len(kernel_size[l]) == n_convs[l]
    print("n_levels", n_levels)
    print("n_convs", n_convs)
    print("n_fmaps", n_fmaps)
    print("kernel_size", kernel_size)
    print("pool_size", pool_size)
    return n_levels, n_convs, n_fmaps, kernel_size, pool_size


def sharing_upsaling_unet(my_specs):
    up_model = dict()
    up_model["layers"] = []
    up_model["connectivity"] = []
    sc = my_specs.sc
    print("SC",sc)
    n_levels = my_specs.n_levels
    print("N_LEVELS", n_levels)
    n_convs = my_specs.n_convs
    n_fmaps = my_specs.n_fmaps
    kernel_size = my_specs.kernel_size
    merge_index = []
    curr_shape = np.array([16, 64, 64])
    for l in range(n_levels):
        for c_c, (n_f, ks) in enumerate(zip(n_fmaps[l], kernel_size[l])):
            up_model['layers'].append(Conv3D(n_f, (ks[0], ks[1]*sc+((ks[1]*sc)%2-1), ks[2]*sc+(ks[2]*sc)%2-1),
                                             kernel_initializer='he_normal', bias_initializer='he_normal',
                                             padding='same', activation='relu',
                                             name='conv_downleg_l{0:}_{1:}'.format(l, c_c)))
            up_model['connectivity'].append(-1)
        if l < n_levels - 1:
            if sc >= 2:
                pool_size = (1, 2, 2)
                up_model['layers'].append(Deconvolution3D(n_f, (ks[0]*sc-1, ks[1]*sc-1, ks[2]*sc-1),
                                    output_shape=(None, n_f,) + tuple(curr_shape*np.array([sc,1,1])),
                                    strides=(sc, 1, 1), kernel_initializer='he_normal', padding='same',
                                    activation='relu', name='transconv_downleg_l{0:}'.format(l)))
                up_model['connectivity'].append(-1)
                last_layer_index = -2
                sc /= 2
            else: # downsampling in all dimensions now
                sc = 1
                pool_size = (2, 2, 2)
                last_layer_index = -1

            #merge_shapes.append(list(layers[-1].get_shape().as_list()[spatial_slice]))
            merge_index.append(len(up_model['layers']))
            up_model['layers'].append(MaxPooling3D(pool_size=pool_size, padding='same', name='maxpool_l{0:}'.format(l)))
            curr_shape = curr_shape/np.array(pool_size)
            up_model['connectivity'].append(last_layer_index)

    def merge_caller(tensors, indices):
        if K.image_data_format() == 'channels_last':
            ch_axis = -1
        else:
            ch_axis = 1
        return concatenate([tensors[indices[0]], tensors[indices[1]]], axis=ch_axis)

    for l in range(n_levels-1)[::-1]:
        curr_shape = curr_shape * np.array([2*sc, 2, 2])
        up_model['layers'].append(Deconvolution3D(n_fmaps[l][0], (kernel_size[l][-1][0], kernel_size[l][-1][1],
                                                  kernel_size[l][-1][2]),
                                                  output_shape=(None, n_fmaps[l][0],) + tuple(curr_shape),
                                                  strides=(2*sc, 2, 2), kernel_initializer='he_normal', padding='same',
                                                  activation='relu', name='transconv_upleg_l{0:}'.format(l)))
        up_model['connectivity'].append(-1)

        up_model['layers'].append(merge_caller)
        up_model['connectivity'].append((-(len(up_model['layers'])-merge_index[l]), -1))

        for c_c, (n_f, ks) in enumerate(zip(n_fmaps[l], kernel_size[l])):
            up_model['layers'].append(Conv3D(n_f, (ks[0], ks[1], ks[2]), kernel_initializer='he_normal',
                                             bias_initializer='he_normal', padding='same', activation='relu',
                                             name='conv_upleg_l{0:}_{1:}'.format(l, c_c)))
            up_model['connectivity'].append(-1)

    up_model['layers'].append(Conv3D(1, (3, 3, 3), kernel_initializer='he_normal', bias_initializer='he_normal',
                                     border_mode='same', name='conv_final'))
    up_model['connectivity'].append(-1)

    return up_model


def upscaling_unet(my_specs, layers):
    """adds layers of a 3D-SRUnet"""

    sc = my_specs.sc
    n_levels = my_specs.n_levels
    n_convs = my_specs.n_convs
    n_fmaps = my_specs.n_fmaps
    kernel_size = my_specs.kernel_size
    merge_shapes = []
    merge_index = []

    if K.image_data_format() == 'channels_last':
        spatial_slice = np.s_[1:-1]
    else:
        spatial_slice = np.s_[2:]

    for l in range(n_levels):   # downstream
        for c_c, (n_f, ks) in enumerate(zip(n_fmaps[l], kernel_size[l])):
            layers.append(Conv3D(n_f, (ks[0], ks[1]*sc+(ks[1]*sc)%2-1, ks[2]*sc+(ks[2]*sc)%2-1),
                                 kernel_initializer='he_normal', bias_initializer='he_normal', padding='same',
                                 activation='relu')(layers[-1]))
        if l < n_levels - 1:  # intermediate upsampling (doesn't happen for bottom)
            if sc >= 2:
                pool_size = (1, 2, 2)  # the later downsampling doesn't happen
                layers.append(
                    Deconvolution3D(n_f, (ks[0]*sc-1, ks[1]*sc-1, ks[2]*sc-1),
                                    output_shape=(None, n_f, ) + tuple(layers[-1].get_shape().as_list()[
                                                                         spatial_slice]*np.array([sc, 1, 1])),
                                    strides=(sc, 1, 1), kernel_initializer='he_normal', padding='same',
                                    activation='relu',
                                    name='transconv_downleg_l{0:}'.format(l))(layers[-1]))

                last_layer_index = -2
                sc /= 2
            else: # downsampling in all dimensions now
                sc = 1
                pool_size = (2, 2, 2)
                last_layer_index = -1
            merge_shapes.append(list(layers[-1].get_shape().as_list()[spatial_slice]))
            merge_index.append(len(layers)-1)
            layers.append(MaxPooling3D(pool_size=pool_size, padding='same',
                                       name='maxpool_l{0:}'.format(l))(layers[last_layer_index]))

    for l in range(n_levels-1)[::-1]:
        layers.append(Deconvolution3D(n_fmaps[l][0], (kernel_size[l][-1][0], kernel_size[l][-1][1], kernel_size[l][
            -1][2]), output_shape=(None, n_fmaps[l][0],) + tuple(layers[-1].get_shape().as_list()[spatial_slice] *
                                                                 np.array([2 * sc, 2, 2])), strides=(2*sc, 2, 2),
                                      kernel_initializer='he_normal', padding='same', activation='relu',
                                      name='transconv_upleg_l{0:}'.format(l))(layers[-1]))
        offset = np.array(merge_shapes[l]) - np.array(layers[-1].get_shape().as_list()[spatial_slice])
        if np.any(offset%2 != 0):
            warnings.warn('For seamless tiling you need to choose a different input shape or kernel size')
        layers.append(Cropping3D(cropping=((offset[0]/2, offset[0]/2), (offset[1]/2, offset[1]/2),
                                           (offset[2]/2, offset[2]/2)),
                                 name='cropping_l{0:}'.format(l))(layers[merge_index[l]]))
        if K.image_data_format() == 'channels_last':
            ch_axis = -1
        else:
            ch_axis = 1
        layers.append(concatenate([layers[-1], layers[-2]], axis=ch_axis, name='merging_l{0:}'.format(l)))

        for c_c, (n_f, ks) in enumerate(zip(n_fmaps[l], kernel_size[l])):
            layers.append(Conv3D(n_f, (ks[0], ks[1], ks[2]), kernel_initializer='he_normal',
                                 bias_initializer='he_normal', padding='same', activation='relu',
                                 name='conv_upleg_l{0:}_{1:}'.format(l, c_c))(layers[-1]))

    layers.append(Conv3D(1, (3, 3, 3), kernel_initializer='he_normal', bias_initializer='he_normal', padding='same',
                         name='conv_final')(layers[-1]))
    return layers


def sparsecoding(my_specs, layers, input_shape=(64, 64, 16)):
    """adds layers of a sparsecoding network (FSRCNN)"""

    if K.image_data_format() == 'channels_last':
        spatial_slice = [1, 2, 3]
    else:
        spatial_slice = [2, 3, 4]

    layers.append(Conv3D(my_specs.d, (5, 13, 13), kernel_initializer='he_normal',
                         bias_initializer='he_normal', padding='same')(layers[-1]))
    layers.append(PReLU(alpha_initializer=keras.initializers.Constant(value=0.05), shared_axes=spatial_slice)
                  (layers[-1]))
    layers.append(Conv3D(my_specs.s, (1, 1, 1), kernel_initializer='he_normal',
                         bias_initializer='he_normal', padding='same')(layers[-1]))
    layers.append(PReLU(alpha_initializer=keras.initializers.Constant(value=0.05), shared_axes=spatial_slice)
                  (layers[-1]))

    for k in range(my_specs.m):
        layers.append(Conv3D(my_specs.s, (3, 9, 9), kernel_initializer='he_normal',
                             bias_initializer='he_normal', padding='same')(layers[-1]))
        layers.append(PReLU(alpha_initializer=keras.initializers.Constant(value=0.05), shared_axes=spatial_slice)
                      (layers[-1]))
    layers.append(Conv3D(my_specs.d, (1, 1, 1), kernel_initializer='he_normal',
                         bias_initializer='he_normal', padding='same')(layers[-1]))
    layers.append(PReLU(alpha_initializer=keras.initializers.Constant(value=0.05), shared_axes=spatial_slice)
                  (layers[-1]))
    spatial_output_shape = tuple(np.array(input_shape)*np.array((my_specs.sc, 1, 1)))
    if K.image_data_format() == 'channels_last':
        output_shape = (None,) + spatial_output_shape + (1,)
    else:
        output_shape = (None, 1,) + spatial_output_shape
    print(spatial_output_shape)
    layers.append(Deconvolution3D(1, (13, 13, 13), output_shape=output_shape, strides=(my_specs.sc, 1, 1),
                                  padding='same', kernel_initializer=keras.initializers.random_normal(stddev=0.001))
                  (layers[-1]))
    return layers


def main():
    #sparse_coding()
    n_levels, n_convs, n_fmaps, kernel_size, pool_size = specify_unet(n_levels=3, n_convs=3,
                                                                      n_fmaps=dict(start=64, mult=2),
                                                                      kernel_size=(3, 3, 3), pool_size=(2, 2, 2))


if __name__ == '__main__':
    main()
