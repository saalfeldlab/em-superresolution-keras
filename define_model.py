from __future__ import print_function
import numpy as np
import json

from keras.models import Model
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, concatenate, Input, AveragePooling3D, Permute, add, \
    Lambda, BatchNormalization
from keras.layers.advanced_activations import PReLU
import keras.initializers
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K


class LRSchedule(Callback):
    def __init__(self, base_lr, decay=-0.5, start_epoch=0, adapt_interval=1):
        super(LRSchedule, self).__init__()
        self.base_lr = base_lr
        self.start_epoch = 1 + start_epoch
        self.decay = decay
        self.adapt_interval = adapt_interval

    def set_start_epoch(self, new_start_epoch):
        self.start_epoch = new_start_epoch

    def on_epoch_begin(self, epoch, logs=None):
        print(epoch)
        new_lr = self.base_lr * (self.start_epoch+epoch//self.adapt_interval)**self.decay
        K.set_value(self.model.optimizer.lr, new_lr)


def merge_caller(tensors, indices):
    if K.image_data_format() == 'channels_last':
        ch_axis = -1
    else:
        ch_axis = 1
    return concatenate([tensors[indices[0]], tensors[indices[1]]], axis=ch_axis)


def mean_squared(y_true, y_pred):
    return K.mean(K.square(y_pred))


def dummy_loss(y_true, y_pred):
    return K.zeros((1, ))


def gaussian_init(shape, dtype=None):
    return K.random_normal(shape, stddev=5e-5, dtype=dtype)


class IsoNet(object):
    def __init__(self, scaling_factor, input_shape, simulate=True, from_groundtruth=True):
        self.scaling_factor = scaling_factor
        self.input_shape = np.array(input_shape)
        self.simulate = simulate
        self.from_groundtruth = from_groundtruth
        self.model_pre = None
        self.model = None
        self.scheduler = None

    def fullyconv_unet_simple_spec(self, height, width, depth, iso_kernel_size=(3, 3, 3), batchnorm=False):
        model_pre = dict()
        model_pre["layers"] = []
        model_pre["connectivity"] = []
        curr_shape = self.input_shape
        curr_width = width
        remaining_factor = self.scaling_factor
        merge_index = []
        if K.image_data_format() == 'channels_last':
            batchnormaxis = -1
        else:
            batchnormaxis = 1

        for level in range(height):
            for conv_no in range(depth):
                curr_ks = (iso_kernel_size[0],
                           iso_kernel_size[1] * remaining_factor + (iso_kernel_size[1] * remaining_factor) % 2 - 1,
                           iso_kernel_size[2] * remaining_factor + (iso_kernel_size[2] * remaining_factor) % 2 - 1)
                model_pre["layers"].append(Conv3D(curr_width, curr_ks,
                                                  kernel_initializer='he_normal',
                                                  padding='same',
                                                  activation='relu',
                                                  name='conv_downleg_l{0:}_c{1:}'.format(level, conv_no)))
                model_pre["connectivity"].append(-1)
                if batchnorm:
                    model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                    model_pre["connectivity"].append(-1)
            if level < height - 1:  # going to the next level
                if remaining_factor >= 2:
                    pool_size = (1, 2, 2)
                    curr_ks = (iso_kernel_size[0] * remaining_factor - 1,
                               iso_kernel_size[1] * remaining_factor - 1,
                               iso_kernel_size[2] * remaining_factor - 1)
                    # if K.image_data_format() == 'channels_last':
                    #    deconv_output = (None,) + tuple(curr_shape * np.array([remaining_factor, 1,
                    # 1])) + (curr_width,)
                    # else:
                    #    deconv_output = (None, curr_width,) + tuple(curr_shape * np.array([remaining_factor, 1, 1]))
                    model_pre["layers"].append(Conv3DTranspose(curr_width, curr_ks, strides=(remaining_factor, 1, 1),
                                                               padding='same', activation='relu',
                                                               kernel_initializer='he_normal_transposed',
                                                               name='transconv_downloeg_l{0:}'.format(level)))
                    # model_pre["layers"].append(Deconvolution3D(curr_width, curr_ks,
                    #                                           output_shape=deconv_output,
                    #                                           strides=(remaining_factor, 1,1),
                    #                                           kernel_initializer='he_normal',
                    #                                           padding='same',
                    #                                           activation='relu',
                    #                                           name='transconv_downleg_l{0:}'.format(level)))
                    model_pre["connectivity"].append(-1)
                    if batchnorm:
                        model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                        model_pre["connectivity"].append(-1)
                    last_layer_index = -2
                    remaining_factor /= 2
                else:
                    remaining_factor = 1
                    pool_size = (2, 2, 2)
                    last_layer_index = -1

                merge_index.append(len(model_pre["layers"]))
                curr_shape = curr_shape / np.array(pool_size)
                model_pre["connectivity"].append(last_layer_index)
                curr_width *= 2
                model_pre["layers"].append(Conv3D(curr_width, curr_ks, kernel_initializer='he_normal',
                                                  padding='same', activation='relu', strides=pool_size,
                                                  name='strided_conv_l{0:}'.format(
                        level)))
                if batchnorm:
                    model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                    model_pre["connectivity"].append(-1)

        for level in range(height - 1)[::-1]:
            curr_shape = curr_shape * np.array([2 * remaining_factor, 2, 2])
            curr_width /= 2
            # if K.image_data_format() == 'channels_last':
            #    deconv_output = (None, ) + tuple(curr_shape) + (curr_width,)
            # else:
            #    deconv_output = (None, curr_width, ) + tuple(curr_shape)
            model_pre["layers"].append(Conv3DTranspose(curr_width, iso_kernel_size,
                                                       strides=(2 * remaining_factor, 2, 2),
                                                       kernel_initializer='he_normal_transposed',
                                                       padding='same',
                                                       activation='relu',
                                                       name='transconv_upleg_l{0:}'.format(level)))

            # model_pre["layers"].append(Deconvolution3D(curr_width, iso_kernel_size,
            #                                           output_shape=deconv_output,
            #                                           strides=(2*remaining_factor, 2, 2),
            #                                           kernel_initializer='he_normal',
            #                                           padding='same',
            #                                           activation='relu',
            #                                           name='transconv_upleg_l{0:}'.format(level)))
            model_pre["connectivity"].append(-1)
            if batchnorm:
                model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                model_pre["connectivity"].append(-1)
            model_pre["layers"].append(merge_caller)
            model_pre["connectivity"].append((-(len(model_pre["layers"]) - merge_index[level]), -1))

            for conv_no in range(depth):
                model_pre["layers"].append(Conv3D(curr_width, iso_kernel_size,
                                                  kernel_initializer='he_normal',
                                                  padding='same', activation='relu',
                                                  name='conv_upleg_l{0:}_c{1:}'.format(level, conv_no)))
                model_pre["connectivity"].append(-1)
                if batchnorm:
                    model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                    model_pre["connectivity"].append(-1)

        model_pre["layers"].append(Conv3D(1, iso_kernel_size,
                                          kernel_initializer='he_normal', padding='same', activation='relu',
                                          name='conv_final'))
        model_pre["connectivity"].append(-1)
        self.model_pre = model_pre

    def unet_simple_spec(self, height, width, depth, iso_kernel_size=(3, 3, 3), batchnorm=False):
        #todo change to include scaling axis
        model_pre = dict()
        model_pre["layers"] = []
        model_pre["connectivity"] = []
        curr_shape = self.input_shape
        curr_width = width
        remaining_factor = self.scaling_factor
        merge_index = []
        if K.image_data_format() == 'channels_last':
            batchnormaxis = -1
        else:
            batchnormaxis = 1

        for level in range(height):
            for conv_no in range(depth):
                curr_ks = (iso_kernel_size[0],
                           iso_kernel_size[1] * remaining_factor + (iso_kernel_size[1] * remaining_factor) % 2 - 1,
                           iso_kernel_size[2] * remaining_factor + (iso_kernel_size[2] * remaining_factor) % 2 - 1)
                model_pre["layers"].append(Conv3D(curr_width, curr_ks,
                                                  kernel_initializer='he_normal',
                                                  padding='same',
                                                  activation='relu',
                                                  name='conv_downleg_l{0:}_c{1:}'.format(level, conv_no)))
                model_pre["connectivity"].append(-1)
                if batchnorm:
                    model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                    model_pre["connectivity"].append(-1)
            if level < height - 1:  # going to the next level
                if remaining_factor >= 2:
                    pool_size = (1, 2, 2)
                    curr_ks = (iso_kernel_size[0] * remaining_factor - 1,
                               iso_kernel_size[1] * remaining_factor - 1,
                               iso_kernel_size[2] * remaining_factor - 1)
                    #if K.image_data_format() == 'channels_last':
                    #    deconv_output = (None,) + tuple(curr_shape * np.array([remaining_factor, 1,
                    # 1])) + (curr_width,)
                    #else:
                    #    deconv_output = (None, curr_width,) + tuple(curr_shape * np.array([remaining_factor, 1, 1]))
                    model_pre["layers"].append(Conv3DTranspose(curr_width, curr_ks, strides=(remaining_factor, 1, 1),
                                                               padding='same', activation='relu',
                                                               kernel_initializer='he_normal_transposed',
                                                               name='transconv_downloeg_l{0:}'.format(level)))
                    #model_pre["layers"].append(Deconvolution3D(curr_width, curr_ks,
                    #                                           output_shape=deconv_output,
                    #                                           strides=(remaining_factor, 1,1),
                    #                                           kernel_initializer='he_normal',
                    #                                           padding='same',
                    #                                           activation='relu',
                    #                                           name='transconv_downleg_l{0:}'.format(level)))
                    model_pre["connectivity"].append(-1)
                    if batchnorm:
                        model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                        model_pre["connectivity"].append(-1)
                        last_layer_index = -3
                    else:
                        last_layer_index = -2
                    remaining_factor /= 2
                else:
                    remaining_factor = 1
                    pool_size = (2, 2, 2)
                    last_layer_index = -1

                merge_index.append(len(model_pre["layers"]))
                model_pre["layers"].append(MaxPooling3D(pool_size=pool_size, padding='same',
                                                        name='maxpool_l{0:}'.format(level)))
                curr_shape = curr_shape/np.array(pool_size)
                model_pre["connectivity"].append(last_layer_index)
                curr_width *= 2
        for level in range(height-1)[::-1]:
            curr_shape = curr_shape * np.array([2*remaining_factor, 2, 2])
            curr_width /= 2
            #if K.image_data_format() == 'channels_last':
            #    deconv_output = (None, ) + tuple(curr_shape) + (curr_width,)
            #else:
            #    deconv_output = (None, curr_width, ) + tuple(curr_shape)
            model_pre["layers"].append(Conv3DTranspose(curr_width, iso_kernel_size,
                                                       strides=(2 * remaining_factor, 2, 2),
                                                       kernel_initializer='he_normal_transposed',
                                                       padding='same',
                                                       activation='relu',
                                                       name='transconv_upleg_l{0:}'.format(level)))
            #model_pre["layers"].append(Deconvolution3D(curr_width, iso_kernel_size,
            #                                           output_shape=deconv_output,
            #                                           strides=(2*remaining_factor, 2, 2),
            #                                           kernel_initializer='he_normal',
            #                                           padding='same',
            #                                           activation='relu',
            #                                           name='transconv_upleg_l{0:}'.format(level)))
            model_pre["connectivity"].append(-1)
            if batchnorm:
                model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                model_pre["connectivity"].append(-1)
            model_pre["layers"].append(merge_caller)
            model_pre["connectivity"].append((-(len(model_pre["layers"]) - merge_index[level]), -1))

            for conv_no in range(depth):
                model_pre["layers"].append(Conv3D(curr_width, iso_kernel_size,
                                                  kernel_initializer='he_normal',
                                                  padding='same', activation='relu',
                                                  name='conv_upleg_l{0:}_c{1:}'.format(level, conv_no)))
                model_pre["connectivity"].append(-1)
                if batchnorm:
                    model_pre["layers"].append(BatchNormalization(axis=batchnormaxis))
                    model_pre["connectivity"].append(-1)

        model_pre["layers"].append(Conv3D(1, iso_kernel_size,
                                          kernel_initializer='he_normal', padding='same', activation='relu',
                                          name='conv_final'))
        model_pre["connectivity"].append(-1)
        self.model_pre = model_pre

    def fsrcnn_spec(self, d, s, m):
        if K.image_data_format() == 'channels_last':
            spatial_slice = [1, 2, 3]
        else:
            spatial_slice = [2, 3, 4]

        model_pre = dict()
        model_pre["layers"] = []
        model_pre["connectivity"] = []
        model_pre["layers"].append(Conv3D(d, (5, 13, 13),
                                          kernel_initializer='he_normal',
                                          padding='same',
                                          activation='linear'))
        model_pre["connectivity"].append(-1)
        model_pre["layers"].append(PReLU(alpha_initializer='zeros',
                                         shared_axes=spatial_slice))
        model_pre["connectivity"].append(-1)
        model_pre["layers"].append(Conv3D(s, (1, 1, 1),
                                          kernel_initializer='he_normal',
                                          padding='same',
                                          activation='linear'))
        model_pre["connectivity"].append(-1)
        model_pre["layers"].append(PReLU(alpha_initializer='zeros',
                                         shared_axes=spatial_slice))
        model_pre["connectivity"].append(-1)

        for mapping in range(m):
            model_pre["layers"].append(Conv3D(s, (3, 9, 9),
                                              kernel_initializer='he_normal',
                                              padding='same',
                                              activation='linear'))
            model_pre["connectivity"].append(-1)
            model_pre["layers"].append(PReLU(alpha_initializer='zeros',
                                             shared_axes=spatial_slice))
            model_pre["connectivity"].append(-1)

        model_pre["layers"].append(Conv3D(d, (1, 1, 1),
                                          kernel_initializer='he_normal',
                                          padding='same',
                                          activation='linear'))
        model_pre["connectivity"].append(-1)
        model_pre["layers"].append(PReLU(alpha_initializer='zeros',
                                         shared_axes=spatial_slice))
        model_pre["connectivity"].append(-1)
        #deconv_output = self.input_shape*np.array([self.scaling_factor, 1, 1])
        #if K.image_data_format() == 'channels_last':
        #    deconv_output = (None, ) + tuple(deconv_output) + (1,)
        #else:
        #    deconv_output = (None, 1,) + tuple(deconv_output)

        model_pre["layers"].append(Conv3DTranspose(1, (13, 13, 13),
                                                   strides=(self.scaling_factor, 1, 1),
                                                   padding='same',
                                                   kernel_initializer='he_normal_transposed',#gaussian_init,
                                                   # ''#'he_normal',
                                                   activation='linear'))
#        model_pre["layers"].append(Deconvolution3D(1, (13, 13, 13),
#                                                   output_shape=deconv_output,
#                                                   strides=(self.scaling_factor, 1, 1),
#                                                   padding='same',
#                                                   kernel_initializer=keras.initializers.random_normal(
#                                                       stddev=np.float32(5e-05)),
#                                                   activation='linear'))

        model_pre["connectivity"].append(-1)
        self.model_pre = model_pre

    def training_scheme(self):
        if self.model_pre is None:
            raise ValueError("model needs to be specified first")

        if self.simulate:
            actual_input_shape = self.input_shape * np.array([self.scaling_factor, 1, 1])
        else:
            actual_input_shape = self.input_shape

        if K.image_data_format() == 'channels_last':
            actual_input_shape = tuple(actual_input_shape) + (1,)
        else:
            actual_input_shape = (1,) + tuple(actual_input_shape)

        tensors = [Input(shape=actual_input_shape), ]
        if self.simulate:
            x = AveragePooling3D(pool_size=(self.scaling_factor, 1, 1),
                                            strides=(self.scaling_factor, 1, 1),
                                            padding='valid')
            tensors.append(x(tensors[-1]))

        inputavg_idx_1 = len(tensors) - 1

        for layer, connectivity in zip(self.model_pre["layers"], self.model_pre["connectivity"]):
            if isinstance(connectivity, int):
                tensors.append(layer(tensors[connectivity]))
            elif isinstance(connectivity, tuple):
                tensors.append(layer(tensors, connectivity))
            else:
                raise TypeError("type of connectivity entry should be int or tuple")

        if self.from_groundtruth:
            self.model = Model(inputs=[tensors[0]], outputs=[tensors[-1]])
        else:

            pred_idx_1 = len(tensors) - 1
            tensors.append(AveragePooling3D(pool_size=(self.scaling_factor, 1, 1),
                                            strides=(self.scaling_factor, 1, 1),
                                            padding='valid')(tensors[-1]))
            predavg_idx_1 = len(tensors) - 1

            # first rotation zyx -> yzx
            tensors.append(Permute((1, 3, 2, 4), name='premute_yzx_forward')(tensors[pred_idx_1]))
            tensors.append(AveragePooling3D(pool_size=(self.scaling_factor, 1, 1),
                                            strides=(self.scaling_factor, 1, 1),
                                            padding='valid')(tensors[-1]))
            inputavg_idx_1_2 = len(tensors) - 1
            for layer, connectivity in zip(self.model_pre["layers"], self.model_pre["connectivity"]):
                if isinstance(connectivity, int):
                    tensors.append(layer(tensors[connectivity]))
                elif isinstance(connectivity, tuple):
                    tensors.append(layer(tensors, connectivity))
                else:
                    raise TypeError("type of connectivity entry should be int or tuple")
            tensors.append(AveragePooling3D(pool_size=(self.scaling_factor, 1, 1),
                                            strides=(self.scaling_factor, 1, 1),
                                            padding='valid')(tensors[-1]))
            predavg_idx_1_2 = len(tensors) - 1
            tensors.append(Permute((1, 3, 2, 4), name='permute_yzx_backward')(tensors[-2]))
            pred_idx_1_2 = len(tensors) - 1

            #second rotation zyx -> xyz
            tensors.append(Permute((1, 4, 3, 2), name='permute_xyz_forward')(tensors[pred_idx_1]))
            tensors.append(AveragePooling3D(pool_size=(self.scaling_factor, 1, 1),
                                            strides=(self.scaling_factor, 1, 1),
                                            padding='valid')(tensors[-1]))
            inputavg_idx_1_3 = len(tensors) - 1
            for layer, connectivity in zip(self.model_pre["layers"], self.model_pre["connectivity"]):
                if isinstance(connectivity, int):
                    tensors.append(layer(tensors[connectivity]))
                elif isinstance(connectivity, tuple):
                    tensors.append(layer(tensors, connectivity))
                else:
                    raise TypeError("type of connectivity entry should be int or tuple")
            tensors.append(AveragePooling3D(pool_size=(self.scaling_factor, 1, 1),
                                            strides=(self.scaling_factor, 1, 1),
                                            padding='valid')(tensors[-1]))
            predavg_idx_1_3 = len(tensors) - 1
            tensors.append(Permute((1, 4, 3, 2), name='permute_xyz_backward')(tensors[-2]))
            pred_idx_1_3 = len(tensors) - 1

            # prepare loss
            tensors.append(Lambda(lambda x: -x)(tensors[pred_idx_1_2]))
            tensors.append(add([tensors[-1], tensors[pred_idx_1]], name='diff_1_2'))
            diff_idx_1_2 = len(tensors) - 1

            tensors.append(Lambda(lambda x: -x)(tensors[pred_idx_1_3]))
            tensors.append(add([tensors[-1], tensors[pred_idx_1]], name='diff_1_3'))
            diff_idx_1_3 = len(tensors) - 1

            tensors.append(Lambda(lambda x: -x)(tensors[inputavg_idx_1]))
            tensors.append(add([tensors[-1], tensors[predavg_idx_1]], name='diff_avg_1'))
            diff_idx_avg_1 = len(tensors) - 1

            tensors.append(Lambda(lambda x: -x)(tensors[inputavg_idx_1_2]))
            tensors.append(add([tensors[-1], tensors[predavg_idx_1_2]], name='diff_avg_1_2'))
            diff_idx_avg_1_2 = len(tensors) - 1

            tensors.append(Lambda(lambda x: -x)(tensors[inputavg_idx_1_3]))
            tensors.append(add([tensors[-1], tensors[predavg_idx_1_3]], name='diff_avg_1_3'))
            diff_idx_avg_1_3 = len(tensors) - 1

            outputs = [tensors[diff_idx_1_2], tensors[diff_idx_1_3], tensors[diff_idx_avg_1],
                       tensors[diff_idx_avg_1_2], tensors[diff_idx_avg_1_3]]

            if self.simulate:
                tensors.append(Lambda(lambda x: -x)(tensors[0]))
                tensors.append(add([tensors[-1], tensors[pred_idx_1]], mode='sum', name='diff_gt'))
                diff_idx_gt = len(tensors)-1
                outputs.append(tensors[diff_idx_gt])

            self.model = Model(inputs=[tensors[0]], output=outputs)

    def load_weights(self, weightfile):
        self.model.load_weights(weightfile, by_name=True)

    def compile(self, lr, adapt_interval):
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        if self.from_groundtruth:
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            losses = {'diff_1_2': mean_squared,
                      'diff_1_3': mean_squared,
                      'diff_avg_1': mean_squared,
                      'diff_avg_1_2': mean_squared,
                      'diff_avg_1_3': mean_squared}

            if self.simulate:
                losses['diff_gt'] = dummy_loss
                metrics = {'diff_gt': mean_squared}
            else:
                metrics = None

            loss_weights = {'diff_1_2': 0.5 / 2.,
                            'diff_1_3': 0.5 / 2.,
                            'diff_avg_1': 0.5 / 2.,
                            'diff_avg_1_2': 0.5 / 4.,
                            'diff_avg_1_3': 0.5 / 4.},

            self.model.compile(optimizer=optimizer,
                               loss=losses,
                               loss_weights=loss_weights,
                               metrics=metrics)

        self.scheduler = LRSchedule(lr, adapt_interval=adapt_interval)
        print(self.model.summary())

    def save_json(self, path_to_file):
        json_string = self.model.to_json()
        with open(path_to_file, 'wb') as f:
            json.dump(json_string, f)
