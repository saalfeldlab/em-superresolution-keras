from __future__ import print_function
import CNN_models
from keras.models import Model
from keras.layers import AveragePooling3D, Permute
from keras.layers import Input, merge
from keras.optimizers import Adam
from keras import backend as K
import numpy as np


def learn_from_groundtruth_shared(input_shape, my_specs, lr):
    sc = my_specs.sc
    input_shape_ch = tuple(np.array(input_shape)* np.array([sc, 1, 1]))
    if K.image_dim_ordering() == 'tf':
        input_shape_ch = input_shape_ch + (1,)
    else:
        input_shape_ch = (1,) + input_shape_ch
    up_model = CNN_models.sharing_upsaling_unet(my_specs)
    print(up_model['connectivity'])
    print(up_model['layers'])
    tensors = [Input(shape=input_shape_ch),]
    tensors.append(AveragePooling3D(pool_size=(sc, 1, 1), strides=(sc, 1, 1), border_mode='valid')(tensors[-1]))
    for l, c in zip(up_model['layers'], up_model['connectivity']):
        print(l, c)
        if isinstance(c, int):
            tensors.append(l(tensors[c]))
        elif isinstance(c, tuple):
            tensors.append(l(tensors, c))
        else:
            print("A")
        #print(len(tensors))
    model = Model(input=tensors[0], output=tensors[-1])
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    print(model.summary())
    return model


def learn_without_groundtruth_simulated(input_shape, my_specs, lr):
    sc = my_specs.sc
    input_shape_ch = tuple(np.array(input_shape)* np.array([sc, 1, 1]))
    if K.image_dim_ordering() == 'tf':
        input_shape_ch = input_shape_ch + (1,)
    else:
        input_shape_ch = (1,) + input_shape_ch
    up_model = CNN_models.sharing_upsaling_unet(my_specs)
    print(up_model['connectivity'])
    print(up_model['layers'])

    tensors = [Input(shape=input_shape_ch),]
    tensors.append(AveragePooling3D(pool_size=(sc, 1, 1), strides=(sc, 1, 1), border_mode='valid')(tensors[-1]))

    for l, c in zip(up_model['layers'], up_model['connectivity']):
        print(l, c)
        if isinstance(c, int):
            tensors.append(l(tensors[c]))
        elif isinstance(c, tuple):
            tensors.append(l(tensors, c))
        else:
            print("A")
    pred_idx_first = len(tensors)-1

    tensors.append(Permute((1, 3,2,4), name='permute_yzx_forward')(tensors[-1]))
    for l, c in zip(up_model['layers'], up_model['connectivity']):
        print(l, c)
        if isinstance(c, int):
            tensors.append(l(tensors[c]))
        elif isinstance(c, tuple):
            tensors.append(l(tensors, c))
        else:
            print("A")
    tensors.append(Permute((1, 3,2,4), name='permute_yzx_backward')(tensors[-1]))
    pred_idx_1_2 = len(tensors)-1

    tensors.append(Permute((1, 4,3,2), name='permute_xyz_forward')(tensors[pred_idx_first]))
    for l, c in zip(up_model['layers'], up_model['connectivity']):
        print(l, c)
        if isinstance(c, int):
            tensors.append(l(tensors[c]))
        elif isinstance(c, tuple):
            tensors.append(l(tensors, c))
        else:
            print("A")
    tensors.append(Permute((1, 4,3,2), name='permute_xyz_backward')(tensors[-1]))
    pred_idx_1_3 = len(tensors)-1

    model = Model(input=[tensors[0]],
                  output=[tensors[pred_idx_first], tensors[pred_idx_1_2], tensors[pred_idx_1_3]])

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse')
    print(model.summary())
    return model


def learn_from_groundtruth(input_shape, my_specs, lr):
    """set up learning from groundtruth, i.e. downscale artificially and reconstruct the HR"""
    sc = my_specs.sc
    input_shape_ch = tuple(np.array(input_shape) * np.array([sc, 1, 1]))
    if K.image_dim_ordering()=='tf':
        input_shape_ch = input_shape_ch + (1,)
    else:
        input_shape_ch = (1,) + input_shape_ch

    layers = [Input(shape=input_shape_ch)]
    layers.append(AveragePooling3D(pool_size=(sc, 1, 1), strides=(sc, 1, 1), border_mode='valid')(layers[-1]))
    if my_specs.model_type == 'unet' or my_specs.model_type == 'u-net':
        layers = CNN_models.upscaling_unet(my_specs, layers)
    elif my_specs.model_type == 'fsrcnn' or my_specs.model_type == 'sparsecoding':
        layers = CNN_models.sparsecoding(my_specs, layers, input_shape)
    model = Model(input=layers[0], output=layers[-1])
    #optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    print(model.summary())
    return model


#def build_training_scheme(input_shape, cnn_model_specs, sc):
#    def add_downscaling_layer(in_layer, sc=4):
#        return AveragePooling3D(pool_size=(1, 1, sc), strides=(1, 1, sc), border_mode='valid')(in_layer)
#
#    def add_permute_layer(in_layer, axes):
#        return Permute(axes)(in_layer)
#
#    CNN = cnn_model_specs.get_upscaling_unet(cnn_model_specs)
#    orig_data = Input(shape=input_shape)
#
#    sim_data = add_downscaling_layer(orig_data, sc)
#    orig_predicition = CNN(sim_data)
#    sim_from_prediction = add_downscaling_layer(orig_predicition, sc)
#
#    permute_for_y = add_permute_layer(orig_predicition, (2, 3))
#    sim_ydata = add_downscaling_layer(permute_for_y, sc)
#    upscale_ydata = CNN(sim_ydata)
#    backpermute_ydata = add_permute_layer(upscale_ydata, (2, 3))
#
#    permute_for_x = add_permute_layer(orig_predicition, (1, 3))
#    sim_xdata = add_downscaling_layer(permute_for_x, sc)
#    upscale_xdata = CNN(sim_xdata)
#    backpermute_xdata = add_permute_layer(upscale_xdata, (1, 3))
#
#    full_model = Model(input=orig_data, output=[sim_data, orig_predicition, sim_from_prediction, backpermute_ydata,
#                                                backpermute_xdata])
#    return full_model


if __name__ == '__main__':
    mycnnspecs =CNN_models.CNNspecs(model_type='U-Net', n_levels=3, n_convs=3, n_fmaps=dict(start=64, mult=2))
    #learn_from_groundtruth((100, 106, 106), mycnnspecs)
    learn_without_groundtruth_simulated((16, 64, 64), mycnnspecs, 10**(-4))
