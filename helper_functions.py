from CNN_models import CNNspecs
import numpy as np
from training_scheme import learn_from_groundtruth
import h5py
import fnmatch, os
from PIL import Image
from keras.models import Sequential
from keras.layers import AveragePooling3D


def zero_padding_effect(input_shape):
    model = learn_from_groundtruth(input_shape, CNNspecs(sc=4), 10**(-5))
    y = model.predict(np.ones((1,)+input_shape), batch_size=1)
    zero_padding_h5 = h5py.File('zero_padding60.h5', 'w-')
    zero_padding_h5.create_dataset('raw', data=y)
    zero_padding_h5.close()


def init_downscale_model(input_size, sc, get_output_shape=False):
    model = Sequential()
    model.add(AveragePooling3D(pool_size=sc, border_mode='valid', input_shape=input_size))
    if get_output_shape:
        return model.get_output_shape_at(0)[1:]
    else:
        return model


def h5_from_tiffs(downsample_factor=(8,2,2), percentage_test=30):
    path = '/nrs/saalfeld/hanslovskyp/CutOn4-15-2013_ImagedOn1-27-2014/aligned/substacks/' \
           '1300-3449/4000x2500+5172+1416/20151031_004930/out/04/render/'
    h5_path = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-8iso/'
    h5_file_train = 'training.h5'
    h5_file_validation = 'validation.h5'
    file_format = '{:04d}.tif'

    file_list = fnmatch.filter(os.listdir(path), '*.tif')
    dim_z = len(file_list)

    example_img = Image.open(path+file_list[0])
    dim_yx = np.array(np.array(example_img).shape)

    dim_x_train = int(((1-percentage_test/100.)*dim_yx[1]//downsample_factor[0])*downsample_factor[0])
    dim_x_test = ((dim_yx[1]-dim_x_train)//downsample_factor[0])*downsample_factor[0]
    dim_x_max = dim_x_train + dim_x_test

    dim_z_max = int((dim_z//downsample_factor[-1])*downsample_factor[-1])

    del example_img

    train_array = np.squeeze(np.zeros(init_downscale_model((1, dim_x_train, dim_yx[0], dim_z_max), downsample_factor,
                                      get_output_shape=True)))
    valid_array = np.squeeze(np.zeros(init_downscale_model((1, dim_x_test, dim_yx[0], dim_z_max), downsample_factor,
                                      get_output_shape=True)))
    train_model = init_downscale_model(input_size=(1, downsample_factor[0], dim_yx[0], dim_x_train),
                                       sc=downsample_factor)
    valid_model = init_downscale_model(input_size=(1, downsample_factor[0], dim_yx[0], dim_x_test),
                                       sc=downsample_factor)

    i = 0
    while i < dim_z_max:
        print(i)
        imarray = np.zeros((downsample_factor[0], dim_yx[0], dim_yx[1]))
        for k in range(downsample_factor[0]):
            imarray[k, :, :] = Image.open(path+file_format.format(i))
            i += 1
        train_array[i/downsample_factor[0]-1, :,:] = \
            np.squeeze(train_model.predict(imarray[np.newaxis, np.newaxis, :, :, : dim_x_train]))
        valid_array[i/downsample_factor[0]-1, :,:] = np.squeeze(valid_model.predict(imarray[np.newaxis,
                                                                                          np.newaxis,:,:,
                                                                                      dim_x_train:dim_x_max]))
    h5f_train = h5py.File(h5_path+h5_file_train, 'w-')
    h5f_train.create_dataset('raw', data=np.squeeze(train_array))
    h5f_train.close()
    h5f_valid = h5py.File(h5_path+h5_file_validation, 'w-')
    h5f_valid.create_dataset('raw', data=np.squeeze(valid_array))
    h5f_valid.close()
    del train_array

if __name__ == '__main__':
    h5_from_tiffs((4, 1, 1))