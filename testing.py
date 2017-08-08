from __future__ import print_function

#from keras.layers import Convolution3D
#from keras_contrib.layers import Deconvolution3D
from keras.layers import Conv3D, Conv3DTranspose

from keras.models import Sequential
import numpy as np
m =Sequential()
init='he_normal'
#m.add(Conv3D(300,(16,16,16), kernel_initializer=init, padding='same', activation='relu', input_shape=(100,100, 100,1)))
m.add(Conv3D(300, (16,16,16), kernel_initializer=init, padding='same', input_shape=(100,100,100, 2)))
m.add(Conv3DTranspose(20, (3,6,8), strides=(2,2,2), padding='same', kernel_initializer=init))
#m.add(Convolution3D(300,16,16,16, init=init, border_mode='same', activation='relu', input_shape=(2,100,100,100)))
#m.add(Deconvolution3D(20,3,6,8, output_shape=(None, 20,100,100,100), subsample=(2,2,2), border_mode='same', init=init))
m.compile(optimizer='adam', loss='mse')
for layer in m.layers:
    weights = layer.get_weights()
    print(type(weights))
    print(type(weights[0]))
    print(len(weights))
    for w in weights:
        print(np.mean(w), np.std(w), w.shape)
        print(np.min(w), np.max(w))




