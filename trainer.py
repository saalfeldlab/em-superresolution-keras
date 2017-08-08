from __future__ import print_function
import numpy as np
import h5py
#import cPickle as pickle
import pickle
import time
import utils

from keras.callbacks import Callback
from keras import backend as K


class LossHistory(Callback):
    def __init__(self, exp_path, save_interval):
        super(LossHistory, self).__init__()
        self.save_interval = save_interval
        self.exp_path = exp_path
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_interval == 0:
            with open(self.exp_path + 'loss_history{:02d}.p'.format(epoch//self.save_interval), 'w') as f:
                pickle.dump(self.losses, f)


class RunlengthSaver(Callback):
    def __init__(self, exp_path, save_interval):
        super(RunlengthSaver, self).__init__()
        self.start = 0
        self.runlength = []
        self.exp_path = exp_path
        self.save_interval = save_interval

    def on_train_begin(self, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.runlength.append(time.time() - self.start)
        if epoch % self.save_interval == 0:
            with open(self.exp_path + 'runlength{0:}.p'.format(epoch//self.save_interval), 'w') as f:
                pickle.dump(self.runlength, f)


class ModelSaver(Callback):
    def __init__(self, exp_path, save_interval):
        super(ModelSaver, self).__init__()
        self.save_interval = save_interval
        self.exp_path = exp_path

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_interval == 0:
            self.model.save_weights(self.exp_path + 'weights{:02d}.h5'.format(epoch))
            with open(self.exp_path + 'opt_state{:02d}.p'.format(epoch), 'w') as f:
                pickle.dump(self.model.optimizer.get_weights(), f)
            self.model.save(self.exp_path + 'model_state{:02d}.h5'.format(epoch))


class Trainer(object):
    def __init__(self, model, exp_path, training_source, validation_source, save_interval=10, bs=6,
                 steps_per_epoch=22*22, cubic=False, normalize=True):
        self.model = model
        self.bs = bs
        self.steps_per_epoch = steps_per_epoch
        self.cubic = cubic
        self.normalize = normalize
        if K.image_data_format() == 'channels_last':
            input_shape = self.model.model.input_shape[1:-1]
        else:
            input_shape = self.model.model.input_shape[2:]
        self.valid_generator = self.h5_data_generator(validation_source, input_shape,
                                                      num_outputs=len(self.model.model.output_names))
        self.train_generator = self.h5_data_generator(training_source, input_shape,
                                                      num_outputs=len(self.model.model.output_names))
        self.callbacks = [LossHistory(exp_path, save_interval=save_interval),
                          ModelSaver(exp_path, save_interval=save_interval),
                          RunlengthSaver(exp_path, save_interval=save_interval)]

    def run(self, epochs, start_epoch=0):
        epoch_history = self.model.model.fit_generator(self.train_generator,
                                                       epochs=epochs,
                                                       validation_data=self.valid_generator,
                                                       callbacks=self.callbacks + [self.model.scheduler],
                                                       steps_per_epoch=self.steps_per_epoch,
                                                       validation_steps=20,
                                                       max_queue_size=2*self.bs,
                                                       initial_epoch=start_epoch)#, workers=4, use_multiprocessing=True)

    def h5_data_generator(self, source, input_shape, num_outputs):
        train_ds = h5py.File(source, 'r')['raw']

        while True:
            if self.cubic:
                sample_downsampled = np.empty((self.bs,) + (input_shape[0]//self.model.scaling_factor,) + input_shape[
                                                                                                         1:])
            batch = np.empty((self.bs,)+input_shape)

            if self.cubic:
                batch_cubicup = np.empty((self.bs,) + input_shape)
            z_start = np.random.random_integers(0, train_ds.shape[0] - input_shape[0] - 1, self.bs)
            y_start = np.random.random_integers(0, train_ds.shape[1] - input_shape[1] - 1, self.bs)
            x_start = np.random.random_integers(0, train_ds.shape[2] - input_shape[2] - 1, self.bs)
            for k in range(self.bs):
                train_ds.read_direct(batch,
                                     np.s_[z_start[k]:z_start[k] + input_shape[0],
                                           y_start[k]:y_start[k] + input_shape[1],
                                           x_start[k]:x_start[k] + input_shape[2]],
                                     np.s_[k, :, :, :])
                if self.cubic:
                    sample_downsampled[k, :, :, :] = utils.downscale_manually(batch[k, :, :, :],
                                                                              factor=self.model.scaling_factor,
                                                                              axis=0)
                    batch_cubicup[k, :, :, :] = utils.cubic_up(sample_downsampled[k, :, :, :],
                                                               self.model.scaling_factor, axis=0)

            if K.image_data_format() == 'channels_last':
                ch_axis = -1
            else:
                ch_axis = 1

            if self.cubic:
                gt = np.expand_dims(batch_cubicup, ch_axis)
            batch = np.expand_dims(batch, ch_axis)
            if num_outputs == 1:
                if not self.cubic:
                    gt = batch
            else:
                gt = np.zeros((self.bs, 1, 1, 1, 1))
            if self.normalize:
                batch = (batch/255.).astype('float32')
                gt = (gt/255.).astype('float32')
            yield ([batch], [gt] * num_outputs)
