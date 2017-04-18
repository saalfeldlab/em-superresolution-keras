from CNN_models import CNNspecs
from training_scheme import learn_from_groundtruth, learn_from_groundtruth_shared, learn_without_groundtruth_simulated
import numpy as np
import h5py
import datetime, time
import os
from keras.callbacks import Callback
import cPickle as pickle
from keras import backend as K
import json
import utils


class LRSchedule(Callback):
    def __init__(self, base_lr, decay=-0.5):
        super(LRSchedule, self).__init__()
        self.base_lr = base_lr
        self.start_epoch = 1
        self.decay = decay

    def set_start_epoch(self, new_start_epoch):
        self.start_epoch = new_start_epoch

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.base_lr * (self.start_epoch+epoch)**self.decay
        K.set_value(self.model.optimizer.lr, new_lr)


class ModelSaver(Callback):
    def __init__(self, save_interval, saving_path, exp_name, start):
        super(ModelSaver, self).__init__()
        self.save_interval=save_interval
        self.saving_path = saving_path
        self.exp_name = exp_name
        self.start = start
        self.runlength = []
        self.training_only = []

    def on_train_begin(self, logs=None):
        self.runlength.append(time.time() - self.start)
        self.training_only.append(time.time() - self.start)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch > 1) and (epoch % self.save_interval == 0):

            self.model.save_weights(self.saving_path + self.exp_name +
                                    'weights{:02d}.h5'.format(epoch/self.save_interval))
            with open(self.saving_path + self.exp_name +
                              'opt_state{:02d}.p'.format(epoch/self.save_interval), 'w') as f:
                pickle.dump(self.model.optimizer.get_weights(), f)
            self.model.save(self.saving_path + self.exp_name + 'model_state{:02d}.h5'.format(epoch/self.save_interval))
            self.runlength.append(time.time() - self.start)


class LossHistory(Callback):
    def __init__(self, save_interval, saving_path, exp_name):
        super(LossHistory, self).__init__()
        self.save_interval = save_interval
        self.saving_path = saving_path
        self.exp_name = exp_name
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1 and epoch % self.save_interval == 0:
            with open(self.saving_path + self.exp_name +
                              'loss_history{:02d}.p'.format(epoch/self.save_interval), 'w') as f:
                pickle.dump(self.losses, f)


def h5_data_generator_same(train_h5path, io_shape=(100,100,100), bs=1, num_outputs=1):
    """data generator for random coordinates from h5 file"""
    train_ds = h5py.File(train_h5path, 'r')['raw']

    ## generate a random crop
    print train_ds.shape

    while True:
        batch = np.empty(io_shape+(bs,))
        z_start = np.random.random_integers(0, train_ds.shape[0]-io_shape[0]-1, bs)
        y_start = np.random.random_integers(0, train_ds.shape[1]-io_shape[1]-1, bs)
        x_start = np.random.random_integers(0, train_ds.shape[2]-io_shape[2]-1, bs)

        for k in range(bs):
            train_ds.read_direct(batch, np.s_[z_start[k]: z_start[k] + io_shape[0],
                                              y_start[k]: y_start[k] + io_shape[1],
                                              x_start[k]: x_start[k] + io_shape[2]],
                                 np.s_[:, :, :, k])
        if K.image_dim_ordering() == 'tf':
            batch = np.swapaxes(np.expand_dims(batch, 0), 0, 4)/255.
            if num_outputs==1:
                gt = batch
            else:
                gt = np.zeros((bs, 1, 1, 1, 1))
        else:
            batch = np.transpose(np.expand_dims(batch, -1), (3, 4, 0, 1, 2))/255.
            if num_outputs==1:
                gt = batch
            else:
                gt = np.zeros((bs, 1, 1, 1, 1))
        yield ([batch], [gt]*num_outputs)


def h5_data_generator_same_cubic(train_h5path, io_shape=(64,64,64), bs=1, sc=4):
    """data generator for random coordinates from h5 file and cubic upsampling"""
    train_ds = h5py.File(train_h5path, 'r')['raw']

    while True:
        batch = np.empty((bs,) + io_shape)
        batch_cubicup = np.empty((bs, )+ io_shape)
        z_start = np.random.random_integers(0, train_ds.shape[0] - io_shape[0] - 1, bs)
        y_start = np.random.random_integers(0, train_ds.shape[1] - io_shape[1] - 1, bs)
        x_start = np.random.random_integers(0, train_ds.shape[2] - io_shape[2] - 1, bs)

        for k in range(bs):
            train_ds.read_direct(batch, np.s_[z_start[k]: z_start[k] + io_shape[0],
                                              y_start[k]: y_start[k] + io_shape[1],
                                              x_start[k]: x_start[k] + io_shape[2]],
                                 np.s_[k, :, :, :])
            sample_downsampled = utils.downscale_manually(batch[k,:,:,:].squeeze()/255., factor=sc, axis=0)
            batch_cubicup[k,:,:,:] = utils.bicubic_up(sample_downsampled, sc, axis=0)

        if K.image_dim_ordering() == 'tf':
            batch = np.expand_dims(batch, -1)/255.
            batch_cubicup = np.expand_dims(batch_cubicup, -1)

        else:
            batch = np.expand_dims(batch, 1)/255.
            batch_cubicup = np.expand_dims(batch_cubicup, 1)
        yield (batch, batch_cubicup)


def h5_data_generator(train_h5path, input_shape=(100, 106,106), output_shape=(4,4,4), bs=10):
    """data generator for getting random coordinates from h5 file but moving in intervals of output_shape (i.e. no
    overlap of output)"""
    train_ds = h5py.File(train_h5path, 'r')['raw']

    ## generate a random crop
    print train_ds.shape
    tds_dims = np.array(train_ds.shape)[:3] // np.array(output_shape)
    print tds_dims
    print tds_dims * 4
    while True:
        batch = np.empty(input_shape+(bs,))
        z_start = np.random.random_integers(0, tds_dims[0]-input_shape[0]//output_shape[0]-1, bs)
        y_start = np.random.random_integers(0, tds_dims[1]-input_shape[1]//output_shape[1]-1, bs)
        x_start = np.random.random_integers(0, tds_dims[2]-input_shape[2]//output_shape[2]-1, bs)
        for k in range(bs):
            train_ds.read_direct(batch, np.s_[z_start[k]*output_shape[0]:z_start[k]*output_shape[0]+input_shape[0],
                                              y_start[k]*output_shape[1]:y_start[k]*output_shape[1]+input_shape[1],
                                              x_start[k]*output_shape[2]:x_start[k]*output_shape[2]+input_shape[2], 0],
                                 np.s_[:, :, :, k])
        if K.image_dim_ordering() == 'tf':
            batch = (np.swapaxes(np.expand_dims(batch, 0), 0, 4)/255.)
            gt = batch[:, (input_shape[0]-output_shape[0])/2: (input_shape[0]-output_shape[0])/2 + output_shape[0],
                       (input_shape[1]-output_shape[1])/2: (input_shape[1]-output_shape[1])/2 + output_shape[1],
                       (input_shape[2]-output_shape[2])/2: (input_shape[2]-output_shape[2])/2 + output_shape[2], :]
        else:
            batch = (np.transpose(np.expand_dims(batch, -1), (3, 4, 0, 1, 2))/255.)
            gt = batch[:, :, (input_shape[0]-output_shape[0])/2: (input_shape[0]-output_shape[0])/2 + output_shape[0],
                       (input_shape[1]-output_shape[1])/2: (input_shape[1]-output_shape[1])/2 + output_shape[1],
                       (input_shape[2]-output_shape[2])/2: (input_shape[2] - output_shape[2])/2 + output_shape[2]]

        yield (batch, gt)


def finetuning_no_gt(exp_name, exp_no, ep_no, lr=10**(-5), arch='Unet', n_l=None, n_c=None, n_f=None, d=None, s=None,
                     m=None, bs=6, epoch_sessions=50, saving_interval=22, batches_per_epoch=22):
    start = time.time()
    train_h5path ='/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-10isozyx/training.h5'
    valid_h5path = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-10isozyx/validation.h5'
    saving_path = '../results_keras/'


    mycnnspecs = CNNspecs(model_type=arch, n_levels=n_l, n_convs=n_c, n_fmaps=dict(start=n_f, mult=2), d=d, s=s, m=m)
    nogt_model = learn_without_groundtruth_simulated((16,64,64), mycnnspecs, lr)
    nogt_model.load_weights(utils.get_model_path(exp_name, exp_no, ep_no), by_name=True)
    if K.image_dim_ordering() == 'tf':
        input_zyx = nogt_model.input_shape[1:-1]
        output_zyx = nogt_model.output_shape[1:-1]
    else:
        input_zyx = nogt_model.input_shape[2:]
        output_zyx = nogt_model.output_shape[2:]
    if exp_no!=0:
        exp_name+='{:04d}/'.format(exp_no)
    exp_name+='/finetuning{:03d}e-4/'.format(ep_no)
    os.mkdir(saving_path+exp_name)
    with open(saving_path+exp_name+'nogt_model_def_json.txt', 'wb') as outfile:
        json.dump(nogt_model.to_json(), outfile)
    mygen_train = h5_data_generator_same(train_h5path, io_shape=input_zyx, bs=bs, num_outputs=len(
        nogt_model.output_names))
    mygen_valid = h5_data_generator_same(valid_h5path, io_shape=input_zyx, bs=bs, num_outputs=len(
        nogt_model.output_names))
    runlength = []
    training_only = []
    runlength.append(time.time()-start)
    training_only.append(time.time()-start)
    scheduler = LRSchedule(K.get_value(nogt_model.optimizer.lr))
    model_saver = ModelSaver(saving_interval, saving_path, exp_name, start)
    history = LossHistory(saving_interval, saving_path, exp_name)

    epoch_history = nogt_model.fit_generator(mygen_train, samples_per_epoch=bs*batches_per_epoch,
                                           nb_epoch=saving_interval*epoch_sessions, nb_worker=1,
                                           validation_data=mygen_valid, nb_val_samples=60,
                                           callbacks=[history, model_saver, scheduler],
                                           max_q_size=bs*2)
    training_only.append(time.time()-runlength[-1]-start)
    print(training_only[-1])
    with open(saving_path+exp_name+'epoch_history.p', 'w') as f:
        pickle.dump(epoch_history.history, f)
    with open(saving_path+exp_name+'run_times.p', 'w') as f:
        pickle.dump((model_saver.runlength, model_saver.training_only), f)


def train(exp_name=None, d=None, s=None, m=None, lr=10 ** (-5), n_l=None, n_f=None, n_c=None, arch='UNet', bs=6,
          epoch_sessions=50, saving_interval=22, batches_per_epoch=22):
    """training routine for an upscaling network"""

    start = time.time()
    train_h5path = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    valid_h5path = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    mycnnspecs = CNNspecs(model_type=arch, n_levels=n_l, n_convs=n_c, n_fmaps=dict(start=n_f,mult=2), d=d, s=s, m=m)
    #gt_model = learn_without_groundtruth_simulated((16, 64, 64), mycnnspecs, lr)
    gt_model = learn_from_groundtruth((16,64,64), mycnnspecs, lr)
    saving_path = '../results_keras/'
    if exp_name == None:
        exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")+'/'
    elif os.path.isdir(saving_path+exp_name):
        count = 1
        exp_name = exp_name+'{:04d}/'
        while os.path.isdir(saving_path+exp_name.format(count)):
            count += 1
        exp_name = exp_name.format(count)
    else:
        exp_name += '/'
    os.mkdir(saving_path+exp_name)

    json_string = gt_model.to_json()
    with open(saving_path+exp_name+'model_def_json.txt', 'wb') as outfile:
        json.dump(json_string, outfile)

    if K.image_dim_ordering() == 'tf':
        input_zyx = gt_model.input_shape[1:-1]
        output_zyx = gt_model.output_shape[1:-1]
    else:
        input_zyx = gt_model.input_shape[2:]
        output_zyx = gt_model.output_shape[2:]

    mygen_train = h5_data_generator_same(train_h5path, io_shape=input_zyx, bs=bs, num_outputs=len(
        gt_model.output_names))
    mygen_valid = h5_data_generator_same(valid_h5path, io_shape=input_zyx, bs=bs, num_outputs=len(gt_model.output_names))

    runlength = []
    training_only = []
    runlength.append(time.time()-start)
    training_only.append(time.time()-start)
    scheduler = LRSchedule(K.get_value(gt_model.optimizer.lr))
    model_saver = ModelSaver(saving_interval, saving_path, exp_name, start)
    history = LossHistory(saving_interval, saving_path, exp_name)

    epoch_history = gt_model.fit_generator(mygen_train, samples_per_epoch=bs*batches_per_epoch,
                                           nb_epoch=saving_interval*epoch_sessions, nb_worker=1,
                                           validation_data=mygen_valid, nb_val_samples=60,
                                           callbacks=[history, model_saver, scheduler],
                                           max_q_size=bs*2)
    training_only.append(time.time()-runlength[-1]-start)
    print(training_only[-1])
    with open(saving_path+exp_name+'epoch_history.p', 'w') as f:
        pickle.dump(epoch_history.history, f)
    with open(saving_path+exp_name+'run_times.p', 'w') as f:
        pickle.dump((model_saver.runlength, model_saver.training_only), f)


def print_model_summary(exp_name=None, d=None, s=None, m=None, lr=10 ** (-5), n_l=None, n_f=None, n_c=None,
                        arch='UNet', **kwargs):
    """instead of training just print the model summary (allows the same inputs as train()) for convenience"""
    mycnnspecs = CNNspecs(model_type=arch, n_levels=n_l, n_convs=n_c, n_fmaps=n_f, d=d, s=s, m=m)
    gt_model = learn_from_groundtruth((16, 64, 64), mycnnspecs, lr)


def unet_training():
    n_l = 3 # 4 3 2
    n_f = 64 # 64 32
    n_c = 2  # 3 2
    lrexp = -4# -4 -5 -6 -7
    #name = 'unet_zyx_nl{0:}_nc{1:}_nf{2:}'.format(n_l, n_c, n_f)
    name='test_shared'
    train(name, n_l=n_l, n_c=n_c, n_f=n_f, lr=10 ** lrexp, arch="UNet", epoch_sessions=50, bs=2)


def fsrcnn_training():
    d = 280 # 240 280
    s = 48  # 48 64
    m = 4  # 2 3 4
    name = 'FSRCNN_d{0:}_s{1:}_m{2:}_49again'.format(d, s, m)
    train(name, d=d, s=s, m=m, lr=10 ** (-5), arch="FSRCNN", epoch_sessions=50, saving_interval=22)


if __name__ == '__main__':
    #finetuning_no_gt('unet_cubic_nl4_nc2_nf64', 1, 49, n_l=4, n_c=2, n_f=64, epoch_sessions=50, bs=3,
    #                 saving_interval=22, batches_per_epoch=22*2, lr=10**(-4))


    #unet_training()
    #train('Unet_best_zyx_cubic', n_l=4, n_c=2, n_f=64, lr=10**(-4), arch="UNet", epoch_sessions=50)
    fsrcnn_training()
    #print_model_summary()
