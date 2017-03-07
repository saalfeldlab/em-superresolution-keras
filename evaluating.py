from __future__ import print_function
import os
import json
import h5py
import itertools
import numpy as np
import time
from CNN_models import gaussian_init
from keras.models import load_model, model_from_json
from keras_contrib.layers import Deconvolution3D
from keras import backend as K
from CNN_models import CNNspecs
import utils

if K.image_dim_ordering() == 'tf':
    spatial_slice = np.s_[1:-1]
else:
    spatial_slice = np.s_[2:]


class Evaluator:

    def __init__(self, model_path, save_path,
                 data_path='/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16iso/validation_and_test.h5',
                 sc=4.):
        self.data_path = data_path
        self.model_path = model_path
        self.save_path = save_path
        self.data = h5py.File(data_path, 'r')['raw']
        self.model = None
        self.output_file = None
        self.sc = sc

    def get_patch(self, coord):
        """routine to get the patch belonging to a single coordinate"""
        inp = self.model.input_shape[spatial_slice]
        patch = np.empty(inp+(1,))

        # 1 added in second coordinate to get the right length in case of odd inp. Doesn't hurt for even
        # shorter than int(np.ceil())
        coord_slice = np.s_[coord[0]-inp[0]/2:coord[0]+(inp[0]+1)/2,
                            coord[1]-inp[1]/2:coord[1]+(inp[1]+1)/2,
                            coord[2]-inp[2]/2:coord[2]+(inp[2]+1)/2, 0]

        self.data.read_direct(patch, coord_slice, np.s_[:, :, :, 0])
        if K.image_dim_ordering() == 'tf':
            patch = np.swapaxes(np.expand_dims(patch, 0), 0, 4)/255.
        else:
            patch = np.transpose(np.expand_dims(patch, -1), (3, 4, 0, 1, 2))/255.
        return patch

    def test_data_generator(self, ignore_border, bs, safety_margin=(0, 0)):
        """generates data as required by keras"""
        o_shape = self.model.output_shape[spatial_slice]
        border_remainder = (np.array(self.data.shape[:-1])-np.array([0, 0, safety_margin[0] + safety_margin[1]]) -
                            2*np.array(ignore_border)) % (np.array(o_shape) - 2*np.array(ignore_border))
        x_extra = [] if border_remainder[0] == 0 else [int(self.data.shape[0] - o_shape[0]/2)]
        y_extra = [] if border_remainder[1] == 0 else [int(self.data.shape[1] - o_shape[1]/2)]
        if border_remainder[2] == 0:
            z_extra = []
        else:
            z_extra = self.data.shape[2] - o_shape[2]/2 - safety_margin[1]
            z_last = range(safety_margin[0] + o_shape[2]/2, self.data.shape[2] - o_shape[2]/2 - safety_margin[1],
                           o_shape[2] - 2*ignore_border[2])[-1]
            overlap_to_last = o_shape[2] - 2*ignore_border[2] - (z_extra-z_last)
            z_extra = [int(z_extra - (self.sc - overlap_to_last%self.sc))]

        batch = np.zeros(self.model.input_shape[spatial_slice]+(bs,))
        k = 0
        l = 0
        for x in range(o_shape[0]/2, self.data.shape[0] - o_shape[0]/2, o_shape[0] - 2*ignore_border[0]) + x_extra:
            print('x', x)
            for y in range(o_shape[1]/2, self.data.shape[1] - o_shape[1]/2, o_shape[1] - 2*ignore_border[1]) + y_extra:
                print('y', y)
                for z in range(safety_margin[0] + o_shape[2]/2, self.data.shape[2] - o_shape[2]/2 - safety_margin[1],
                               o_shape[2] - 2*ignore_border[2]) + z_extra:
                    print('.', z, end='')

                    batch[:, :, :, k] = np.squeeze(self.get_patch((x, y, z)))
                    if k == bs-1:
                        if K.image_dim_ordering() == 'tf':
                            batch = np.swapaxes(np.expand_dims(batch, 0), 0, 4)
                        else:
                            batch = np.transpose(np.expand_dims(batch, -1), (3, 4, 0, 1, 2))
                        l += 1
                        yield batch
                        k = 0
                        batch = np.zeros(self.model.input_shape[spatial_slice]+(bs,))
                    else:
                        k += 1
                print('\n')

        if K.image_dim_ordering() == 'tf':
            batch = np.swapaxes(np.expand_dims(batch, 0), 0, 4)
        else:
            batch = np.transpose(np.expand_dims(batch, -1), (3, 4, 0, 1, 2))
        l += 1
        print(l)

        yield batch
        while True:
            l += 1
            print('justzeros',l)
            yield np.zeros((bs,1,)+self.model.input_shape[spatial_slice])

    def parallel_coordinates_generator(self, ignore_border, safety_margin=(0,0)):
        """generates the coordinates in the same order as test_data_generator (necessary because keras doesn't allow
        for additional outputs of the generator)"""
        # with enumerate
        o_shape = self.model.output_shape[spatial_slice]
        border_remainder = (np.array(self.data.shape[:-1])-np.array([0, 0, safety_margin[0]+safety_margin[1]]) -
                            2*np.array(ignore_border)) % (np.array(o_shape)-2*np.array(ignore_border))
        x_extra = [] if border_remainder[0] == 0 else [self.data.shape[0] - o_shape[0] / 2]
        y_extra = [] if border_remainder[1] == 0 else [self.data.shape[1] - o_shape[1] / 2]
        if border_remainder[2] == 0:
            z_extra = []
        else:
            z_extra = self.data.shape[2]-o_shape[2]/2 - safety_margin[1]
            z_last = range(safety_margin[0] + o_shape[2]/2, self.data.shape[2] - o_shape[2] / 2 - safety_margin[1],
                           o_shape[2] - 2*ignore_border[2])[-1]

            overlap_to_last = o_shape[2] - 2 * ignore_border[2]-(z_extra-z_last)
            z_extra = [z_extra-(4-overlap_to_last%4)]
        print(x_extra, y_extra, z_extra)
        time.sleep(5)
        k = 1
        for x in range(o_shape[0]/2, self.data.shape[0] - o_shape[0]/2, o_shape[0] - 2*ignore_border[0]) + x_extra:
            for y in range(o_shape[1]/2, self.data.shape[1] - o_shape[1]/2, o_shape[1] - 2*ignore_border[1]) + y_extra:
                for z in range(safety_margin[0] + o_shape[2]/2, self.data.shape[2] - o_shape[2]/2 - safety_margin[1],
                               o_shape[2] - 2*ignore_border[2]) + z_extra:
                    yield k, (x, y, z)
                    k += 1

    def load_model(self, json_file=None):
        """model specification is read from json file"""
        if json_file is None:
            exp_path = os.path.dirname(self.model_path)
            json_file = exp_path+'/model_def_json.txt'

        model_def_file = open(json_file, 'r')
        self.model = model_from_json(json.load(model_def_file), custom_objects={'gaussian_init': gaussian_init})
        model_def_file.close()
        self.model.load_weights(self.model_path, by_name=True)
        #self.model = load_model(self.model_path, custom_objects={'gaussian_init':gaussian_init })
        return self.model

    def reset_save_path(self, save_path):
        """resets the save path in case one instant should run several evaluations"""
        self.save_path = save_path

    def prepare_output_file(self, shape, dset_name='raw'):
        """make h5 file and dataset"""
        self.output_file = h5py.File(self.save_path, 'w-')
        self.output_file.create_dataset(dset_name, data=np.zeros(shape))


    def single_evaluation(self, coord):
        """generate prediction for a single patch"""
        self.load_model()
        self.prepare_output_file(self.model.output_shape[spatial_slice])

        patch = self.get_patch(coord)
        prediction = np.squeeze(self.model.predict(patch,1))
        self.output_file.create_dataset('original', data=np.squeeze(patch))
        self.output_file['raw'].write_direct(prediction, np.s_[:, :, :], np.s_[:, :, :])
        self.output_file.close()

    def multiple_evaluations(self, coords):
        """generate predictions for patches belonging to a list of coordinates (not implemented)"""
        pass

    def run_full_evaluation(self, inner_cube, bs, safety_margin=(0,0)):
        """generate prediction for a whole dataset"""
        self.load_model()
        ignore_border = (np.array(self.model.input_shape[spatial_slice]) - np.array(inner_cube) )/ 2.
        if not np.all([ib.is_integer() for ib in ignore_border]):
            raise ValueError('The shape of inner_cube is not valid')
        else:
            ignore_border = tuple([int(ib) for ib in ignore_border])
        print(ignore_border)
        #self.prepare_output_file(self.data.shape[:-1])
        predict = np.zeros(self.data.shape[:-1])
        self.output_file = h5py.File(self.save_path, 'w-')
        batch_generator = self.test_data_generator(ignore_border, bs, safety_margin=safety_margin)
        corresponding_coords_generator, counter = itertools.tee(self.parallel_coordinates_generator(ignore_border,
                                                                                                    safety_margin))
        #get length
        for num_coords_to_process, _ in counter:
            pass
        chunk_size = bs
        processed_examples = 0
        print(num_coords_to_process%chunk_size)
        #num_coords_to_process = (num_coords_to_process/chunk_size)*chunk_size
        print(num_coords_to_process)
        print(chunk_size)
        print(bs)
        chunk_of_predictions = self.model.predict_generator(batch_generator, val_samples=num_coords_to_process)

        for (processed_examples, coord), pred in zip(corresponding_coords_generator, chunk_of_predictions):
            pred = pred[0, ignore_border[0]: -ignore_border[0], ignore_border[1]: -ignore_border[1],
                        ignore_border[2]: -ignore_border[2]]
            predict[coord[0] - pred.shape[0]/2: coord[0] + (pred.shape[0] + 1)/2,
                    coord[1] - pred.shape[1]/2: coord[1] + (pred.shape[1] + 1)/2,
                    coord[2] - pred.shape[2]/2: coord[2] + (pred.shape[2] + 1)/2] = pred

        self.output_file.create_dataset('raw', data=predict)
        self.output_file.close()


def shifted_evaluation(exp_name, run, cp):
    for mode in ['validation', 'test']:
        modelp = utils.get_model_path(exp_name, exp_no=run, ep_no=cp)
        shifted_evaluator = Evaluator(modelp, '', utils.get_data_path(mode))
        for shift in range(int(shifted_evaluator.sc)):
            savep = utils.get_save_path(exp_name, exp_no=run, ep_no=cp, mode=mode, add='_shift'+str(shift))
            shifted_evaluator.reset_save_path(savep)
            shifted_evaluator.run_full_evaluation(inner_cube=(48,48,24), bs=6, safety_margin=(shift, -shift))

def Unet_evaluation(ep_no=49):
    n_l = 4
    n_f = 64
    n_c = 3
    run = 0
    for mode in ['validation', 'test']:
        modelp = utils.get_model_path('Unet_nl{0:}_nc{1:}_nf{2:}'.format(n_l, n_c, n_f), exp_no=run, ep_no=ep_no)
        savep = utils.get_save_path('Unet_nl{0:}_nc{1:}_nf{2:}'.format(n_l, n_c, n_f), exp_no=run, ep_no=ep_no,
                                   mode=mode)
        simple_evaluator = Evaluator(modelp, savep, utils.get_data_path(mode))
        simple_evaluator.run_full_evaluation(inner_cube=(48, 48, 24), bs=6)


def single_fsrcnn_evaluation(ep_no=49, exp_no=2, d=240, s=64, m=3):
    for mode in ['validation', 'test']:
        modelp = utils.get_model_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(d, s, m), exp_no=exp_no, ep_no=ep_no)
        savep = utils.get_save_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(d, s, m), exp_no=exp_no, ep_no=ep_no, mode=mode)

        simple_evaluator = Evaluator(modelp, savep, utils.get_data_path(mode))
        simple_evaluator.run_full_evaluation(inner_cube=(48, 48, 24), bs=6)


def run_single_evaluation(run, ep_no, exp_name=None, d=None, s=None, m=None, lr=10 ** (-5), n_l=None, n_f=None,
                          n_c=None, arch='UNet', bs=6, **kwargs):
    mycnnspecs = CNNspecs(model_type=arch, n_levels=n_l, n_convs=n_c, n_fmaps=n_f, d=d, s=s, m=m)
    for mode in ['validation', 'test']:
        modelp = utils.get_model_path(exp_name, exp_no=run, ep_no=ep_no)
        savep = utils.get_save_path(exp_name, exp_no=run, ep_no=ep_no, mode=mode)
        simple_evaluator = Evaluator(modelp, savep, utils.get_data_path(mode))
        simple_evaluator.run_full_evaluation(inner_cube=(48, 48, 24), bs=bs)


def fsrcnn_hyperparameter_evaluation(ep_no=12):
    for d in [240, 280]:
        for s in [48, 64]:
            for m in [2,3,4]:
                for run in range(2):
                    run_single_evaluation(run, ep_no, exp_name='FSRCNN_d{0:}_s{1:}_m{2:}'.format(d, s, m), d=d, s=s,
                                          m=m)
                    for mode in ['validation', 'test']:
                        modelp = utils.get_model_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(d, s, m), exp_no=run,
                                                      ep_no=ep_no)
                        savep = utils.get_save_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(d, s, m), exp_no=run, ep_no=ep_no,
                                              mode=mode)
                        simple_evaluator = Evaluator(modelp, savep, utils.get_data_path(mode))
                        simple_evaluator.run_full_evaluation(inner_cube=(48, 48, 24), bs=6)


def evaluate_per_saved_epoch():
    for no in range(1,49):
        run_single_evaluation(ep_no=no, exp_no=2, d=240, s=64, m=3)


def main():
    simple_evaluator = Evaluator(utils.get_model_path('Unet_nl2_nc2_nf32_dc1_lrexp-6',0,6), 'test_unetdeconv.h5')
    simple_evaluator.run_full_evaluation(inner_cube=(48,48,24), bs=2)

if __name__=='__main__':

    #single_FSRCNN_evaluation()
    #FSRCNN_evaluation()
    #Unet_evaluation()
    #evaluate_whole_run()
    shifted_evaluation('Unet_nl4_nc2_nf64_dc1', 0, 49)
    # main()