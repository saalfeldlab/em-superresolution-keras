from __future__ import print_function
import os
import json
import h5py
import itertools
import numpy as np
import time
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json
from keras.utils import generic_utils
from keras_contrib.layers import Deconvolution3D
from keras import backend as K
import utils


if K.image_data_format() == 'channels_last':
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
        patch = np.empty(inp)

        # 1 added in second coordinate to get the right length in case of odd inp. Doesn't hurt for even
        # shorter than int(np.ceil())
        coord_slice = np.s_[coord[0]-inp[0]/2:coord[0]+(inp[0]+1)/2,
                            coord[1]-inp[1]/2:coord[1]+(inp[1]+1)/2,
                            coord[2]-inp[2]/2:coord[2]+(inp[2]+1)/2]

        self.data.read_direct(patch, coord_slice, np.s_[:, :, :])
        return patch

    def test_data_generator(self, ignore_border, bs, safety_margin=(0, 0)):
        """generates data as required by keras"""
        o_shape = self.model.output_shape[spatial_slice]
        border_remainder = (np.array(self.data.shape)-np.array([safety_margin[0] + safety_margin[1], 0, 0]) -
                            2*np.array(ignore_border)) % (np.array(o_shape) - 2*np.array(ignore_border))
        x_extra = [] if border_remainder[2] == 0 else [int(self.data.shape[2] - o_shape[2]/2)]
        y_extra = [] if border_remainder[1] == 0 else [int(self.data.shape[1] - o_shape[1]/2)]
        if border_remainder[0] == 0:
            z_extra = []
        else:
            z_extra = self.data.shape[0] - o_shape[0]/2 - safety_margin[1]
            z_last = range(safety_margin[0] + o_shape[0]/2, self.data.shape[0] - o_shape[0]/2 - safety_margin[1],
                           o_shape[0] - 2*ignore_border[0])[-1]
            overlap_to_last = o_shape[0] - 2*ignore_border[0] - (z_extra-z_last)
            z_extra = [int(z_extra - (self.sc - overlap_to_last%self.sc))]

        batch = np.zeros((bs,)+self.model.input_shape[spatial_slice])
        k = 0
        l = 0

        for z in range(safety_margin[0] + o_shape[0]/2, self.data.shape[0] - o_shape[0]/2 - safety_margin[1],
                       o_shape[0] - 2*ignore_border[0]) + z_extra:
            print('\nz', z)
            for y in range(o_shape[1]/2, self.data.shape[1] - o_shape[1]/2, o_shape[1] - 2*ignore_border[1]) + y_extra:
                print('\ny', y)
                for x in range(o_shape[2]/2, self.data.shape[2] - o_shape[2]/2, o_shape[2] - 2*ignore_border[2]) + \
                        x_extra:
                    print('.', end='')

                    batch[k, :, :, :] = self.get_patch((z, y, x))/255.

                    if k == bs-1:
                        if K.image_data_format() == 'channels_last':
                            batch = np.expand_dims(batch, -1)
                        else:
                            batch = np.expand_dims(batch, 1)
                        l += 1
                        yield batch
                        k = 0
                        batch = np.zeros((bs,)+self.model.input_shape[spatial_slice])
                    else:
                        k += 1
                print('\n')
        if k > 0:
            batch = batch[:k, :, :, :]

            if K.image_data_format() == 'channels_last':
                batch = np.expand_dims(batch, -1)
            else:
                batch = np.expand_dims(batch, 1)
            yield batch

    def parallel_coordinates_generator(self, ignore_border, safety_margin=(0, 0)):
        """generates the coordinates in the same order as test_data_generator (necessary because keras doesn't allow
        for additional outputs of the generator)"""
        # with enumerate
        o_shape = self.model.output_shape[spatial_slice]
        border_remainder = (np.array(self.data.shape)-np.array([safety_margin[0]+safety_margin[1], 0, 0]) -
                            2*np.array(ignore_border)) % (np.array(o_shape)-2*np.array(ignore_border))
        x_extra = [] if border_remainder[2] == 0 else [self.data.shape[2] - o_shape[2] / 2]
        y_extra = [] if border_remainder[1] == 0 else [self.data.shape[1] - o_shape[1] / 2]
        if border_remainder[0] == 0:
            z_extra = []
        else:
            z_extra = self.data.shape[0]-o_shape[0]/2 - safety_margin[1]
            z_last = range(safety_margin[0] + o_shape[0]/2, self.data.shape[0] - o_shape[0] / 2 - safety_margin[1],
                           o_shape[0] - 2*ignore_border[0])[-1]

            overlap_to_last = o_shape[0] - 2 * ignore_border[0]-(z_extra-z_last)
            z_extra = [int(z_extra-(self.sc-overlap_to_last % self.sc))]

        time.sleep(5)
        k = 1
        #x last
        for z in range(safety_margin[0] + o_shape[0]/2, self.data.shape[0] - o_shape[0]/2 - safety_margin[1],
                       o_shape[0] - 2 * ignore_border[0]) + z_extra:
            for y in range(o_shape[1]/2, self.data.shape[1] - o_shape[1]/2, o_shape[1]-2*ignore_border[1]) + y_extra:
                for x in range(o_shape[2]/2, self.data.shape[2] - o_shape[2]/2, o_shape[2] - 2*ignore_border[2]) + \
                        x_extra:
                    yield k, (z, y, x)
                    k += 1

    def load_model(self, json_file=None):
        """model specification is read from json file"""
        if json_file is None:
            exp_path = os.path.dirname(self.model_path)
            json_file = exp_path+'/model_def_json.txt'

        model_def_file = open(json_file, 'r')
        self.model = model_from_json(json.load(model_def_file))
        model_def_file.close()
        self.model.load_weights(self.model_path)
        #os.path.dirname(self.model_path)+'/finetuning047/model_state28.h5',
        # by_name=True)
        #self.model = load_model(self.model_path, custom_objects={'gaussian_init':gaussian_init })
        print("INPUT:", self.model.input_shape)
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
        if K.image_data_format() == 'channels_last':
            patch = patch[np.newaxis, :, :, :, np.newaxis]
        else:
            patch = patch[np.newaxis, np.newaxis, :, :, :]
        prediction = np.squeeze(self.model.predict(patch, 1))
        self.output_file.create_dataset('original', data=patch)
        self.output_file['raw'].write_direct(prediction, np.s_[:, :, :], np.s_[:, :, :])
        self.output_file.close()

    def multiple_evaluations(self, coords):
        """generate predictions for patches belonging to a list of coordinates (not implemented)"""
        pass

    def run_full_evaluation(self, inner_cube, bs, safety_margin=(0, 0)):
        """generate prediction for a whole dataset"""
        self.load_model()
        ignore_border = (np.array(self.model.input_shape[spatial_slice]) - np.array(inner_cube))/2.
        if not np.all([ib.is_integer() for ib in ignore_border]):
            raise ValueError('The shape of inner_cube is not valid')
        else:
            ignore_border = tuple([int(ib) for ib in ignore_border])
        self.prepare_output_file(self.data.shape)

        batch_generator = self.test_data_generator(ignore_border, bs, safety_margin=safety_margin)
        corresponding_coords_generator, counter = itertools.tee(self.parallel_coordinates_generator(ignore_border,
                                                                                                    safety_margin))
        #get length
        for num_coords_to_process, _ in counter:
            pass
        print("NUM PROCESS", num_coords_to_process)
        for batch in batch_generator:
            pred_batch = self.model.predict_on_batch(batch)
            for sample, (processed_examples, coord) in zip(pred_batch, corresponding_coords_generator):
                if K.image_data_format()=='channels_last':
                    self.output_file['raw'].write_direct(sample, np.s_[ignore_border[0]: -ignore_border[0],
                                                                       ignore_border[1]: -ignore_border[1],
                                                                       ignore_border[2]: -ignore_border[2], 0],
                                                    np.s_[coord[0] - (sample.shape[0] - 2*ignore_border[0]) / 2:
                                                          coord[0] + (sample.shape[0] - 2*ignore_border[0] + 1) / 2,
                                                          coord[1] - (sample.shape[1] - 2*ignore_border[1]) / 2:
                                                          coord[1] + (sample.shape[1] - 2*ignore_border[1] + 1) / 2,
                                                          coord[2] - (sample.shape[2] - 2*ignore_border[2]) / 2:
                                                          coord[2] + (sample.shape[2] - 2*ignore_border[2] + 1) / 2])
                else:
                    self.output_file['raw'].write_direct(sample, np.s_[0, ignore_border[0]: -ignore_border[0],
                                                                       ignore_border[1]: -ignore_border[1],
                                                                       ignore_border[2]: -ignore_border[2]],
                                                     np.s_[coord[0] - (sample.shape[1] - 2*ignore_border[0]) / 2:
                                                           coord[0] + (sample.shape[1] - 2*ignore_border[0] + 1) / 2,
                                                           coord[1] - (sample.shape[2] - 2*ignore_border[1]) / 2:
                                                           coord[1] + (sample.shape[2] - 2*ignore_border[1] + 1) / 2,
                                                           coord[2] - (sample.shape[3] - 2*ignore_border[2]) / 2:
                                                           coord[2] + (sample.shape[3] - 2*ignore_border[2] + 1) / 2])
        self.output_file.close()


def shifted_evaluation(exp_name, run, cp, resolution=16):
    for mode in ['validation', 'test']:
        modelp = utils.get_model_path(exp_name, exp_no=run, ep_no=cp)
        shifted_evaluator = Evaluator(modelp, '', utils.get_data_path(mode, resolution))
        for shift in range(int(shifted_evaluator.sc)):
            savep = utils.get_save_path(exp_name, exp_no=run, ep_no=cp, mode=mode, add='_shift'+str(shift))
            shifted_evaluator.reset_save_path(savep)
            shifted_evaluator.run_full_evaluation(inner_cube=(24, 48, 48), bs=6, safety_margin=(shift, -shift))


def run_evaluation(exp_name, run, ep_no, inner_cube=(24, 48, 48), bs=6, resolution=16, add_to_save_path='',
                   add_to_data_path='zyx'):
    for mode in ['validation', 'test']:
        modelp = utils.get_model_path(exp_name, exp_no=run, ep_no=ep_no)
        savep = utils.get_save_path(exp_name, exp_no=run, ep_no=ep_no, mode=mode, add=add_to_save_path)
        simple_evaluator = Evaluator(modelp, savep, utils.get_data_path(mode, resolution, add=add_to_data_path))
        simple_evaluator.run_full_evaluation(inner_cube=inner_cube, bs=bs)


def fsrcnn_hyperparameter_evaluation(ep_no=49):
    k = 0
    # ep_nos = [186, 161, 140, 168, 141, 121, 162, 142, 126, 148, 125, 110]
    for d in [240, 280]:
        for s in [48, 64]:
            for m in [2, 3, 4]:
                run_evaluation('fsrcnn50e-4_d{0:}_s{1:}_m{2:}'.format(d, s, m), 0, ep_no, inner_cube=(24, 48, 48),
                               resolution=16)

                k += 1


def evaluate_per_saved_epoch(max_epoch, exp_name, run, ep_no, inner_cube=(24, 48, 48), bs=6):
    for no in range(1, max_epoch):
        run_evaluation(exp_name, run, ep_no, inner_cube=inner_cube, bs=bs)


if __name__ == '__main__':
    # single_FSRCNN_evaluation()

    # FSRCNN_evaluation()
    #run_evaluation('fsrcnn_test', 2, 1, bs=6)
    fsrcnn_hyperparameter_evaluation(ep_no=49)

    # run_evaluation('Unet_best_zyx', 3, 28, inner_cube=(24, 48, 48), resolution=10)

    # evaluate_whole_run()
    # simple_eval = Evaluator(utils.get_model_path('unet_zyx_nl2_nc2_nf32', 0, 49), utils.get_save_path(
    #     'unet_zyx_nl2_nc2_nf32', 0, 49), utils.get_data_path('validation', 10, add='zyx'))
    # simple_eval.run_full_evaluation(inner_cube=(24,48,48), bs=6)
    # shifted_evaluation('Unet_nl4_nc2_nf64_dc1', 0, 49)
    # main()
