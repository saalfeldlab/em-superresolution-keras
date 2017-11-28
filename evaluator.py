from __future__ import print_function, division

import numpy as np
import h5py
import itertools

from keras import backend as K


class Evaluator(object):
    def __init__(self, model, inner_cube_size=(24,48,48), data_path=None, dset_name='raw', normalize=True):
        self.model = model
        self.data_path = data_path
        self.data = h5py.File(data_path, 'r')[dset_name]
        self.inner_cube_size = inner_cube_size
        self.safety_margin = ((0, 0, 0), (0, 0, 0))
        self.pred_dset_name = 'raw'
        self.normalize = normalize
        if K.image_data_format() == 'channels_last':
            self.spatial_slice = np.s_[1:-1]
        else:
            self.spatial_slice = np.s_[2:]

    def _prepare_output_file(self, path, shape):
        self.output_file = h5py.File(path, 'w-')
        self.output_file.create_dataset(self.pred_dset_name, data=np.zeros(shape))

    def set_data_path(self, data_path, dset_name='raw'):
        self.data_path = data_path
        self.data = h5py.File(data_path, 'r')[dset_name]

    def set_safety_margin(self, safety_margin):
        self.safety_margin = safety_margin

    def set_pred_dset_name(self, name):
        self.pred_dset_name = name

    def get_patch(self, coord):
        patch = np.empty(self.model.model.input_shape[self.spatial_slice])
        coord_slice = []
        patch_slice = []
        for axis, axis_coord in enumerate(coord):
            coord_slice.append(np.s_[axis_coord - self.model.model.input_shape[self.spatial_slice][axis] // 2 :
                                     axis_coord + (self.model.model.input_shape[self.spatial_slice][axis] + 1) // 2])
            patch_slice.append(np.s_[:])
        coord_slice = tuple(coord_slice)
        patch_slice = tuple(patch_slice)
        self.data.read_direct(patch, coord_slice, patch_slice)
        if self.normalize:
            patch /= 255.
        return patch

    def batch_generator(self, ignore_border, bs, coordinate_generator):
        #todo change to include model's scaling axis

        batch = np.zeros((bs,)+self.model.model.input_shape[self.spatial_slice])
        k=0
        for l, coord in coordinate_generator:
            batch[k] = self.get_patch(coord)
            if k == bs-1:
                if K.image_data_format() == 'channels_last':
                    batch = np.expand_dims(batch, -1)
                else:
                    batch = np.expand_dims(batch, 1)
                yield batch
                k = 0
                batch = np.zeros((bs,) + self.model.model.input_shape[self.spatial_slice])
            else:
                k += 1

        if k > 0:
            batch = batch[:k, :, :, :]
            if K.image_data_format() == 'channels_last':
                batch = np.expand_dims(batch, -1)
            else:
                batch = np.expand_dims(batch, 1)
            yield batch

    def coordinate_generator(self, ignore_border):
        output_shape = self.model.model.output_shape[self.spatial_slice]

        #calculate how much is not covered after covering the data with cubes
        border_remainder = (np.array(self.data.shape) - 2 * np.array(ignore_border) -
                            np.array(self.safety_margin)[0] + np.array(self.safety_margin)[1]) % (np.array(
                            output_shape) - 2 * np.array(ignore_border))

        extras = [[]] * len(border_remainder)
        for axis in range(len(border_remainder)):
            if border_remainder[axis] != 0:  # need additional sample points to cover all data
                extras[axis] = [int(self.data.shape[axis] - output_shape[axis]//2 - self.safety_margin[1][axis])]
                if axis == 0:
                    last = range(self.safety_margin[0][axis]+output_shape[axis]//2, self.data.shape[axis] -
                                 output_shape[axis]//2 - self.safety_margin[1][axis], output_shape[axis] -
                                 2*ignore_border[axis])[-1]
                    overlap = output_shape[axis] - 2*ignore_border[axis] - (extras[axis][0]-last)
                    extras[axis] = [int(extras[axis] - (self.model.scaling_factor - overlap
                                                        % self.model.scaling_factor))]

        k = 1
        coord_seq = [[]] * len(border_remainder)
        for axis in range(len(border_remainder)):
            coord_seq[axis] = range(self.safety_margin[0][axis] + np.array(output_shape)[axis]//2,
                                    self.data.shape[axis] - np.array(output_shape)[axis]//2 - self.safety_margin[1][axis],
                                    np.array(output_shape)[axis] - 2 * np.array(ignore_border)[axis]) + extras[axis]

        for coord in itertools.product(coord_seq[0], coord_seq[1], coord_seq[2]):
            yield k, coord
            k += 1

    def run(self, target_path, bs):
        ignore_border = (np.array(self.model.input_shape[self.spatial_slice]) - np.array(self.inner_cube_size))/2.
        if not np.all([ib.is_integer for ib in ignore_border]):
            raise ValueError("The shape of inner_cube is not valid for the model's input shape")
        ignore_border = tuple([int(ib) for ib in ignore_border])
        self._prepare_output_file(target_path, self.data.shape)
        cg, cg_for_bg = itertools.tee(self.coordinate_generator(ignore_border))
        bg = self.batch_generator(ignore_border, bs, cg_for_bg)
        for batch in bg:
            predicted_batch = self.model.model.predict_on_batch(batch)
            for sample, (processed_examples, coord) in zip(predicted_batch, cg):
                self.output_file[self.pred_dset_name].write_direct(
                sample,
                np.s_[ignore_border[0]: self.inner_cube_size[0]+ignore_border[0],
                      ignore_border[1]: self.inner_cube_size[1]+ignore_border[1],
                      ignore_border[2]: self.inner_cube_size[2]+ignore_border[2],
                      0],
                np.s_[coord[0] - (sample.shape[0] - 2 * ignore_border[0]) // 2:
                      coord[0] + (sample.shape[0] - 2 * ignore_border[0] + 1) // 2,
                      coord[1] - (sample.shape[1] - 2 * ignore_border[1]) // 2:
                      coord[1] + (sample.shape[1] - 2 * ignore_border[1] + 1) // 2,
                      coord[2] - (sample.shape[2] - 2 * ignore_border[2]) // 2:
                      coord[2] + (sample.shape[2] - 2 * ignore_border[2] + 1) // 2])

        self.output_file.close()
