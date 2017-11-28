from __future__ import print_function
import numpy as np
import scipy.misc
import datetime
import os
import cv2


def cut_to_sc(arr, scaling_factor, axis):
    slicing = [np.s_[:]]*arr.ndim
    slicing[axis] = np.s_[:(arr.shape[axis]-arr.shape[axis]%scaling_factor)]
    return arr[slicing]


def get_bg_borders(arr, bg=0):
    bg_borders = []
    for axis in range(arr.ndim):
        slice_to_check = [np.s_[:]] * arr.ndim

        slice_check_forward_idx = 0
        slice_to_check[axis] = slice_check_forward_idx
        while np.all(arr[slice_to_check] == bg):
            slice_check_forward_idx += 1
            slice_to_check[axis] = slice_check_forward_idx
        slice_check_backward_idx = -1
        slice_to_check[slice_check_backward_idx] = slice_check_backward_idx
        while np.all(arr[slice_to_check] == bg):
            slice_check_backward_idx -= 1
            slice_to_check[axis] = slice_check_backward_idx
        slice_check_backward_idx += 1 # like this it can just be used for slicing
        bg_borders.append((slice_check_forward_idx, slice_check_backward_idx))
    return bg_borders


def cut_to_size(arr, cut_borders):
    slicing = [np.s_[:]] * arr.ndim
    for axis, cut_border_per_axis in enumerate(cut_borders):
        if cut_border_per_axis[1] == 0:
            slicing[axis] = np.s_[cut_border_per_axis[0]:]
        else:
            slicing[axis] = np.s_[cut_border_per_axis[0]:cut_border_per_axis[1]]
    return arr[slicing]


def compute_psnr(arr1, arr2):
    mse = np.sum((arr1 - arr2)**2) / arr1.size
    psnr = -10 * np.log10(mse)
    return psnr


def compute_wpsnr(arr, gt, scaling_factor=4., axis=0):
    downscaled = downscale_manually(gt, scaling_factor, axis)
    cubic = cubic_up(downscaled, scaling_factor, axis)
    weighting = 0.5 * (cubic - gt)**2 / (2 * np.max((cubic - gt)**2))
    wmse = np.sum((arr - gt)**2 * weighting) / arr.size
    wpsnr = -10 * np.log10(wmse)
    return wpsnr


def downscale_manually(arr, factor, axis):
    down_shape = np.array(arr.shape)
    down_shape[axis] = int(down_shape[axis]//factor)
    avg_array = np.zeros(down_shape)

    for k in range((int(factor))):
        slicing = [np.s_[:]]*arr.ndim
        slicing[axis] = np.s_[k:down_shape[axis]*factor:int(factor)]
        avg_array += arr[slicing]
    return avg_array/factor


def cubic_up(arr, factor, axis):
    if arr.ndim < 2:
        raise ValueError("arr must be at least 2dim")
    sliceaxis = 1 if axis == 0 else 0

    # compute size of array after reshaping and initialize with zeros
    nd_factor = np.ones(arr.ndim)
    nd_factor[axis] *= factor
    new_shape = (np.array(arr.shape)*(nd_factor)).astype(int)
    resized_arr = np.zeros(new_shape)
    new_shape = tuple(np.delete(new_shape, sliceaxis, 0).astype(int))

    # for iteration construct the shape of each resized image
    if arr.ndim==3:
        new_shape_cv = np.delete(new_shape, sliceaxis, 0).astype(int)
        #print(new_shape)
    else:
        new_shape_cv = new_shape
    new_shape[sliceaxis]=1
    new_shape_cv = tuple(new_shape_cv[::-1])
    for k in range(arr.shape[sliceaxis]):
        sliceobj = tuple([slice(None)]*sliceaxis+[slice(k, k+1, None)])
        resized_arr[sliceobj] = np.reshape(cv2.resize(arr[sliceobj].squeeze().astype('float'),new_shape_cv,
                                                          interpolation=cv2.INTER_CUBIC), new_shape)
        #resized_arr[sliceobj] = np.expand_dims(imresize(arr[sliceobj].squeeze(), new_shape,
        #                                                           interp='bicubic'), axis=sliceaxis)
    return resized_arr


def get_exppath(saving_path, exp_name=None):
    if saving_path[-1] != '/':
        saving_path += '/'
    if exp_name is None:
        exp_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")+'/'
    elif os.path.isdir(saving_path+exp_name):
        count = 1
        exp_name = exp_name+'{:04d}/'
        while os.path.isdir(saving_path+exp_name.format(count)):
            count += 1
        exp_name = exp_name.format(count)
    else:
        if exp_name[-1] != '/':
            exp_name += '/'
    exp_path = saving_path + exp_name
    os.mkdir(exp_path)
    return exp_path
