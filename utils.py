import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import scipy.misc
import os


def get_model_path(exp_name, exp_no, ep_no):
    if exp_no != 0:
        exp_name += '{:04d}'.format(exp_no)
    return '/groups/saalfeld/home/heinrichl/Projects/results_keras/' + exp_name + '/model_state{:02d}.h5'.format(ep_no)


def get_weight_path(exp_name, exp_no, ep_no):
    if exp_no != 0:
        exp_name += '{:04d}'.format(exp_no)
    return '/groups/saalfeld/home/heinrichl/Projects/results_keras/' + exp_name + '/weights{:02d}.h5'.format(ep_no)


def get_save_path(exp_name, exp_no, ep_no, mode='validation', add=''):
    if exp_no != 0:
        exp_name += '{:04d}'.format(exp_no)
    return '/groups/saalfeld/home/heinrichl/Projects/results_keras/' + exp_name+'/' + mode + '{:02d}'.format(
        ep_no) + '_'+add+'.h5'


def get_data_path(mode, resolution=16, add=''):
    return '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-{0:d}iso'.format(resolution)+add+'/'+mode+'.h5'


def fix_prelu_json_files(run, exp_name=None):
    if exp_name is None:
        exp_name = 'FSRCNN_d{0:}_s{1:'
    exp_path = os.path.dirname(get_model_path(exp_name, exp_no=run, ep_no=1))
    json_file = exp_path + '/model_def_json.txt'
    f = open(json_file, 'r')
    config_str = f.read()
    f.close()
    new_config_str = config_str.replace('\\"init\\": \\"zero\\", \\"trainable\\": true',
                                        '\\"init\\": \\"zero\\", \\"trainable\\": true, \\"shared_axes\\": [2, 3, 4]')
    f = open(json_file, 'w')
    f.write(new_config_str)
    f.close()
    old_backup = open(exp_path + '/model_def_json_orig.txt', 'w')
    old_backup.write(config_str)
    old_backup.close()


def running_mean(arr, window):
    curr_sum = 0
    squared_sum = 0
    mean_arr = np.zeros_like(arr)
    std_arr = np.zeros_like(arr)
    for i in range(window):
        curr_sum += arr[i]
        squared_sum += arr[i]**2
        mean_arr[i] = curr_sum/(i+1.)
        std_arr[i] = (squared_sum/(i+1.)-mean_arr[i]**2)**0.5
    for i in range(window, len(arr)):
        curr_sum += - arr[i-window] + arr[i]
        squared_sum += - arr[i-window]**2 + arr[i]**2
        mean_arr[i] = curr_sum/window
        std_arr[i] = (squared_sum/window-mean_arr[i]**2)**0.5

    return mean_arr, std_arr


def downscale_manually(arr, factor=4., axis=0):
    down_shape = list(arr.shape)
    down_shape[axis] = int(down_shape[axis]/factor)
    avg_array = np.zeros(down_shape)
    #reduction_slice = [np.s_[:]]*arr.ndim
    #reduction_slice[axis] = np.s_[:down_shape[axis]*factor]
    #arr = arr[reduction_slice]
    for k in range(int(factor)):
        slicing = [np.s_[:]]*arr.ndim
        slicing[axis] = np.s_[k::factor]
        avg_array += arr[slicing]
    return avg_array/factor


def bicubic_up(arr, factor, axis):
    assert arr.ndim > 1

    sliceaxis = 1 if axis == 0 else 0

    nd_factor = np.ones(arr.ndim)
    nd_factor[axis] *= factor
    new_shape = (np.array(arr.shape)*nd_factor).astype(int)
    resized_arr = np.zeros(new_shape)

    new_shape = tuple(np.delete(new_shape, sliceaxis, 0).astype(int))

    for k in range(arr.shape[sliceaxis]):
        sliceobj = tuple([slice(None)]*sliceaxis+[slice(k, k+1, None)])
        resized_arr[sliceobj] = np.expand_dims(scipy.misc.imresize(arr[sliceobj].squeeze(), new_shape,
                                                                   interp='bicubic', mode='F'), axis=sliceaxis)
    return resized_arr


def get_cut_borders(arr):
    cut_borders = []
    for axis in range(arr.ndim):
        slicing_idx = [np.s_[:], ]*arr.ndim

        slice_forward = 0
        slicing_idx[axis] = slice_forward

        while np.all(arr[slicing_idx] == 0):
            slice_forward += 1
            slicing_idx[axis] = slice_forward
        slice_backward = -1
        slicing_idx[axis] = slice_backward
        while np.all(arr[slicing_idx] == 0):
            slice_backward -= 1
            slicing_idx[axis] = slice_backward
        slice_backward += 1
        cut_borders.append((slice_forward, slice_backward))
    return cut_borders


def cut_to_same_size(zero_bordered, to_be_same_sizes):
    for arr in to_be_same_sizes:
        print arr.shape
        print zero_bordered.shape
        assert arr.ndim == zero_bordered.ndim
        assert arr.shape == zero_bordered.shape
    p_cb = get_cut_borders(zero_bordered)
    for axis, p_cb_axis in enumerate(p_cb):
        slicing_idx = [np.s_[:]] * zero_bordered.ndim

        if p_cb_axis[1] == 0:
            slicing_idx[axis] = np.s_[p_cb_axis[0]:]
        else:
            slicing_idx[axis] = np.s_[p_cb_axis[0]:p_cb_axis[1]]

        zero_bordered = zero_bordered[slicing_idx]
        for k, arr in enumerate(to_be_same_sizes):
            to_be_same_sizes[k] = arr[slicing_idx]
    return zero_bordered, to_be_same_sizes


def cut_to_sc(arr, sc, axis):
    slicing = [np.s_[:]]*arr.ndim
    slicing[axis] = np.s_[:(arr.shape[axis]-arr.shape[axis]%sc)]
    return arr[slicing]


def get_epoch_losses(exp_name):
    pfile_epoch = open('../results_keras/' + exp_name + '/epoch_history.p', 'r')
    d = pickle.load(pfile_epoch)
    pfile_epoch.close()
    return d['loss'], d['val_loss']


def get_all_losses(exp_name, cp):
    pfile_loss = open('../results_keras/' + exp_name + '/loss_history{:02}.p'.format(cp), 'r')
    all_losses = pickle.load(pfile_loss)
    pfile_loss.close()
    return all_losses


def get_training_time(exp_name):
    pfile_time = open('../results_keras/' + exp_name + '/run_times.p', 'r')
    training_times, _ = pickle.load(pfile_time)
    pfile_time.close()
    return training_times


def get_time_per_epoch(exp_name):
    training_times = get_training_time(exp_name)
    return training_times[-1]/(len(training_times)-1)


def plot_final_loss(exp_name, save_interval=22, smoothing=22):
    epoch_losses, epoch_val_losses = get_epoch_losses(exp_name)
    print(len(epoch_losses), len(epoch_val_losses))
    all_losses = get_all_losses(exp_name, cp=int(len(epoch_losses)/save_interval)-1)
    print(len(all_losses))
    mean_r, std_r = running_mean(all_losses, window=smoothing)
    epochs = range(1, len(all_losses), save_interval)

    plt.style.use('seaborn-whitegrid')
    ax, = plt.semilogy(mean_r, label='training_loss')
    plt.fill_between(range(len(mean_r)), mean_r - std_r, mean_r + std_r, alpha= 0.5)
    plt.semilogy(epochs, epoch_val_losses[:-save_interval+1], label='validation_loss', ls='--')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    #plot_final_loss('Unet_best_zyx0003')
    for d in [240, 280]:
        for s in [48,64]:
            for m in [2,3,4]:
                fix_prelu_json_files(0,'FSRCNN_d{0:}_s{1:}_m{2:}_100h'.format(d,s,m))