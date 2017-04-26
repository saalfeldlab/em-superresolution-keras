from __future__ import print_function
import numpy as np
import h5py
from matplotlib import pyplot as plt
import utils
import tabulate
import os


def mse(arr1, arr2):
    assert arr1.shape == arr2.shape
    return np.sum((arr1 - arr2) ** 2) / arr1.size


def se_arr(arr1, arr2):
    assert arr1.shape == arr2.shape
    return (arr1-arr2)**2


def evaluate_per_slice(error_arr):
    per_slice_error = []
    for k in range(error_arr.shape[0]):
        per_slice_error.append(np.sum(error_arr[k, :, :])/error_arr[k, :, :].size)
    return per_slice_error


def run_eval(groundtruth, prediction, sc=4, axis=0):
    groundtruth= utils.cut_to_sc(groundtruth, sc, axis)
    prediction = utils.cut_to_sc(prediction, sc, axis)
    downscaled = utils.downscale_manually(groundtruth, sc, axis)
    print("original shape:", groundtruth.shape, ", downscaled shape:", downscaled.shape)
    cubic = utils.bicubic_up(downscaled, sc, axis)
    prediction, [groundtruth, cubic] = utils.cut_to_same_size(prediction, [groundtruth, cubic])
    print("mean --", "prediction:", prediction.mean(), "groundtruth:", groundtruth.mean())
    print("standard deviation --", "prediction:", prediction.std(), "groundtruth:", groundtruth.std())
    assert prediction.shape == groundtruth.shape
    assert prediction.shape == cubic.shape
    mse_error = mse(prediction, groundtruth)
    cubic_error = mse(cubic, groundtruth)
    psnr_error = 10 * np.log10(mse_error)
    error_psnr_cubic = 10 * np.log10(cubic_error)
    print("PSNR:", psnr_error)
    print("cubic PSNR:", error_psnr_cubic)
    cubic_weighting = se_arr(cubic, groundtruth)
    cubic_weighting = 0.5 + cubic_weighting/(np.max(cubic_weighting)*2)
    weighted_error_arr = se_arr(prediction, groundtruth) * cubic_weighting
    weighted_mse_error = np.sum(weighted_error_arr)/groundtruth.size
    weighted_psnr_error = 10*np.log10(weighted_mse_error)
    print("wPSNR:", weighted_psnr_error)
    return mse_error, psnr_error, weighted_mse_error, weighted_psnr_error


def run_per_slice_eval(groundtruth, prediction, avg=True, sc=4.):
    downscaled = utils.downscale_manually(groundtruth, sc)
    cubic = utils.bicubic_up(downscaled, sc, 0)
    prediction, [groundtruth, cubic] = utils.cut_to_same_size(prediction, [groundtruth, cubic])
    raw_error_arr = se_arr(prediction, groundtruth)
    cubic_weighting = se_arr(cubic, groundtruth)
    cubic_weighting = 0.5 + cubic_weighting/(np.max(cubic_weighting)*2)
    weighted_error_arr = raw_error_arr * cubic_weighting

    raw_error_per_slice = evaluate_per_slice(raw_error_arr)
    weighted_error_per_slice = evaluate_per_slice(weighted_error_arr)
    if avg:
        raw_error_per_slice, _ = utils.running_mean(raw_error_per_slice, sc)
        weighted_error_per_slice, _ = utils.running_mean(weighted_error_per_slice, sc)
    plt.plot(raw_error_per_slice)
    plt.plot(weighted_error_per_slice)
    plt.show()


def cubic_main(mode='validation', sc = 4.):
    filename = utils.get_save_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(240, 64, 2), exp_no=2, ep_no=49, mode=mode)
    prediction = np.array(h5py.File(filename, 'r')['raw'])
    gt = np.array(
        h5py.File('/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16iso/'+mode+'.h5', 'r')[
            'raw']) / 255.
    gt = np.squeeze(gt)
    downscaled = utils.downscale_manually(gt, sc)

    bicubic = utils.bicubic_up(downscaled, sc, 0)
    prediction, [bicubic] = utils.cut_to_same_size(prediction, [bicubic])
    mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr = run_eval(gt, bicubic)
    return mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr


def per_slice_main(filename, mode='validation'):
    pred = h5py.File(filename, 'r')['raw']
    gt = np.array(h5py.File('/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16iso/'+mode+'.h5', 'r')[
                      'raw'])/255.
    pred = pred[:, :, :]
    run_per_slice_eval(np.squeeze(gt), pred, avg=False)


def main(filename, mode='validation_and_test', axis=0, res=16, add=''):

    pred = h5py.File(filename, 'r')['raw']
    gt = np.array(
         h5py.File('/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-{0:d}iso'.format(res)+add+'/'+mode+'.h5',
                   'r')['raw']) / 255.
    pred = pred[:, :, :]
    mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr = run_eval(np.squeeze(gt), pred, axis=axis)
    return mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr


def evaluate_fsrcnn(cp=49, res=16):
    resultlist = []
    k=0
    #ep_nos = [186, 161, 140, 168, 141, 121, 162, 142, 126, 148, 125, 110]
    for d in [240, 280]:
        for s in [48, 64]:
            for m in [2, 3, 4]:
                name='fsrcnn50_d{0:}_s{1:}_m{2:}'.format(d, s, m)
                #for run in range(2):
                resultlist.append([d, s, m])
                run = 0
                for mode in ['validation', 'test']:
                    savep = utils.get_save_path(name, exp_no=run, ep_no=cp, mode=mode)
                    resultlist[-1] += list(main(savep, mode, res=res, axis=0, add='zyx'))
                k += 1
    as_str = tabulate.tabulate(resultlist, headers=['d', 's', 'm', 'mse valid', 'psnr valid', 'bc_mse valid',
                                           'bc_psnr valid', 'mse test', 'psnr test', 'bc_mse test', 'bc_psnr test'])

    f = open('../results_keras/summaries/fsrcnn50.txt', 'w')
    f.write(as_str)
    f.close()


def main_evaluate_shift(exp_name, run, cp, sc=4):
    resultlist = []
    for shift in range(sc):
        resultlist.append([shift])
        for mode in ['validation', 'test']:
            savep = utils.get_save_path(exp_name, exp_no=run, ep_no=cp, mode=mode, add='_shift' + str(shift))
            resultlist[-1] += list(main(savep, mode))
    as_str = tabulate.tabulate(resultlist, headers=['shift', 'mse valid', 'psnr valid', 'bc_mse valid',
                                                    'bc_psnr valid', 'mse test', 'psnr test', 'bc_mse test',
                                                    'bc_psnr test'])
    shift_file = open(os.path.dirname(savep)+'/shift_evaluation_'+str(cp)+'.txt', 'w')
    shift_file.write(as_str)
    shift_file.close()


def main_evaluate_unets(cp= 49):
    resultlist = []
    for n_l in [4, 3, 2]:
        for n_f in [64, 32]:
            for n_c in [3, 2]:
                deconv=True
                name = 'Unet_nl{0:}_nc{1:}_nf{2:}_dc{3:}'.format(n_l, n_c, n_f, int(deconv))
                run = 0
                resultlist.append([n_l, n_f, n_c, run])
                for mode in ['validation', 'test']:
                    savep = utils.get_save_path(name, exp_no=run, ep_no=cp, mode=mode)
                    resultlist[-1] += list(main(savep, mode))

    as_str = tabulate.tabulate(resultlist, headers=['num_levels', 'start_num_filters', 'num_convs', 'run',
                                                    'mse valid', 'psnr valid', 'bc_mse valid', 'bc_psnr valid',
                                                    'mse test', 'psnr test', 'bc_mse test', 'bc_psnr test'])
    f = open('../results_keras/summaries/Unet_eval.txt', 'w')
    f.write(as_str)
    f.close()


def main_evaluate_checkpoints(name='FSRCNN_d{0:}_s{1:}_m{2:}'.format(240, 64, 3), run=2):
    resultlist = []
    for cp in range(1, 50):
        resultlist.append([cp])
        for mode in ['validation', 'test']:
            savep = utils.get_save_path(name, exp_no=run, ep_no=cp)
            resultlist[-1] += main(savep, mode)
    as_str = tabulate.tabulate(resultlist, headers=['it', 'mse valid', 'psnr valid', 'bc_mse valid', 'bc_psnr valid',
                                                    'mse test', 'psnr test', 'bc_mse test', 'bc_psnr_test'])

    summary_file = open(os.path.dirname(utils.get_save_path(name, run, cp))+'/checkpoint_eval.txt', 'w')
    summary_file.write(as_str)
    summary_file.close()


if __name__ == '__main__':
    #main('test_4divisible_FSRCNN.h5')
    #main_evaluate_fsrcnn_longrun()
    #main_evaluate_unets()
    #evaluate_runs()
    #cubic_main()
    #per_slice_main(utils.get_save_path('Unet_nl4_nc2_nf64_dc1',0,49), 'validation')
    #main_evaluate_shift('Unet_nl4_nc2_nf64_dc1', 0, 49)

    #evaluate_fsrcnn(49, res=16)
    #mode = 'validation'
    #cp = 28
    #run = 2
    #res = 10
    #name = 'FSRCNN_d{0:}_s{1:}_m{2:}'.format(240, 64, 2)

    #savep = utils.get_save_path(name, exp_no=run, ep_no=cp, mode=mode, add='new')
    #main(savep, mode, res=res, axis=2)

    evaluate_fsrcnn(49,16)
    #main(utils.get_save_path('Unet_best_zyx',3,28, add='w-gtzyx'), mode='validation', res=10,
    #     add='zyx', axis=0)

    #main(utils.get_save_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(240, 64, 2), exp_no=2, ep_no=49, mode='test'),
    # mode='test')