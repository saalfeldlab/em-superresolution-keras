from __future__ import print_function
import numpy as np
import h5py
from matplotlib import pyplot as plt
import utils
import tabulate
import os
import cPickle as pickle

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
    groundtruth = utils.cut_to_sc(groundtruth, sc, axis)
    prediction = utils.cut_to_sc(prediction, sc, axis)
    downscaled = utils.downscale_manually(groundtruth, sc, axis)
    print(downscaled.shape)
    bicubic = utils.bicubic_up(downscaled, sc, axis)
    prediction, [groundtruth, bicubic] = utils.cut_to_same_size(prediction, [groundtruth, bicubic])
    print("std", prediction.std(), groundtruth.std())
    print("mean", prediction.mean(), groundtruth.mean())
    assert prediction.shape == groundtruth.shape
    assert prediction.shape == bicubic.shape
    MSE = mse(prediction, groundtruth)
    cubic_MSE = mse(bicubic, groundtruth)
    PSNR = 10*np.log10(MSE)
    cubic_PSNR = 10*np.log10(cubic_MSE)
    print("MSE: ", MSE)
    print("PSNR:", PSNR)

    bicubic_weighting = se_arr(bicubic, groundtruth)
    bicubic_weighting = 0.5+bicubic_weighting/(np.max(bicubic_weighting)*2)
    weighted_error_arr = se_arr(prediction, groundtruth) * bicubic_weighting
    weighted_cubic_error_arr = se_arr(bicubic, groundtruth)* bicubic_weighting
    wMSE = np.sum(weighted_error_arr)/groundtruth.size
    wPSNR= 10*np.log10(wMSE)
    cubic_wMSE = np.sum(weighted_cubic_error_arr)/groundtruth.size
    cubic_wPSNR = 10 *np.log10(cubic_wMSE)
    print("wMSE:", wMSE)
    print("wPSNR", wPSNR)
    print("cubic PSNR:", cubic_PSNR)
    print("cubic wPSNR:", cubic_wPSNR)
    return MSE, PSNR, wMSE, wPSNR


def run_per_slice_eval(groundtruth, prediction, avg=True, sc=4.):
    downscaled = utils.downscale_manually(groundtruth, sc)
    bicubic = utils.bicubic_up(downscaled, sc, 0)
    prediction, [groundtruth, bicubic] = utils.cut_to_same_size(prediction, [groundtruth, bicubic])
    raw_error_arr = se_arr(prediction, groundtruth)
    bicubic_weighting = se_arr(bicubic, groundtruth)
    print(np.max(bicubic_weighting))
    bicubic_weighting = 0.5 + bicubic_weighting/(np.max(bicubic_weighting)*2)
    weighted_error_arr = raw_error_arr * bicubic_weighting

    raw_error_per_slice = evaluate_per_slice(raw_error_arr)
    weighted_error_per_slice = evaluate_per_slice(weighted_error_arr)
    if avg:
        raw_error_per_slice, _ = utils.running_mean(raw_error_per_slice, sc)
        weighted_error_per_slice, _ = utils.running_mean(weighted_error_per_slice, sc)
    plt.plot(raw_error_per_slice)
    plt.plot(weighted_error_per_slice)
    plt.show()


def bicubic_main(mode='validation', sc = 4.):
    filename = utils.get_save_path('FSRCNN_d{0:}_s{1:}_m{2:}'.format(240, 64, 2), exp_no=2, ep_no=49, mode=mode)
    prediction = np.array(h5py.File(filename, 'r')['raw'])
    gt = np.array(
        h5py.File('/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16iso/'+mode+'.h5', 'r')[
            'raw']) / 255.
    gt= np.squeeze(gt)
    downscaled = utils.downscale_manually(gt, sc)

    bicubic = utils.bicubic_up(downscaled, sc, 0)
    prediction, [bicubic] = utils.cut_to_same_size(prediction, [bicubic])
    mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr = run_eval(gt, bicubic)
    return mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr


def per_slice_main(filename, mode='validation'):
    pred = h5py.File(filename, 'r')['raw']
    gt = np.array(h5py.File('/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16iso/'+mode+'.h5', 'r')[
                      'raw'])/255.
    pred = pred[:,:,:]
    run_per_slice_eval(np.squeeze(gt), pred, avg=False)


def main(filename, mode='validation_and_test', axis=0, res=16, add=''):

    pred = h5py.File(filename, 'r')['raw']
    gt = np.array(
        h5py.File('/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-{0:d}iso'.format(res)+add+'/'+mode+'.h5',
                  'r')[
            'raw']) / 255.
    pred= pred[:,:,:]
    mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr = run_eval(np.squeeze(gt), pred, axis=axis)
    return mse, psnr, bicubic_weighted_mse, bicubic_weighted_psnr


def evaluate_fsrcnn(cp=12, res=16):
    resultlist = []
    k=0
    ep_nos = [186, 161, 140, 168, 141, 121, 162, 142, 126, 148, 125, 110]
    for d in [240, 280]:
        for s in [48, 64]:
            for m in [2, 3, 4]:
                name='FSRCNN_d{0:}_s{1:}_m{2:}_100h'.format(d,s,m)
                #for run in range(2):
                resultlist.append([d, s, m])
                run = 0
                for mode in ['validation', 'test']:
                    savep = utils.get_save_path(name, exp_no=run, ep_no=cp, mode=mode, add='255')
                    resultlist[-1] += list(main(savep, mode, res=res, axis=2))
                k += 1
    as_str = tabulate.tabulate(resultlist, headers=['d', 's', 'm', 'mse valid', 'psnr valid', 'bc_mse valid',
                                           'bc_psnr valid', 'mse test', 'psnr test', 'bc_mse test', 'bc_psnr test'])

    file = open('../results_keras/summaries/FSRCNN_eval_after49_sc01_new.txt', 'w')
    file.write(as_str)
    file.close()


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


def main_evaluate_fsrcnn_longrun(run=0, cp=49):
    resultlist = []
    for d in [240, 280]:
        for s in [48, 64]:
            for m in [2, 3, 4]:
                name='longFSRCNN_d{0:}_s{1:}_m{2:}_3868b61_lr-4_init5e-5'.format(d, s, m)
                resultlist.append([d, s, m])
                for mode in ['validation', 'test']:
                    savep = utils.get_save_path(name, exp_no=run, ep_no=cp, mode=mode, add='icT')
                    resultlist[-1] += list(main(savep, mode, add='zyx', res=16, axis=0))
    as_str = tabulate.tabulate(resultlist, headers=['d', 's', 'm', 'mse valid', 'psnr valid', 'bc_mse valid',
                                                    'bc_psnr valid', 'mse test', 'psnr test', 'bc_mse test',
                                                    'bc_psnr test'])
    file = open('../../results_keras/summaries/FSRCNN_eval_longrun_{0:}icT.txt'.format(cp), 'w')
    file.write(as_str)
    file.close()


def main_evaluate_unets(cp= 49):
    resultlist = []
    for n_l in [2,3,4]:
        for n_f in [32, 64]:
            for n_c in [2,3 ]:

                name = 'longUnet_nl{0:}_nf{1:}_nc{2:}_3868b61_scheduler10'.format(n_l, n_f, n_c)
                run = 0
                resultlist.append([n_l, n_f, n_c])
                for mode in ['validation', 'test']:
                    savep = utils.get_save_path(name, exp_no=run, ep_no=cp, mode=mode, add='icT')
                    resultlist[-1] += list(main(savep, mode, res=16, add='zyx'))

    as_str = tabulate.tabulate(resultlist, headers=['num_levels', 'start_num_filters', 'num_convs', 'run',
                                                    'mse valid', 'psnr valid', 'bc_mse valid', 'bc_psnr valid',
                                                    'mse test', 'psnr test', 'bc_mse test', 'bc_psnr test'])
    file = open('../../results_keras/summaries/longUnet_eval_scheduler10icT_{0}.txt'.format(cp), 'w')
    file.write(as_str)
    file.close()


def main_evaluate_checkpoints(name='FSRCNN_d{0:}_s{1:}_m{2:}'.format(240, 64, 3), run=2):
    resultlist = []
    psnrs = dict()
    psnrs['validation'] = []
    psnrs['test'] = []
    psnrs['training_subset'] = []
    psnrs['validation_wPSNR'] = []
    psnrs['test_wPSNR'] = []
    psnrs['training_subset_wPSNR'] = []
    psnrs['cp'] = []
    for cp in range(10, 91,10)+range(91,101,1):
        print(cp)
        resultlist.append([cp])
        psnrs['cp'].append(cp)
        for mode in ['validation', 'test', 'training_subset']:
            print(mode)
            savep = utils.get_save_path(name, exp_no=run, mode=mode, ep_no=cp-1, add='all')
            x = main(savep, mode, axis=0, res=16, add='zyx')
            resultlist[-1] += x

            psnrs[mode].append(x[1])
            psnrs[mode+'_wPSNR'].append(x[3])
    with open(os.path.dirname(savep)+'/checkpointer.p', 'w') as f:
        pickle.dump(psnrs, f)
    as_str = tabulate.tabulate(resultlist, headers=['it', 'mse valid', 'psnr valid', 'bc_mse valid', 'bc_psnr valid',
                                                    'mse test', 'psnr test', 'bc_mse test', 'bc_psnr_test'])


    #summary_file = open(os.path.dirname(utils.get_save_path(name, run, cp))+'/checkpoint_eval.txt', 'w')
    #summary_file.write(as_str)
    #summary_file.close()


if __name__ == '__main__':
    #main('test_4divisible_FSRCNN.h5')
    #main_evaluate_fsrcnn_longrun(cp=149)
    #main_evaluate_unets(cp=99)
    #for d in [240,280]:
    #    for s in [48,64]:
    #        for m in [2,3,4]:
    main('/groups/saalfeld/home/heinrichl/Projects/results_keras/new_wogt_3-32-30002/validation15_.h5',
         mode='validation', axis=0, res=16, add='zyx')

    #
    #n_l = 4
    #n_f = 64
    #n_c = 3
    #print(n_l, n_f, n_c)
    #main_evaluate_checkpoints('longUnet_nl{0:}_nf{1:}_nc{2:}_3868b61_scheduler10'.format(n_l, n_f, n_c), run=0)
    #name='longUnet_nl{0:}_nf{1:}_nc{2:}_3868b61_scheduler10'.format(n_l, n_f, n_c)
    #run=0
    #mode='test'
    #cp=100
    #savep = utils.get_save_path(name, exp_no=run, mode=mode, ep_no=cp - 1, add='all')
    #x = main(savep, mode, axis=0, res=16, add='zyx')
    #for conf in [(4,64,2), (4,64,3), (3,64,2), (3,64,3)]:
    #    exp_name = 'longUnet_nl{0:}_nf{1:}_nc{2:}_3868b61_scheduler10'.format(*conf)
    #    for ep in [49,99]:
    #        print(conf, ep)
    #        savep = utils.get_save_path(exp_name, exp_no=0, ep_no=ep, mode='training')
    #        print(savep)
    #        main(savep, mode='training', axis=0, add='zyx')


            #simple_evaluator = Evaluator(modelp, savep, utils.get_data_path('training', 16, add='zyx'))
            #simple_evaluator.run_full_evaluation(inner_cube=(24,48,48), bs=6)
    #evaluate_runs()
    #bicubic_main()
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
    #main('/nearline/saalfeld/larissa/older_keras_results/Unet_nl3_nc2_nf64_dc1/validation49_.h5', mode='validation',
    #     res=16, add = '', axis=2)
    #main(utils.get_save_path('FSRCNN_d240_s64_m3_3868b61',0,49, mode='test'), mode='test', res=16,
    #     add='zyx', axis=0)
    #main('/groups/saalfeld/home/heinrichl/Projects/results_keras/Unet_nl3_nf64_nc2_3868b61_lrconst-4/validation49_.h5',
    #     mode='validation', add='zyx')
    #main_evaluate_fsrcnn_longrun(0, 149)

    #main('/groups/saalfeld/home/heinrichl/Projects/results_keras/FSRCNN_const1e-5_240482/test49_.h5',
    #     mode='test', add='zyx')