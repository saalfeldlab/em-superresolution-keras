from __future__ import print_function

import numpy as np
import os

from define_model import *
from trainer import *
import utils
from evaluator import *


def training_unet_simulated_w_gt():
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    name = 'Unet_4-64-3'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name)
    print("output results and logs to:", exp_path)

    m = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=True)
    m.unet_simple_spec(4, 64, 3, batchnorm=False)
    #m.fullyconv_unet_simple_spec(3,32,2)
    m.training_scheme()
    m.compile(1e-04, 10)
    m.save_json(exp_path+'model_def_json.txt')

    t = Trainer(m, exp_path, trainingfile, validationfile, cubic=False)
    t.run(151)

    e = Evaluator(m, data_path=validationfile)
    e.run(exp_path+'validation150.h5', t.bs)


def upper_bound_wo_gt_eval():
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    exp_path = results_dir + 'Unet_henormaltransposed_orig/'

    m = IsoNet(4, (16,64,64), simulate=True, from_groundtruth=False)
    m.unet_simple_spec(3, 32, 3)
    m.training_scheme()
    t = Trainer(m, exp_path, trainingfile, validationfile, cubic=False)

    loss = []
    diff_1_2_loss = []
    diff_1_3_loss = []
    diff_avg_1_loss = []
    diff_avg_1_2_loss = []
    diff_avg_1_3_loss = []
    diff_gt_mean_squared = []

    loss_val = []
    diff_1_2_loss_val = []
    diff_1_3_loss_val = []
    diff_avg_1_loss_val = []
    diff_avg_1_2_loss_val = []
    diff_avg_1_3_loss_val = []
    diff_gt_mean_squared_val = []
    for epoch in range(10, 151, 10):
        print("Epoch:", epoch)
        m.load_weights(exp_path+'weights{0:}.h5'.format(epoch))
        m.compile(1e-04, 10)
        l, d12, d13, da, da12, da13, _, dgt, lv, d12v, d13v, dav, da12v, da13v, _, dgtv = t.evaluate(100)
        loss.append(l)
        diff_1_2_loss.append(d12)
        diff_1_3_loss.append(d13)
        diff_avg_1_loss.append(da)
        diff_avg_1_2_loss.append(da12)
        diff_avg_1_3_loss.append(da13)
        diff_gt_mean_squared.append(dgt)
        loss_val.append(lv)
        diff_1_2_loss_val.append(d12v)
        diff_1_3_loss_val.append(d13v)
        diff_avg_1_loss_val.append(dav)
        diff_avg_1_2_loss_val.append(da12v)
        diff_avg_1_3_loss_val.append(da13v)
        diff_gt_mean_squared_val.append(dgtv)
    with open('upperbound.p', 'w') as f:
        pickle.dump((loss, diff_1_2_loss, diff_1_3_loss, diff_avg_1_loss, diff_avg_1_2_loss, diff_avg_1_3_loss,
                 diff_gt_mean_squared, loss_val, diff_1_2_loss_val, diff_1_3_loss_val, diff_avg_1_loss_val,
                 diff_avg_1_2_loss_val, diff_avg_1_3_loss_val, diff_gt_mean_squared_val),
                f)
    #for k in range(360):
    #for steps in range(30,361, 30):
        #print(k)
        #x, y =t.evaluate(1)
        #train.append(x)
        #val.append(y)
        #print([np.mean([out[i] for out in train]) for i in range(len(x))], [np.std([out[i] for out in train]) for i in
        #                                                                          range(len(x))])
        #print([np.mean([out[i] for out in val]) for i in range(len(y))], [np.std([out[i] for out in val]) for i in
        #                                                                         range(len(y))])


def generate_evaluation():
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    exp_path = '/nrs/saalfeld/heinrichl/results_keras/Unet3-32-2_wogt_10cubic/finetuning_avg10weights_lrs1/'
    model = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=True)
    model.unet_simple_spec(3,32,2)
    model.training_scheme()
    model.load_weights(exp_path+'weights20.h5')
    e = Evaluator(model, data_path=validationfile)
    e.run(exp_path+'validation20.h5', 6)


def evaluate_prediction():
    groundtruthfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    predictionfile = '/nrs/saalfeld/heinrichl/results_keras/Unet3-32-2_wogt_10cubic/finetuning_avg10weights_lrs1' \
                     '/validation30.h5'
    gt_arr = h5py.File(groundtruthfile, 'r')['raw']
    pred_arr = h5py.File(predictionfile, 'r')['raw']
    gt_arr = utils.cut_to_sc(gt_arr, 4, 0)
    pred_arr = utils.cut_to_sc(pred_arr, 4, 0)
    border = utils.get_bg_borders(pred_arr)
    gt_arr = utils.cut_to_size(gt_arr, border)
    pred_arr = utils.cut_to_size(pred_arr, border)
    print(utils.compute_psnr(pred_arr, gt_arr))
    print(utils.compute_wpsnr(pred_arr, gt_arr))


def training_fsrcnn_simulated_w_gt():
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    name = 'FSRCNN_henormaltransposed_nottrunc'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name)
    print("output results and logs to:", exp_path)
    d = 240
    s = 64
    m = 4
    mod = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=True)
    mod.fsrcnn_spec(d, s, m)
    mod.training_scheme()
    mod.compile(1e-04, 1)
    mod.save_json(exp_path+'model_def_json.txt')
    t = Trainer(mod, exp_path, trainingfile, validationfile, cubic=False)
    t.run(151)

    e = Evaluator(mod, data_path=validationfile)
    e.run(exp_path+'validation150.h5', t.bs)


def continue_fsrcnn_training():
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    name = 'FSRCNN_initfromold'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name)
    print("output results and logs to:", exp_path)
    d = 240; s=64; m=4
    mod = IsoNet(4,(16,64,64),simulate=True, from_groundtruth=True)
    mod.fsrcnn_spec(d,s,m)
    mod.training_scheme()
    mod.compile(1e-04, 1)
    mod.save_json(exp_path+'model_def_json.txt')
    mod.load_weights('/nrs/saalfeld/heinrichl/results_keras/longFSRCNN_d240_s64_m4_3868b61_lr-4_init5e-5'
                     '/weights02_keras2format.h5')
    t = Trainer(mod, exp_path, trainingfile, validationfile, cubic=False)
    t.run(151, start_epoch=2)


def continue_unet_training():
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    name = 'Unet3-32-2wogt_testwithnewinit/finetuning'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    exp_path = results_dir + name + '/'
    #exp_path = utils.get_exppath(results_dir, exp_name=name)
    print("output results and logs to:", exp_path)
    mod = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=False)
    mod.unet_simple_spec(3, 32, 2)
    mod.training_scheme()
    mod.compile(1e-04, 1)
    #mod.save_json(exp_path+'model_def_json.txt')
    mod.load_weights('/nrs/saalfeld/heinrichl/results_keras/Unet3-32-2wogt_testwithnewinit/finetuning/weights80.h5')
    t = Trainer(mod, exp_path, trainingfile, validationfile, cubic=False)
    t.run(151, start_epoch=80)


def training_unet_simulated_wo_gt():
    results_dir ='/nrs/saalfeld/heinrichl/results_keras/'
    name_cubic = 'Unet3-32-3_wogt_10cubic'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name_cubic)#results_dir+name_cubic+'/'
    #'/nrs/saalfeld/heinrichl/results_keras/Unetwogt_testwithnewinit0007/'#
    finetune_exp_path = exp_path + 'finetuning_avg10weight/'
    os.mkdir(finetune_exp_path)
    print("output results and logs to:", exp_path)

    pretraining_epochs = 11
    finetuning_epochs = 140
    h = 3
    w = 32
    d = 3

    m_cubic = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=True)
    m_cubic.unet_simple_spec(h, w, d)
    m_cubic.training_scheme()
    m_cubic.compile(1e-04, 10)
    m_cubic.save_json(exp_path+'pretrain_model_def_json.txt')

    t_cubic = Trainer(m_cubic, exp_path, trainingfile, validationfile, cubic=True)
    t_cubic.run(pretraining_epochs)

    e_cubic = Evaluator(m_cubic, data_path=validationfile)
    e_cubic.run(exp_path + 'cubicvalidation{0:}.h5'.format(pretraining_epochs), t_cubic.bs)

    m = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=False)
    m.unet_simple_spec(h, w, d)
    m.training_scheme()
    m.compile(1e-04, 10)
    m.save_json(finetune_exp_path + 'model_def_json.txt')
    m.load_weights(exp_path+'weights{0:}.h5'.format(pretraining_epochs-1))

    t = Trainer(m, finetune_exp_path, trainingfile, validationfile)
    t.run(pretraining_epochs + finetuning_epochs, start_epoch=pretraining_epochs)

    m_cubic.load_weights(finetune_exp_path + 'weights{0:}.h5'.format(pretraining_epochs + finetuning_epochs - 1))
    e_finetuned = Evaluator(m_cubic, data_path=validationfile)
    e_finetuned.run(finetune_exp_path + 'validation{0:}.h5'.format(pretraining_epochs + finetuning_epochs), t.bs)


def training_unet_from_cubic():
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    name_cubic = 'Unet3-32-3_wogt_10cubic'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = results_dir + name_cubic + '/'
    finetune_exp_path = exp_path + 'finetuning_avg10weights_lrs1/'
    os.mkdir(finetune_exp_path)
    print("output results and logs to:", exp_path)

    pretraining_epochs = 11
    finetuning_epochs = 140
    h = 3
    w = 32
    d = 3

    m = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=False)
    m.unet_simple_spec(h, w, d)
    m.training_scheme()
    m.compile(1e-04, 1)
    m.save_json(finetune_exp_path+'model_def_json.txt')
    m.load_weights(exp_path+'weights{0:}.h5'.format(pretraining_epochs-1))

    t = Trainer(m, finetune_exp_path, trainingfile, validationfile)
    t.run(pretraining_epochs + finetuning_epochs, start_epoch=pretraining_epochs)

    m_pred = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=True)
    m_pred.unet_simple_spec(h, w, d)
    m_pred.training_scheme()
    m_pred.compile(1e-04, 10)
    m_pred.load_weights(finetune_exp_path + 'weights{0:}.h5'.format(pretraining_epochs + finetuning_epochs - 1))
    e_pred = Evaluator(m_pred, data_path=validationfile)
    e_pred.run(finetune_exp_path + 'validation{0:}.h5'.format(pretraining_epochs + finetuning_epochs - 1), t.bs)



def prediction():
    epoch=50
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    exp_path='/nrs/saalfeld/heinrichl/results_keras/Unetwogt_testwithnewinit0007/'
    mod = IsoNet(4, (16,64,64), simulate=True, from_groundtruth=True)
    mod.unet_simple_spec(3,32,3)
    mod.training_scheme()
    mod.compile(1e-04, 10)
    mod.load_weights(exp_path+'weights{0:}.h5'.format(epoch))
    e = Evaluator(mod, data_path=validationfile)
    e.run(exp_path+'cubicvalidation{0:}.h5'.format(epoch), 6)

def patches():
    #def nn(arr, scale, axis):
    #    scale_nd = np.ones(arr.ndim)
    #    scale_nd[axis] = scale
    #    new_shape = np.array(arr.shape)*scale_nd
    #    arr_new = np.zeros(new_shape)
    #    for k in range(arr.shape[axis]):
    #        sliceobj = tuple([slice(None)]*axis+[slice(k, k+1, None)])
    #        arr_new[sliceobj] = np.repeat()

    import scipy.ndimage
    import scipy.misc
    expno=8
    im = scipy.ndimage.imread('/groups/saalfeld/saalfeldlab/posters/miccai-2017/with_groundtruth/exp{0:}_new/gt.png'
                              ''.format(expno)
                              )[:,
         :,0]/255.
    print(im.shape)
    print(np.max(im), np.min(im))
    im_down = utils.downscale_manually(im, 4, 0)
    im_cubic = utils.cubic_up(im_down, 4, 0)
    im_nn = np.repeat(im_down, 4, 0)
    print(im_down.shape)
    print(im_cubic.shape)
    print(im_nn.shape)
    scipy.misc.imsave('/groups/saalfeld/saalfeldlab/posters/miccai-2017/with_groundtruth/exp{'
                      '0:}_new/cubic.png'.format(expno), im_cubic)
    scipy.misc.imsave('/groups/saalfeld/saalfeldlab/posters/miccai-2017/with_groundtruth/exp{'
                      '0:}_new/nn.png'.format(expno), im_nn)
if __name__ == '__main__':
    #prediction()
    #training_unet_simulated_wo_gt()
    #training_unet_from_cubic()
    #patches()
    generate_evaluation()
    #evaluate_prediction()
    #continue_unet_training()
    #training_fsrcnn_simulated_w_gt()
    #continue_fsrcnn_training()