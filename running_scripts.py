from __future__ import print_function

import numpy as np
import os

from define_model import *
from trainer import *
import utils
from evaluator import *


def training_unet_simulated_w_gt():
    results_dir = '/nrs/saalfeld/heinrichl/results_keras/'
    name = 'Unet_henormaltransposed_orig'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name)
    print("output results and logs to:", exp_path)

    m = IsoNet(4, (16, 64, 64), simulate=True, from_groundtruth=True)
    m.unet_simple_spec(3, 32, 3)
    m.training_scheme()
    m.compile(1e-04, 10)
    m.save_json(exp_path+'model_def_json.txt')

    t = Trainer(m, exp_path, trainingfile, validationfile, cubic=False)
    t.run(151)

    e = Evaluator(m, data_path=validationfile)
    e.run(exp_path+'validation150.h5', t.bs)


def generate_evaluation():
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    exp_path = '/nrs/saalfeld/heinrichl/results_keras/FSRCNN_deleteme/'
    model = IsoNet(4, (16,64,64), simulate=True, from_groundtruth=True)
    model.fsrcnn_spec(240,64,4)
    model.training_scheme()
    model.load_weights(exp_path+'weights06.h5')
    e = Evaluator(model, data_path=validationfile)
    e.run(exp_path+'validation06.h5', 6)


def evaluate_prediction():
    groundtruthfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'
    predictionfile = '/nrs/saalfeld/heinrichl/results_keras/FSRCNN_defaultinit/validation49.h5'
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
    name = 'Unet_orig_initfromold'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name)
    print("output results and logs to:", exp_path)
    mod = IsoNet(4,(16,64,64),simulate=True, from_groundtruth=True)
    mod.unet_simple_spec(3,32,3)
    mod.training_scheme()
    mod.compile(1e-04, 1)
    mod.save_json(exp_path+'model_def_json.txt')
    mod.load_weights('/nrs/saalfeld/heinrichl/results_keras/longUnet_nl3_nf32_nc3_3868b61_scheduler10'
                     '/weights02_keras2format.h5')
    t = Trainer(mod, exp_path, trainingfile, validationfile, cubic=False)
    t.run(151, start_epoch=2)


def training_unet_simulated_wo_gt():
    results_dir ='/nrs/saalfeld/heinrichl/results_keras/'
    name_cubic = 'Unetwogt_test_keras2'
    trainingfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/training.h5'
    validationfile = '/nrs/saalfeld/heinrichl/SR-data/FIBSEM/downscaled/bigh5-16isozyx/validation.h5'

    exp_path = utils.get_exppath(results_dir, exp_name=name_cubic)
    finetune_exp_path = exp_path + 'finetuning/'
    os.mkdir(finetune_exp_path)
    print("output results and logs to:", exp_path)

    pretraining_epochs = 51
    finetuning_epochs = 100
    h = 4
    w = 64
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
    m.save_json(finetune_exp_path + 'pretrain_model_def_json.txt')
    m.load_weights(exp_path+'weights{0:}.h5'.format(pretraining_epochs-1))

    t = Trainer(m, finetune_exp_path)
    t.run(pretraining_epochs + finetuning_epochs, start_epoch=pretraining_epochs)

    m_cubic.load_weights(finetune_exp_path + 'weights{0:}.h5'.format(pretraining_epochs + finetuning_epochs - 1))
    e_finetuned = Evaluator(m_cubic, data_path=validationfile)
    e_finetuned.run(finetune_exp_path + 'validation{0:}.h5'.format(pretraining_epochs + finetuning_epochs))


if __name__ == '__main__':
    training_unet_simulated_w_gt()
    #continue_unet_training()
    #training_fsrcnn_simulated_w_gt()
    #continue_fsrcnn_training()