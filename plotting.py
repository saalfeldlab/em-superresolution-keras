import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
def fsrcnn_table():
    data = dict()

    data['d240_s48_m2'] = dict()
    data['d240_s48_m2']['PSNR'] = [34.04,34.62,34.78]
    data['d240_s48_m2']['nP'] = [1.8e06]*3

    data['d240_s48_m3'] = dict()
    data['d240_s48_m3']['PSNR'] = [34.03,34.71,35.02]
    data['d240_s48_m3']['nP'] = [2.4e06]*3

    data['d240_s48_m4'] = dict()
    data['d240_s48_m4']['PSNR'] = [34.05,34.40,34.49]
    data['d240_s48_m4']['nP'] = [2.9e06]*3

    data['d240_s64_m2'] = dict()
    data['d240_s64_m2']['PSNR'] = [34.31,34.62,34.78]
    data['d240_s64_m2']['nP'] = [2.8e06]*3

    data['d240_s64_m3'] = dict()
    data['d240_s64_m3']['PSNR'] = [34.20,34.45,34.58]
    data['d240_s64_m3']['nP'] = [3.7e06]*3

    data['d240_s64_m4'] = dict()
    data['d240_s64_m4']['PSNR'] = [34.52,35.02,35.27]
    data['d240_s64_m4']['nP'] = [4.7e06]*3

    data['d280_s48_m2'] = dict()
    data['d280_s48_m2']['PSNR'] = [33.64,34.50,34.66]
    data['d280_s48_m2']['nP'] = [2.0e06]*3

    data['d280_s48_m3'] = dict()
    data['d280_s48_m3']['PSNR'] = [34.03,34.66,34.81]
    data['d280_s48_m3']['nP'] = [2.6e06]*3

    data['d280_s48_m4'] = dict()
    data['d280_s48_m4']['PSNR'] = [34.12,34.57,34.75]
    data['d280_s48_m4']['nP'] = [3.1e06]*3

    data['d280_s64_m2'] = dict()
    data['d280_s64_m2']['PSNR'] = [34.34,34.41,34.77]
    data['d280_s64_m2']['nP'] = [2.9e06]*3

    data['d280_s64_m3'] = dict()
    data['d280_s64_m3']['PSNR'] = [34.34,34.91,35.08]
    data['d280_s64_m3']['nP'] = [3.9e06]*3

    data['d280_s64_m4'] = dict()
    data['d280_s64_m4']['PSNR'] = [33.93,34.38,35.06]
    data['d280_s64_m4']['nP'] = [4.9e06]*3


    colors = ['cyan', 'orange', 'crimson']
    symbols = ['x', 'o']
    linestyles = ['--', '-']

    for k,m in enumerate([2,3,4]):
        c = colors[k]
        for l,d in enumerate([240, 280]):
            symbol = symbols[l]
            for i,s in enumerate([48,64]):
                ls = linestyles[i]
                dictstr = 'd{0:}_s{1:}_m{2:}'.format(d,s,m)
                plt.plot(data[dictstr]['nP'], data[dictstr]['PSNR'],marker=symbol, color=c, ls=ls)


    legend = []
    for k,m in enumerate([2,3,4]):
        c = colors[k]
        legend.append(mpatches.Patch(color=c, label="m = {}".format(m)))
    for l, d in enumerate([240, 280]):
        symbol = symbols[l]
        legend.append(mlines.Line2D([], [], color='k', marker=symbol, label="d = {}".format(d)))

    for i, s in enumerate([48,64]):
        legend.append(mlines.Line2D([], [], color='k', ls = linestyles[i], label="s = {}".format(s)))
    plt.legend(handles=legend, prop={'size':25})
    plt.xlabel('# parameters', fontsize=25)
    plt.ylabel('PSNR', fontsize=25)
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    plt.show()


def unet_getlosses():
    import cPickle as pickle
    losses = dict()
    k=0
    ep_nos = [149,149,149,145,149,149,149,126,149,149,149,129]
    for n_l in [2,3,4]:
        for n_f in [32, 64]:
            for n_c in [2,3]:
                with open('/groups/saalfeld/home/heinrichl/Projects/results_keras/longUnet_nl{0:}_nf{1:}_nc{'
                          '2:}_3868b61_scheduler10/loss_history{3:}.p'.format(n_l,n_f,n_c,ep_nos[k]), 'r') as f:
                    losses['n_l{0:}_nf{1:}_nc{2:}'.format(n_l,n_f,n_c)]=pickle.load(f)
                k+=1
    return losses


def fsrcnn_getvallosses():
    import cPickle as pickle
    losses = dict()
    for m in [2,3,4]:
        for d in [240, 280]:
            for s in [48,64]:
                with open('/groups/saalfeld/home/heinrichl/Projects/results_keras/longFSRCNN_d{0:}_s{1:}_m{'
                          '2:}_3868b61_lr-4_init5e-5/checkpointer.p'.format(d,s,m), 'r') as f:
                    losses['d{0:}_s{1:}_m{2:}'.format(d,s,m)]=pickle.load(f)
    return losses


def unet_getvallosses():
    import cPickle as pickle
    losses = dict()
    for n_l in [2,3,4]:
        for n_f in [32, 64]:
            for n_c in [2,3]:
                with open('/groups/saalfeld/home/heinrichl/Projects/results_keras/longUnet_nl{0:}_nf{1:}_nc{'
                          '2:}_3868b61_scheduler10/checkpointer.p'.format(n_l,n_f,n_c), 'r') as f:
                    losses['nl{0:}_nf{1:}_nc{2:}'.format(n_l,n_f,n_c)]=pickle.load(f)
    return losses

def fsrcnn_getlosses():
    import cPickle as pickle
    losses = dict()
    for m in [2,3,4]:
        for d in [240, 280]:
            for s in [48,64]:
                with open('/groups/saalfeld/home/heinrichl/Projects/results_keras/longFSRCNN_d{0:}_s{1:}_m{'
                          '2:}_3868b61_lr-4_init5e-5/loss_history149.p'.format(d,s,m), 'r') as f:
                    losses['d{0:}_s{1:}_m{2:}'.format(d,s,m)]=pickle.load(f)
    return losses


def convert_to_psnr(losses):
    import numpy as np
    psnr_losses = dict()
    for key in losses.iterkeys():
        psnr_losses[key] = -10 * np.log10(losses[key])
    return psnr_losses


def plot_trainingcurves(losses, adaptrate=1, log = False):
    import pandas as pd
    import numpy as np
    for key, value in sorted(losses.iteritems()):
        plt.plot(pd.rolling_mean(np.array(value), 22*22), label=key)
    for adapt in range(0,151,adaptrate):
        plt.axvline(x=22*22*adapt, color='k', linestyle='--', linewidth=.2)
    for snapshots in [50,100,150]:
        plt.axvline(x=22*22*snapshots, color='r', linestyle='--', linewidth = .5)
    plt.legend(prop= {'size': 25},loc=4)
    if log:
        plt.semilogy()
    plt.xlabel('# iterations', fontsize=25)
    plt.ylabel('PSNR', fontsize=25)
    ax = plt.gca()
    ax.grid()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(20)

    plt.show()


def fsrcnn_plot_training_and_validationcurves(losses, adaptrate, log=False):
    import pandas as pd
    import numpy as np
    import cPickle as pickle
    train_losses = convert_to_psnr(fsrcnn_getlosses())
    val_losses = fsrcnn_getvallosses()
    for d in [240,280]:
        for s in [48,64]:
            plt.figure()
            plt.title('d: {0:}, s:{1:}'.format(d,s))

            for m, c in zip([2,3,4], ['cyan', 'orange', 'crimson']):
                print(d,s,m)
                print(val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['cp'])
                #plt.plot(pd.rolling_mean(np.array(train_losses['d{0:}_s{1}_m{2}'.format(d,s,m)]), 22*22),
                #         label='train m = '+str(m), color=c)

                for mode,ls in zip(['training_subset', 'test', 'validation'], ['-', '-.', '--']):
                    plt.plot([cp * 22 * 22 for cp in val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['cp']],
                            [-p for p in val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)][mode]],
                            label=mode+' m = ' + str(m), color=c, linestyle=ls)
            for y, ls in zip([32.79,33.22, 33.20], ['-', '-.', '--']):
                plt.axhline(y=y, color='k', linestyle=ls)

            plt.legend(prop={'size': 25}, loc=4)
            for adapt in range(0, 151, adaptrate):
                plt.axvline(x=22 * 22 * adapt, color='k', linestyle='--', linewidth=.2)
            for snapshots in [50, 100, 150]:
                plt.axvline(x=22 * 22 * snapshots, color='r', linestyle='--', linewidth=.5)
            if log:
                plt.semilogy()
            plt.xlabel('# iterations', fontsize=25)
            plt.ylabel('PSNR', fontsize=25)
            ax = plt.gca()

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
    plt.show()
def unet_plot_training_and_validationcurves(losses, adaptrate, log=False):
    import pandas as pd
    import numpy as np
    import cPickle as pickle
    train_losses = convert_to_psnr(fsrcnn_getlosses())
    val_losses = unet_getvallosses()
    for n_f in [32,64]:
        for n_c in [2,3]:
            plt.figure()
            plt.title('n_f: {0:}, n_c:{1:}'.format(n_f,n_c))

            for n_l, c in zip([2,3,4], ['cyan', 'orange', 'crimson']):
                print(n_l,n_f,n_c)
                print(val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['cp'])
                #plt.plot(pd.rolling_mean(np.array(train_losses['d{0:}_s{1}_m{2}'.format(d,s,m)]), 22*22),
                #         label='train m = '+str(m), color=c)

                for mode,ls in zip(['training_subset', 'test', 'validation'], ['-', '-.', '--']):
                    plt.plot([cp * 22 * 22 for cp in val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['cp']],
                            [-p for p in val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f,n_c)][mode]],
                            label=mode+' n_l = ' + str(n_l), color=c, linestyle=ls)
            for y, ls in zip([32.79,33.22, 33.20], ['-', '-.', '--']):
                plt.axhline(y=y, color='k', linestyle=ls)

            plt.legend(prop={'size': 25}, loc=4)
            for adapt in range(0, 151, adaptrate):
                plt.axvline(x=22 * 22 * adapt, color='k', linestyle='--', linewidth=.2)
            for snapshots in [50, 100, 150]:
                plt.axvline(x=22 * 22 * snapshots, color='r', linestyle='--', linewidth=.5)
            if log:
                plt.semilogy()
            plt.xlabel('# iterations', fontsize=25)
            plt.ylabel('PSNR', fontsize=25)
            ax = plt.gca()

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(20)
    plt.show()

def unet_findmax():
    import numpy as np
    n_l=4
    n_f=64
    n_c=3


    val_losses = unet_getvallosses()

    indices1 = val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['cp'].index(90)
    indices2 = val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['cp'].index(100)
    print(indices1, indices2)
    max_idx = np.argmin(val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['validation'][
                        indices1:indices2+1])+indices1
    print(max_idx)
    print(val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['cp'][max_idx])
    print("training_subset",val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['training_subset'][max_idx],
          val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['training_subset_wPSNR'][max_idx])
    print("validation", val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['validation'][max_idx],
          val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['validation_wPSNR'][max_idx])
    print("test", val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['test'][max_idx],
          val_losses['nl{0:}_nf{1}_nc{2}'.format(n_l, n_f, n_c)]['test_wPSNR'][max_idx])


def fsrcnn_findmax():
    import numpy as np
    d=240
    s=64
    m=3
    val_losses = fsrcnn_getvallosses()

    indices1 = val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['cp'].index(90)
    indices2 = val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['cp'].index(100)
    print(indices1, indices2)
    max_idx = np.argmin(val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['validation'][indices1:indices2+1])+indices1
    print(max_idx)
    print(val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['cp'][max_idx])
    print("training_subset",val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['training_subset'][max_idx],
          val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['training_subset_wPSNR'][max_idx])
    print("validation", val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['validation'][max_idx],
          val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['validation_wPSNR'][max_idx])
    print("test", val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['test'][max_idx],
          val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['test_wPSNR'][max_idx])
    #print(val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['cp'][indices1:indices2])
    #print(np.max(val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['training_subset'][indices1:indices2]))


    #print(val_losses['d{0:}_s{1}_m{2}'.format(d, s, m)]['validation'][indices1:indices2])

if __name__=='__main__':
    unet_plot_training_and_validationcurves(0,10)
    #unet_findmax()
    #plot_trainingcurves(convert_to_psnr(unet_getlosses()), adaptrate=10)
    #plot_trainingcurves(convert_to_psnr(fsrcnn_getlosses()))
