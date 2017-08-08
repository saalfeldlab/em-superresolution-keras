import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_loss_values(loss_file):
    with open(loss_file, 'r') as f:
        return pickle.load(f)


def plot_loss(loss_file, psnr=True, log=False):
    loss_values = get_loss_values(loss_file)
    if psnr:
        loss_values = [-10*np.log10(v) for v in loss_values]
    plt.plot(loss_values)
    if log:
        plt.semilogy()
    plt.xlabel('iterations')
    if psnr:
        plt.ylabel('PSNR')
    else:
        plt.ylabel('MSE')
    plt.grid()


def plot_multiple_losses(list_of_loss_files, psnr=True, log=False, labels=None, list_of_iterations=None,
                         smoothing=22*22):
    if not isinstance(list_of_loss_files, list):
        list_of_loss_files = [list_of_loss_files]
    #if labels is not None:
    if not isinstance(labels, list):
       labels = [labels]
    if not isinstance(list_of_iterations, list):
        list_of_iterations = [list_of_iterations]
    else:
        if not any(isinstance(iterations, list) or iterations is None for iterations in list_of_iterations):
            list_of_iterations = [list_of_iterations]

    if len(labels) < len(list_of_loss_files):
        labels.extend([None]*(len(list_of_loss_files)-len(labels)))
    elif len(labels) > len(list_of_loss_files):
        raise ValueError("The given numbers of labels ({0:}) is larger than the given number of losses {1:})".format(
            len(labels), len(list_of_loss_files)))
    if len(list_of_iterations) < len(list_of_loss_files):
        print("yup")
        list_of_iterations.extend([None]*(len(list_of_loss_files)-len(list_of_iterations)))
    elif len(list_of_iterations) > len(list_of_loss_files):
        raise ValueError("list_of_iterations has more entries ({0:}) than the list_of_loss)files ({1:})".format(len(
            list_of_iterations), len(list_of_loss_files)))
    list_of_loss_values = list()
    for loss_file in list_of_loss_files:
        list_of_loss_values.append(get_loss_values(loss_file))
    for idx, iterations in enumerate(list_of_iterations):
        if iterations is None:
            list_of_iterations[idx] = range(1, len(list_of_loss_values[idx])+1)
    if psnr:
        list_of_loss_values = [[-10*np.log10(v) for v in loss_values] for loss_values in list_of_loss_values]
    if smoothing:
        list_of_loss_values = [pd.rolling_mean(np.array(loss_values), smoothing) for loss_values in list_of_loss_values]
    for iterations, loss_values, label in zip(list_of_iterations, list_of_loss_values, labels):
        plt.plot(iterations, loss_values, label=label)
    if log:
        plt.semilogy()
    if psnr:
        plt.ylabel('PSNR', fontsize=25)
    else:
        plt.ylabel('MSE', fontsize=25)

    plt.xlabel('iterations', fontsize=25)
    plt.legend(prop={'size': 25})

    ax = plt.gca()
    ax.grid()
    for ticklabels in (ax.get_xticklabels() + ax.get_yticklabels()):
        ticklabels.set_fontsize(20)
    plt.show()

