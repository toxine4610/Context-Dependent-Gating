import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import pickle
import os, sys

NUM_TASKS = 20

def load_data(subfolder):

    variables = {}
    omegas = {}
    for fn in os.listdir('./analysis/saves/{}/'.format(subfolder)):
        if 'var' in fn:
            variables[fn[4:-4]] = np.load('./analysis/saves/{}/{}'.format(subfolder, fn))
        elif 'om' in fn:
            omegas[fn[3:-4]] = np.load('./analysis/saves/{}/{}'.format(subfolder, fn))

    return variables, omegas


def yield_tasks(dataset):
    keys = dataset.keys()
    for t in range(NUM_TASKS):
        offset = 3 if t < 10 else 4
        task_data = {key[offset:]:dataset[key] for key in keys if 't{}_'.format(t) in key}
        yield t, task_data


def scalar_bins(set):
    for t, d in yield_tasks(set):
        fig, ax = plt.subplots(3,2, sharex=True, sharey=True, figsize=[10,8])
        for n, e in enumerate(d.keys()):
            bins = np.linspace(np.min(d[e]), np.max(d[e]), 100)
            ax[n%3,n//3].hist(d[e].flatten(), bins, facecolor='g', alpha=0.75)
            ax[n%3,n//3].set_title('{}'.format(e))
            ax[n%3,n//3].set_yscale('symlog')

        plt.suptitle('Omega Distributions, Task {}'.format(t))
        plt.savefig('./analysis/plots/gated_om_t{}.png'.format(t))
        plt.clf()


def mean_setup(set):
    arrays = {}
    for t, d in yield_tasks(set):
        for k in d.keys():
            arrays[k] = arrays[k] + [d[k]] if k in arrays else [d[k]]

    for k in d.keys():
        arrays[k] = np.stack(arrays[k], axis=-1)

    return arrays


def mean_std(set, name, fn):
    arrays = mean_setup(set)

    fig, ax = plt.subplots(3,2, sharex=True, sharey=True, figsize=[10,8])

    for n, k in enumerate(arrays.keys()):
        #print(n, k)
        """
        if '2W' in k:
            variable_mean = np.mean(arrays[k], axis=-1)
            variable_std  = np.std(arrays[k], axis=-1)

            colors = plt.cm.get_cmap('magma', NUM_TASKS)
            for i in range(NUM_TASKS):
                ax[n%3,n//3].scatter(variable_mean[:,i], variable_std[:,i], s=1, c=colors(i), label=i)

            ax[n%3,n//3].set_title(k)
            ax[n%3,n//3].legend()

        else:
        """
        if True:
            variable_mean = np.mean(arrays[k], axis=-1).flatten()
            variable_std  = np.std(arrays[k], axis=-1).flatten()

            ax[n%3,n//3].scatter(variable_mean, variable_std, s=0.1, c='k')
            ax[n%3,n//3].set_title(k)

    ax[2,0].set_xlabel('Variable Mean')
    ax[2,1].set_xlabel('Variable Mean')

    ax[0,0].set_ylabel('Variable Std Dev')
    ax[1,0].set_ylabel('Variable Std Dev')
    ax[2,0].set_ylabel('Variable Std Dev')

    plt.suptitle('$\omega$ Means vs. Std Dev for\n{} Variables Across 10 Tasks'.format(name))
    #plt.savefig('./analysis/plots/scatter_std_{}.png'.format(fn))
    #plt.clf()
    plt.show()


def histograms():
    for fn in ['gated10', 'ungated10']:
        variables, omegas = load_data(fn)
        arrays = mean_setup(omegas)
        for k in arrays.keys():
            if not '2W' in k:
                pass
            else:
                #print(k, arrays[k].shape)
                data = np.transpose(np.array(arrays[k]),[1,0,2])
                #print(data.shape)
                #quit()
                max = np.max(data)
                bins = 30
                binning = np.linspace(0,max,bins)

                for i in range(20):
                    fig, ax = plt.subplots(4,3, sharex=True, sharey=True, figsize=[10,8])
                    for j in range(10):
                        ax[j%4,j//4].hist(data[i,j,:], bins=binning)
                        ax[j%4,j//4].set_title('Synapse ({},{})'.format(i,j))
                    plt.savefig('./analysis/{}_synapse_hist{}.png'.format(fn, i))
                    plt.clf()


def mnist_csweep_analysis():
    fig, ax = plt.subplots(3,2, sharex=True, sharey=True, figsize=[10,8])

    for g, n, col, counts in zip(['Gated', 'Ungated', 'Ungated (Small $c \\leq 0.02$)'], [3,4,5], ['r','g','b'], [11, 11, 11]):
        x = pickle.load(open('./savedir/mnist_csweep{}.pkl'.format(n), 'rb'))
        v = 0


        for c_id, c in enumerate(np.linspace(0, 0.1, counts)):
            key = 'c{}_v{}'.format(c_id, v)
            omega_c = x[key]['par']['omega_c']
            accuracy = x[key]['accuracy'][-1]

            norms = {}
            mean_norm = []
            for i, k in enumerate(x[key]['task_records']['norms'].keys()):
                norms[k] = x[key]['task_records']['norms'][k]
                mean_norm.append(np.mean(x[key]['task_records']['norms'][k]))
                #mean_norm = np.mean(mean_norm)

                ax[i%3,i//3].scatter(np.mean(norms[k]), accuracy, c=col, s=5)
                ax[i%3,i//3].set_title(k)

            if c_id == 0:
                ax[2,1].scatter(0,1, c=col, s=5, label=g)


        # Repeats last point but adds label without duplication
        #ax[0,0].scatter(mean_norm, accuracy, c=col, s=5, label=g)

    ax[0,0].set_ylabel('Accuracy')
    ax[1,0].set_ylabel('Accuracy')
    ax[2,0].set_ylabel('Accuracy')
    ax[2,0].set_xlabel('Mean Norm')
    ax[2,1].set_xlabel('Mean Norm')

    plt.legend()
    plt.show()


    """
        ax[0].scatter(c, accuracy, label=c_id, c='b', s=5)
        ax[1].scatter(c, mean_norm, label=c_id, c='g', s=5)

    ax[0].set_title('Accuracies Across $c$')
    ax[0].set_xlabel('$c$')
    ax[0].set_ylabel('Accuracy')

    ax[1].set_title('Mean Norms Across $c$')
    ax[1].set_xlabel('$c$')
    ax[1].set_ylabel('Average Norm')
    ax[1].set_yscale('symlog')
    ax[1].set_ylim(2e-5,3e-4)
    ax[1].set_yticks([2e-5,5e-5,1e-4,2e-4,3e-4])

    plt.grid(True)
    plt.show()
    """



mnist_csweep_analysis()
