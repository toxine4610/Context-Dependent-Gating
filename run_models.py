import tensorflow as tf
import numpy as np

np.random.seed(97813)

from parameters import *
import model
import sys, os
import pickle

def try_model(save_fn,gpu_id,taskrange):

    try:
        # Run model
        return model.main(save_fn, gpu_id, taskrange)
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')

###############################################################################
###############################################################################
###############################################################################


mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

csweep_mnist_updates = {
    'layer_dims'            : [784, 2000, 2000, 10],
    'n_tasks'               : 2,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 5000, #5*3906,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

small_mnist_updates = {
    'layer_dims'            : [784, 400, 400, 10],
    'n_tasks'               : 3,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 100,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

mnist_updates = {
    'layer_dims'            : [784, 400, 400, 10],
    'n_tasks'               : 12,
    'task'                  : 'mnist',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 2000,
    'drop_keep_pct'         : 0.5,
    'input_drop_keep_pct'   : 1.0,
    'multihead'             : False
    }

cifar_updates = {
    'layer_dims'            : [4096, 1000, 1000, 5],
    'n_tasks'               : 20,
    'task'                  : 'cifar',
    'save_dir'              : './savedir/',
    'n_train_batches'       : 977,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

imagenet_updates = {
    'layer_dims'            : [4096, 2000, 2000, 10],
    'n_tasks'               : 100,
    'task'                  : 'imagenet',
    'save_dir'              : './savedir/ImageNet/',
    'n_train_batches'       : 977*2,
    'input_drop_keep_pct'   : 1.0,
    'drop_keep_pct'         : 0.5,
    'multihead'             : False
    }

# updates for multi-head network, cifar only
multi_updates = {'layer_dims':[4096, 1000, 1000, 100], 'multihead': True}
imagenet_multi_updates = {'layer_dims':[4096, 2000, 2000, 1000], 'multihead': True}


# updates for split networks
mnist_split_updates = {'layer_dims':[784, 3665, 3665, 10], 'multihead': False}
cifar_split_updates = {'layer_dims':[4096, 1164, 1164, 5], 'multihead': False}
imagenet_split_updates = {'layer_dims':[4096, 3665, 3665, 10], 'multihead': False}


def recurse_best(data_dir, prefix):


    # Get filenames
    name_and_data = []
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix):
            x = pickle.load(open(data_dir + full_fn, 'rb'))
            name_and_data.append((full_fn, x['accuracy_full'][-1], x['par']['omega_c']))

    # Find number of c's and v's
    cids = []
    vids = []
    for (f, _, _) in name_and_data:
        if f[-9].isdigit():
            c = f[-9:-7]
        else:
            c = f[-8]
        if c == 'R':
            # don't look at existing files generated by this function
            print('Ignoring ', f)
            continue
        if c not in cids:
            cids.append(c)
        if f[-5] not in vids:
            vids.append(f[-5])

    print(name_and_data)
    print(cids)
    print(vids)

    # Scan across c's and v's for accuracies
    accuracies = np.zeros((len(cids)))
    count = np.zeros((len(cids)))
    omegas = np.zeros((len(cids)))
    cids = sorted(cids)
    vids = sorted(vids)

    for (c_id, v_id) in product(range(len(cids)), range(len(vids))):
        text_c = 'omega'+str(cids[c_id])
        text_v = '_v'+str(vids[v_id])
        for full_fn in os.listdir(data_dir):
            if full_fn.startswith(prefix) and text_c in full_fn and text_v in full_fn:
                print('c_id', c_id)
                x = pickle.load(open(data_dir + full_fn, 'rb'))
                accuracies[int(c_id)] += x['accuracy_full'][-1]
                count[int(c_id)] += 1
                omegas[int(c_id)] = x['par']['omega_c']

    accuracies /= count
    print('accuracies ', accuracies)

    ind_sorted = np.argsort(accuracies)
    print('Sorted ind ', ind_sorted)
    if ind_sorted[-1] > ind_sorted[-2] or ind_sorted[-1] == len(ind_sorted)-1: # to the right
        cR = (omegas[ind_sorted[-1]] + omegas[ind_sorted[-1]-1])/2
    else:
        cR = (omegas[ind_sorted[-1]] + omegas[ind_sorted[-1]+1])/2

    print('omegas ', omegas)
    print('cR = ', cR)


    # Get optimal parameters
    for full_fn in os.listdir(data_dir):
        if full_fn.startswith(prefix) and 'omega'+cids[ind_sorted[0]] in full_fn:
            opt_pars = pickle.load(open(data_dir + full_fn, 'rb'))['par']

    # Update parameters and run versions

    update_parameters(opt_pars)
    update_parameters({'omega_c' : cR})
    for i in range(5):
        save_fn = prefix + '_omegaR_v' + str(i) + '.pkl'
        print('save_fn', save_fn)
        print('save_dir', opt_pars['save_dir'])
        try_model(save_fn, sys.argv[1])
        print(save_fn, cR)


def run_base():
    update_parameters(imagenet_updates)
    update_parameters({'gating_type': None,'gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_c': 0.0, 'omega_xi': 0.01})

    update_parameters({'reset_weights': True})
    for i in range(0,5):
        save_fn = 'imagenet_weight_reset_v' + str(i) + '.pkl'
        #save_fn = 'imagenet_base_omega0_v' + str(i) + '.pkl'
        try_model(save_fn, gpu_id)

    update_parameters(imagenet_multi_updates)
    for i in range(1,0):
        save_fn = 'imagenet_baseMH_v' + str(i) + '.pkl'
        try_model(save_fn, gpu_id)

def run_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': None,'gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(1,0):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

    update_parameters(imagenet_multi_updates)
    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_MH_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_partial_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'partial','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(1,5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_partial_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_XdG_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(0,1):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_XdG_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_split_SI():

    omegas = [0.2, 0.5, 1, 2, 5]

    update_parameters(imagenet_updates)
    update_parameters(imagenet_split_updates)
    update_parameters({'gating_type': 'split','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(1,len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_SI_split_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_split_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200]

    update_parameters(imagenet_updates)
    update_parameters(imagenet_split_updates)
    update_parameters({'gating_type': 'split','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'pathint', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(0,1):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_split_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_partial_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'partial','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'EWC', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_partial_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

def run_XdG_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'EWC', 'omega_xi': 0.01})

    for i in range(5):
        for j in range(len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_XdG_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)


def run_EWC():

    omegas = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    update_parameters(imagenet_updates)
    update_parameters({'gating_type': None,'gate_pct': 0.80, 'input_drop_keep_pct': 1.0, \
        'stabilization': 'EWC', 'omega_xi': 0.01})

    for i in range(1,0):
        for j in range(len(omegas)-1, len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)

    update_parameters(imagenet_multi_updates)
    for i in range(5):
        for j in range(len(omegas)-1, len(omegas)):
            update_parameters({'omega_c': omegas[j]})
            save_fn = 'imagenet_EWC_MH_omega' + str(j) + '_v' + str(i) + '.pkl'
            try_model(save_fn, gpu_id)


def run_csweep():
    update_parameters(csweep_mnist_updates)
    update_parameters({'gating_type' : None, 'gate_pct' : 0.0, 'omega_xi' : 0.01})

    gen_gating()
    try_model('baseline_savefn', gpu_id, range(0,1))

    results = {}
    for i in range(1):
        for c_id, c in enumerate(np.linspace(0, 0.002, 11)):

            np.random.seed(97813)
            update_parameters({'omega_c' : c})
            tf.set_random_seed(47789)

            save_fn = 'mnist_pathint_csweep.pkl'
            results['c{}_v{}'.format(c_id, i)] = try_model(save_fn, gpu_id, range(1,2))

    pickle.dump(results, open(par['save_dir']+'mnist_csweep5.pkl', 'wb'))


# Second argument will select the GPU to use
# Don't enter a second argument if you want TensorFlow to select the GPU/CPU
try:
    gpu_id = sys.argv[1]
    print('Selecting GPU ', gpu_id)
except:
    gpu_id = None


#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_EWC_split')
#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_SI_split')
#run_EWC()
#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_EWC_omega')
#recurse_best('/home/masse/Context-Dependent-Gating/savedir/ImageNet/', 'imagenet_SI_XdG')

#run_EWC()
#run_base()
#run_SI()
#run_split_EWC()
#run_split_SI()
#run_partial_SI()
#run_XdG_SI()
#run_partial_EWC()
#run_XdG_EWC()
run_csweep()

quit()




print('ImageNet - Synaptic Stabilization = SI - Gating = 80%')
update_parameters(imagenet_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.80, 'input_drop_keep_pct': 1.0})
update_parameters({'stabilization': 'pathint', 'omega_c': 1.0, 'omega_xi': 0.01})
update_parameters({'train_convolutional_layers': True})
save_fn = 'imagenet_SI.pkl'
try_model(save_fn, gpu_id)
quit()


print('MNIST - Synaptic Stabilization = SI - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'pathint', 'omega_c': 0.035, 'omega_xi': 0.01})
save_fn = 'mnist_SI.pkl'
try_model(save_fn, gpu_id)

print('MNIST - Synaptic Stabilization = EWC - Gating = 80%')
update_parameters(mnist_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.8, 'input_drop_keep_pct': 0.8})
update_parameters({'stabilization': 'EWC', 'omega_c': 10})
save_fn = 'mnist_EWC.pkl'
try_model(save_fn, gpu_id)

print('CIFAR - Synaptic Stabilization = SI - Gating = 75%')
update_parameters(cifar_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.75, 'input_drop_keep_pct': 1.0})
update_parameters({'stabilization': 'pathint', 'omega_c': 0.2, 'omega_xi': 0.01})
update_parameters({'train_convolutional_layers': True})
save_fn = 'cifar_SI.pkl'
try_model(save_fn, gpu_id)

print('CIFAR - Synaptic Stabilization = EWC - Gating = 75%')
update_parameters(cifar_updates)
update_parameters({'gating_type': 'XdG','gate_pct': 0.75, 'input_drop_keep_pct': 1.0})
update_parameters({'stabilization': 'EWC', 'omega_c': 10})
update_parameters({'train_convolutional_layers': False})
save_fn = 'cifar_EWC.pkl'
try_model(save_fn, gpu_id)
