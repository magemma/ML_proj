"""
    train_classification: main file for training multiple nns
    for solving a classification task
    ML 2019 Project
    Dipartimento di Informatica Università di Pisa
    Authors: R. Manetti, G. Martini
    We declare that the content of this file is entirelly
    developed by the authors
"""

import csv
import numpy as np
from nn_utilities import *
from optimizer import get_optimizer
import argparse
import config
import os
import matplotlib.pylab as plt
from tqdm import tqdm

num_epochs = 5000

parser = argparse.ArgumentParser(
    description='Training and validation of neural network.')

parser.add_argument("--folds",
                    metavar="k",
                    default=5,
                    help="Number of folds for cross-validation")

parser.add_argument("--train-file",
                    required=True,
                    metavar="train_file",
                    help="path of train file")
                    
parser.add_argument("--optimizer",
                    required=True,
                    metavar="opt",
                    help="acronym for optimizer")

parser.add_argument("--extra",
                    metavar="extra",
                    default="",
                    help="extra text for output path")


parser.add_argument("--results-folder",
                    required=True,
                    metavar="results_folder",
                    help="folder where to save results file")

parser.add_argument("--plots-folder",
                    required=True,
                    metavar="plots_folder",
                    help="path of plots folder")
                    
parser.add_argument("--check-gradient",
                    action="store_true",
                    help="Enable gradient checking")

args = vars(parser.parse_args())
#print(args)
  
if not os.path.exists(args['plots_folder']+"/"):
    os.makedirs(args['plots_folder']+"/")

if not os.path.exists(args['results_folder']+"/"):
    os.makedirs(args['results_folder']+"/")
        
with open(args['train_file'], 'r') as f:
    data_str = [x[0].split()[:-1] for x in csv.reader(f)]
    
#Convert to float
data = [tuple(map(int, x)) for x in data_str]

x_ori = np.array([np.array(l[1:]) for l in data])
x_one_hot = one_of_k(x_ori)
x = [np.array([l[1:]]).T for l in x_one_hot]
y = [np.array([l[0]]).T for l in data]

configs = config.configs[str(args['optimizer'])+str(args['extra'])]

#Write header of results file

configs[0]['k'] = args['folds']

with open(args['results_folder'] + "/" + args['extra'] +args['optimizer'] + ".csv", 'w') as f:
    configs[0]['score'] = 0
    w = csv.DictWriter(f, configs[0].keys())
    w.writeheader()

num_conf = 0

#For each configuration (of the grid search)
for config in configs:
    config['k'] = args['folds']
    #print("Running configuration " + str(num_conf) + "/" + str(len(configs)))
    model_score = 0
    #config['num_epochs'] = 0
    #config['loss'] = 1000
    
    #k-fold CV
    folds = cv(len(x), args['folds'], x, y)
    list_loss = [[] for e in range(num_epochs)]
    list_loss_val = [[] for e in range(num_epochs)]
    list_acc = [[] for e in range(num_epochs)]
    list_acc_val = [[] for e in range(num_epochs)]
    #print(len(list_loss))
    
    #Setting plots info
    fig, (ax1, ax2) = plt.subplots(2 , 1)
    
    for fold in tqdm(range(len(folds))):
        #print("Fold {}".format(fold))
        x_train = folds[fold]['x_tr']
        y_train = folds[fold]['y_tr']
        x_val = folds[fold]['x_val']
        y_val = folds[fold]['y_val']

        network = nn(config, len(x_train[0]), len(y_train[0]))

        #Obs: the bias in the input vector is added inside the function

        if args['check_gradient']:
            grad_check(x_train, y_train, network.weights, network.act_f)
        #print(type(config['optimizer']))
        optimizer = get_optimizer(config['optimizer'])

        #for epoch in range(config['num_epochs']):
        #removed the number of epochs from grid search
        #loss_data = {}
        #acc_data = {}
        for epoch in range(num_epochs):
            #print("Epoch {}".format(epoch))
            #note: compute_epoch updates weights in place
            weights, l_epoch, a_epoch = compute_epoch(x_train, y_train,
                                        config["l_rate"], network.weights,
                                        len(x_train), config["lambda"],
                                        network.act_f, optimizer,
                                        config['batch_size'], True)
            l_val, a_val = evaluate_model(x_val, y_val, weights,  network.act_f, dict(tp=0, fp=0, tn=0, fn=0), True)
            #loss_val[epoch] = l_val
            list_loss_val[epoch].append(l_val)
            list_loss[epoch].append(l_epoch)
            #acc_val[epoch] = a_val
            list_acc_val[epoch].append(a_val)
            list_acc[epoch].append(a_epoch)
            #print("Accuracy: {}".format(a_epoch)
            #print(old_weights[0] == network.weights[0])
            #store info only when the loss decreases
            #if l_epoch < config['loss']:
            #    config['num_epochs'] = epoch
            #    config['loss'] = l_epoch
            best_weights = [np.copy(curr) for curr in network.weights]
                
            #loss_data[epoch] = l_epoch
            #acc_data[epoch] = a_epoch
        #Adding score for this fold
        ms, _ = evaluate_model(x_val, y_val, best_weights,
                network.act_f, dict(tp=0, fp=0, tn=0, fn=0), True)
        model_score += ms                              
        #Create first subplot
        #plt.subplot(312)
        #Plot loss curve
        #loss_lists = sorted(loss_data.items()) # sorted by key, return a list of tuples
        #lh_axis, lv_axis = zip(*loss_lists) # unpack a list of pairs into two tuples
        #ax2.set_xlabel('Number of epochs')
        #ax2.set_ylabel('Log loss')
        #ax2.plot(lh_axis, lv_axis)
        
        #Create second subplot
        #plt.subplot(313)
        #Plot accuracy curve
        #acc_lists = sorted(acc_data.items()) # sorted by key, return a list of tuples
        #ah_axis, av_axis = zip(*acc_lists) # unpack a list of pairs into two tuples
        #ax3.set_xlabel('Number of epochs')
        #ax3.set_ylabel('Accuracy')
        #ax3.plot(ah_axis, av_axis)


    #Average over all the folds
    model_score /= len(folds)
    
    #Store score in config dictionary
    
    config['score'] = model_score

    #print(plot_data)
    curr_config = "k" + str(args['folds'])
    curr_config += "lr" + str(config['l_rate'])
    curr_config += "nhn" + str(config['num_hidden_neurons'])
    curr_config += "wd" + str(config['weights_distr'])
    curr_config += "bs" + str(config['batch_size'])
    curr_config += "af" + str(config['act_f'])
    curr_config += "l" + str(config['lambda'])
    curr_config += "o" + str(config['optimizer'])
    #print(curr_config)
    
    #print(list_loss)
    mean_l = np.array([np.mean(el) for el in list_loss])
    std_dev_l = np.array([np.std(el) for el in list_loss])
    mean_a = np.array([np.mean(el) for el in list_acc])
    std_dev_a = np.array([np.std(el) for el in list_acc])
    
    mean_l_val = np.array([np.mean(el) for el in list_loss_val])
    std_dev_l_val = np.array([np.std(el) for el in list_loss_val])
    mean_a_val = np.array([np.mean(el) for el in list_acc_val])
    std_dev_a_val = np.array([np.std(el) for el in list_acc_val])
    #print(mean_l)
    #print(std_dev_l)
    #plt.subplot(311)
    #ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('Log loss')
    ax1.plot(range(num_epochs), mean_l, color='steelblue', linestyle='dashed', lw=2, label='training set')
    ax1.fill_between(range(num_epochs), mean_l + std_dev_l, mean_l -  std_dev_l, facecolor='steelblue', alpha=0.4)
    
    ax1.plot(range(num_epochs), mean_l_val, color='orange', lw=2, label='validation set')
    ax1.fill_between(range(num_epochs), mean_l_val + std_dev_l_val, mean_l_val -  std_dev_l_val, facecolor='orange', alpha=0.4)
    ax1.legend(prop={'size': 8})
    
    ax2.set_xlabel('Number of epochs')
    ax2.set_ylabel('Accuracy')
    ax2.plot(range(num_epochs), mean_a,  color='steelblue', linestyle='dashed', lw=2, label='training set')
    ax2.fill_between(range(num_epochs), mean_a + std_dev_a, mean_a -  std_dev_a, facecolor='steelblue', alpha=0.4)
    
    ax2.plot(range(num_epochs), mean_a_val, 'orange', lw=2, label='validation set')
    ax2.fill_between(range(num_epochs), mean_a_val + std_dev_a_val, mean_a_val -  std_dev_a_val, facecolor='orange', alpha=0.4)
    ax2.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig(args['plots_folder'] + "/" + curr_config +".svg")
    plt.close(fig)
    with open(args['results_folder'] + "/" + args['extra'] + args['optimizer'] + ".csv", 'a') as f:
        w = csv.DictWriter(f, config.keys())
        w.writerow(config)
    num_conf += 1
#Grid search ended
print("Trained on all configurations")
