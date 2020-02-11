"""
    main: main file for training a neural network for a
    regression task
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

num_epochs = 10

##########MAIN################

parser = argparse.ArgumentParser(
    description='Training and validation of neural network.')

parser.add_argument("--folds",
                    metavar="k",
                    default=5,
                    help="Number of folds for cross-validation")

parser.add_argument("--train-file",
                    metavar="train_file",
                    default="ML-CUP19-TR-shuffled.csv",
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
    #Ignore comment lines
    data_str = [x[1:] for x in csv.reader(f) if len(x) > 1]

#Convert to float
data = [tuple(map(float, x)) for x in data_str]

#Convert to column vectors
x = [np.array([l[:-2]]).T for l in data]
y = [np.array([l[-2:]]).T for l in data]

configs = config.configs[str(args['optimizer'])+str(args['extra'])]

#Write header of results file

configs[0]['k'] = args['folds']

with open(args['results_folder'] + "/" + args['extra']+ args['optimizer'] + ".csv", 'w') as f:
    configs[0]['score'] = 0
    w = csv.DictWriter(f, configs[0].keys())
    w.writeheader()

num_conf = 0

for config in configs:
    #print("Running configuration " + str(num_conf) + "/" + str(len(configs)))
    config['k'] = args['folds']
    model_score = 0
    #config['num_epochs'] = 0
    #config['loss'] = 1000
    
    #k-fold CV
    folds = cv(len(x), args['folds'], x, y)

    list_loss = [[] for e in range(num_epochs)]
    list_loss_val = [[] for e in range(num_epochs)]
    
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
        plot_data = {}
        for epoch in range(num_epochs):
            #print("Epoch {}".format(epoch))
            #note: compute_epoch updates weights in place
            weights, l_epoch, a= compute_epoch(x_train, y_train,
                                    config["l_rate"], network.weights,
                                    len(x_train), config["lambda"],
                                    network.act_f, optimizer,
                                    config['batch_size'])
            #print(old_weights[0] == network.weights[0])
            #store info only when the loss decreases
            #if l_epoch < config['loss']:
            #    config['num_epochs'] = epoch
            #    config['loss'] = l_epoch
            l_val, a_val = evaluate_model(x_val, y_val, weights,  network.act_f, dict(tp=0, fp=0, tn=0, fn=0))
            list_loss_val[epoch].append(l_val)
            list_loss[epoch].append(l_epoch)
            best_weights = [np.copy(curr) for curr in network.weights]
                
            #plot_data[epoch] = l_epoch
        #Adding score for this fold
        ms, _ = evaluate_model(x_val, y_val, best_weights,
                network.act_f, dict(tp=0, fp=0, tn=0, fn=0))
        model_score += ms        
        #Plot curve
        #lists = sorted(plot_data.items()) # sorted by key, return a list of tuples
        #h_axis, v_axis = zip(*lists) # unpack a list of pairs into two tuples
        #plt.xlabel('Number of epochs')
        #plt.ylabel('MEE')
        #plt.plot(h_axis, v_axis)

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
    
    mean_l = np.array([np.mean(el) for el in list_loss])
    std_dev_l = np.array([np.std(el) for el in list_loss])
    mean_l_val = np.array([np.mean(el) for el in list_loss_val])
    std_dev_l_val = np.array([np.std(el) for el in list_loss_val])
    
    plt.xlabel('Number of epochs')
    plt.ylabel('MEE')
    plt.plot(range(num_epochs), mean_l, color='steelblue', linestyle='dashed', lw=2, label='training set')
    plt.fill_between(range(num_epochs), mean_l + std_dev_l, mean_l -  std_dev_l, facecolor='steelblue', alpha=0.4)
    plt.plot(range(num_epochs), mean_l_val, color='orange', lw=2, label='validation set')
    plt.fill_between(range(num_epochs), mean_l_val + std_dev_l_val, mean_l_val -  std_dev_l_val, facecolor='orange', alpha=0.4)
    plt.legend(prop={'size': 10})
    plt.tight_layout()
    #Exporting plot
    plt.savefig(args['plots_folder'] + "/" + curr_config +".svg")
    plt.clf()
    with open(args['results_folder'] + "/"+ args['extra'] + args['optimizer'] + ".csv", 'a') as f:
        w = csv.DictWriter(f, config.keys())
        w.writerow(config)
    num_conf += 1
#Grid search ended
print("Trained on all configurations")
