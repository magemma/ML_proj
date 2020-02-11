"""
    test_regression: main file for selecting the best
    configuration of hyperparameters for solving a classification
    task, training the best model and run it on the test set
    ML 2019 Project
    Dipartimento di Informatica Università di Pisa
    Authors: R. Manetti, G. Martini
    We declare that the content of this file is entirelly
    developed by the authors
"""

import os
import glob
import argparse
import pandas as pd
import csv
import numpy as np
from nn_utilities import *
import ast
from optimizer import get_optimizer
import pickle
import matplotlib.pylab as plt
from tqdm import tqdm


num_epochs = 1000
num_best = 5
parser = argparse.ArgumentParser(description='Testing of neural network for a classification task.')

parser.add_argument("--train-file",
                    required=True,
                    metavar="train_file",
                    help="path of train file")

parser.add_argument("--results-file",
                    required=True,
                    metavar="results_file",
                    help="path of results file")

parser.add_argument(
    "--network-file",
    required=True,
    metavar="network_file",
    help="path where to write/read the network (activation function + weights)"
)

parser.add_argument("--test-file",
                    required=True,
                    metavar="test_file",
                    help="path of test file")

parser.add_argument("--output-file",
                    required=True,
                    metavar="output_file",
                    help="path of output file")

args = vars(parser.parse_args())

#Clear output file
if os.path.exists(args["output_file"]):
    os.remove(args["output_file"])



#take all slides in folder
filename = [i for i in glob.glob(args['results_file'])]

#combine all files in the list
df = pd.concat([pd.read_csv(f) for f in filename])

df.sort_values(by=['score'], inplace=True)

df = df.iloc[:, :-1]
#print(df.head(1))

best_configs = [df.iloc[[i]].to_dict('records')[0] for i in range(num_best)]

#Add loss and accuracy to dict
#print(best_configs)
best_configs = [{**d, "Loss":0, "Accuracy":0} for d in best_configs]
#print(best_configs)

#Write header of output file
with open(args["output_file"], "w") as f:
    w = csv.DictWriter(f, list(best_configs[0].keys()))
    w.writeheader()

#print("Best configuration: ", best_config)

#Reading train data
with open(args['train_file'], 'r') as f:
    data_str = [x[0].split()[:-1] for x in csv.reader(f)]

    #Convert to float
    data = [tuple(map(int, x)) for x in data_str]

    x_ori = np.array([np.array(l[1:]) for l in data])
    x_one_hot = one_of_k(x_ori)
    x_tr = [np.array([l[1:]]).T for l in x_one_hot]
    y_tr = [np.array([l[0]]).T for l in data]

#Reading test data
with open(args['test_file'], 'r') as f:
    data_str = [x[0].split()[:-1] for x in csv.reader(f)]

    #Convert to float
    data = [tuple(map(int, x)) for x in data_str]

    x_ori = np.array([np.array(l[1:]) for l in data])
    x_one_hot = one_of_k(x_ori)
    x_ts = [np.array([l[1:]]).T for l in x_one_hot]
    y_ts = [np.array([l[0]]).T for l in data]

#Manipulation on input strings

#print(best_configs)

for config in best_configs:
    config['num_hidden_neurons'] = ast.literal_eval(config['num_hidden_neurons'])
    
    config['optimizer'] = ast.literal_eval(config['optimizer'][1:-1])

#print(best_configs)

networks = [nn(best_configs[i], len(x_tr[0]), len(y_tr[0])) for i in range(num_best)]

#Dump best network
with open(args["network_file"], "wb") as n:
    pickle.dump(networks[0], n)

optimizers = [get_optimizer(best_configs[i]['optimizer']) for i in range(num_best)]

#Training on whole training set
print("Training model")

o = 0


for best_config in best_configs:
    network = networks[o]
    optimizer = optimizers[o]
    o += 1
    print("Configuration{}".format(o))
    #Initialize dss for plots
    loss_train = {}
    acc_train = {}
    loss_test = {}
    acc_test = {}
    for epoch in tqdm(range(num_epochs)):
          weights, l_train, a_train = compute_epoch(x_tr, y_tr, best_config["l_rate"], network.weights,
                                  len(x_tr), best_config["lambda"], network.act_f,
                                  optimizer, best_config['batch_size'], True)
                                  
          loss_train[epoch] = l_train
          acc_train[epoch] = a_train

          l_test, a_test = evaluate_model(x_ts, y_ts, weights,  network.act_f, dict(tp=0, fp=0, tn=0, fn=0), True)
          loss_test[epoch] = l_test
          acc_test[epoch] = a_test                            

    #Setting plots info
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    #Create first subplot
    #Plot loss curve
    loss_lists_tr = sorted(loss_train.items()) # sorted by key, return a list of tuples
    lh_axis_tr, lv_axis_tr = zip(*loss_lists_tr) # unpack a list of pairs into two tuples
    loss_lists_ts = sorted(loss_test.items()) # sorted by key, return a list of tuples
    lh_axis_ts, lv_axis_ts = zip(*loss_lists_ts) # unpack a list of pairs into two tuples
    ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('MEE')
    ax1.plot(lh_axis_tr, lv_axis_tr, 'b', label='training', alpha=0.7)
    ax1.plot(lh_axis_ts, lv_axis_ts, 'r', label='test', alpha=0.7)
    ax1.legend()

    #Create second subplot
    #Plot accuracy curve
    acc_lists_tr = sorted(acc_train.items()) # sorted by key, return a list of tuples
    ah_axis_tr, av_axis_tr = zip(*acc_lists_tr) # unpack a list of pairs into two tuples
    acc_lists_ts = sorted(acc_test.items()) # sorted by key, return a list of tuples
    ah_axis_ts, av_axis_ts = zip(*acc_lists_ts) # unpack a list of pairs into two tuples
    ax2.set_xlabel('Number of epochs')
    ax2.set_ylabel('Accuracy')

    ax2.plot(ah_axis_tr, av_axis_tr, 'b', label='training', alpha=0.7)
    ax2.plot(ah_axis_ts, av_axis_ts, 'r', label='test', alpha=0.7)
    ax2.legend()
    plt.tight_layout()

    #Export plot
    plt.savefig("best" + str(o) + "_" + args["output_file"] + ".svg")
    plt.close(fig)

    #Evaluate on the test set
    print("Running on test set")

    #Computing scores on test set
    model_loss, model_acc = evaluate_model(x_ts, y_ts, weights, network.act_f,
                                             dict(tp=0, fp=0, tn=0, fn=0), True)
    
    #Update loss and accuracy keys
    best_config["Loss"] = model_loss
    best_config["Accuracy"] = model_acc
    with open(args["output_file"], "a") as f:
        w = csv.DictWriter(f, best_config.keys())
        w.writerow(best_config)
    
    #with open(args["output_file"], "a") as f:
    #      f.write("Loss: {}".format(model_loss))
    #      f.write("Accuracy: {}".format(model_acc))
f.close()
