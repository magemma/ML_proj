"""
    k_plot_test_reg: main file for selecting the best
    configuration of hyperparameters for solving a classification
    task, training the best model k times on a subset of the
    training set and then tested on a disjoint subset of the
    same training set.
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

num_epochs = 100

parser = argparse.ArgumentParser(description='Testing of neural network for classification.')

parser.add_argument("--train-file",
                    metavar="train_file",
                    help="path of train file")

parser.add_argument("--trials",
                    metavar="k",
                    default=2,
                    help="Number of instances of the winner model to train and test")

parser.add_argument("--results-file",
                    required=True,
                    metavar="results_file",
                    help="path of results file")

parser.add_argument("--fig-name",
                    metavar="fig_name",
                    help="name of the file where to store the plot")

parser.add_argument(
    "--network-file",
    metavar="network_file",
    help="path where to write/read the network (activation function + weights)"
)

parser.add_argument("--test-file",
                    metavar="test_file",
                    help="path of test file")

args = vars(parser.parse_args())

filename = [i for i in glob.glob(args['results_file'])]

#print(all_filenames)

#combine all files in the list
df = pd.concat([pd.read_csv(f) for f in filename])

#Make score field float 64 instead of string
df['score'] = pd.to_numeric(df['score'], errors='coerce')

df.sort_values(by=['score'], inplace=True)

df = df.iloc[:, :-1]
#print(df.head(1))

best_config = df.head(1).to_dict('records')[0]

print("Best configuration: ", best_config)

with open(args['train_file'], 'r') as f:
    data_str = [x[0].split()[:-1] for x in csv.reader(f)]
    
    #Convert to float
    data = [tuple(map(int, x)) for x in data_str]

    x_ori = np.array([np.array(l[1:]) for l in data])
    x_one_hot = one_of_k(x_ori)
    x_train = [np.array([l[1:]]).T for l in x_one_hot]
    y_train = [np.array([l[0]]).T for l in data]

with open(args['test_file'], 'r') as f:
    data_str = [x[0].split()[:-1] for x in csv.reader(f)]
    
    #Convert to float
    data = [tuple(map(int, x)) for x in data_str]

    x_ori = np.array([np.array(l[1:]) for l in data])
    x_one_hot = one_of_k(x_ori)
    x_test = [np.array([l[1:]]).T for l in x_one_hot]
    y_test = [np.array([l[0]]).T for l in data]

#Manipulation on input strings
best_config['num_hidden_neurons'] = ast.literal_eval(
    best_config['num_hidden_neurons'])

best_config['optimizer'] = eval(best_config['optimizer'][1:-1])

optimizer = get_optimizer(best_config['optimizer'])

list_loss = [[] for e in range(num_epochs)]
list_loss_test = [[] for e in range(num_epochs)]

list_acc = [[] for e in range(num_epochs)]
list_acc_test = [[] for e in range(num_epochs)]

model_score = 0
for t in range(args['trials']):
    network = nn(best_config, len(x_train[0]), len(y_train[0]))

    #Training on whole training set
    print("Training model")
    for epoch in tqdm(range(num_epochs)):
        weights, l_train, a_train = compute_epoch(x_train, y_train,
                                            best_config["l_rate"], network.weights,
                                            len(x_train), best_config["lambda"],
                                            network.act_f, optimizer,
                                            best_config['batch_size'], True)
        l_test, a_test = evaluate_model(x_test, y_test, weights,  network.act_f, dict(tp=0, fp=0, tn=0, fn=0), True)
        #if (epoch != 0):
        list_loss_test[epoch].append(l_test)
        list_loss[epoch].append(l_train)
        list_acc_test[epoch].append(a_test)
        list_acc[epoch].append(a_train)
            
    model_score += l_test

    with open(args["network_file"]+str(t)+".pkl", "wb") as n:
        pickle.dump(network, n)

model_score /= args['trials']
print("The loss on shuffled test set is {}".format(model_score))

#Setting plots info
fig, (ax1, ax2) = plt.subplots(2, 1)
    
#Plotting loss
mean_l = np.array([np.mean(el) for el in list_loss])
std_dev_l = np.array([np.std(el) for el in list_loss])

mean_l_test = np.array([np.mean(el) for el in list_loss_test])
std_dev_l_test = np.array([np.std(el) for el in list_loss_test])

ax1.set_ylabel('Log loss')
ax1.plot(range(num_epochs), mean_l, color='steelblue', linestyle='dashed', lw=2, label='training set')
ax1.fill_between(range(num_epochs), mean_l + std_dev_l, mean_l -  std_dev_l, facecolor='steelblue', alpha=0.4)
ax1.plot(range(num_epochs), mean_l_test, 'orange', lw=2, label='test set')
ax1.fill_between(range(num_epochs), mean_l_test + std_dev_l_test, mean_l_test -  std_dev_l_test, facecolor='orange', alpha=0.4)
ax1.legend()

#Plotting accuracy
mean_a = np.array([np.mean(el) for el in list_acc])
std_dev_a = np.array([np.std(el) for el in list_acc])

mean_a_test = np.array([np.mean(el) for el in list_acc_test])
std_dev_a_test = np.array([np.std(el) for el in list_acc_test])

ax2.set_xlabel('Number of epochs')
ax2.set_ylabel('Accuracy')
ax2.plot(range(num_epochs), mean_a, color='steelblue', linestyle='dashed', lw=2, label='training set')
ax2.fill_between(range(num_epochs), mean_a + std_dev_a, mean_a -  std_dev_a, facecolor='steelblue', alpha=0.4)
ax2.plot(range(num_epochs), mean_a_test, 'orange', lw=2, label='test set')
ax2.fill_between(range(num_epochs), mean_a_test + std_dev_a_test, mean_a_test -  std_dev_a_test, facecolor='orange', alpha=0.4)
ax2.legend()

plt.savefig(args['fig_name'])
plt.close(fig)
