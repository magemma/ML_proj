"""
    test_regression: main file for selecting the best
    configuration of hyperparameters for solving a regression
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

parser = argparse.ArgumentParser(description='Testing of neural network for a regression task.')

parser.add_argument("--train-file",
                    metavar="train_file",
                    default="ML-CUP19-TR-shuffled.csv",
                    help="path of train file")

parser.add_argument("--results-file",
                    required=True,
                    metavar="results_path",
                    help="path of results file")

parser.add_argument(
    "--network-file",
    metavar="network_file",
    default="cup_net.pickle",
    help="path where to write/read the network (activation function + weights)"
)

parser.add_argument("--test-file",
                    metavar="test_file",
                    default="ML-CUP19-TS-shuffled.csv",
                    help="path of test file")

parser.add_argument("--output-file",
                    metavar="output_file",
                    default="manetti-martini-ML-CUP19-TS.csv",
                    help="path of output file")

args = vars(parser.parse_args())

filename = [i for i in glob.glob(args['results_file'])]

#combine all files in the list
df = pd.concat([pd.read_csv(f) for f in filename])

df.sort_values(by=['score'], inplace=True)

df = df.iloc[:, :-1]
#print(df.head(1))

best_config = df.head(1).to_dict('records')[0]

print("Best configuration: ", best_config)

with open(args['train_file'], 'r') as f:
    #Ignore comment lines
    data_str = [x[1:] for x in csv.reader(f) if len(x) > 1]

    #Convert to float
    data = [tuple(map(float, x)) for x in data_str]

    #Convert to column vectors
    x = [np.array([l[:-2]]).T for l in data]
    y = [np.array([l[-2:]]).T for l in data]

#Manipulation on input strings
best_config['num_hidden_neurons'] = ast.literal_eval(
    best_config['num_hidden_neurons'])

best_config['optimizer'] = eval(best_config['optimizer'][1:-1])

#print(best_config["num_hidden_neurons"])
network = nn(best_config, len(x[0]), len(y[0]))
#print(best_config['optimizer'])
optimizer = get_optimizer(best_config['optimizer'])

#Training on whole training set
print("Training model")
plot_data = {}
for epoch in tqdm(range(200)):
    weights, l_epoch, _ = compute_epoch(x, y,
                                        best_config["l_rate"], network.weights,
                                        len(x), best_config["lambda"],
                                        network.act_f, optimizer,
                                        best_config['batch_size'])
    plot_data[epoch] = l_epoch
#Plot curve
lists = sorted(plot_data.items())  # sorted by key, return a list of tuples
h_axis, v_axis = zip(*lists)  # unpack a list of pairs into two tuples
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.plot(h_axis, v_axis)

#Storing plot
plt.savefig("best_regression.svg")
plt.clf()

with open(args["network_file"], "wb") as n:
    pickle.dump(network, n)

#Evaluate on the test set
print("Running on test set")

#Reading test data
with open(args['test_file'], 'r') as f:
    #Ignore comment lines
    data_str = [x for x in csv.reader(f) if len(x) > 1]

#Convert to int/float
data = [(int(x[0]), tuple(map(float, x[1:]))) for x in data_str]

output = []
for i, x_in in data:
    x = np.array([x_in]).T
    _, _, y = feed_forward_no_output(x, network.weights, network.act_f)
    output.append((i, ) + tuple(y.flat))
with open(args["output_file"], "w") as f:
    wr = csv.writer(f)
    wr.writerows(output)
