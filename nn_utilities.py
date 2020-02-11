"""
    nn_utilities: utility functions for neural network implementation
    ML 2019 Project
    Dipartimento di Informatica Università di Pisa
    Authors: R. Manetti, G. Martini
    We declare that the content of this file is entirelly
    developed by the authors
"""

import numpy as np
import random
import csv
from activation_functions import *
from loss_functions import *


weights_boundary = 0.01

def shuffle(a, b):
    """ This function shuffles two numpy array of the same
    length using the same permutation.
    Obs: since the arrays have same length, the same number
    of calls to the random number generator is made. By
    resetting the state we are guaranteed that the random
    number generator will give the same results in the
    second call to shuffle().
    
    @param a: first numpy array
    @param b: second numpy array
    
    @return a_shuffled and b_shuffled
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def cv(n_samples, k, x_data, y_data):
    """
    K-fold cross validation function which partitions the
    training set into training and validation subsets.
    Before splitting the data a shuffling phase is held.
    @param n_samples: the number of training examples
    @param k:  total number of folds
    @param x_data: numpy array of inputs
    @param y_data: numpy array of target variables

    @return a dictionary of the arrays containing examples 
        and target for training and validation
    """
    assert k != 1
    n_samples_fold = int(n_samples / k)
    folds = {}
    #Shuffling phase
    x_data, y_data = shuffle(x_data, y_data)
    #print("number of examples in dataset: {}".format(n_samples))
    for i in range(k - 1):
        folds[i] = {}
        offset = i * n_samples_fold
        folds[i]['x_val'] = x_data[offset:offset + n_samples_fold]
        folds[i]['x_tr'] = x_data[:offset - 1] + x_data[offset +
                                                        n_samples_fold + 1:]

        #print("#samples validation fold: {}".format(len(folds[i]['x_val'])))
        folds[i]['y_val'] = y_data[offset:offset + n_samples_fold]
        folds[i]['y_tr'] = y_data[:offset - 1] + y_data[offset +
                                                        n_samples_fold + 1:]

    #Outside for to take the remainder of the division by k
    folds[k - 1] = {}
    folds[k - 1]['x_val'] = x_data[(k - 2) * n_samples_fold:n_samples]
    folds[k - 1]['x_tr'] = x_data[:(k - 2) * n_samples_fold - 1]
    folds[k - 1]['y_val'] = y_data[(k - 2) * n_samples_fold:n_samples]
    folds[k - 1]['y_tr'] = y_data[:(k - 2) * n_samples_fold - 1]
    #print("#samples validation fold: {}".format(len(folds[i]['x_val'])))
    return folds


def weighted_inputs_sum(W, x):
    """Matrix-vector product representing the weighted inputs
    of a unit
    @param W: weights matrix
    @param x: inputs vector
    @return the result of the product
    """
    return W.dot(x)


def hidden_layer(W, x, act_f):
    """
    Computation of a whole hidden layer of the neural network
    @param W: the weights matrix
    @param x: the inputs of a layer
    @param act_f: instance of activation_function class

    @return: both the output of the multiplication (p) and the 
        output of the layer (f_sigma(p)
    """
    p = weighted_inputs_sum(W, x)
    return p, act_f.compute(p)


def feed_forward_no_output(x, weights, act_f, clas=False):
    """
    Whole functioning of feed forward network.
    @param x: the input of the nn
    @param weights: the list of matrices representing the weights
    @param act_f: instance of activation_function class
    @param clas: boolean for indicating if the task is
        classification or not
    @return list of p values (one for each hidden layer)
            list of x values (one for each hidden layer) and
            y values corresponding to input x values
    Notice that the layers are 0-based, just like lists
    in Python for simplicity of notation
    """
    #print(x.shape)
    inputs = []
    ps = []
    print(len(x))
    #Add "bias"-component to input array
    x_bias = np.concatenate((x, [[1]]))
    inputs.append(x_bias)
    for l in range(0, len(weights) - 1):  #excluding last layer
        p, x = hidden_layer(weights[l], x_bias, act_f)
        #Add "bias"-component to input array
        x_bias = np.concatenate((x, [[1]]))
        ps.append(p)
        inputs.append(x_bias)
    #We arrived to the last layer
    #If the problem is regression hence no activation function
    #print(clas)
    if clas:
        #istantiate symmetric (in [0, +1]) activation function
        act_f = logistic()
        xx, x = hidden_layer(weights[-1], x_bias, act_f)
        #print(x)
        #NO ROUNDING FOR CROSS ENTROPY LOSS
        #Round the output value to nearest int -> values in {0, 1}
        #x = np.rint(x)
        #print(x)
    else:
        x = weighted_inputs_sum(weights[-1], x_bias)
    inputs.append(x)
    return ps, inputs, x


def feed_forward(x, y, weights, act_f, scores, clas=False):
    """
    Whole functioning of feed forward network.
    @param x: the input of the nn
    @param y: the objective function's value corresponding to x
    @param weights: the list of matrices representing the weights
    @param act_f: instance of activation_function class
    @param scores: dictionary of number of elements in the 
        different quadrants of the confusion matrix
    @param clas: boolean for indicating if the task is
        classification or not
    @return list of p values (one for each hidden layer)
            list of x values (one for each hidden layer)
            loss function between the output of the nn and the
                objective function
            updates (in classification case) on scores dict
    Notice that the layers are 0-based, just like lists
    in Python for simplicity of notation
    """
    ps, inputs, y_out = feed_forward_no_output(x, weights, act_f, clas)
    #Check for non converging case, returning a super high loss
    if np.isnan(y_out[0][0]):
        #incrementing fp without chaning acc (tn + tp / all)
        scores['fp'] += 1
        return ps, inputs, 100, scores
    loss_f = mee()
    if clas:
        loss_f = log_loss()
    L = loss_f.compute(y, y_out)
    #print(loss_f.name)
    #If we are in the classification case eval results 4 acc
    if clas:
        #print(scores)
        #print(int(y_out[0][0]))
        #print(int(y[0]))
        out = np.rint(y_out)
        if (out[0][0] == 1) and (y[0] == 1):
            #true positive
            scores['tp'] += 1
        if (out[0][0] == 0) and (y[0] == 0):
            #true negative
            scores['tn'] += 1
        if (out[0][0] == 1) and (y[0] == 0):
            #false positive
            scores['fp'] += 1
        if (out[0][0] == 0) and (y[0] == 1):
            #false negative
            scores['fn'] += 1
    return ps, inputs, L, scores

def back_x(W, df_dp):
    """
    Computation of the partial derivative of the composition
    between the weighted input function and a function f
    with respect to vector x
    @param W: numpy bidimensional array representing the
        weights matrix (of layer k)
    @param df_dp: numpy array representing the partial
        derivative of the composition between the weighted input
        sum and the function f with respect to the former
    @return the array of derivatives
    """
    return W.T.dot(df_dp)


def back_w(df_dp, x):
    """
    Computation of the partial derivative of the composition
    between the weighted input function and a function f
    with respect to matrix W
    @param df_dp: numpy array representing the partial
        derivative of the composition between the weighted input
        sum and the function f with respect to the former
    @param x: numpy array representing the inputs of the
        current layer
    @return the matrix of the partial derivatives
    """
    return df_dp.dot(x.T)


def back_step(dL_dxk, pk, Wk, xkmin1, act_f):
    """
    Computation of a backpropagation step that returns the
    couple (dL/dW, dL/dx)
    @param dL_dxk: the numpy array of the partial derivatives
        of the loss function with respect to the inputs of a
        layer k
    @param pk: vector p at the k-th layer
    @param Wk: matrix W at the k-th layer
    @param xkmin1: output of the (k-1)-th layer
    @param act_f: instance of activation_function class
    
    @return (dL_dWk, dL_dxk1) the couple of the partial derivatives
        of L with respect to both W and x
    """
    dfp_p = act_f.chain_rule(dL_dxk, pk)
    dL_dxk1 = back_x(Wk, dfp_p)
    dL_dWk = back_w(dfp_p, xkmin1)
    return dL_dxk1, dL_dWk


def backpropagation(L, ps, inputs, weights, y, lam, act_f, clas=False):
    """
    Computation of the backpropagation of the partial derivative
    across the whole neural network.
    @param L: loss of the output value
    @param ps: list of p values
    @param inputs: list of inputs of all the layers
    @param weights: list of weight matrices (one per layer)
    @param y: list of numpy arrays (target variable)
    @param lam: lambda for L2 regularization
    @param act_f: instance of activation_function class
    @param clas: boolean for indicating if the task is
        classification or not
    @return the list of the partial derivatives of the loss
        with respect to the weights tensor (one matrix per layer)
    """
    #Istantiare class of loss function
    loss_f = mee()
    if clas:
        loss_f = log_loss()
    dL_dweights = []
    #the last element of inputs array is the output of the nn
    dL_dout = loss_f.partial(y, inputs[-1])
    #print(type(dL_dout[0][0]))
    dL_dweights.append(back_w(dL_dout, inputs[-2]) + lam * weights[-1])
    #Compute the partial derivative of the loss with respect to
    #the input of the last layer
    dL_dx = back_x(weights[-1], dL_dout)
    for l in range(len(weights) - 2, -1, -1):  #last extreme excluded
        #ps contains len(weights)-1 elements
        dL_dx, dL_dW = back_step(dL_dx, ps[l], weights[l], inputs[l], act_f)
        dL_dweights.append(dL_dW + lam * weights[l])

    #Reversing the order of the partial derivatives of the
    #loss with respect to the weight
    dL_dweights.reverse()
    return dL_dweights


def compute_epoch(x,
                  y,
                  l_rate,
                  weights,
                  n_samples,
                  lam,
                  act_f,
                  optimizer,
                  b_size,
                  clas=False):
    """
    Computation of an epoch (feedforward + backpropagation)
    @param x: list of input vectors
    @param y: list of target values
    @param weights: list of weight matrices
    @param n_samples: the number of examples belonging to 
        the training set
    @param lam: lambda for L2 regularization
    @param act_f: instance of activation_function class
    @param optimizer: optimizer to be used to train the network
    @param b_size: integer representing the batch size
    @param clas: boolean for indicating if the task is
        classification or not
    @return weights: matrix of updated weights
    @return loss: loss of the model in this epoch
    @return accuracy: makes sense only in the case of classification
    """
    total_loss = 0
    dL_dweights = [0 for a in range(len(weights))]
    weights_for_gradient = optimizer.gradient_calculation_point(weights)
    scores = dict(tp=0, fp=0, tn=0, fn=0)
    for i in range(n_samples):
        ps, inputs, L, scores = feed_forward(x[i], y[i], weights_for_gradient,
                                             act_f, scores, clas)
        #print(scores)
        dL_dweights_curr = backpropagation(L, ps, inputs, weights_for_gradient,
                                           y[i], lam, act_f, clas)
        for a in range(len(dL_dweights)):
            dL_dweights[a] += dL_dweights_curr[a]
        total_loss += L
        #If the previous batch has been exhausted
        if ((i != 0) and ((i % b_size == 0) or (i == n_samples - 1))):
            #Update the weights
            optimizer.update_weights(l_rate, b_size, dL_dweights, weights)
            #Reset the partial derivatives for next batch
            dL_dweights = [0 for a in range(len(weights))]
            # Compute the next update point.
            weights_for_gradient = optimizer.gradient_calculation_point(
                weights)
    accuracy = 0
    if clas:
        accuracy = (scores['tp'] + scores['tn']) / (
            scores['tp'] + scores['tn'] + scores['fp'] + scores['fn'])
    #Print row of loss for that epoch
    #print(total_loss / n_samples)
    return weights, total_loss / n_samples, accuracy


def evaluate_model(x, y, weights, act_f, scores, clas=False):
    """
    This function returns the score of our model (represented
    through the weights of the nn) on the set (x,y)
    @param x: numpy array representing the inputs of the whole 
        training set
    @param y: numpy array representing the target values of the 
        whole training set
    @weigths: the tensor representing the weights of the nn
    @act_f: instance of the activation function
    @scores: dictionary used for computing the accuracy
        (makes sense only in classification)
    @param clas: boolean for indicating if the task is
        classification or not
    @return loss and accuracy for such model
    """
    total_loss = 0
    for i in range(len(x)):
        ps, inputs, L, scores = feed_forward(x[i], y[i], weights, act_f,
                                             scores, clas)
        accuracy = 0
        if clas:
            accuracy = (scores['tp'] + scores['tn']) / (
                scores['tp'] + scores['tn'] + scores['fp'] + scores['fn'])
        total_loss += L

    return total_loss / len(x), accuracy


def print_model_row(config, f):
    """This procedure prints a row containing the parameters
    of the model, the best achieved score and the number
    of epochs needed for that
    @param config: dictionary containing the values of the
        hyperparameters defining a model
    @param f: path of output file"""

    w = csv.DictWriter(f, config.keys())
    w.writeheader()
    w.writerow(config)
    #print(config, flush=True)


class nn(object):
    def __init__(self, config, n_in, n_out):
        #Weights initialization with Glorot scheme
        fans = [n_in] + config["num_hidden_neurons"] + [n_out]
        weights = []
        for i in range(len(fans) - 1):
            #fans[i+1] is the number of neurons of next layer
            #fans[i] is the number of neurons of current layer
            if (config["weights_distr"] == "uniform"):
                weights.append(
                    np.random.uniform(
                        -math.sqrt(6) / (math.sqrt(fans[i] + fans[i + 1])),
                        math.sqrt(6) / (math.sqrt(fans[i] + fans[i + 1])),
                        (fans[i + 1], fans[i] + 1)))
            elif (config["weights_distr"] == "gaussian"):
                var = 2 / (fans[i] + fans[i + 1])
                std_dev = math.sqrt(var)
                weights.append(
                    np.random.normal(0.0, std_dev,
                                     (fans[i + 1], fans[i] + 1)))
            elif (config["weights_distr"] == "base"):
                weights.append(
                    np.random.uniform(-1 * weights_boundary, weights_boundary, (fans[i + 1], fans[i] + 1)))
            
            else:
                raise Exception("WrongDistrException")
        self.weights = weights
        if config['act_f'] == 'relu':
            self.act_f = relu()
        if config['act_f'] == 'softplus':
            self.act_f = softplus()
        if config['act_f'] == 'tanh':
            self.act_f = tanh()
        if config['act_f'] == 'logistic':
            self.act_f = logistic()


def one_of_k(data):
    """This function transforms the dataset into one hot
        notation.
        Notice that this function has been copied from the web
        @param data: the numpy array containing the input of
            a training example
        @return the numpy array representing the one hot
            encoding of such input x
    """
    dist_values = np.array(
        [np.unique(data[:, i]) for i in range(data.shape[1])])
    new_data = []
    First_rec = True
    for record in data:
        #print(record)
        new_record = []
        First = True
        indice = 0
        for attribute in record:
            new_attribute = np.zeros(len(dist_values[indice]), dtype=int)
            for j in range(len(dist_values[indice])):
                if dist_values[indice][j] == attribute:
                    new_attribute[j] += 1
            if First:
                new_record = new_attribute
                First = False
            else:
                new_record = np.concatenate((new_record, new_attribute),
                                            axis=0)
            indice += 1
        #print(new_record)
        if First_rec:
            new_data = np.array([new_record])
            First_rec = False
        else:
            new_data = np.concatenate((new_data, np.array([new_record])),
                                      axis=0)
    return new_data


def grad_check(x, y, weights, act_f):
    """
    Procedure for checking the correctness of the computation of
    the gradient.
    @param x: numpy array containing all inputs in TR
    @param y: numpy array representing the corresponding target
        values
    @param weights: tensor of weights
    @param act_f: instance of activation function 
    """
    import sys
    import copy
    total_loss = 0
    eps = 1e-5
    max_err = 0
    ps, inputs, L = feed_forward(x[0], y[0], weights, act_f)
    dL_dweights = backpropagation(L, ps, inputs, weights, y[0], 0, act_f)
    for j in range(len(weights)):
        for a in range(weights[j].shape[0]):
            for b in range(weights[j].shape[1]):
                wpe = copy.deepcopy(weights)
                wme = copy.deepcopy(weights)
                wpe[j][a][b] += eps
                wme[j][a][b] -= eps
                _, _, Lpe = feed_forward(x[0], y[0], wpe, act_f)
                _, _, Lme = feed_forward(x[0], y[0], wme, act_f)
                d_est = (Lpe - Lme) / (2 * eps)
                max_err = max(abs(d_est - dL_dweights[j][a][b]), max_err)
    if max_err > 1e-5:
        print("Max error: {} with activation {}".format(max_err, act_f.name))
