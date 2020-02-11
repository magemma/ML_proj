"""
    loss_functions: classes and interfaces for dealing
    with loss functions
    ML 2019 Project
    Dipartimento di Informatica Università di Pisa
    Authors: R. Manetti, G. Martini
    We declare that the content of this file is entirelly
    developed by the authors
"""

import math
import numpy as np
import sys


class loss_interface(object):
    def __init__(self):
        pass

    def compute(self, a, b):
        raise Exception("NotImplementedException")

    def partial(self, df_dp, p):
        raise Exception("NotImplementedException")
        
class mee(loss_interface):
    name = "Mean Euclidean Error"

    def __init__(self):
        loss_interface.__init__(self)

    def compute(self, y, o):
        """
        Implementation of Mean Euclidean Error loss function
        @param y: numpy array representing the target value
        @param o: numpy array representing the output of the model
        @return the value of the function on (y, o)
        """
        return (np.linalg.norm(o - y, 2))**2
                
    def partial(self, y, o):
        """
        Computation of the partial derivative of the MEE loss
        function with respect to vector o
        @param y: numpy array representing the target variable
        @param o: numpy array representing the output
        @return dL/do
        """
        return 2 * (o - y)
        
        
class log_loss(loss_interface):
    name = "Cross-entropy loss function"
    epsilon = 1e-12

    def __init__(self):
        loss_interface.__init__(self)

    def compute(self, y, o):
        """
        Implementation of Cross-entropy loss function
        @param y: numpy array representing the target value
        @param o: numpy array representing the output of the model
        @return the value of the function on (y, o)
        """
        #print(y, o)
        #print(-(y[0]*np.log(o[0][0]+1e-9) - (1-y[0])*np.log(1-o[0][0]+1e-9)))
        o = np.clip(o, self.epsilon, 1. - self.epsilon)
        #print("prediction in loss: {}".format(o[0][0]))
        return -1*( y[0]*np.log(o[0][0]) + (1-y[0])*np.log(1-o[0][0]) )
                
    def partial(self, y, o):
        """
        Computation of the partial derivative of the log loss
        function with respect to vector o
        @param y: numpy array representing the target variable
        @param o: numpy array representing the output
        @return dL/do
        """
        o = np.clip(o, self.epsilon, 1. - self.epsilon)
        #print("prediction in partial loss: {}".format(o[0][0]))
        #print(y)
        #a = -1 * y[0]/o[0][0] + (1-y[0])/(1-o[0][0])
        #print(type(np.array([[a]])))
        #print(type(2 * (o-y)))
        return np.array([[-1 * y[0]/o[0][0] + (1-y[0])/(1-o[0][0])]])
                
        
