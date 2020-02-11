"""
    activation_functions: classes and interfaces for dealing
    with activation functions
    ML 2019 Project
    Dipartimento di Informatica Università di Pisa
    Authors: R. Manetti, G. Martini
    We declare that the content of this file is entirelly
    developed by the authors
"""

import math
import numpy as np
import sys


class f_sigma_interface(object):
    def __init__(self):
        pass

    def compute(self, p):
        raise Exception("NotImplementedException")

    def chain_rule(self, df_dp, p):
        raise Exception("NotImplementedException")


class logistic(f_sigma_interface):
    name = "logistic"

    def __init__(self):
        f_sigma_interface.__init__(self)

    def compute(self, p):
        """Implementation of sigmoid activation function
        @param p: numpy array representing the input of f_sigma
        @return the value of the function on p
        """
        #print(type(1 / (1 + np.exp(-p))))
        #print(type(np.array([[5]])))
        l = []
        for i in range(len(p)):
           #If p negative and very big i abs value -> overflow
            if ((-1 * p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                a = sys.float_info.max
            else:
                a = np.exp(- p[i][0])
            l.append([1/(1+a)])
        return np.array(l)
                
    def chain_rule(self, df_dp, p):
        """
        Computation of the partial derivative of the composition
        of a function f and the sigmoid activation function with
        respect to p
        @param df_dp: numpy array representing the partial
            derivative of f with respect to p
        @param p: numpy array
        @return the partial derivative of the composition
            of f with relu with respect to vector p
        """
        
        l = []
        for i in range(len(p)):
           #If p negative and very big i abs value -> overflow
            if ((-1 * p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                a = sys.float_info.max
            else:
                a = np.exp(- p[i][0])
            if ((p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                b = sys.float_info.max
            else:
                b = np.exp(p[i][0]) 
            l.append([2+b+a])
        l = np.array(l)
        # Drop known-term value (bias) from part_fp
        return np.divide(df_dp[:-1], l)
        #return np.divide(df_dp[:-1], (2 + np.exp(p) + np.exp(-p)))


class tanh(f_sigma_interface):
    name = "tanh"

    def __init__(self):
        f_sigma_interface.__init__(self)

    def compute(self, p):
        """Implementation of hyperbolic tangent activation function
        @param p: numpy array representing the input of f_sigma
        @return the value of the function in p
        """
        l = []
        for i in range(len(p)):
           #If p negative and very big i abs value -> overflow
            if ((-1 * p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                a = sys.float_info.max
            else:
                a = np.exp(- p[i][0])
            if ((p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                b = sys.float_info.max
            else:
                b = np.exp(p[i][0]) 
            l.append([(b-a)/(b+a)])
        l = np.array(l)
        return l
        #return (np.exp(p) - np.exp(-p)) / (np.exp(p) + np.exp(-p))

    def chain_rule(self, df_dp, p):
        """
        Computation of the partial derivative of the composition
        of a function f and the sigmoid activation function with
        respect to p
        @param df_dp: numpy array representing the partial
            derivative of f with respect to p
        @param p: numpy array
        @return the partial derivative of the composition
            of f with relu with respect to vector p
        """
        l = []
        for i in range(len(p)):
           #If p negative and very big i abs value -> overflow
            if ((-2 * p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                a = sys.float_info.max
            else:
                a = np.exp(-2* p[i][0])
            if (2*(p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                b = sys.float_info.max
            else:
                b = np.exp(2*p[i][0]) 
            l.append([2+b+a])
        l = np.array(l)
        # Drop known-term value (bias) from part_fp
        return np.divide(4 * df_dp[:-1], l)
        # Drop known-term value (bias) from part_fp.
        #return np.divide(4 * df_dp[:-1], (np.exp(2 * p) + np.exp(-2 * p) + 2))


class relu(f_sigma_interface):
    name = "relu"

    def __init__(self):
        f_sigma_interface.__init__(self)

    def compute(self, p):
        """Efficient implementation of RELU activation function
        @param p: numpy array representing the input of RELU
        @return the value of the function on p
        """
        #print(p.shape)
        return p * (p >= 0)

    def chain_rule(self, df_dp, p):
        """
        Computation of the partial derivative of the composition
        of a function f and the relu activation function with
        respect to p
        @param df_dp: numpy array representing the partial
            derivative of f with respect to p
        @param p: numpy array
        @return the partial derivative of the composition
            of f with relu with respect to vector p
        """
        # Drop known-term value (bias) from part_fp.
        out = df_dp[:-1] * (p >= 0)
        #print(out.shape)
        return out


class softplus(f_sigma_interface):
    name = "softplus"

    def __init__(self):
        f_sigma_interface.__init__(self)

    def compute(self, p):
        """Efficient implementation of softplus activation function
        @param p: numpy array representing the input of softplus
        @return the value of the function on p
        """
        l = []
        for i in range(len(p)):
           #If p negative and very big i abs value -> overflow
            if ((p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                b = sys.float_info.max
            else:
                b = np.exp(p[i][0]) 
            l.append([1+b])
        l = np.array(l)
        return np.log(l)
        #return np.log(1 + np.exp(p))

    def chain_rule(self, df_dp, p):
        """
        Computation of the partial derivative of the composition
        of a function f and the softmax activation function with
        respect to p
        @param df_dp: numpy array representing the partial
            derivative of f with respect to p
        @param p: numpy array
        @return the partial derivative of the composition
            of f with relu with respect to vector p
        """
        l = []
        for i in range(len(p)):
           #If p negative and very big in abs value -> overflow
            if ((-1 * p[i][0]) > np.log(sys.float_info.max) -1):
                #print(p)
                a = sys.float_info.max
            else:
                a = np.exp(-1 * p[i][0])
            l.append([1+a])
        l = np.array(l)
        # Drop known-term value (bias) from part_fp
        return np.divide(df_dp[:-1], l)
        # Drop known-term value (bias) from part_fp.
        #return np.divide(df_dp[:-1], 1 + np.exp(-p))
