"""
    optimizer: classes and interfaces for dealing
    with different optimizers
    ML 2019 Project
    Dipartimento di Informatica Università di Pisa
    Authors: R. Manetti, G. Martini
    We declare that the content of this file is entirelly
    developed by the authors
"""
import copy
import numpy as np


class optimizer_interface(object):
    def __init__(self):
        pass

    def gradient_calculation_point(self, weigths):
        raise Exception("NotImplementedException")

    def update_weights(self, lr, b_size, grad, weights):
        raise Exception("NotImplementedException")


class trivial_momentum:
    def __init__(self, alpha):
        self.v = []
        self.alpha = alpha

    def gradient_calculation_point(self, weights):
        return weights

    # modifies weights in-place
    def update_weights(self, lr, b_size, grad, weights):
        if not self.v:
            self.v = [0 for _ in weights]
        for i in range(len(weights)):
            self.v[i] = self.alpha * self.v[i] - lr * grad[i] / b_size
            weights[i] += self.v[i]


class nesterov_momentum:
    def __init__(self, alpha):
        self.v = []
        self.alpha = alpha

    def gradient_calculation_point(self, weights):
        if not self.v:
            self.v = [0 for _ in weights]
        pt = copy.deepcopy(weights)
        for i in range(len(weights)):
            pt[i] += self.alpha * self.v[i]
        return pt

    # modifies weights in-place
    def update_weights(self, lr, b_size, grad, weights):
        for i in range(len(weights)):
            self.v[i] = self.alpha * self.v[i] - lr * grad[i] / b_size
            weights[i] += self.v[i]


class adam:
    def __init__(self, beta1, beta2, eps):
        self.m = []
        self.v = []
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def gradient_calculation_point(self, weights):
        return weights

    # modifies weights in-place
    def update_weights(self, lr, b_size, grad, weights):
        if not self.v:
            self.v = [0 for _ in weights]
            self.m = [0 for _ in weights]
        self.t += 1
        for i in range(len(weights)):
            scaled_grad = grad[i] / b_size
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * scaled_grad
            self.v[i] = self.beta2 * self.v[i] + (
                1 - self.beta2) * scaled_grad * scaled_grad
            mhat = self.m[i] / (1 - self.beta1**self.t)
            vhat = self.v[i] / (1 - self.beta2**self.t)
            weights[i] -= lr * np.divide(mhat, np.sqrt(vhat) + self.eps)


def get_optimizer(opt):
    if opt[0] == 'trivial':
        return trivial_momentum(*opt[1:])
    if opt[0] == 'adam':
        return adam(*opt[1:])
    if opt[0] == 'nesterov':
        return nesterov_momentum(*opt[1:])
    assert False
