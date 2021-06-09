#!/usr/bin/python3

import numpy as np

class Functions(object):


    def error(self, pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def softmax(self, s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def cross_entropy(self, pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples

    def relu_backward(self, dA, activation_cache):
        #z/dw
        return dA * (1. * (activation_cache > 0))

    def sigmoid_backward(self, dA, activation_cache):
        #z/dw 
        k, f= self.sigmoid(activation_cache)
        return dA * (k*(1- k))

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z)),Z

    def relu(self, Z):
        return Z*(Z>0), Z
