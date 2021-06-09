#!/usr/bin/python3

import numpy as np

from functions import Functions

fu1 = Functions()

class forwardPropagation(object):

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache

    
    def linear_activation_forward(self, A_prev, W, b, activation):

        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = fu1.sigmoid(Z)
        elif activation =="relu":
            A, activation_cache = fu1.relu(Z)
        assert (A.shape == W.shape[0], A_prev.shape[1])
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):

        caches = []
        A = X
        L = len(parameters) // 2   

        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev,parameters['W'+str(l)],\
                    parameters['b'+str(l)],"relu")
            caches.append(cache)
            
        AL, cache = self.linear_activation_forward(A,parameters['W'+str(L)],\
                parameters['b'+str(L)],"sigmoid")
        caches.append(cache)

        return AL, caches
