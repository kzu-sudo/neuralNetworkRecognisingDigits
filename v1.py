#!/usr/bin/python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
import pylab as pl

from functions import Functions
from forward import forwardPropagation
from backward import backwardPropagation


fu1 = Functions()
forwardProp = forwardPropagation()
backwardProb = backwardPropagation()

class neural_network(object):
    def __init__(self, X_train, Y_train, y_train, y_test, X_test, n_samples,\
            layer_dims=[64, 60, 10, 10], learning_rate=0.01, num_iterations=30000,\
            print_cost=True):

        self.X_train = X_train
        self.Y_train = Y_train
        self.y_train = y_train
        self.y_test = y_test
        self.X_test = X_test
        self.n_samples = n_samples

        self.layer_dims = layer_dims 
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
    

    def initialize_parameters_deep(self):
        np.random.seed(11)
        parameters = {}
        L = len(self.layer_dims) 
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l],self.layer_dims[l-1])*0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l],1))
        assert (parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l-1]))
        assert (parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
        return parameters
        
    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -(1/m)*np.sum((Y*np.log(AL)+(1-Y)*np.log(1-AL)))
        cost=np.squeeze(cost)
        assert (cost.shape == ())

        return cost
    
    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2  # number of layers in the neural network
        for l in range(1, L + 1, 1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - self.learning_rate * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - self.learning_rate * grads["db" + str(l)]

        return parameters
    

    def L_layer_model(self, X, Y, layer_dims):
        np.random.seed(1)
        costs = [] 
        parameters = self.initialize_parameters_deep()
        for i in range(0, self.num_iterations):
            AL, caches = forwardProp.L_model_forward(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = backwardProb.L_model_backward(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads)
            if self.print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if self.print_cost and i % 1000 == 0:
                costs.append(cost)

        plt.plot(np.squeeze(costs)) #plot the cost
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()

        return parameters

    def predict_L_layer(self, X,parameters):
        AL,caches = forwardProp.L_model_forward(X,parameters)
        prediction=np.argmax(AL,axis=0)

        return prediction.reshape(1,prediction.shape[0])

    def start(self, digits, sc, y):
        print(f"layer: {self.layer_dims}")

        parameters = self.L_layer_model(self.X_train, self.Y_train, self.layer_dims)
        predictions_train_L = self.predict_L_layer(self.X_train, parameters)
        print("Training Accuracy : "+ str(np.sum(predictions_train_L==self.y_train)/self.y_train.shape[1] * 100)+" %")
        predictions_test_L=self.predict_L_layer(self.X_test,parameters)
        print("Testing Accuracy : "+ str(np.sum(predictions_test_L==self.y_test)/self.y_test.shape[1] * 100)+" %")

        #test function 
        for j in range(15):
            i=random.randint(0,self.n_samples)
            pl.gray()
            pl.matshow(digits.images[i])
            pl.show()
            img=digits.images[i].reshape((64,1)).T
            img = sc.transform(img)
            img=img.T
            predicted_digit=self.predict_L_layer(img,parameters)
            print('Predicted digit is : '+str(predicted_digit))
            print('True digit is: '+ str(y[i]))
