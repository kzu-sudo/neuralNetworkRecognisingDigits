import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

class Neural_network(object):
    def __init__(self, x, y, digits,  m = 0, lr = 0.1, cost = []):
        self.digits = digits
        self.x = x
        self.y = y

        self.lr = lr
        self.cost = cost
        self.m = self.x.shape[0]
        neurons = 100
        self.params = {
                "w1" : np.random.randn(x.shape[1], neurons),
                "b1" : np.zeros((1, neurons)),
                "w2" : np.random.randn(neurons, neurons),
                "b2" : np.zeros((1, neurons)),
                "w3" : np.random.randn(neurons, y.shape[1]),
                "b3" : np.zeros((1, y.shape[1]))
                }

    def decide(self,s, activation_function):
        if activation_function == "sigmoid":
            return self.sigmoid(s)
        elif activation_function == "softmax":
            return self.softmax(s)
        elif activation_function == "relu":
            return self.relu(s)

    def relue_derv(self, s):
        return (1. *(s > 0))

    def relu(self, s):
        return s*(s>0)

    def softmax(self, s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def cross_entropy(self, pred, real):
        return (pred - real) / real.shape[0]

    def error(self, pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def print_cost(self):
        plt.plot(np.squeeze(self.cost))
        plt.ylabel('cost')
        plt.xlabel("iterations (per 1000)")
        plt.title("Learning rate = " + str(self.lr))
        plt.show()

    def feedforward(self):
        L = len(self.params) // 2
        self.f = {}
        cache = self.x
        for l in range(1, L):
            z = np.dot(cache, self.params["w" + str(l)]) + self.params["b" + str(l)]
            self.f["a" + str(l)] = self.decide(z, "relu")#you could use softmax
            cache = self.f["a" + str(l)]

        z3 = np.dot(cache, self.params["w3"]) + self.params["b3"]
        self.f["a3"] = self.decide(z3, "softmax")


    def backprop(self):
        loss = self.error(self.f["a3"], self.y)
        self.cost.append(loss)
        print('Error :', loss)
        L = len(self.params) // 2
        self.k = {}
        self.k["ad3"] = self.cross_entropy(self.f["a3"], self.y)

        for l in reversed(range(1, 3)):
            self.k["zd" + str(l)] = np.dot(self.k["ad" + str(l + 1)], self.params["w" + str(l + 1)].T)
            self.k["ad" + str(l)] = self.k["zd" + str(l)] * self.relue_derv(self.f["a" + str(l)])

        for l in reversed(range(2, 4)):
            self.params["w" + str(l)] -= self.lr * np.dot(self.f["a" + str(l - 1)].T, self.k["ad" + str(l)])
            self.params["b" + str(l)] -= self.lr * np.sum(self.k["ad" + str(l)], axis=0)

        self.params["w1"] -= self.lr * np.dot(self.x.T, self.k["ad1"])
        self.params["b1"] -= self.lr * np.sum(self.k["ad1"], axis=0)


    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.f["a3"].argmax()

def show_exapmple(xx):
    f = xx.reshape((8, 8))
    plt.matshow(f)
    plt.gray()
    plt.show()

def start_function():

    dig = load_digits()
    onehot_target = pd.get_dummies(dig.target)
    x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

    model = Neural_network(x_train/16.0, np.array(y_train), dig)
    epochs = 10000
    for x in range(epochs):
        model.feedforward()
        model.backprop()

    def get_acc(x, y, t=False):
        acc = 0
        index = 0
        model.print_cost()
        for xx,yy in zip(x, y):
            s = model.predict(xx)
            if t and index < 3:
                index += 1
                print(f"real value {np.argmax(yy)}")
                print(f" predictd value {s}")
                show_exapmple(xx)
            if s == np.argmax(yy):
                acc +=1
        return acc/len(x)*100

    print("Training accuracy : ", get_acc(x_train/16, np.array(y_train), False))
    print("Test accuracy : ", get_acc(x_val/16, np.array(y_val), True))

if __name__ == '__main__':
    start_function()
