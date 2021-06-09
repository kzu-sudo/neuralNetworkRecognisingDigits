#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from v1 import neural_network

def start():
    digits = load_digits()
    
    images_and_labels=list(zip(digits.images,digits.target))
    plt.figure(figsize=(5,5))

    for index,(image,label) in enumerate(images_and_labels[:15]):
        plt.subplot(3,5,index+1)
        plt.axis('off')
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title('%i' % label)

    pl.show()

    n_samples=len(digits.images)
    print("number of samples:"+ str(n_samples))

    x=digits.images.reshape((n_samples,-1))
    print("shape of input "+str(x.shape))
    y=digits.target
    print("shape of target vector :"+str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train = X_train.T
    X_test = X_test.T
    y_train=y_train.reshape(y_train.shape[0],1)
    y_test=y_test.reshape(y_test.shape[0],1)
    y_train=y_train.T
    y_test=y_test.T

   

    Y_train_ = np.zeros((10,y_train.shape[1]))
    for i in range(y_train.shape[1]):
        Y_train_[y_train[0,i],i] = 1

    Y_test_ = np.zeros((10,y_test.shape[1]))
    for i in range(y_test.shape[1]):
        Y_test_[y_test[0,i],i] = 1

    n_x=X_train.shape[0]
    n_h=10
    n_y=Y_train_.shape[0]
    

    m = neural_network(X_train, Y_train_, y_train, y_test, X_test, n_samples)
    m.start(digits, sc, y)

if __name__ == '__main__':
    start()
