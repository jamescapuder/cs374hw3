import numpy as np
import pandas as pd
#import sklearn as sk
import sys
import csv
import random
import math
from sklearn import datasets
from sknn.mlp import Classifier, Layer

def data_prep(seed_, prop_, num_hidden_):
    iris = datasets.load_iris()
    iris_targ = [[x] for x in iris.target]
    x_train = np.concatenate((iris.data, iris_targ), 1)
    np.random.seed(seed_)
    np.random.shuffle(x_train)
    iris_data = np.array([x[:3] for x in x_train])
    iris_targ = np.array([x[4] for x in x_train])
    x = iris_data[:math.floor(len(iris_data)*prop_)]
    y = iris_targ[:math.floor(len(iris_targ)*prop_)]
    nn = Classifier(layers=[Layer("Sigmoid", units = num_hidden_), Layer("Softmax")],learning_rate=0.001,n_iter=2000 )
    nn.fit(x, y)
    y_test = iris_targ[math.floor(len(iris_targ)*prop_):]
    y_pred = nn.predict(iris_data[math.floor(len(iris_data)*prop_):])
    #pred_act = np.concatenate((y_test, y_pred), 1)
    print(y_pred)
    print(y_test)
    
def main():
    try:
        fpath_ = sys.argv[1]
        seed_ = eval(sys.argv[2])
        prop_ = eval(sys.argv[3])
        num_iters_ = eval(sys.argv[4])
        num_hidden_ = eval(sys.argv[5])
        #data_prep(seed_, prop_)
        #datar  = read_dat(setv)
        #pre_proc(setv,num_hidden_, fpath_,prop_,seed_)
        exp_3()
    except IndexError:
        print("usage: python3 neuralnets.py <fpath><seed><proportion><num_iterations><num_hidden>")


def exp_3():
    for i in [5,10,20,50]:
        for j in range(1,11):
            print("hidden: %d, seed %d" %(i, j+80))
            data_prep(j+80, .7, i)

main()
