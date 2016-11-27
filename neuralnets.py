import numpy as np
import pandas as pd
#import sklearn as sk
import sys
import csv
import random
import math
from sklearn import datasets
from sknn.mlp import Classifier, Layer




def data_prep(seed_, prop_):
    iris = datasets.load_iris()
    iris_targ = [[x] for x in iris.target]
    #print(iris_full)
    x_train = np.concatenate((iris.data, iris_targ), 1)
    np.random.seed(seed_)
    np.random.shuffle(x_train)
    #print(x_train)
    iris_data = np.array([x[:3] for x in x_train])
    iris_targ = np.array([x[4] for x in x_train])
    x = iris_data[:math.floor(len(iris_data)*prop_)]
    y = iris_targ[:math.floor(len(iris_targ)*prop_)]
    #print(x)
    #print(y)
    nn = Classifier(layers=[Layer("Sigmoid", units = 50), Layer("Softmax")],learning_rate=0.001,n_iter=1000 )
    nn.fit(x, y)
    y_example = nn.predict(iris_data[math.floor(len(iris_data)*prop_):])
    print(y_example)
    
def main():
    try:
        fpath_ = sys.argv[1]
        seed_ = eval(sys.argv[2])
        prop_ = eval(sys.argv[3])
        num_iters_ = eval(sys.argv[4])
        num_hidden_ = eval(sys.argv[5])
        data_prep(seed_, prop_)
        #datar  = read_dat(setv)
        #pre_proc(setv,num_hidden_, fpath_,prop_,seed_)
    except IndexError:
        print("usage: python3 neuralnets.py <fpath><seed><proportion><num_iterations><num_hidden>")
main()
