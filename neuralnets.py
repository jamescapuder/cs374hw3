import numpy as np
import pandas as pd
#import sklearn as sk
import sys
import csv
import random
import math
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer as DV
from sknn.mlp import Classifier, Layer

def proc_tuple(data_tup):
    count=0
    for i in range(len(data_tup[0])):
        if (data_tup[0][i]==data_tup[1][i]):
            count+=1
    ret = count/len(data_tup[0])
    return ret
    

def data_prep(fpath_, seed_, prop_, num_hidden_, num_iters_):
    iris = datasets.load_digits()
    iris_targ = [[x] for x in iris.target]
    insts = []
    with open(fpath_, 'r') as f:
        reader = csv.reader(f)
        keys = next(reader)
        for row in reader:
            temp = {}
            for i in range(len(row)):
                temp[keys[i]] = row[i]
            insts.append(temp)
    #print(insts)
    vectorizer = DV( sparse = False )
    one_hot = vectorizer.fit_transform(insts)
    np.random.seed(seed_)
    np.random.shuffle(one_hot)
    attrs = np.array([x[1:] for x in one_hot])
    labs = np.array([x[0] for x in one_hot])
    x = attrs[:math.floor(len(attrs)*prop_)]
    y = labs[:math.floor(len(labs)*prop_)]
    nn = Classifier(layers=[Layer("Sigmoid", units = num_hidden_), Layer("Softmax")],learning_rate=0.001,n_iter=num_iters_ )
    nn.fit(x, y)
    y_test = labs[math.floor(len(labs)*prop_):]
    y_pred = nn.predict(attrs[math.floor(len(attrs)*prop_):])
    #pred_act = np.concatenate((y_test, y_pred), 1)
    return (y_pred.flatten().tolist(), y_test.tolist())
    
def main():
    try:
        fpath_ = sys.argv[1]
        seed_ = eval(sys.argv[2])
        prop_ = eval(sys.argv[3])
        num_iters_ = eval(sys.argv[4])
        num_hidden_ = eval(sys.argv[5])
        #tupuru = data_prep(fpath_, seed_, prop_,num_iters_, num_hidden_)
        exp_1()
    except IndexError:
        print("usage: python3 neuralnets.py <fpath><seed><proportion><num_iterations><num_hidden>")

def exp_1():
    with open("exp1monks.csv", 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['seed', 'accuracy'])
        for i in range(480,510):
            print(i)
            towrite = proc_tuple(data_prep("monks1.csv", i, .7, 20,5000))
            wr.writerow([i,towrite])
            print("\a")

def exp_4():
    exp_garb = {}
    with open("exp_4_digit_2.csv",'w') as f:
        wr = csv.writer(f)
        wr.writerow(['iterations', 'proportion','seed', 'accuracy'])
        for i in [1000,2000,5000,10000]:
            temp = []
            for j in range(1,11):
                print("hidden: %d, seed %d" %(i, j+80))
                #temp.append(data_prep(j+80, .7, i))
                towrite = proc_tuple(data_prep(j+80, .7, 10, i))
                wr.writerow([i, .7,j+80, towrite])
                #print(proc_tuple(data_prep(j+8, .7, i, 1000)))
            print("\a")
        #print(map(proc_tuple, temp))
        #exp_garb[i] = temp
        #print(temp)
    #for k,v in exp_garb.items():
    #    print(map(proc_tuple, v))
    print("\a")

main()
