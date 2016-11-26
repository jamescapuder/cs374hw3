import numpy as np
import pandas as pd
#import sklearn as sk
import sys
import csv
import random
import math
from tensorflow.contrib import learn
#import tensorflow.contrib.skflow as learn
import tensorflow as tf
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import layers


def pre_proc(dat, hidden, fpath_,prop_, seed_):
    labels = dat["label"]
    dat = dat[[i for i in list(dat.columns) if i != 'label']]
    attribs = {}
    tempdir = "/"+fpath_[:len(fpath_)-4]+"/temp"
    print(tempdir)
    print(dat.describe())
    #labs = pd.get_dummies(labels)
    #dat = pd.get_dummies(dat)
    #for col in list(dat.columns.values):
        #attribs[col] = pd.get_dummies(dat[col])
        
        #attribs[col] = tf.contrib.layers.sparse_column_with_keys(column_name=col, keys=dat[col].unique())
    print(len(dat.columns.values))
    feat_cols = [tf.contrib.layers.real_valued_column("", dimension=len(dat.columns.values))]
    X_train, X_dev, y_train, y_dev = train_test_split(dat, labels, test_size=(1-prop_), random_state=seed_)
    #feature_cols = list(attribs.values())

    dnn = tf.contrib.learn.DNNClassifier(feature_columns=feat_cols, hidden_units=[hidden], n_classes=2, model_dir=tempdir)
    dnn.fit(X_train, y_train, steps=1000)
    
    
    
#def read_dat(datvect):
    #training = pd.DataFrame(datvect[1], columns=datvect[0])
    #testing = pd.DataFrame(datvect[2], columns=datvect[0])
    #eturn [training, testing]


def data_prep(fpath_, seed_, prop_):
    #testfile = "test"+fpath_+str(seed_)+".csv"
    #trainfile = "train"+fpath_+str(seed_)+".csv"
    # with open(fpath_, 'r') as file:
    #     contain = []
    #     readr = csv.reader(file)
    #     header = next(readr)
    #     for row in readr:
    #         contain.append(row)
    #     random.seed(seed_)
    #     random.shuffle(contain)
    #     train = contain[0:math.floor(prop_*len(contain))]
    #     test = contain[math.floor(prop_*len(contain)):len(contain)-1]
    # trainarray = np.array(train)
    # testarray = np.array(test)
    dat = pd.read_csv(fpath_)
    #labels = dat["label"]
    return dat

def main():
    try:
        fpath_ = sys.argv[1]
        seed_ = eval(sys.argv[2])
        prop_ = eval(sys.argv[3])
        num_iters_ = eval(sys.argv[4])
        num_hidden_ = eval(sys.argv[5])
        setv = data_prep(fpath_,seed_,prop_)
        #datar  = read_dat(setv)
        pre_proc(setv,num_hidden_, fpath_,prop_,seed_)
    except IndexError:
        print("usage: python3 neuralnets.py <fpath><seed><proportion><num_iterations><num_hidden>")
main()
