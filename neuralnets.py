import tensorflow as tf
import sys
import csv
import random
import math

def read_dat(datvect):
    training = tf.contrib.learn.datasets.base.load_csv_with_header(filename=datvect[0],target_dtype=tf.string,features_dtype=tf.string)
    testing = tf.contrib.learn.datasets.base.load_csv_with_header(filename=datvect[1], target_dtype=tf.string,features_dtype=tf.string)
    print(training)
    print(testing)


def data_prep(fpath_, seed_, prop_):
    testfile = "test"+fpath_+str(seed_)+".csv"
    trainfile = "train"+fpath_+str(seed_)+".csv"
    with open(fpath_, 'r') as file:
        contain = []
        readr = csv.reader(file)
        header = next(readr)
        for row in readr:
            contain.append(row)
        random.seed(seed_)
        random.shuffle(contain)
        train = contain[0:math.floor(prop_*len(contain))]
        test = contain[math.floor(prop_*len(contain)):len(contain)-1]
    with open(trainfile,'w') as file:
        wr = csv.writer(file)
        wr.writerow(header)
        for i in train:
            wr.writerow(i)
    with open(testfile,'w') as file:
        wr2 = csv.writer(file)
        wr2.writerow(header)
        for i in test:
            wr2.writerow(i)
    return [trainfile,testfile]

def main():
    try:
        fpath_ = sys.argv[1]
        seed_ = eval(sys.argv[2])
        prop_ = eval(sys.argv[3])
        num_iters_ = eval(sys.argv[4])
        num_hidden_ = eval(sys.argv[5])
        setv = data_prep(fpath_,seed_,prop_)
        read_dat(setv)
    except IndexError:
        print("usage: python3 neuralnets.py <fpath><seed><proportion><num_iterations><num_hidden>")
    
main()
