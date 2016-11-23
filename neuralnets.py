import sklearn
import pandas as pd
import sys
import csv
import random
import math

def read_dat(datvect):
    training = pd.read_csv(datvect[0])
    testing = pd.read_csv(datvect[1])
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
        train = [header].extend(contain[0:math.floor(prop_*len(contain))])
        test = [header].extend(contain[math.floor(prop_*len(contain)):len(contain)-1])
        
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
