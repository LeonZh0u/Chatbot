import logging
import re
import sys
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)
if __name__ == "__main__":
    txt = pd.read_csv("task3_train.txt",sep="\t",names=list("abc"),error_bad_lines=False)
    print(txt.head)
    dir_path = os.path.abspath(os.path.dirname(__file__))
    result = [txt]
    files = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]
    
    for filename in sorted(files):
        if (filename[-3:]=="csv"):
            print(filename)
            data = pd.read_csv(filename, sep='\t',names=list("abc"))
            print(data.head())
            result.append(data)
            
    combined = pd.concat(result)
    combined.columns=["question1","question2","label"]
    combined.to_csv("combined.tsv")
    X= combined.drop("label",axis=1)
    y= combined.label
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    X_train["label"]=y_train
    X_test["label"]=y_test
    X_val["label"]=y_val
    
    
    X_train.to_csv("train.tsv")
    X_test.to_csv("test.tsv")
    X_val.to_csv("dev.tsv")

    
                
                