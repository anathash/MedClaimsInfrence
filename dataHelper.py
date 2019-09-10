import csv
import math
import os
import random
from enum import Enum

import pandas as pd
from sklearn.model_selection import train_test_split


class Split(Enum):
    BY_QUERY = 1
    BY_GROUP = 2


class Data:
    def __init__(self, xtrain, ytrain, xtest, ytest, train_queries = None, test_queries = None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.train_queries = train_queries
        self.test_queries = test_queries


def gen_test_train_set_group_split(input_dir, train_size):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv")]
    df = pd.concat([pd.read_csv(input_dir + '\\' + f) for f in example_files], ignore_index=True)
    train_df, test_df = train_test_split(df, train_size=train_size)
    xtrain, ytrain = split_x_y(train_df)
    xtest, ytest = split_x_y(test_df)
    return Data(xtrain, ytrain, xtest, ytest)


def csv_to_df(input_dir, fnames):
    queries = {f: pd.read_csv(input_dir + '\\' + f) for f in fnames}
    dfs = pd.concat(list(queries.values()), ignore_index=True)
    x, y = split_x_y(dfs)
    return queries, x, y


def gen_test_train_set_query_split(input_dir, train_percent):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv")]
    train_files_num = int(math.ceil(train_percent * len(example_files)))
    random.shuffle(example_files)
    train_fnames = example_files[:train_files_num]
    test_fnames = example_files[train_files_num:]
    train_queries, xtrain, ytrain = csv_to_df(input_dir, train_fnames)
    test_queries, xtest, ytest = csv_to_df(input_dir, test_fnames)
    return Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,test_queries=test_queries, train_queries= train_queries)


def split_x_y(df):
    df = df.apply(pd.to_numeric)
    datatest_array = df.values
    xset = datatest_array[:, 1:]
    yset = datatest_array[:,0]
    return xset, yset


def prepare_dataset(split, input_dir, train_size):
    if split == Split.BY_GROUP:
        return gen_test_train_set_group_split(input_dir, train_size)
    if split == Split.BY_QUERY:
        return gen_test_train_set_query_split(input_dir, train_size)
        # split x and y (feature and target)


def get_class(score):
    if score <= 1.5:
        return 1
    elif score <= 3:
        return 3
    else:
        return 5
