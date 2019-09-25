import csv
import math
import numpy as np
import os
import random
from enum import Enum
from itertools import permutations

import pandas as pd
from sklearn.model_selection import train_test_split


class Method(Enum):
    PAIRS_ALL = 1
    PAIRS_QUERY = 2
    GROUP = 3

class Split(Enum):
    BY_QUERY = 1
    BY_GROUP = 2


class Stats:
    def __init__(self, mae, acc):
        self.acc = acc
        self.mae = mae

class Data:
    def __init__(self, xtrain, ytrain, xtest, ytest, train_queries = None, test_queries = None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.train_queries = train_queries
        self.test_queries = test_queries


class Fnames:
    def __init__(self, test, train):
        self.test = test
        self.train = train

def shrink_classes(df):
    stance_cols = [col for col in df if col.startswith('stance_score')]
    for col in stance_cols:
        for i in range(0, len(df[col])):
            val = df.loc[i, col]
            if val == 2:
                df.loc[i, col] = 1
            if val == 4:
                df.loc[i, col] = 5
                #df[col][i] = 5


def get_data(queries, test_query, method, shrink_scores=False):
    train_queries = {x: y for x, y in queries.items() if x != test_query}
    test_queries = {test_query: queries[test_query]}
    test_df = test_queries[test_query].apply(pd.to_numeric)
    train_dfs = pd.concat(train_queries.values(), ignore_index=True)
    train_dfs = train_dfs.apply(pd.to_numeric)

    if shrink_scores:
        shrink_classes(test_df)
        shrink_classes(train_dfs)
    __, xtrain, ytrain = split_x_y(train_dfs, method)
    __, xtest, ytest = split_x_y(test_df, method)
    return Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest, test_queries=test_queries,
                     train_queries=train_queries)


def gen_test_train_set_group_split(input_dir, train_size, shrink_scores):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv")]
    df = pd.concat([pd.read_csv(input_dir + '\\' + f) for f in example_files], ignore_index=True)
    df = df.apply(pd.to_numeric)
    if shrink_scores:
        shrink_classes(df)
    train_df, test_df = train_test_split(df, train_size=train_size)
    xtrain, ytrain = split_x_y(train_df)
    xtest, ytest = split_x_y(test_df)
    return Data(xtrain, ytrain, xtest, ytest)


def csv_to_df(input_dir, fnames, method, shrink_scores=False):
    queries = {f: pd.read_csv(input_dir + '\\' + f) for f in fnames}
    dfs = pd.concat(list(queries.values()), ignore_index=True)
    df = dfs.apply(pd.to_numeric)
    if shrink_scores:
        shrink_classes(df)
    __, x, y = split_x_y(dfs, method)
    return queries, x, y


def gen_loo_fnames(input_dir, method):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x!= 'all.csv']
    data = []
    for i in range(0, len(example_files)):
        left_out = example_files[i]
        train_fnames = [x for x in example_files if x != left_out]
        data.append(Fnames(test= [left_out], train = train_fnames))
    return data

def get_data2(input_dir, fnames, method):
    train_queries, xtrain, ytrain = csv_to_df(input_dir, fnames.train, method)
    test_queries, xtest, ytest = csv_to_df(input_dir, fnames.test, method)
    return Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,test_queries=test_queries, train_queries= train_queries)

def gen_test_train_set_query_split_loo2(input_dir, method):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x!= 'all.csv']
    data = []
    for i in range(0, len(example_files)):
        lef_out = example_files[i]
        train_fnames = [x for x in example_files if x != lef_out]
        test_fnames = [lef_out]
        train_queries, xtrain, ytrain = csv_to_df(input_dir, train_fnames, method)
        test_queries, xtest, ytest = csv_to_df(input_dir, test_fnames, method)
        data.append(Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,test_queries=test_queries, train_queries= train_queries))
    return data


def get_queries(input_dir, method):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x != 'all.csv']
    queries = {f: pd.read_csv(input_dir + '\\' + f) for f in example_files}
    return queries


def gen_test_train_set_query_split_loo(input_dir, method):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x != 'all.csv']
    data = []
    for i in range(0, len(example_files)):
        left_out = example_files[i]
        train_fnames = [x for x in example_files if x != left_out]
        test_fnames = [left_out]
        train_queries, xtrain, ytrain = csv_to_df(input_dir, train_fnames, method)
        test_queries, xtest, ytest = csv_to_df(input_dir, test_fnames, method)
        data.append(Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,test_queries=test_queries, train_queries= train_queries))
    return data

def gen_test_train_set_query_split(input_dir, train_percent, shrink_scores, excluded=[]):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x not in excluded]
    train_files_num = int(math.ceil(train_percent * len(example_files)))
    random.shuffle(example_files)
    train_fnames = example_files[:train_files_num]
    test_fnames = example_files[train_files_num:]
    train_queries, xtrain, ytrain = csv_to_df(input_dir, train_fnames, shrink_scores)
    test_queries, xtest, ytest = csv_to_df(input_dir, test_fnames, shrink_scores)
    return Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,test_queries=test_queries, train_queries= train_queries)



def test_query_set(method, queries, model):
    errors = []
    accurate = 0
    for query, df in queries.items():
        __, x, y = split_x_y(df, method)
        y_predicted = model.predict(x)
        mean_prediction = np.mean(y_predicted)
        actual_value = np.mean(y)
        prediction = get_class(mean_prediction)
        if prediction - actual_value == 0:
            accurate += 1
        errors.append(np.math.fabs(actual_value - prediction))

        print(' predicted value:' + str(mean_prediction))
        print(' actual value: ' + str(actual_value))
    mae = np.mean(errors)
    acc = accurate / len(queries)
    print('Total absolute squared error:' + str(mae))
    print(' Accuracy:' + str(acc))
    return Stats(mae=mae, acc=acc)

def split_x_y(df, method):
    stance = None
    if method == Method.PAIRS_QUERY:
        stance = df.filter(items=['stance_score1', 'stance_score2'])
        df = df.drop(columns=['stance_score1', 'stance_score2'])
    datatest_array = df.values
    xset = datatest_array[:, 1:]
    yset = datatest_array[:,0]
    return stance, xset, yset

def prepare_dataset_loo(input_dir, method):
    return gen_test_train_set_query_split_loo(input_dir, method)

def prepare_dataset(split, input_dir, train_size, shrink_scores=False, excluded = []):
    if split == Split.BY_GROUP:
        return gen_test_train_set_group_split(input_dir, train_size, shrink_scores)
    if split == Split.BY_QUERY:
        return gen_test_train_set_query_split(input_dir, train_size, shrink_scores, excluded)
        # split x and y (feature and target)


def get_class(score): #TODO - define welll
    if score < 2.5:
        return 1
    elif score < 4:
        return 3
    else:
        return 5
