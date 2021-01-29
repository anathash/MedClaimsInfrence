import csv
import math

import numpy as np
import os
import random
from enum import Enum

import pandas as pd
from sklearn.model_selection import train_test_split

#LABEL_FILE = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\queries.csv'

class Method(Enum):
    PAIRS_ALL = 1
    PAIRS_QUERY = 2
    GROUP = 3
    GROUP_ALL = 4


class ValToClassMode(Enum):
    THREE_CLASSES_PESSIMISTIC = 1
    THREE_CLASSES_OPTIMISTIC = 2
    FOUR_CLASSES = 3
    W_H = 4
    BINARY = 5

VAL_TO_CLASS_DICT = {ValToClassMode.THREE_CLASSES_PESSIMISTIC:{1:1, 2:1, 3:3, 4:3, 5:5},
                     ValToClassMode.THREE_CLASSES_OPTIMISTIC:{1:1, 2:1, 3:3, 4:5, 5:5},
                     ValToClassMode.BINARY:{1:1, 2:1, 3:1, 4:5, 5:5}
                     }
REJECT = 1
NEUTRAL = 3
INITIAL_SUPPORT = 4
SUPPORT = 5



#LABEL_PREDICTION_FUNCS = {Method.PAIRS_ALL: group_prediction,
#                          Method.PAIRS_ALL: pairs_prediction,
#                          Method.PAIRS_QUERY:pairs_prediction}


RANK_METHODS = {Method.GROUP_ALL: ['group'],
                Method.GROUP: ['group'],
                Method.PAIRS_QUERY:['voting', 'avg_label'],
                Method.PAIRS_ALL: ['voting', 'avg_label']}

class Split(Enum):
    BY_QUERY = 1
    BY_GROUP = 2

class Stats:
    def __init__(self, mae, acc, predictions):
        self.acc = acc
        self.mae = mae
        self.predictions = predictions


class Prediction:
    def __init__(self, mean_prediction, class_prediction):
        self.mean_prediction = mean_prediction
        self.class_prediction = class_prediction


class DHMajorityClassifier:
    def __init__(self, majority_file):
        self.predictions = {}
        with open(majority_file, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                value = int(row['majority_value'])
                self.predictions[row['query']] = Prediction(value,get_class(value) )

    def get_predictions(self):
        return self.predictions


class Data:
    def __init__(self, xtrain, ytrain, stance_train, xtest, ytest,stance_test, train_queries = None, test_queries = None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.train_queries = train_queries
        self.test_queries = test_queries
        self.stance_train = stance_train
        self.stance_test = stance_test


class FNames:
    def __init__(self, test, train):
        self.test = test
        self.train = train


#def shrink_classes(df):
#    return
#    stance_cols = [col for col in df if col.startswith('stance_score')]
#    for col in stance_cols:
#        for i in range(0, len(df[col])):
#            val = df.loc[i, col]
#            if val == 2:
#                df.loc[i, col] = 1
#            if val == 4:
#                df.loc[i, col] = 5


def get_all_pairs_train_dfs(queries, test_query):
    df = queries['all.csv']
    testq_split = test_query.split('.')[0]
    train = df.loc[(df.query1 != testq_split) & (df.query2 != testq_split)]
    return train.drop(columns=['query1', 'query2'])

def get_data_new(queries, test_query, method, shrink_scores=False):
    train_queries = {x: y for x, y in queries.items() if x != test_query and test_query not in x}
    test_queries = queries[test_query]
    print(train_queries)
    if method == Method.PAIRS_QUERY:
        test_df = [x.apply(pd.to_numeric) for x in test_query]
        train_dfs = [pd.concat(x.values(), ignore_index=True) for x in train_queries]
        train_dfs = [x.apply(pd.to_numeric) for x in test_queries]
    else:
        test_df = test_query.apply(pd.to_numeric)
        train_dfs = pd.concat(train_queries.values(), ignore_index=True)
        train_dfs = train_dfs.apply(pd.to_numeric)

    stance_train, xtrain, ytrain = split_x_y(train_dfs, method)
    stance_test, xtest, ytest = split_x_y(test_df, method)
    return Data(xtrain=xtrain, ytrain=ytrain, stance_train = stance_train, xtest=xtest, ytest=ytest,
                stance_test = stance_test, test_queries=test_queries, train_queries=train_queries)



def get_data(queries, test_query, method, shrink_scores=False):
    train_queries = {x: y for x, y in queries.items() if x != test_query and test_query not in x}
    test_queries = {test_query: queries[test_query]}
    test_df = test_queries[test_query].apply(pd.to_numeric)
    if method == Method.PAIRS_ALL:
        train_dfs = get_all_pairs_train_dfs(queries, test_query)
    else:
        train_dfs = pd.concat(train_queries.values(), ignore_index=True)
    train_dfs = train_dfs.apply(pd.to_numeric)

#    if shrink_scores:
#        shrink_classes(test_df)
#        shrink_classes(train_dfs)
    stance_train, xtrain, ytrain = split_x_y(train_dfs, method)
    stance_test, xtest, ytest = split_x_y(test_df, method)
    return Data(xtrain=xtrain, ytrain=ytrain, stance_train = stance_train, xtest=xtest, ytest=ytest,
                stance_test = stance_test, test_queries=test_queries, train_queries=train_queries)


def gen_test_train_set_group_split(input_dir, train_size, shrink_scores):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv")]
    df = pd.concat([pd.read_csv(input_dir + '\\' + f) for f in example_files], ignore_index=True)
    df = df.apply(pd.to_numeric)
#    if shrink_scores:
#        shrink_classes(df)
    train_df, test_df = train_test_split(df, train_size=train_size)
    xtrain, ytrain = split_x_y(train_df)
    xtest, ytest = split_x_y(test_df)
    return Data(xtrain, ytrain, xtest, ytest)


def gen_loo_fnames(input_dir, method):
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x!= 'all.csv']
    data = []
    for i in range(0, len(example_files)):
        left_out = example_files[i]
        train_fnames = [x for x in example_files if x != left_out]
        data.append(FNames(test= [left_out], train = train_fnames))
    return data

def get_data2(input_dir, fnames, method):
    train_queries, xtrain, ytrain = csv_to_df(input_dir, fnames.train, method)
    test_queries, xtest, ytest = csv_to_df(input_dir, fnames.test, method)
    return Data(xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest,test_queries=test_queries, train_queries= train_queries)


def csv_to_df(input_dir, fnames, method, shrink_scores=False):
    queries = {f: pd.read_csv(input_dir + '\\' + f) for f in fnames}
    dfs = pd.concat(list(queries.values()), ignore_index=True)
    df = dfs.apply(pd.to_numeric)
#    if shrink_scores:
#        shrink_classes(df)
    __, x, y = split_x_y(dfs, method)
    return queries, x, y

def get_queries(input_dir, method):
    if method == Method.GROUP_ALL:
        queries = {}
        df = pd.read_csv(input_dir + '\\group_features.csv')
        for i in range(0, len(df.index)):
            qname = df.iloc[i].loc['query']
            queries[qname] = df.iloc[i].to_frame().transpose().drop(columns=['query'])
        return  queries
    filenames = os.listdir(input_dir)
    example_files = [x for x in filenames if x.endswith(".csv") and x != 'all.csv']
    if method == Method.PAIRS_ALL:
        example_files.append('all.csv')
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


def group_prediction(y_predicted, class2val):
    mean_prediction = np.mean(y_predicted)
    class_prediction = get_class(mean_prediction, class2val)
    return Prediction(mean_prediction=mean_prediction, class_prediction = class_prediction)


def pairs_prediction(stance, y_predicted):
    ranking = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    acc = 0
    for i in range(0, len(y_predicted)):
        cmp = y_predicted[i]
        #cmp = random.randrange(2)
        if not (cmp ==0.0 or cmp ==1.0 or cmp ==2.0):
            assert False
        if cmp == 0.0:
            ranking[stance.stance_score1[i]] += 1
            acc += 1
        elif cmp == 1.0:
            ranking[stance.stance_score1[i]] += 1
            acc += 1
        elif cmp == 2.0:
            ranking[stance.stance_score2[i]] += 1
            acc += 1
    sorted_ranks = sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)
    voting_mean = sorted_ranks[0][0]
    voting_class = get_class(voting_mean)
    avg_mean = sum([k*v for k, v in ranking.items()]) / acc
    avg_class = get_class(avg_mean)
    return {'voting': Prediction(mean_prediction= voting_mean, class_prediction = voting_class),
            'avg_label':Prediction(mean_prediction= avg_mean, class_prediction = avg_class)}
    return voting_label, int(round(avg_label))


#def get_labels():
#    labels = {}
#    with open(LABEL_FILE, encoding='utf-8', newline='') as queries_csv:
#        reader = csv.DictReader(queries_csv)
#        for row in reader:
#            labels[row['short query']] = int(row['label'])
#    return labels


def test_query_set_pairs(method, queries, model):
    labels = get_labels()
    errors = {'voting': 0,'avg_label':0}
    accurate = {'voting':0,'avg_label':0}
    for query, df in queries.items():
        if query == 'all.csv':
            continue
        stance, x, y = split_x_y(df, method)
        y_predicted = model.predict(x)
        predictions = pairs_prediction(stance, y_predicted)
        query_name = query.split('.')[0]
        #actual_value = np.mean(y)
        actual_class = labels[query_name]
        for rank_method, prediction in predictions.items():
            if prediction.class_prediction - actual_class == 0:
                accurate[rank_method] += 1
            errors[rank_method] += np.math.fabs(actual_class - prediction.mean_prediction)
            print(rank_method + ' predicted value:' + str(prediction.mean_prediction))
            print('Actual class: ' + str(actual_class))
    mae = {x: (y/ len(queries)) for x,y in errors.items()}
    mean_acc = {x: (y/len(queries)) for x,y in accurate.items()}
    for rank_method in accurate.keys():
        print(rank_method + ' total mean absolute error:' + str(mean_acc[rank_method]))
        print(rank_method + ' accuracy:' + str(mean_acc[rank_method]))
    return {'voting':Stats(mae=mae['voting'], acc=mean_acc['voting']),
            'avg_label': Stats(mae=mae['avg_label'], acc=mean_acc['avg_label'])}

def test_group_query_set(queries, learner, class2val, method, labels):
    errors = 0
    accurate = 0
    predictions = {}
    for query, df in queries.items():
        if not labels:
            __, x, y = split_x_y(df, Method.GROUP)
            y_predicted = learner.predict(x)
        else:
            y_predicted = learner.predict(df)
            y = labels[query]
        prediction = group_prediction(y_predicted, class2val)
        predictions[query] = prediction
        actual_value = group_prediction(y, class2val)
        if prediction.class_prediction - actual_value.class_prediction == 0:
            accurate += 1
        errors += np.math.fabs(actual_value.mean_prediction - prediction.mean_prediction)

#        print(' class predicted value:' + str(prediction.class_prediction))
#        print(' mean_prediction predicted value:' + str(prediction.mean_prediction))
#        print(' actual value: ' + str(actual_value))
    mae = errors/ len(queries)
    acc = accurate / len(queries)
#    print('Total absolute squared error:' + str(mae))
#    print(' Accuracy:' + str(acc))
    return {'group':Stats(mae=mae, acc=acc, predictions = predictions)}


def test_query_set(method, queries, learner, class2val, labels):
    return test_group_query_set(queries, learner, class2val, method, labels)



def split_x_y(df, method):
    stance = None
    if method == Method.PAIRS_QUERY or method == Method.PAIRS_ALL:
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


def get_class(score, mode:ValToClassMode): #TODO - define welll
    int_score = int(round(score))
    return  VAL_TO_CLASS_DICT[mode][int_score]
 #   val_to_class= {}
 #   val_to_class[ValToClassMode.THREE_CLASSES_PESSIMISTIC] = {-2:0, -1:0,1:REJECT,2:REJECT,3:NEUTRAL,4:NEUTRAL,5:SUPPORT} #neutral_wins
 #   val_to_class[ValToClassMode.THREE_CLASSES_OPTIMISTIC] = {-2:0, -1: 0, 1: REJECT, 2: REJECT, 3: NEUTRAL, 4: SUPPORT, 5: SUPPORT}  # support_wins
 #   val_to_class[ValToClassMode.FOUR_CLASSES] = {-2:0, -1: 0, 1: REJECT, 2: REJECT, 3: NEUTRAL, 4: INITIAL_SUPPORT, 5: SUPPORT}  # 4 classess
  #  val_to_class[ValToClassMode.W_H] = {-2:0, -1: 0, 0:0, 1: 1, 2: 2, 3: 3}  # 3 classess
#    val_to_class[ValToClassMode.W_H] = {-2:0, -1: 0, 0:0, 1: REJECT, 2: NEUTRAL, 3: SUPPORT}  # 3 classess
 #   return val_to_class[mode][int_score]



#def get_class(score, mode:ValToClassMode): #TODO - define welll
#    return score



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


def get_queries_from_df(df):
    queries = {}
    for i in range(0, len(df.index)):
        qname = df.iloc[i].loc['query']
        qname = qname.strip()
        queries[qname] = df.iloc[i].to_frame().transpose().drop(columns=['query']).apply(pd.to_numeric)
    return queries

def get_queries_from_pairs_df(df):
    queries = {}
    for i in range(0, len(df.index)):
        id = df.iloc[i].loc['id']
        qname = id.split('_')[0].strip()
        if not qname in queries:
            queries[qname] = []
        queries[qname].append(df.iloc[i].to_frame().transpose().drop(columns=['id']).apply(pd.to_numeric))
    return queries

def create_report_file(report_fname,queries, learners, predictions, majority_classifier,labels):
    with open(report_fname, 'w', encoding='utf-8', newline='') as out:
        model_names = [learner.model_name() for learner in learners]
        model_names.append('majority')
        predictions['majority'] = majority_classifier.get_predictions()
        fieldnames = ['query', 'value_label', 'class_label']
        fieldnames.extend([model_name+'_class' for model_name in model_names])
        fieldnames.extend([model_name+'_value' for model_name in model_names])
        fieldnames.extend([model_name+'_acc' for model_name in model_names])
        fieldnames.extend([model_name+'_error' for model_name in model_names])
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for q in queries:
            value_label = labels[q]
            class_label = get_class(value_label)
            row = {'query':q, 'value_label': value_label, 'class_label':class_label}
            for model_name in model_names:
                row[model_name+"_class"] = predictions[model_name][q].class_prediction
                row[model_name + "_value"] = predictions[model_name][q].mean_prediction
                row[model_name + "_acc"] = int(class_label == predictions[model_name][q].class_prediction)
                row[model_name + "_error"] = math.fabs(value_label - predictions[model_name][q].mean_prediction)
            writer.writerow(row)

