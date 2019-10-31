# Data Preprocessing

# Importing the Library
import csv
from collections import OrderedDict

import numpy as np
import pandas
import pandas as pd
from sklearn import metrics, svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from learning import dataHelper
from learning.dataHelper import Stats, RANK_METHODS, get_queries_from_df, create_report_file, MajorityClassifier

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

def visualize(model):
    feature_cols1 =['h_index_mean', 'h_index_std', 'stance_score_mean',
     'stance_score_std', 'current_score_mean', 'current_score_std',
     'recent_weighted_citation_count_mean',
     'recent_weighted_citation_count_std', 'recent_weighted_h_index_mean',
     'recent_weighted_h_index_std', 'citation_count_mean',
     'citation_count_std', 'contradicted_by_later_mean',
     'contradicted_by_later_std', 'num_ir']
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['1', '5'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    Image(graph.create_png())


def learn(regressor, data):
    feature_cols_stance = ['stance_1','stance_2','stance_3','stance_4','stance_5','h_index1_mean','current_score1_mean','recent_weighted_citation_count1_mean','recent_weighted_h_index1_mean','citation_count1_mean','h_index2_mean','current_score2_mean','recent_weighted_citation_count2_mean','recent_weighted_h_index2_mean','citation_count2_mean','h_index3_mean','current_score3_mean','recent_weighted_citation_count3_mean','recent_weighted_h_index3_mean','citation_count3_mean','h_index4_mean','current_score4_mean','recent_weighted_citation_count4_mean','recent_weighted_h_index4_mean','citation_count4_mean','h_index5_mean','current_score5_mean','recent_weighted_citation_count5_mean','recent_weighted_h_index5_mean','citation_count5_mean','rel']
    fc2 = ['stance_1','stance_3','stance_5','h_index1_mean','current_score1_mean','recent_weighted_citation_count1_mean',
           'recent_weighted_h_index1_mean','citation_count1_mean','contradicted_by_later1_mean','h_index3_mean',
           'current_score3_mean','recent_weighted_citation_count3_mean','recent_weighted_h_index3_mean','citation_count3_mean',
           'contradicted_by_later3_mean','h_index5_mean','current_score5_mean',
           'recent_weighted_citation_count5_mean','recent_weighted_h_index5_mean','citation_count5_mean','contradicted_by_later5_mean','num_ir']

    # Fitting Simple Linear Regression model to the data set
    #linear_regressor = LinearRegression()
    X = data.xtrain
    y = data.ytrain
    regressor.fit(X,y)
    #tree_rules = export_text(regressor, feature_names=feature_cols_stance)
    #print(tree_rules)
    #visualize(regressor)
    # Predicting a new result
    y_pred = regressor.predict(data.xtest)

    df = pd.DataFrame({'Actual': data.ytest.flatten(), 'Predicted': y_pred.flatten()})
    #print(df)

    r_sq = regressor.score(data.xtrain, data.ytrain)
    print('coefficient of determination:', r_sq)

 #   print('Mean Absolute Error:', metrics.mean_absolute_error(data.ytest, y_pred))
 #   print('Mean Squared Error:', metrics.mean_squared_error(data.ytest, y_pred))
 #   print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data.ytest, y_pred)))


def x_fold(num_folds, queries):
    keys = list(queries.keys())
    test_queries = []
    queries_per_fold = round(len(queries) / num_folds)
    start_index = 0
    for i in range(0, num_folds):
        end_index = min(len(keys), start_index+queries_per_fold)
        fold = [keys[i] for i in range(start_index, end_index)]
        test_queries.append(fold)
        start_index += queries_per_fold
    return test_queries

def test_models(models, queries, split, method):
    test_queries = x_fold(10,queries)
    train_acc = {}
    test_acc = {}
    train_mae = {}
    test_mae = {}
    queries_stats = {}
    for test_query in queries:
        queries_stats[test_query] = {}
        for model in models:
            model_name = type(model).__name__
            queries_stats[test_query][model_name] = {}
            for rm in RANK_METHODS[method]:
                queries_stats[test_query][model_name][rm] = 0

    predictions = {}
    for model in models:
        model_name = type(model).__name__
        predictions[model_name] = {}
        print('\n \n' + model_name)
        train_acc[model_name] = {x: 0 for x in RANK_METHODS[method]}
        test_acc[model_name] = {x: 0 for x in RANK_METHODS[method]}
        train_mae[model_name] = {x: 0 for x in RANK_METHODS[method]}
        test_mae[model_name] = {x: 0 for x in RANK_METHODS[method]}
        for test_query in queries:
            if test_query == 'all.csv':
                continue
            data = dataHelper.get_data(queries, test_query, method)
            learn(model, data)
            if split == dataHelper.Split.BY_QUERY:
                print(test_query)
  #              print('\n \n Train Queries')
                stats = dataHelper.test_query_set(method, data.train_queries, model)
                for rm, stats in stats.items():
                    train_acc[model_name][rm] += stats.acc
                    train_mae[model_name][rm] += stats.mae
   #             print('\n \n Test Queries')
                stats = dataHelper.test_query_set(method, data.test_queries, model)
                predictions[model_name][test_query] = stats['group'].predictions[test_query]
                for rm, stats in stats.items():
                    test_acc[model_name][rm] += stats.acc
                    test_mae[model_name][rm] += stats.mae
                    queries_stats[test_query][model_name][rm] += stats.mae
        for rm in RANK_METHODS[method]:
            train_acc[model_name][rm] /= len(queries)
            test_acc[model_name][rm] /= len(queries)
            train_mae[model_name][rm] /= len(queries)
            test_mae[model_name][rm] /= len(queries)


    #train_acc = OrderedDict(sorted(train_acc.items(), key=lambda x: x[1], reverse=True))
    #test_acc = OrderedDict(sorted(test_acc.items(), key=lambda x: x[1], reverse=True))
    #print(' \n\nTrain')
    #for model in train_acc.keys():
        #print('model: ' + model + ' acc: ' + str(train_acc[model]) + ' mae: ' + str(train_mae[model]))
        #for rm in RANK_METHODS[method]:

    #print(' \n\nTest')
    for model in test_acc.keys():
        print ('model: '  + model + '\n')
        #print('model: ' + model + ' acc: ' + str(test_acc[model]) + ' mae: ' + str(test_mae[model]))
        for rm in RANK_METHODS[method]:
            print (rm + '\n')
            print(' train acc: ' + str(train_acc[model][rm]) + ' test mae: ' + str(train_mae[model][rm]))
            print(' test acc: ' + str(test_acc[model][rm]) + ' test mae: ' + str(test_mae[model][rm]))
        print ('query stats:')
    for q in queries:
        q_stat = {}
        q_stat['query_name'] = q

        #print(q)
        stats = ' '
        for model in test_acc.keys():
            q_stat[model] = queries_stats[q][model]['group']
            stats += model + ': '
            for rm in RANK_METHODS[method]:
                stats += str(queries_stats[q][model][rm]) + ', '
        return predictions
        #print(stats)
        #print()




def run_infrence(input_dir, method, models):
    #data = dataHelper.prepare_dataset_loo(input_dir, method)
    queries = dataHelper.get_queries(input_dir, method)
    #DecisionTreeClassifier(random_state=0), svm.SVC(gamma='scale')
    test_models(models, queries, dataHelper.Split.BY_QUERY, method)

def test_query_set2(model, queries, method ):
    errors = []
    accurate = 0
    for query, df in queries.items():
        __, X, Y = dataHelper.split_x_y(df, method)
        predicted_y = model.predict(X)
        mean_prediction = np.mean(predicted_y)
        actual_value = np.mean(Y)
        prediction = dataHelper.get_class(mean_prediction)
        if prediction - actual_value == 0:
            accurate += 1
        errors.append(np.math.fabs(actual_value - prediction))
        print(query)
        print(' predicted value:' + str(np.mean(predicted_y)))
        print(' actual value: ' + str(np.mean(Y)))
    mae = np.mean(errors)
    acc = accurate / len(queries)
    print('Total mean absolute mean error:' + str(mae))
    print(' Accuracy:' + str(acc))
    return Stats(mae, acc)


def group():
 models = [#DecisionTreeRegressor(random_state=0),
              LinearRegression(fit_intercept=False, normalize=True),
              #svm.SVC(gamma='scale'),
              DecisionTreeClassifier(random_state=0)]
 input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\group3'
 run_infrence(input_dir, dataHelper.Method.GROUP, models)


def pairs():

    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\normed\\group8'
    models = [DecisionTreeClassifier(random_state=0),
     DecisionTreeRegressor(random_state=0)]
       #           svm.SVC(gamma='scale')]
    #              DecisionTreeClassifier(random_state=0)]
    run_infrence(input_dir, dataHelper.Method.PAIRS_QUERY, models)

def group_all():
    #input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\normed\\group7'
    input_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael\\by_group'
    models = [#DecisionTreeRegressor(random_state=0),
              DecisionTreeClassifier(random_state=0)]
              #LinearRegression()]

    #df = pd.read_csv(input_dir + '\\group_features.csv')
    df = pd.read_csv(input_dir + '\\group_features_by_stance.csv')
    #df = pd.read_csv(input_dir + '\\group_features_by_paper_type.csv')
    queries = get_queries_from_df(df)
    labels = {q:int(queries[q].label) for q in queries}
    mc = MajorityClassifier(input_dir + '\\majority.csv')
    predictions = test_models(models, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL)
    create_report_file(input_dir + '\\group_features_by_stance_report.csv',queries=queries,
                       models = models,predictions=predictions,majority_classifier=mc,labels=labels
                       )

def main():
    pandas.set_option('display.max_rows', 50)
    pandas.set_option('display.max_columns', 50)
    pandas.set_option('display.width', 1000)  # Clerical work:
    #pairs()
    #group()
    group_all()
    #    models = [DecisionTreeRegressor(random_state=0),
#              LinearRegression(fit_intercept=False, normalize=True),
#              svm.SVC(gamma='scale'),
#              DecisionTreeClassifier(random_state=0)]

    #input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\perm\\group1'
    #run_infrence(input_dir, dataHelper.Method.GROUP)

    # Importing the dataset
#    filenames = os.listdir(input_dir)
#    excluded = [x for x in filenames if x.endswith(".csv")]
#    for f in excluded:
#        print(f)
#        data = dataHelper.prepare_dataset(dataHelper.Split.BY_QUERY, input_dir, 0.8, shrink_scores=False)
#        models = [DecisionTreeClassifier(random_state=0), DecisionTreeRegressor(random_state=0)]
        #models = [DecisionTreeClassifier(random_state=0), DecisionTreeRegressor(random_state=0),  LinearRegression(fit_intercept=False, normalize=True), svm.SVC(gamma='scale')]
#        test_models(models, data)


if __name__ == '__main__':
    main()
