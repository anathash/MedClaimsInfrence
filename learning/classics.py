# Data Preprocessing

# Importing the Library
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from learning import dataHelper
from learning.dataHelper import Stats


def learn(regressor, data):
    # Fitting Simple Linear Regression model to the data set
    #linear_regressor = LinearRegression()
    X = data.xtrain
    y = data.ytrain
    regressor.fit(X,y)
    # Predicting a new result
    y_pred = regressor.predict(data.xtest)

    df = pd.DataFrame({'Actual': data.ytest.flatten(), 'Predicted': y_pred.flatten()})
    #print(df)

    r_sq = regressor.score(data.xtrain, data.ytrain)
    print('coefficient of determination:', r_sq)

    print('Mean Absolute Error:', metrics.mean_absolute_error(data.ytest, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(data.ytest, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(data.ytest, y_pred)))

def test_models(models, queries, split, method):
    train_acc = {}
    test_acc = {}
    train_mae = {}
    test_mae = {}
    for model in models:
        model_name = type(model).__name__
        print('\n \n' + model_name)
        train_acc[model_name] = 0
        test_acc[model_name] = 0
        train_mae[model_name] = 0
        test_mae[model_name] = 0
        for test_query in queries:
            data = dataHelper.get_data(queries, test_query, method)
            learn(model, data)
            if split == dataHelper.Split.BY_QUERY:
                print('\n \n Train Queries')
                stats = dataHelper.test_query_set(method, data.train_queries, model)
                train_acc[model_name] += stats.acc
                train_mae[model_name] += stats.mae
                print('\n \n Test Queries')
                stats = dataHelper.test_query_set(model, data.test_queries, model)
                test_acc[model_name] += stats.acc
                test_mae[model_name] += stats.mae

        train_acc[model_name] /= len(queries)
        test_acc[model_name] /= len(queries)
        train_mae[model_name] /= len(queries)
        test_mae[model_name] /= len(queries)

    train_acc = OrderedDict(sorted(train_acc.items(), key=lambda x: x[1], reverse=True))
    test_acc = OrderedDict(sorted(test_acc.items(), key=lambda x: x[1], reverse=True))
    print (' \n\nTrain')
    for model in train_acc.keys():
        print('model: ' + model + ' acc: ' + str(train_acc[model]) + ' mae: ' + str(train_mae[model]))
    print(' \n\nTest')
    for model in test_acc.keys():
        print('model: ' + model + ' acc: ' + str(test_acc[model]) + ' mae: ' + str(test_mae[model]))

def run_infrence(input_dir, method):
    #data = dataHelper.prepare_dataset_loo(input_dir, method)
    queries = dataHelper.get_queries(input_dir, method)
    #DecisionTreeClassifier(random_state=0), svm.SVC(gamma='scale')
    models = [DecisionTreeRegressor(random_state=0),
              LinearRegression(fit_intercept=False, normalize=True),
              svm.SVC(gamma='scale'),
              DecisionTreeClassifier(random_state=0)]
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

def main():
    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\perm\\group1'
    run_infrence(input_dir, dataHelper.Method.GROUP)

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
