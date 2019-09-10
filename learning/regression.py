# Data Preprocessing

# Importing the Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import dataHelper


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


    #df1 = df.head(25)
    #df1.plot(kind='bar',figsize=(16,10))
    #plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    #plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    #plt.show()

    #plt.scatter(data.xtest, data.ytest,  color='gray')
    #plt.plot(data.xtest, y_pred, color='red', linewidth=2)
    #plt.show()
    # Visualising the Decision Tree Regression results

    #X_grid = np.arange(min(X), max(X), 0.1)
    #X_grid = X_grid.reshape((len(X_grid), 1))
    #plt.scatter(X, y, color = 'red')
    #plt.plot(X, regressor.predict(X_grid), color = 'blue')
    #plt.title('Truth or Bluff (Decision Tree Regression)')
    #plt.xlabel('Position level')
    #plt.ylabel('Stance')
    #plt.show()

def test_query_set(model, queries ):
    errors = []
    accurate = 0
    for query, df in queries.items():
        X, Y = dataHelper.split_x_y(df)
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
    print('Total absolute squared error:' + str(np.mean(errors)))
    print(' Accuracy:' + str(accurate / len(queries)))

def test_models(models, data ):
    for model in models:
        print('\n \n' + type(model).__name__)
        learn(model, data)
        print('\n \n Train Queries')
        test_query_set(model, data.train_queries)
        print('\n \n Test Queries')
        test_query_set(model, data.test_queries)

def main():
    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\group1'
    # Importing the dataset
    data = dataHelper.prepare_dataset(dataHelper.Split.BY_QUERY, input_dir, 0.5)
    models = [DecisionTreeRegressor(random_state=0),  LinearRegression(fit_intercept=False, normalize=True)]
    test_models(models, data)

if __name__ == '__main__':
    main()
