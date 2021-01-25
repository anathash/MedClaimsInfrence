from collections import Counter
from statistics import mean
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text

class MultipleCls:
    def __init__(self,  knn, rfc, features = None, resample = False):
        #self.selector = SelectFromModel(estimator=model)
        self.models = {'knn':knn,'rfc':rfc}
        self.features = features
        self.resample = resample
        self.mname = 'Ensemble'

    def learn(self, data):
        X = data.xtrain
        y = data.ytrain
        if self.resample:
            over = SMOTE()
            X_fit, y_fit = over.fit_resample(X, y)
            print(Counter(y_fit))
            self.models['knn'].fit(X_fit,y_fit)
            self.models['rfc'].fit(X_fit,y_fit)

        else:
            print(Counter(y))
            self.models['knn'].fit(X,y)
            self.models['rfc'].fit(X,y)


    def predict(self,x):
        knn_predict = self.models['knn'].predict(x)
        rfc_predict = self.models['rfc'].predict(x)

        return min(rfc_predict, knn_predict)


    def model_name(self):
        return self.mname

