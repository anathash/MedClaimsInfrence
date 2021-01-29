from collections import Counter
from statistics import mean
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

from learning.majorityClassifier import MajorityClassifier


class EnsembleLearner:
    def __init__(self, val2class, features = None, resample = False):
        self.models = {'knn': KNeighborsClassifier(n_neighbors=5),
                       'rfc': RandomForestClassifier(random_state=0),
                       'knn_smote': KNeighborsClassifier(n_neighbors=5),
                       'rfc_smote': RandomForestClassifier(random_state=0),
                       'knn_1_3': KNeighborsClassifier(n_neighbors=5),
                       'rfc_1_3': RandomForestClassifier(random_state=0),
                       'knn_3_5': KNeighborsClassifier(n_neighbors=5), #3/5?
                       'rfc_3_5': RandomForestClassifier(random_state=0),
                       'maj': MajorityClassifier(val2class)}
        self.features = features
        self.resample = resample
        self.mname = 'EnsembleLearner'

    def learn(self, data):
        X = data.xtrain
        y = data.ytrain
        over = SMOTE()
        X_fit, y_fit = over.fit_resample(X, y)
        self.models['maj'].learn(data)
        self.models['knn'].fit(X, y)
        self.models['rfc'].fit(X, y)
        self.models['knn_smote'].fit(X_fit, y_fit)
        self.models['rfc_smote'].fit(X_fit, y_fit)
        one_
        cls_y = [score_to_class_dict[cls][x] for x in y]


    def predict(self,x):
        knn_predict = self.models['knn'].predict(x)
        rfc_predict = self.models['rfc'].predict(x)
        maj_predict = self.models['maj'].predict(x)
        if maj_predict != 5:
            return maj_predict
        #if knn_predict == 5 or rfc_predict == 5:
        #    return 5

        return min(rfc_predict, knn_predict)


    def model_name(self):
        return self.mname

