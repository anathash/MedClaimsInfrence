from itertools import chain

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text

from learning.dataHelper import ValToClassMode, split_x_y, Method

FOUR_SCORE_TO_CLASS = {1:{1:True,2:True,3:False,4:False,5:False},
                       3:{1:False, 2: False, 3: True, 4: False, 5: False},
                       4:{1:False,2:False,3:False,4:True,5:False},
                       5:{1:False,2:False,3:False,4:False,5:True}}

THREE_SCORE_TO_CLASS_PESS = {1:{1:True,2:True,3:False,4:False,5:False},
                        3:{1:False, 2: False, 3: True, 4: True, 5: False},
                        5:{1:False,2:False,3:False,4:False,5:True}}


THREE_SCORE_TO_CLASS_OPT = {1:{1:True,2:True,3:False,4:False,5:False},
                        3:{1:False, 2: False, 3: True, 4: False, 5: False},
                        5:{1:False,2:False,3:False,4:True,5:True}}

W_H_SCORE_TO_CLASS  = {1:{1:True,2:False,3:False},
                        2:{1:False, 2: True, 3: False},
                        3:{1:False,2:False,3:True}}


SCORE_TO_CLASS = {ValToClassMode.THREE_CLASSES_PESSIMISTIC:THREE_SCORE_TO_CLASS_PESS,
                  ValToClassMode.THREE_CLASSES_OPTIMISTIC:THREE_SCORE_TO_CLASS_OPT,
                  ValToClassMode.FOUR_CLASSES: FOUR_SCORE_TO_CLASS
                  }

class DiffLearnerCls:
    def __init__(self, model):
        self.mname = 'DiffLearnerCls'
        self.diff_model = model

    def learn(self, queries, test_query):
        train_queries = {x: y for x, y in queries.items() if x != test_query and test_query not in x}
        diff_train_queries = [x for x in train_queries.values()]
        diff_train_queries = list(chain(*diff_train_queries))
        train_dfs = pd.concat(diff_train_queries)
        train_dfs = train_dfs.apply(pd.to_numeric)
        stance_train, X, Y = split_x_y(train_dfs, Method.GROUP_ALL)
        self.diff_model.fit(X, Y)

        #learn diff
#X is a list of papers

    def predict(self, x ):
        diffs = []
        for p in x:
            df = p.apply(pd.to_numeric)
            stance_train, X, Y = split_x_y(df, Method.GROUP_ALL)
            prediction = self.diff_model.predict(X)
            diffs.append(X.flat[0] - prediction)
        print(diffs)
        m = np.mean(diffs)
        print(m)
        return min(int(m),5)



    def model_name(self):
        return self.mname

