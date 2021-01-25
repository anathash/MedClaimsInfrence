import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text

from learning.dataHelper import ValToClassMode

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


class ExpectedValLearner:
    def __init__(self):
        self.mname = 'ExpectedValLearner'
        self.model = GaussianNB()

    def learn(self, data):
        X = data.xtrain
        y = data.ytrain
        self.model.fit(X, y)

    def predict(self,x):
        probs =  self.model.predict_proba(x)
        exp = 0
        for i in range(1,6):
            exp+=probs[0][i-1]*i
        print('exp = ' + str(exp))
        print('predicted val = ' + str(self.model.predict(x)[0]))
        return exp

    def model_name(self):
        return self.mname

