import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

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

W_H_SCORE_TO_CLASS  = {1:{1:True,2:False,3:False},
                        2:{1:False, 2: True, 3: False},
                        3:{1:False,2:False,3:True}}


SCORE_TO_CLASS = {ValToClassMode.THREE_CLASSES_PESSIMISTIC:THREE_SCORE_TO_CLASS_PESS,
                  ValToClassMode.THREE_CLASSES_OPTIMISTIC:THREE_SCORE_TO_CLASS_OPT,
                  ValToClassMode.FOUR_CLASSES: FOUR_SCORE_TO_CLASS
                  }

class MultipleBinaryCls:
    def __init__(self, mode):
        self.mname = 'MultipleBinaryCls'
        self.mode = mode

    def learn(self, data):
        X = data.xtrain
        y = data.ytrain
        self.models = {}
        score_to_class_dict = SCORE_TO_CLASS[self.mode]
        for cls in score_to_class_dict.keys():
            model = RandomForestClassifier(random_state=0)
       #W     model = LogisticRegression()
            func = lambda x: int(score_to_class_dict[cls][x])
            y_func = np.vectorize(func)
           # cls_y = y_func(y)
            cls_y = [score_to_class_dict[cls][x] for x in y]
           # undersample = RandomUnderSampler('majority')
            X_fit, y_fit = X,cls_y
            #X_fit, y_fit = undersample.fit_resample(X, cls_y)
#            print(y)
#            print(cls_y)
          #  model.fit(X,cls_y)
            model.fit(X_fit, y_fit)
            r_sq = model.score(data.xtrain, cls_y)
            print('coefficient of determination:', r_sq)
            self.models[cls] = model
        self.learn_prediction_model(X,y)

#        tree_rules = export_text(self.model, feature_names=feature_cols_stance)

     #   print(tree_rules)

    def learn_prediction_model(self, X, Y):
        x_list = [x.reshape(1, 104) for x in X]
        x_train = []
        for i in range(0, len(x_list)):
            probs = {}
            x = x_list[i]
            for cls, model in self.models.items():
                #            print(model.predict(x))
                probs[cls] = model.predict_proba(x)[0][0]

            x_train.append( np.array(list(probs.values())))
        self.prediction_model  = LinearRegression()
        self.prediction_model.fit(np.array(x_train), Y)

    def predict_old(self,x):
        probs = {}
        for cls, model in self.models.items():
#            print(model.predict(x))
            probs[cls] = model.predict_proba(x)[0][0]
#            print(probs[cls])
#        print(probs)
        sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_probs[0][0]

    def predict(self,x):
        probs = {}
        for cls, model in self.models.items():
            #            print(model.predict(x))
            probs[cls] = model.predict_proba(x)[0][0]
        #            print(probs[cls])
        #        print(probs)
        x_probs  = np.array(list(probs.values())).reshape(1,-1)
        pred =   self.prediction_model.predict(x_probs)
        return min (pred, 5)


    def model_name(self):
        return self.mname

