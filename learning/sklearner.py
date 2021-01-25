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

class SKLearner:
    def __init__(self, model, features = None, resample = False):
        #self.selector = SelectFromModel(estimator=model)
        self.model = model
        self.features = features
        self.resample = resample
        self.mname = type(model).__name__

    def learn(self, data):
        feature_cols_stance = ['stance_1', 'stance_2', 'stance_3', 'stance_4', 'stance_5', 'h_index1_mean',
                               'current_score1_mean', 'recent_weighted_citation_count1_mean',
                               'recent_weighted_h_index1_mean', 'citation_count1_mean', 'h_index2_mean',
                               'current_score2_mean', 'recent_weighted_citation_count2_mean',
                               'recent_weighted_h_index2_mean', 'citation_count2_mean', 'h_index3_mean',
                               'current_score3_mean', 'recent_weighted_citation_count3_mean',
                               'recent_weighted_h_index3_mean', 'citation_count3_mean', 'h_index4_mean',
                               'current_score4_mean', 'recent_weighted_citation_count4_mean',
                               'recent_weighted_h_index4_mean', 'citation_count4_mean', 'h_index5_mean',
                               'current_score5_mean', 'recent_weighted_citation_count5_mean',
                               'recent_weighted_h_index5_mean', 'citation_count5_mean', 'rel']
        X = data.xtrain
        y = data.ytrain
        if self.resample:
            #
            #X_fit, y_fit = over.fit_resample(X,y)
            #under = RandomUnderSampler(sampling_strategy=0.5)
            #steps = [('over', over), ('under', under), ('model', self.model)]
            #pipeline = Pipeline(steps=steps)
            #pipeline.fit_resample(X, y)

            #under = RandomUnderSampler(sampling_strategy={ 3:15, 4:15, 5:15 })
            #pipeline = Pipeline(steps=[('o', over), ('u', under)])
            #undersample = RandomUnderSampler('majority')
            #X_under, y_under = pipeline.fit_resample(X, y)
            #X_under, y_under = undersample.fit_resample(X, y)
            # summarize class distribution
            #over = SMOTE()
            #X_fit, y_fit = over.fit_resample(X, y)
            #under = RandomUnderSampler(sampling_strategy={3: 50})
            over = SMOTE()
            X_fit, y_fit = over.fit_resample(X, y)
            print(Counter(y_fit))
            self.model.fit(X_fit,y_fit)
        else:
            print(Counter(y))
            self.model.fit(X, y)
#        if self.features:
#            tree_rules = export_text(self.model, feature_names=self.features)
#            print(tree_rules)


    def quick_learn(self, queries):
        dfs = pd.concat(queries.values(), ignore_index=True)
        dfs = dfs.apply(pd.to_numeric)
        datatest_array = dfs.values
        X = datatest_array[:, 1:]
        y = datatest_array[:, 0]
        resample = False
        if resample:
            #over = RandomOverSampler(sampling_strategy={1: 30, 2: 30})
            under = RandomUnderSampler(sampling_strategy={ 3:15, 4:15, 5:15 })
            #pipeline = Pipeline(steps=[('o', over), ('u', under)])
            #undersample = RandomUnderSampler('majority')
            #X_under, y_under = pipeline.fit_resample(X, y)
            X, y = under.fit_resample(X, y)
            # summarize class distribution
            print(Counter(y))
#        self.model.fit(X, y)

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_validate(self.model, X, y,  cv=cv, n_jobs=-1)
        print(scores)

    def predict(self,x):
        return self.model.predict(x)

    def model_name(self):
        return self.mname

