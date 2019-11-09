from sklearn.tree import DecisionTreeClassifier, export_text


class SKLearner:
    def __init__(self, model):
        self.model = model
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
        self.model.fit(X,y)
        r_sq = self.model.score(data.xtrain, data.ytrain)
        print('coefficient of determination:', r_sq)
#        tree_rules = export_text(self.model, feature_names=feature_cols_stance)

     #   print(tree_rules)

    def predict(self,x):
        return self.model.predict(x)

    def model_name(self):
        return self.mname

