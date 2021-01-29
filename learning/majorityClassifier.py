from learning.dataHelper import VAL_TO_CLASS_DICT


class MajorityClassifier:
    def __init__(self, mode):
        self.val2class = VAL_TO_CLASS_DICT[mode]
        self.prediction = None
        self.mname = 'majCls'

    def learn(self, data):
        votes = {x:0 for x in self.val2class.values()}
        test = list(data.test_queries.values())[0]
        for i in range(1,6):
            field = 'stance_'+str(i)+'_votes'
            if field in test:
                cls = self.val2class[i]
                votes[cls] += int(test[field])
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        self.prediction = sorted_votes[0][0]

    def predict(self, __):
        return self.prediction

    def model_name(self):
        return self.mname
