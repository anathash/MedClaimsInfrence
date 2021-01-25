import csv

from sklearn.tree import DecisionTreeClassifier, export_text


class FieldsClassifier:
    def __init__(self, fields_weights_dict, filename):
        self.fields_weights_dict = fields_weights_dict

        self.mname = "Fields_Classifier"
        for f in fields_weights_dict.keys():
            self.mname += '_'+f
        self.filename = filename


    def learn(self, data):
        self.predictions = {}
        with open(self.filename, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                vals = {}
                q = row['query']
                for i in range(1,6):
                    vals[i] = 0
                    for f,w in self.fields_weights_dict.items():
                        vals[i] += float(row[f+str(i)+'_mean'])*w
                sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
                self.predictions[q] = list(sorted_vals)[0][0]

    def predict(self,x):
        return self.predictions[x]

    def model_name(self):
        return self.mname

