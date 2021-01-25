import csv
import numpy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

SCORE_TO_CLASS = {ValToClassMode.THREE_CLASSES_PESSIMISTIC:THREE_SCORE_TO_CLASS_PESS,
                  ValToClassMode.THREE_CLASSES_OPTIMISTIC:THREE_SCORE_TO_CLASS_OPT,
                  ValToClassMode.FOUR_CLASSES: FOUR_SCORE_TO_CLASS
                  }


class KMeansClassifier:
    def __init__(self, dist_file):
        self.mname = 'KMeansClassifier'
        self.dist_file = dist_file

    def get_centroids(self):
        self.features = {}
        categories = [1, 2, 3, 4, 5]
        empty_tuple = ('', {('theta' + str(x)): 0.0 for x in categories})
        centroids = {x: empty_tuple for x in categories}
        with open(self.dist_file, 'r', encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                entry = {'theta' + str(x): float(row['theta' + str(x)]) for x in categories}
                entry['thetaz'] = row['thetaz']
                self.features[row['query']] = entry
                for i in categories:
                    ti = 'theta' + str(i)
                    if centroids[i][1][ti] < float(row[ti]):
                        centroids[i] = (row['query'],entry)

        centroid_queries = set([x[0] for x in centroids.values()])
        assert (len(centroid_queries) == 5)
        centroid_features = [list(x[1].values()) for x in centroids.values()]
        return numpy.array(centroid_features)

    def gen_predictions(self):
        categories = [1,2,3,4,5]
        clusters = {0: [], 1: [], 2: [], 3: [], 4: []}
        labels = {}
        cluster_labels = {}
        for query, features in self.features.items():
            features_arr = numpy.array([list(features.values())])
            cluster = self.model.predict(features_arr)
            assert (len(cluster) == 1)
            int_cluster = int(cluster[0])
            clusters[int_cluster].append(features)
            labels[query] = int_cluster
        for index, cluster in clusters.items():
            tau = {x: 0 for x in categories}
            for example in cluster:
                for i in categories:
                    tau[i] += example['theta' + str(i)]
            sorted_tau = sorted(tau.items(), key=lambda kv: kv[1], reverse=True)
            cluster_labels[index] = sorted_tau[0][0]
   #     print(cluster_labels)
        self.predictions = {x: cluster_labels[y] for x, y in labels.items()}

    def gen_cluster_labels_dict(self):
        categories = [1, 2, 3, 4, 5]
        clusters = {0: [], 1: [], 2: [], 3: [], 4: []}
        labels = {}
        self.cluster_labels = {}
        for query, features in self.features.items():
            features_arr = numpy.array([list(features.values())])
            cluster = self.model.predict(features_arr)
            assert (len(cluster) == 1)
            int_cluster = int(cluster[0])
            clusters[int_cluster].append(features)
            labels[query] = int_cluster
        for index, cluster in clusters.items():
            tau = {x: 0 for x in categories}
            for example in cluster:
                for i in categories:
                    tau[i] += example['theta' + str(i)]
            sorted_tau = sorted(tau.items(), key=lambda kv: kv[1], reverse=True)
            self.cluster_labels[index] = sorted_tau[0][0]
        print( self.cluster_labels)

    def learn(self, data):
        centroids = self.get_centroids()
        X = pd.DataFrame.from_dict(self.features.values()).to_numpy()
        self.model = KMeans(n_clusters=5, init=centroids,  n_init=1)
        #self.model.fit(X)
        res =self.model.fit_predict(X)
        #self.gen_predictions()
        self.gen_cluster_labels_dict()
        #print(res)

    def predict(self,x):
      #  print(x)
        #return self.predictions(x)
        #return self.model.predict(x)
        #return self.model.predict(x)[0] +1
        return self.cluster_labels[self.model.predict(x)[0]]


    def model_name(self):
        return self.mname

