from __future__ import print_function

import gc

import numpy as np
from builtins import range

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

from learning import dataHelper
from learning.dataHelper import Method, Stats, RANK_METHODS
from learning.networks import TwoLayersNet, Layer


class LearningParams:
    def __init__(self, optimizer, num_epoch, criterion):
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.criterion = criterion

class NNLearner:

    def __init__(self, data, method,  net, params:LearningParams):
        self.net = net
        self.params = params
        self.data = data
        self.method = method

    def train(self):
        for epoch in range(self.params.num_epoch):
            X = Variable(torch.Tensor(self.data.xtrain).float())
            Y = Variable(torch.Tensor(self.data.ytrain).long())

            # feedforward - backprop
            self.params.optimizer.zero_grad()
            out = self.net(X)
            loss = self.params.criterion(out, Y)
            loss.backward()
            self.params.optimizer.step()

            if (epoch) % 50 == 0:
                print('Epoch [%d/%d] Loss: %.4f'
                      % (epoch + 1, self.params.num_epoch, loss.item()))


    def test_query_set(self, queries):
        errors = []
        accurate = 0
        for query, df in queries.items():
            __, x, y = dataHelper.split_x_y(df, self.method)
            X = Variable(torch.Tensor(x).float())
            out = self.net(X)
            _, predicted = torch.max(out.data, 1)
            print(query)
            y_predicted = predicted.numpy()
            rms = np.sqrt(mean_squared_error(y, y_predicted))
            mean_prediction = np.mean(y_predicted)
            actual_value = np.mean(y)
            prediction = dataHelper.get_class(mean_prediction)
            if prediction - actual_value == 0:
                accurate += 1
            errors.append(np.math.fabs(actual_value - prediction))

            print(' predicted value:' + str(mean_prediction))
            print(' actual value: ' + str(actual_value))
            print('root mean squared error:' + str(rms))
        mae = np.mean(errors)
        acc = accurate / len(queries)
        print('Total absolute squared error:' + str(mae))
        print(' Accuracy:' + str(acc))
        return Stats(mae=mae, acc=acc)

    def predict(self, x):
        X = Variable(torch.Tensor(x).float())
        out = self.net(X)
        _, predicted = torch.max(out.data, 1)
        return predicted.numpy()

    def test(self):
        # get prediction
        X = Variable(torch.Tensor(self.data.xtest).float())
        Y = torch.Tensor(self.data.ytest).long()
        out = self.net(X)
        _, predicted = torch.max(out.data, 1)

        print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / len(predicted)))
        print('Train Queries')
        train_stat = dataHelper.test_query_set(self.method, self.data.train_queries, self)
        print('\n \n \n Test Queries')
        test_stats = dataHelper.test_query_set(self.method, self.data.test_queries, self)
        return {'train_stat': train_stat, 'test_stats':test_stats}


    def learn(self, method):
       # self.prepare_dataset()
        self.train()
        return self.test()


def get_network():
    layers = [Layer(input=18, output=40), Layer(input=40, output=10)]
    #layers = [Layer(input=30, output=60), Layer(input=60, output=10)]
    return TwoLayersNet(layers)

def get_parms(net):
    lr = 0.1
    torch.manual_seed(1234)
    num_epoch = 500
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    return LearningParams(num_epoch=num_epoch, optimizer=optimizer, criterion=criterion)

def learn_shallow_net(method, input_dir,net):
    # choose optimizer and loss function
    test_acc = {x: [] for x in RANK_METHODS[method]}
    test_mae = {x: [] for x in RANK_METHODS[method]}
    train_acc = {x: [] for x in RANK_METHODS[method]}
    train_mae = {x: [] for x in RANK_METHODS[method]}
    if method == Method.GROUP_ALL:
        df = pd.read_csv(input_dir + '\\group_features_by_stance.csv')
        queries = dataHelper.get_queries_from_df(df)
    else:
        queries = dataHelper.get_queries(input_dir, method)
    queries_stats = {}
    for test_query in queries:
        queries_stats[test_query] = {}
        for rm in RANK_METHODS[method]:
            queries_stats[test_query][rm] = 0

    #dataset = dataHelper.prepare_dataset_loo(input_dir, method)
    params = get_parms(net)
    #for fnames in fnames_list:
    for test_query in queries:
        data = dataHelper.get_data(queries, test_query, method)
        collected = gc.collect()
        print("Garbage collector: collected", "%d objects." % collected)
        #data = dataHelper.get_data(input_dir, fnames, method)
        learner = NNLearner(data, method, net=net, params=params)
        res = learner.learn(method)
        for rm in RANK_METHODS[method]:
            test_acc[rm].append(res['test_stats'][rm].acc)
            test_mae[rm].append(res['test_stats'][rm].mae)
            train_acc[rm].append(res['train_stat'][rm].acc)
            train_mae[rm].append(res['train_stat'][rm].mae)
            queries_stats[test_query][rm] +=  res['test_stats'][rm].acc

    for rm in RANK_METHODS[method]:
        test_acc_mean = np.mean(test_acc[rm])
        test_mae_mean = np.mean(test_mae[rm])
        train_acc_mean = np.mean(train_acc[rm])
        train_mae_mean = np.mean(train_mae[rm])
        print(str(method) + ' ' + rm)
        print('--------------------------')
        print(str(method) + ' ' + rm + ' test_acc_mean = ' + str(test_acc_mean))
        print(str(method) + ' ' + rm + ' test_mae_mean = ' + str(test_mae_mean))
        print(str(method) + ' ' + rm + ' train_acc_mean = ' + str(train_acc_mean))
        print(str(method) + ' ' + rm + ' train_mae_mean = ' + str(train_mae_mean))

 #   print('query stats:')
 #   for q in queries:
 #       print(q)
  #      stats = ' '
  #      for rm in RANK_METHODS[method]:
  #          stats += str(queries_stats[q][rm]) + ', '
  #      print(stats)
   #     print()


def group_all():
    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\group7'
    layers = [Layer(input=34, output=40), Layer(input=40, output=16)]
    net = TwoLayersNet(layers)
    learn_shallow_net(Method.GROUP_ALL, input_dir, net)

def pairs():
    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\normed\\group5'
    layers = [Layer(input=12, output=30), Layer(input=30, output=17)]
    net = TwoLayersNet(layers)
    learn_shallow_net(Method.PAIRS_QUERY, input_dir, net)

def main():
  #  group_all()
  #  return
#    pairs()
    input_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael\\by_group'
    #net = get_network()
    #layers = [Layer(input=18, output=40), Layer(input=40, output=10)]
    # layers = [Layer(input=30, output=60), Layer(input=60, output=10)]
    #layers = [Layer(input=14, output=35), Layer(input=35, output=13)]
    #layers = [Layer(input=31, output=65), Layer(input=65, output=10)]
    layers = [Layer(input=78, output=20), Layer(input=20, output=10)]

    net = TwoLayersNet(layers)
    learn_shallow_net(Method.GROUP_ALL, input_dir, net)

if __name__ == '__main__':
    main()




#train

