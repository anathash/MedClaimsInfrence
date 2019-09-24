from __future__ import print_function

import numpy as np
from builtins import range

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

from learning import dataHelper
from learning.dataHelper import Method, Stats
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

        # get accuration
        print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / len(predicted)))
        print('Train Queries')
        #train_stat = self.test_query_set(self.data.train_queries)
        train_stat = dataHelper.test_query_set(self.method, self.data.train_queries, self)
        print('\n \n \n Test Queries')
        #test_stats = self.test_query_set(self.data.test_queries)
        test_stats = dataHelper.test_query_set(self.method, self.data.test_queries, self)
        return {'train_stat':train_stat, 'test_stats':test_stats}


    def learn(self, method):
       # self.prepare_dataset()
        self.train()
        return self.test()


def get_network():
    layers = [Layer(input=18, output=40), Layer(input=40, output=10)]
    return TwoLayersNet(layers)

def get_parms(net):
    lr = 0.1
    torch.manual_seed(1234)
    num_epoch = 1500
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    return LearningParams(num_epoch=num_epoch, optimizer=optimizer, criterion=criterion)

def learn_shallow_net(method):
    # choose optimizer and loss function
    test_acc = []
    test_mae = []
    train_acc = []
    train_mae = []
    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\perm\\group1'
    dataset = dataHelper.prepare_dataset_loo(input_dir, method)
    net = get_network()
    params = get_parms(net)
    for data in dataset:
        learner = NNLearner(data, Method.GROUP, net=net, params=params)
        res = learner.learn(method)
        test_acc.append(res['test_stats'].acc)
        test_mae.append(res['test_stats'].mae)
        train_acc.append(res['train_stat'].acc)
        train_mae.append(res['train_stat'].mae)

    test_acc_mean = np.mean(test_acc)
    test_mae_mean = np.mean(test_mae)
    train_acc_mean = np.mean(train_acc)
    train_mae_mean = np.mean(train_mae)
    print('test_acc_mean = ' + str(test_acc_mean))
    print('test_mae_mean = ' + str(test_mae_mean))
    print('train_acc_mean = ' + str(train_acc_mean))
    print('train_mae_mean = ' + str(train_mae_mean))



def main():
    learn_shallow_net(Method.GROUP)

if __name__ == '__main__':
    main()




#train

