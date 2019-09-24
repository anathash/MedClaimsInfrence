from __future__ import print_function

import numpy as np
from builtins import range

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.autograd import Variable

from learning import dataHelper
from learning.dataHelper import Split

"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""


class AlexNet(nn.Module):

    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class Net(nn.Module):

    def __init__(self, hl):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(18, 40)
        self.fc2 = nn.Linear(40, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LearningParams:
    def __init__(self, optimizer, num_epoch, criterion):
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.criterion = criterion

class Stats:
    def __init__(self, mae, acc):
        self.acc = acc
        self.mae = mae

class NNLearner:

    def __init__(self, train, test, net, params:LearningParams):
        self.net = net
        self.params = params
        # self.split = split
        #self.train_percent = train_percent
        # self.input_dir = input_dir
        #self.data = dataHelper.prepare_dataset_loo(self.split, self.input_dir, self.train_percent)
        #self.data = dataHelper.prepare_dataset_loo(self.input_dir)
        self.train = train
        self.test = test

    """
    SECTION 2 : Build and Train Model

    Multilayer perceptron model, with one hidden layer.
    input layer : 4 neuron, represents the feature of Iris
    hidden layer : 10 neuron, activation using ReLU
    output layer : 3 neuron, represents the class of Iris

    optimizer = stochastic gradient descent with no batch-size
    loss function = categorical cross entropy
    learning rate = 0.01
    epoch = 500
    """
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
            x, y = dataHelper.split_x_y(df)
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
        return  Stats(mae=mae, acc=acc)


    def test(self):
        # get prediction
        X = Variable(torch.Tensor(self.data.xtest).float())
        Y = torch.Tensor(self.data.ytest).long()
        out = self.net(X)
        _, predicted = torch.max(out.data, 1)

        # get accuration
        print('Accuracy of the network %d %%' % (100 * torch.sum(Y == predicted) / len(predicted)))
        print('Train Queries')
        train_stat = self.test_query_set(self.data.train_queries)
        print('\n \n \n Test Queries')
        test_stats = self.test_query_set(self.data.test_queries)
        return {'train_stat':train_stat, 'test_stats':test_stats}


    def learn(self):
       # self.prepare_dataset()
        self.train()
        return self.test()


def learn_shallow_net():
    hl = 70
    lr = 0.1
    torch.manual_seed(1234)
    num_epoch = 1500
    net = Net(hl)
    # choose optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    params = LearningParams(num_epoch=num_epoch, optimizer=optimizer, criterion=criterion)
    test_acc = []
    test_mae = []
    train_acc = []
    train_mae = []
    input_dir = 'C:\\research\\falseMedicalClaims\\examples\\model input\\pubmed\\group1'
    dataset = dataHelper.prepare_dataset_loo(input_dir)
    for data in dataset:
        learner = NNLearner(data,
                            train_percent=0.9, net=net, split = Split.BY_QUERY, params=params)
        res = learner.learn()
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
    learn_shallow_net()

if __name__ == '__main__':
    main()




#train

