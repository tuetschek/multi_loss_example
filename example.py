#!/usr/bin/env python3

"""
Loosely based on https://github.com/yangzhangalmo/pytorch-iris/
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.autograd import Variable
from torch import Tensor


class ClassifNet(nn.Module):

    def __init__(self):
        super(ClassifNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = torch.tanh(self.fc1(X))
        X = self.fc2(X)
        X = self.softmax(X)
        return X

    def train(self, X, y, lr, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print("Epoch %5d: loss = %.4f" % (epoch, loss.data))


class RegressNet(nn.Module):

    def __init__(self):
        super(RegressNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.regress = nn.Linear(10, 1)

    def forward(self, X):
        X = torch.tanh(self.fc1(X))
        X = self.regress(X)
        return X

    def train(self, X, y, lr, epochs):
        crit = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self(X)
            loss = crit(out, y)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print("Epoch %5d: loss = %.4f" % (epoch, loss.data))


class DualCriterionNet(nn.Module):

    def __init__(self):
        super(DualCriterionNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(dim=1)
        self.regress = nn.Linear(10, 1)

    def forward(self, X):
        X = torch.tanh(self.fc1(X))
        X1 = self.softmax(self.fc2(X))
        X2 = self.regress(X)
        return X1, X2

    def train(self, X, y1, y2, lr, epochs):
        crit1 = nn.CrossEntropyLoss()
        crit2 = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            out1, out2 = self(X)
            loss1 = crit1(out1, y1)
            loss2 = crit2(out2, y2)
            loss = loss1 + loss2  # total loss is sum of two losses
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print("Epoch %5d: loss = %.4f | %.4f + %.4f"
                      % (epoch, loss.data, loss1.data, loss2.data))


if __name__ == '__main__':
    # fixed random seed
    torch.manual_seed(123)
    np.random.seed(123)

    # data
    # see https://en.wikipedia.org/wiki/Iris_flower_data_set for details
    print('Loading data...')
    dataset = pd.read_csv('iris.csv')

    # transform the species column to numerics
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2
    dataset.species = dataset.species.astype(int)

    # split data, with two things to predict: petal_width and species
    # using the other columns as features
    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:3]].values,
                                                        dataset[dataset.columns[3:]].values,
                                                        test_size=0.2)
    train_X = Variable(Tensor(train_X).float())
    train_y1 = Variable(Tensor(train_y[:, 1]).long())  # species
    train_y2 = Variable(Tensor(np.expand_dims(train_y[:, 0], axis=1)).float())  # petal_width
    test_X = Variable(Tensor(test_X).float())
    test_y1 = Variable(Tensor(test_y[:, 1]).long())  # species
    test_y2 = Variable(Tensor(np.expand_dims(test_y[:, 0], axis=1)).float())  # petal_width

    # predicting species only (classification)
    print('\nClassification model...')
    cla_net = ClassifNet()
    cla_net.train(train_X, train_y1, lr=0.01, epochs=10001)
    out_species = cla_net(test_X)
    _, out_species = torch.max(out_species, 1)  # taking the max from the output softmax distribution
    print('Classification model -- species accuracy: %.4f'
          % accuracy_score(test_y1, out_species.data))

    # predicting petal width only (regression)
    print('\nRegression model...')
    reg_net = RegressNet()
    reg_net.train(train_X, train_y2, lr=0.01, epochs=10001)
    out_petal_width = reg_net(test_X)
    print('Regression model -- petal width mean absolute error: %.4f'
          % mean_absolute_error(test_y2, out_petal_width.data))

    # dual prediction, with dual loss
    print('\nDual loss classification + regression...')
    dc_net = DualCriterionNet()
    dc_net.train(train_X, train_y1, train_y2, lr=0.01, epochs=10001)

    out_species, out_petal_width = dc_net(test_X)
    _, out_species = torch.max(out_species, 1)
    print('Dual criterion -- species accuracy: %.4f, petal width mean absolute error: %.4f'
          % (accuracy_score(test_y1, out_species.data), mean_absolute_error(test_y2, out_petal_width.data)))
