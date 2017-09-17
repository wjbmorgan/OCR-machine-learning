#author: jiachen guo
#student number:736913

import csv
import numpy as np 
import pandas as pd
from sknn.mlp import Classifier, Layer

def write_result_into_file(name, result):
    csv_file = open(name , 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['Id','Character'])

    writer.writerows(result)
    csv_file.close()

def train_data(file):
    features_train = np.array(file.ix[:35000,9:438])
    labels_train = np.array(file.ix[:35000,1])
    return features_train, labels_train

def valid_data(file):
    features_valid = np.array(file.ix[35001:45000,9:438])
    labels_valid = np.array(file.ix[35001:45000,1])
    return features_valid, labels_valid

def test_data(file):
    features_test = np.array(file.ix[:,9:438])
    return features_test

def correct_rate(predicts,labels_valid):
    corrects = 0
    for i in range(10000):
        if labels_train[i] == predicts[i]:
            corrects += 1
    return corrects / 100000.0

train_data_file = pd.read_csv('train.csv')
test_data_file = pd.read_csv('test.csv')



features_train, labels_train = train_data(train_data_file)
features_valid, labels_valid = valid_data(train_data_file)


features_test = test_data(test_data_file)

nn = Classifier(
        layers=[
            Layer("Sigmoid", units = 429),
            Layer("Sigmoid", units = 300),
            Layer("Sigmoid", units = 150),
            Layer("Softmax", units = 92)],
        n_iter= 1,
        n_stable= 40,
        batch_size= 25,
        learning_rate=0.003,
        learning_rule="momentum",
        valid_size=0.25,
        regularize = "L2",
        normalize = "weights",
        weight_decay = 0.0001,
        loss_type = "mcc",
        verbose=1)

nn.fit(features_train, labels_train)

predicts1 = nn.predict(features_valid)
correctness = correct_rate(predicts1,labels_valid)
print correctness

predicts2 = nn.predict(features_test)



list1 = []
for i in predicts2:
    list1.append(i[0])

Idlist = np.array(test_data_file.ix[:,0])
result = zip(Idlist,list1)
#print result
write_result_into_file("result.csv",result)
#correct = 0
#for i in range(0,1279):
#   if predicts[i] == labels_test[i]:
#       correct += 1
#
#print(correct / 1281.0)

