import pandas as pd
import numpy as np
import csv as csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import BaggingClassifier


def write_result_into_file(name, result):
    csv_file = open(name , 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['Id','Character'])

    writer.writerows(result)
    csv_file.close()

def train_data(file,begin,end):
    X_train = np.array(file.ix[begin:end,9:438])
    X_train /= 255
    Y_train = np.array(file.ix[begin:end,1])
    return X_train, Y_train



def test_data(file,begin,end):
    X_test = np.array(file.ix[begin:end,9:438])
    X_test /= 255
    return X_test


train_data_file = pd.read_csv('train.csv')
test_data_file = pd.read_csv('test.csv')

X_train, Y_train = train_data(train_data_file,0,32000)
X_test = test_data(test_data_file,0,8714)

print (X_train.shape)

print 'Start PCA to 30'
pca = RandomizedPCA(n_components=30, whiten=True).fit(X_train)
X_train=pca.transform(X_train)
X_test=pca.transform(X_test)

print 'Start training'
rbf_svc = BaggingClassifier(svm.SVC(),max_samples=0.5, max_features=0.5)
rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train,Y_train)

print 'Start predicting'
predicts = rbf_svc.predict(X_test)

list1 = []
for i in predicts:
    list1.append(i)

Idlist = np.array(test_data_file.ix[:8714,0])
result = zip(Idlist,list1)

write_result_into_file("result_svm.csv",result)
