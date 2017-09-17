import numpy as np
import pandas as pd
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet


train_file = pd.read_csv('train.csv')
test_file = pd.read_csv('test.csv')

features_train = train_file.ix[:,9:438].values
features_train = np.array(features_train).reshape((-1,1,33,13)).astype(np.uint8)
labels_train = np.array(train_file.ix[:,1]).astype(np.uint8)
features_test = test_file.ix[:,9:438].values
features_test = np.array(features_test).reshape((-1,1,33,13)).astype(np.uint8)

def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],

        input_shape=(None, 1, 33, 13),
        conv1_num_filters=16,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=lasagne.nonlinearities.rectify,

        pool1_pool_size=(2, 2),
        dropout1_p=0.2,

        conv2_num_filters=32,
        conv2_filter_size=(2, 2),
        conv2_nonlinearity=lasagne.nonlinearities.rectify,

        hidden3_num_units=200,

        output_num_units=98,
        output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=0.001,
        update_momentum=0.9,

        max_epochs=n_epochs,
        verbose=1,
    )
    return net1


nn = CNN(15)
nn.fit(features_train, labels_train)

predict = nn.predict(features_test)
Idlist = np.array(test_file.ix[:,0])
np.savetxt('submission.csv', np.c_[Idlist,predict], delimiter=',', header = 'Id,Character', comments = '', fmt='%d')

