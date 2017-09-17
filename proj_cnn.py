#author:jiachen guo
#student number: 736913


import csv
import numpy as np 
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD 

def write_result_into_file(name, result):
    csv_file = open(name , 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['Id','Character'])

    writer.writerows(result)
    csv_file.close()

def train_data(file):
    features_train = np.array(file.ix[:29999,9:438])
    labels_train = np.array(file.ix[:29999,1])
    return features_train, labels_train

def valid_data(file):
    features_valid = np.array(file.ix[30000:32000,9:438])
    labels_valid = np.array(file.ix[30000:32000,1])
    return features_valid, labels_valid

def test_data(file):
    features_test = np.array(file.ix[:,9:438])
    return features_test

def correct_rate(predicts,labels_valid):
    corrects = 0
    for i in range(9999):
        if labels_train[i] == predicts[i]:
            corrects += 1
    return corrects / 10000.0



train_data_file = pd.read_csv('train.csv')
test_data_file = pd.read_csv('test.csv')

#---------------------------------------------------------------------
# reshape
features_train, labels_train = train_data(train_data_file)
features_train = features_train.reshape(30000,33,13,1).astype('float32')

features_valid, labels_valid = valid_data(train_data_file)
features_valid = features_valid.reshape(2001,33,13,1).astype('float32')

features_test = test_data(test_data_file)
features_test = features_test.reshape(8715,33,13,1).astype('float32')

features_train /= 255
features_valid /= 255
features_test /= 255

n_classes = 98

labels_train = to_categorical(labels_train,98)
labels_valid = to_categorical(labels_valid,98)

######################################################################



from keras.models import Sequential
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2

model = Sequential()

n_pool = 2

model.add(Convolution2D(
        100, 4, 4,

        border_mode='valid',
        #W_regularizer= l2(l=0.005),

        input_shape=(33, 13, 1)
))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))

######################################################################

model.add(Convolution2D(
        200, 4, 4,

        border_mode='valid',
        #W_regularizer= l2(l=0.005),
        bias=True,

        input_shape=(15, 5, 1)
))

model.add(Activation('relu'))

# then we apply pooling to summarize the features
# extracted thus far
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))

######################################################################




from keras.layers import Dropout, Flatten, Dense

model.add(Dropout(0.25))

# flatten the data for the 1D layers
model.add(Flatten())

# Dense(n_outputs)
model.add(Dense(300, W_regularizer= l2(l=0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.40))

model.add(Dense(150, W_regularizer= l2(l=0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.40))

# the softmax output layer gives us a probablity for each class
model.add(Dense(n_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.015, decay=1e-6, momentum=0.95, nesterov=True)

model.compile(
    loss='categorical_crossentropy',
    #optimizer= sgd,
    optimizer='adam',
    metrics=['accuracy']
)






# how many examples to look at during each training iteration
batch_size = 30

# how many times to run through the full set of examples
n_epochs = 100

# the training may be slow depending on your computer
model.fit(features_train,
          labels_train,
          batch_size=batch_size,
          nb_epoch=150,
          validation_data=(features_valid, labels_valid))
          

loss, accuracy = model.evaluate(features_valid, labels_valid)
print('loss:', loss)
print('accuracy:', accuracy)


predicts2 = model.predict_classes(features_test)

list1 = []
for i in predicts2:
    list1.append(i)

Idlist = np.array(test_data_file.ix[:,0])
result = zip(Idlist,list1)
#print result
write_result_into_file("result_v3.csv",result)



