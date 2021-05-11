#!/usr/bin/env python
# coding: utf-8

#import libraries
import numpy as np
import pandas as pd
import os
import math
import tensorflow as tf
import sklearn as sklearn
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout

NumLSTM_Layers = 2 #This can be any value >2 (Note that there is always an input layer, and a non return sequence layer.
NumLSTM_Units = 4 #>0
DropoutValue = 0.2 #between 0-1
NumEpochs = 100 #>0
NumBatches = 100#>0
MyVerbose = 1 #0,1,2
MyValSplit = 0.1 #between 0-1
trainDataPath = os.getcwd() + "\SampleData"


def mapStringToFloatList(dataString):
    dataFloat = []
    for i in range(0, len(dataString)):
        if dataString[i] == 'A':
            dataFloat.append([1,0,0,0])
        elif dataString[i] == 'C':
            dataFloat.append([0,1,0,0])
        elif dataString[i] == 'G':
            dataFloat.append([0,0,1,0])
        elif dataString[i] == 'T':
            dataFloat.append([0,0,0,1])
        else:
            print("Invalid Data Input")
            return None
    return dataFloat

#this method accepts a list of string data and will convert all
#datapoints to an float
#this method will return a numpy array of data points
def mapStringDataToFloatData(StringDataList):
    x_training_data = []
    for i in range(0, len(StringDataList)):
        #print(StringDataList[i][1])
        x_training_data.append(mapStringToFloatList(StringDataList[i][1]))
    return x_training_data

def getYValues(StringDataList):
    y_training_data = []
    for i in range(0, len(StringDataList)):
        #print(StringDataList[i][1])
        y_training_data.append(StringDataList[i][2])
    return y_training_data

def DataArrayGenerator(Data):
    #split the entire dataset by the newlines so we get each line of data seperated
    SplitData = []
    for i in range(0, len(Data)):
        SplitData.append(Data[i].split())
    

    np.random.shuffle(SplitData)
    x_training_data = mapStringDataToFloatData(SplitData)
    y_training_data = getYValues(SplitData)
    x_train_numpy = np.array(x_training_data)
    y_train_numpy = np.array(y_training_data)
    y_train_numpy = y_train_numpy.reshape(len(Data),1)
    
    return x_train_numpy,y_train_numpy

def StringToFloatArr(Array):
    tempArray = []
    for i in range(0,len(Array)):
        tempArray.append(float(Array[i][0]))
    return tempArray


f = open(trainDataPath, "r")
TrainingData = []
for x in f:
    TrainingData.append(x)

TrainingData = TrainingData[1:]
x,y = DataArrayGenerator(TrainingData)

training_dataset_length = math.ceil(len(x) * .75)

x_train_data = x[0:training_dataset_length, : ]
y_train_data = y[0:training_dataset_length, : ]
#Test data set
x_test_data = x[training_dataset_length: , : ]
y_test_data = y[training_dataset_length: , : ]
y_test = np.array(StringToFloatArr(y_test_data))
#Reshape the data into 3-D array
x_test = np.reshape(x_test_data, (x_test_data.shape[0],x_test_data.shape[1],4))

#Reshape the data into 3-D array
x_train = np.reshape(x_train_data, (x_train_data.shape[0],x_train_data.shape[1], 4))
y_train = np.array(StringToFloatArr(y_train_data))
print(len(y_train))
print("Training and Test Data Ready")

### Model LSTM
print(len(x_train))

# Initialising the RNN
model = Sequential()

model.add(LSTM(units = NumLSTM_Units,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                return_sequences = True,
                input_shape = (x_train.shape[1],4)))
model.add(Dropout(DropoutValue))

#procedurally add LSTM layers to the Model

for i in range(0, NumLSTM_Layers-2):
    model.add(LSTM(units = NumLSTM_Units,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                bias_initializer="zeros",
                return_sequences = True))
    model.add(Dropout(DropoutValue))
    print(i)
    
model.add(LSTM(units = NumLSTM_Units,activation="tanh",recurrent_activation="sigmoid"))
model.add(Dropout(DropoutValue))

# Adding the output layer
# For Full connection layer we use dense
# As the output is 1D so we use unit=1
model.add(Dense(units = 1, activation='sigmoid'))

model.summary()

#compile and fit the model on A set Number of epochs
model.compile(optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=0.01  ,
    initial_accumulator_value=0.1,
    epsilon=1e-07,
    name="Adagrad"
), loss = 'mean_squared_error', metrics=[tf.keras.metrics.AUC(),'accuracy'])
model.fit(x_train, y_train, epochs = NumEpochs, verbose=MyVerbose, callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
    validation_split=MyValSplit, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=True)

#check predicted values
predictions = model.predict(x_test)
print(predictions)
#Undo scaling
#predictions = scaler.inverse_transform(predictions)


#Calculate RMSE score
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print("Root Mean Squared Error: ")
print(rmse)

rounded_predictions = []
for i in range(0,len(predictions)):
    rounded_predictions.append(round(predictions[i][0]  ,0))

#Calculate Confusion Matrix
print(tf.math.confusion_matrix(y_test,rounded_predictions))
print(predictions[0][1:10])

#print(tf.keras.metrics.AUC(y_test,rounded_predictions))
