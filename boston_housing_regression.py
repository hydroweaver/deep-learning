# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:28:51 2018

@author: karan.verma
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_x, train_y),(test_x, test_y) = boston_housing.load_data()

#feature normalization
mean = train_x.mean(axis=0) #axis = 0 means along the columns, don't get confused!
train_x = train_x - mean

sd = train_x.std(axis=0)
train_x = train_x/sd

#use the same mean and std to also normalize features of test data (not labels)
'''"Note that the quantities used for normalizing the test data are
computed using the training data. You should never use in your
workflow any quantity computed on the test data, even for something
as simple as data normalization."'''

test_x = test_x - mean
test_x = test_x/sd

#book uses a network of 64 units, will try later with 16 as well. Also model used as a function
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    
    #model compilation
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model

#k fold init
k = 4
num_val_samples = len(train_x) // 4 # for returning int val
num_epochs = 500
all_scores = []

for i in range(k):
    print('Processing fold #', i)
    val_x = train_x[i * num_val_samples: (i+1) * num_val_samples]
    val_y = train_y[i * num_val_samples: (i+1) * num_val_samples]
    
    partial_train_x = np.concatenate([train_x[:i*num_val_samples], train_x[(i+1)*num_val_samples:]],axis=0)
    partial_train_y = np.concatenate([train_y[:i*num_val_samples], train_y[(i+1)*num_val_samples:]],axis=0)
    
    model = build_model()
    
    history = model.fit(partial_train_x,
                        partial_train_y,
                        epochs = num_epochs,
                        batch_size = 1,
                        validation_data=(val_x, val_y))
    
    mae_history = history.history['val_mean_absolute_error']
    all_scores.append(mae_history)


for i in len(all_scores):
    print('Average of Epoch#',i,' is ',np.mean(all_scores[i-1]))
          
#plotting the validation score per epoch using all_scores table








