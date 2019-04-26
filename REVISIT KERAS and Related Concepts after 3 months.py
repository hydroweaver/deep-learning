# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:47:35 2019

@author: Karan.Verma
"""

from keras import models
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from random import randint
#import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

for_test = x_test.copy()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = to_categorical(y_train, num_classes=10)

#reshape input data
def reshape_func(arr):
    return np.reshape(arr, (len(arr), 28*28))

x_train = reshape_func(x_train)
x_test = reshape_func(x_test)

#normalise values
def norm(arr):
    return arr/255

x_train = norm(x_train)
x_test = norm(x_test)

#create validation from training data, 30%
split = 0.25

x_train_new, x_train_val = x_train[:int(len(x_train)*(1-split))], x_train[int(len(x_train)*(1-split)):]
y_train_new, y_train_val = y_train[:int(len(y_train)*(1-split))], y_train[int(len(y_train)*(1-split)):]

#create model
model = models.Sequential()
model.add(Dense(32, input_shape=(784,), activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation='softmax'))

#compile model
model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#fit model
history = model.fit(x_train_new,
                   y_train_new,
                   batch_size = 32,
                   epochs = 10,
                   validation_data=(x_train_val, y_train_val),
                   verbose = 1)

#make graphs
fig, ax = plt.subplots(2,1, figsize=(20, 10))
plt.subplot(2,1,1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(np.linspace(1, len(history.history['loss'])+1, len(history.history['loss'])), history.history['loss'], 'b*', label = 'Model Loss Values')
plt.plot(np.linspace(1, len(history.history['loss'])+1, len(history.history['loss'])), history.history['val_loss'], 'r*', label = 'Validation Loss Values')
plt.legend()
plt.subplot(2,1,2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(np.linspace(1, len(history.history['loss'])+1, len(history.history['loss'])), history.history['acc'], 'b-', label = 'Model Accuracy Values')
plt.plot(np.linspace(1, len(history.history['loss'])+1, len(history.history['loss'])), history.history['val_acc'], 'r-', label = 'Validation Accuracy Values')
plt.legend()
plt.show()



#get random values and print the figure and prediction, 10 predictions

plt.close()
fig, ax = plt.subplots(2, 5, figsize=(20,10))

for i in range(1, 11):
        random_num = randint(0, len(x_test) - 1)
        plt.subplot(2, 5, i)
        plt.imshow(for_test[random_num])
        sample = np.reshape(x_test[random_num], (1, 28*28))
        plt.title('Original : {} & Predicted : {}'.format(y_test[random_num], np.argmax(model.predict(sample))))
        #plt.title('Original : {}'.format(y_test[random_num]))