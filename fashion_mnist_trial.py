# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:04:12 2019

@author: Karan.Verma
"""

from keras import models
from keras import layers
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

def shape(arr):
    x = np.reshape(arr, (len(arr), 784))
    return x

e = 40

train_x_reshaped = shape(train_x).astype('float32')/255
train_y = to_categorical(train_y)
train_y = train_y.astype('float32')

test_x_reshaped = shape(test_x).astype('float32')/255
test_y = to_categorical(test_y)
test_y = test_y.astype('float32')

#take out validation data, 25% = 15,000 samples

partial_train_x = train_x_reshaped[:int(len(train_x_reshaped)*0.75)]
partial_train_y = train_y[:int(len(train_y)*0.75)]

val_x = train_x_reshaped[int(len(train_x_reshaped)*0.75):]
val_y = train_y[int(len(train_y)*0.75):]

#make model

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10, activation = 'sigmoid'))

#compile model
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

#model fit

history = model.fit(partial_train_x,
                    partial_train_y,
                    epochs = e,
                    batch_size=512,
                    validation_data=(val_x, val_y))

acc_list = history.history
fig, ax = plt.subplots(2,1, figsize=(20, 10))

plt.subplot(211)
plt.plot(np.arange(1, e+1), acc_list['loss'], label='Training Loss')
plt.plot(np.arange(1, e+1), acc_list['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.subplot(212)
plt.plot(np.arange(1, e+1), acc_list['acc'], label='Training Acc')
plt.plot(np.arange(1, e+1), acc_list['val_acc'], label='Validation Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.legend()
plt.show()

prediction = model.evaluate(test_x_reshaped, test_y)

print(prediction)





