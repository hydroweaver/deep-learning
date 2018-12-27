# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras import models
from keras import layers
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words = 10000)

#get word indices & #reverse the index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, word) for (word, value) in word_index.items()])

#one hot ecoding of the data, 10,000 units long index
def onehot(arr):
    output_arr = np.zeros([len(arr), 10000])    
    for seq, i in enumerate(arr):
        output_arr[seq,i] = 1
    return output_arr

train_x_new = onehot(train_x)
test_x_new = onehot(test_x)

train_y = np.asarray(train_y).astype('float32')
test_y = np.asarray(test_y).astype('float32')

#validation data
partial_train_x = train_x_new[:15000]
val_x = train_x_new[15000:]

partial_train_y = train_y[:15000]
val_y = train_y[15000:]

#create model big and small
model_big = models.Sequential()
model_big.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model_big.add(layers.Dense(16, activation = 'relu'))
model_big.add(layers.Dense(1, activation='sigmoid'))

model_small = models.Sequential()
model_small.add(layers.Dense(4, activation = 'relu', input_shape = (10000,)))
model_small.add(layers.Dense(4, activation = 'relu'))
model_small.add(layers.Dense(1, activation='sigmoid'))

model_super = models.Sequential()
model_super.add(layers.Dense(512, activation = 'relu', input_shape = (10000,)))
model_super.add(layers.Dense(512, activation = 'relu'))
model_super.add(layers.Dense(1, activation='sigmoid'))

#compile models
model_big.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

model_small.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

model_super.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])


#run model big
print('Running Older Model')
history1 = model_big.fit(partial_train_x,
                        partial_train_y,
                        epochs = 20,
                        batch_size = 512,
                        validation_data=(val_x, val_y),
                        verbose=1)

big_acc = history1.history

#run model small
print('\nRunning Newer Model')
history2 = model_small.fit(partial_train_x,
                        partial_train_y,
                        epochs = 20,
                        batch_size = 512,
                        validation_data=(val_x, val_y),
                        verbose=1)

small_acc = history2.history

#run model super
print('\nRunning Super Model')
history3 = model_super.fit(partial_train_x,
                        partial_train_y,
                        epochs = 20,
                        batch_size = 512,
                        validation_data=(val_x, val_y),
                        verbose=1)

super_acc = history3.history


e = range(1,21)
big_acc_val_loss = big_acc['val_loss']
small_acc_val_loss = small_acc['val_loss']
super_acc_val_loss = super_acc['val_loss']

big_acc_tr_acc = big_acc['val_acc']
small_acc_tr_acc = small_acc['val_acc']
super_acc_tr_acc = super_acc['val_acc']

btr_loss = big_acc['loss']
str_loss = small_acc['loss']
su_tr_loss = super_acc['loss']

#plt.plot(e, small_acc_val_loss, 'b', label='Small Model Loss')
plt.plot(e, big_acc_val_loss, '+', label='Large Model Loss')
plt.plot(e, super_acc_val_loss, 'o', label='Super Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

#plt.plot(e, small_acc_tr_acc, 'b', label='Small Model Tr Accu')
plt.clf()
plt.plot(e, big_acc_tr_acc, '+', label='Large Model Tr Accu')
plt.plot(e, super_acc_tr_acc, 'o', label='Super Model Tr Accu')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.show()

plt.clf()
plt.plot(e, btr_loss, '+', label='Large Model')
plt.plot(e, su_tr_loss, 'o', label='Super Model')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.show()

