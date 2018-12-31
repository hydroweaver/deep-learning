# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#initially run with no L2 regularizer, now with l2 = 0.001

#Without L2 regularization test data evaluation : 16 units [0.8112427774882317, 0.84612] 4 units [0.4251603396511078, 0.86644] 512 units [0.9472465605561435, 0.87]
#   With L2 regularization test data evaluation : 16 units [0.4665150016975403, 0.86312] 4 units [0.3792718832683563, 0.87156] 512 units [0.47489984335422514, 0.87348] major reduction in loss on test data
#   With L1 regularization test data evaluation : 16 units [0.485685515127182, 0.87044] 4 units [0.43256368284225466, 0.86312] 512 units [2.962592507019043, 0.86816] 512 model completely useless, 16 and 4 relatively similar performance and acc 
#With L1/L2 regularization test data evaluation : 16 units [0.48758635732650757, 0.87032] 4 units [0.437683063287735, 0.86224] 512 units [2.94838287399292, 0.87736] pretty much the same as above with only l1 regularization
from keras import models
from keras import layers
from keras import regularizers
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words = 10000)

#get word indices & #reverse the index
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, word) for (word, value) in word_index.items()])

#variables to hold values for all models
lyrs = [16, 4, 512]
model_all = []
output_dir_graphs = r'C:\Users\karan.verma\.spyder-py3\deep-learning\graphs'
#putting all losses in the following variable : validation and training as a list
all_loss = []
acc_all = []

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

#create model big and small and then compile and run
for i in range(3):
    model = models.Sequential()
    model.add(layers.Dense(lyrs[i], kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation = 'relu', input_shape = (10000,)))
    model.add(layers.Dense(lyrs[i], activation = 'relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    
    model_all.append(model)
    
    print('\nTraining Model with %d units.--------------------->' % lyrs[i])    
    
    
    history = model.fit(partial_train_x,
                            partial_train_y,
                            epochs = 20,
                            batch_size = 512,
                            validation_data=(val_x, val_y),
                            verbose=1)

    all_loss.append([history.history['val_loss'],history.history['loss']])
    acc_all.append(model.evaluate(test_x_new, test_y))
    


e = range(1,21)
loss_fig, ax = plt.subplots(2,1, figsize=(20,10))

plt.subplot(211)
plt.title('Validation Loss')
plt.plot(e, all_loss[0][0], 'b', label='Original Model')
plt.plot(e, all_loss[1][0], 'g', label='Small Model')
plt.plot(e, all_loss[2][0], 'r', label='Large Model')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()

plt.subplot(212)
plt.title('Training Loss')
plt.plot(e, all_loss[0][1], 'b', label='Original Model')
plt.plot(e, all_loss[1][1], 'g', label='Small Model')
plt.plot(e, all_loss[2][1], 'r', label='Large Model')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()

plt.tight_layout()

loss_fig.savefig('{}/Model Comparison with Regularization IMDB'.format(output_dir_graphs))

