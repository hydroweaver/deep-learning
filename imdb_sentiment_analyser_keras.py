# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:05:42 2018

@author: karan.verma
"""

import numpy as np
from keras import models
from keras.models import load_model
from keras import layers
from keras.datasets import imdb
from keras import optimizers
import matplotlib.pyplot as plt
import operator


# using all word freqeuncies
#(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

# using all word freqeuncies = 10k
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words = 10000)

#convert data to tensor using one hot encoding, essentially putting 1 where there is a value and a 0 otherwise
#split data only after converting otherwise it wont be compatible
x_train = np.zeros([len(train_data),10000])
for number, sequence in enumerate(train_data):
    x_train[number, sequence] = 1

#in the book it is done as a function...but like right now.....ignore it
x_test = np.zeros([len(test_data),10000])
for number, sequence in enumerate(test_data):
    x_test[number, sequence] = 1

#vectorize labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# HAVE NOT RANDOMIZED THE DATA!!!! TRY AND DO THAT LATER
#get validation split, 60% training, 40% validation --> 15000 TRAINING, 10000 VALIDATION
x_train_partial, validation_x_train = x_train[:15000], x_train[15000:]
y_train_partial, validation_y_train = y_train[:15000], y_train[15000:]

#model iterations
activations = ['relu', 'tanh']
#activations = ['relu']

loss_func = ['mse', 'binary_crossentropy']
#loss_func = ['mse']

hidden_layer_units = [16, 32, 64]
#hidden_layer_units = [16]

hidden_layers = [1,2,3]
#hidden_layers = [1,2]

#one metric for all models:
metric_master = {}

for activation in activations:
    for function in loss_func:
        for units in hidden_layer_units:
            for lyrs in hidden_layers:
                
                #print(activation, function, units, lyrs);
                #write model
                model = models.Sequential()
                model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
                
                for i in range(0,lyrs):
                    model.add(layers.Dense(units, activation = activation))    
                    
                model.add(layers.Dense(1, activation = 'sigmoid'))
                
                #optimzers, loss etc.
                model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
                              loss = function,
                              metrics = ['accuracy'])
                
                history = model.fit(x_train_partial,
                                    y_train_partial,
                                    epochs = 4,
                                    batch_size = 512,
                                    validation_data=(validation_x_train, validation_y_train))
                
                #for graphs
                hist_dict = history.history
                validation_loss_values = hist_dict['val_loss']
                validation_acc_values = hist_dict['val_acc']
                training_loss_values = hist_dict['loss']
                training_acc_values = hist_dict['acc']
                
                epochs = np.arange(1,4+1)
                
                loss_fig, ax = plt.subplots(2,1, figsize=(20,10))
                
                plt.subplot(211)
                plt.plot(epochs, validation_loss_values, 'b', label = 'Validation Loss')
                plt.plot(epochs, training_loss_values, 'r', label='Training Loss')
                plt.title('Loss with ACTIVATION %s, LOSS FUNCTION %s, %d HIDDEN LAYERS & %d HIDDEN UNITS' % (activation, function, lyrs, units))
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                
                for i,j in zip(epochs, validation_loss_values):
                    plt.subplot(211).annotate(str(j), xy=(i,j), arrowprops=dict(facecolor='black', shrink=0.05))

                for i,j in zip(epochs, training_loss_values):
                    plt.subplot(211).annotate(str(j), xy=(i,j), arrowprops=dict(facecolor='black', shrink=0.05))
                
                plt.subplot(212)
                plt.plot(epochs, validation_acc_values, 'b', label = 'Validation Accuracy')
                plt.plot(epochs, training_acc_values, 'r', label='Training Accuracy')
                plt.title('Accuracy with ACTIVATION %s, LOSS FUNCTION %s, %d HIDDEN LAYERS & %d HIDDEN UNITS' % (activation, function, lyrs, units))
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                
                for i,j in zip(epochs, validation_acc_values):
                    plt.subplot(212).annotate(str(j), xy=(i,j), arrowprops=dict(facecolor='black', shrink=0.05))

                for i,j in zip(epochs, training_acc_values):
                    plt.subplot(212).annotate(str(j), xy=(i,j), arrowprops=dict(facecolor='black', shrink=0.05))
                
                plt.legend()
                plt.tight_layout()
                
                #show model evaluation
                results = model.evaluate(x_test, y_test)
            
                output_dir_graphs = r'C:\Users\karan.verma\.spyder-py3\deep-learning\graphs'
                output_dir_models = r'C:\Users\karan.verma\.spyder-py3\deep-learning\models'
                
                loss_fig.savefig('{}/ACTIVATION %s, LOSS FUNCTION %s, %d HIDDEN LAYERS & %d HIDDEN UNITS'.format(output_dir_graphs) % (activation, function, lyrs, units))               
                
                #save the model for later use
                model.save('{}/Model %s %s %d %d'.format(output_dir_models) % (activation, function, lyrs, units))
                
                s0_stt = 'Model %s %s %d %d' % (activation, function, lyrs, units)
                metric_master[s0_stt] = results[1]*100
                
                #best model is ACTIVATION relu, LOSS FUNCTION mse, 1 HIDDEN LAYERS & 32 HIDDEN UNITS with 88.19% accuracy

#sort model based on top accuracy and get last element
best_model_name = sorted(metric_master.items(), key=operator.itemgetter(1))[len(metric_master)-1]

best_model = load_model('{}/%s'.format(output_dir_models) % (best_model_name[0]))

#get prediction matrix of this model
prediction = best_model.predict(x_test)

#get word index for testing
word_index = imdb.get_word_index()
reverse_word_index_dict = {}

#create reverse index for testing                
for word, index in word_index.items():
    reverse_word_index_dict[index] = word

#get random number from 25000 choices, reverse it from lookup and print it
#test_Data is equivalent to x_test wrt to example index
rand_choice = np.random.choice(25000)                
empty_string = ''
for index in test_data[rand_choice]:
    empty_string += reverse_word_index_dict.get(index-3, '?') + ' '            
    
empty_string.strip()
print("Accuracy of the best model is: {0:.2f}% \n".format(best_model_name[1]))
print("It's parameters are %s \n" % best_model_name[0])
print('A random sample is as follows: %s \n' % empty_string)
print('Based on this model this sample is : {0:.2f}% positive'.format(prediction[rand_choice][0]*100))