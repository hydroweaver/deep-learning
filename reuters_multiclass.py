# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:32:01 2018

@author: karan.verma
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.datasets import reuters
from keras.models import load_model
import operator

(train_x, train_y), (test_x, test_y) = reuters.load_data(num_words = 10000)

#get word index
reuters_word_index = reuters.get_word_index()

#reverse word index
reverse_word_index = dict([(value, key) for (key, value) in reuters_word_index.items()])

#vectorize train_x and test_x as [len(of array), 10000 for sample / 46 for labels]

def to_vector(arr, one_hot_length):
    interim_vector = np.zeros([len(arr), one_hot_length])
    for seq, record in enumerate(arr):
        interim_vector[seq, record] = 1
    return interim_vector

vector_train_x = to_vector(train_x, 10000)
vector_test_x = to_vector(test_x, 10000)

#make model master
model_master = {}

#set path
output_dir = r'C:\Users\karan.verma\.spyder-py3\deep-learning\reuters_multiclass_graphs_models'

#vectorize train_y and test_y as []
#NEXT TIME TRY TO USE TO_CATEGORICAL
vector_train_y = to_vector(train_y, 46)
vector_test_y = to_vector(test_y, 46)

#get validation set from training data, 80/20 split, since all the data is only 8982 samples !

partial_vector_train_x, val_vector_train_x = vector_train_x[:7185], vector_train_x[7185:]
partial_vector_train_y, val_vector_train_y = vector_train_y[:7185], vector_train_y[7185:]

#define model and iterate on multiple

#activations = ['relu', 'tanh', 'sigmoid']
activations = ['relu', 'sigmoid']

#lyrs = [1, 2, 3, 4]
lyrs = [1]

#units = [16, 32, 64, 128]
units = [64, 128]

#epochs
e = 5


for activation in activations:
    for unit in units:
        for lyr in lyrs:
            
            #FIRST LAYER STUFF HAS NOT BEEN CHECK IN TERMS OF UNITS...CAN THAT BE CHANGED???
            model = models.Sequential()
            model.add(layers.Dense(64, activation = 'relu', input_shape=(10000,)))
            for i in range(lyr):
                model.add(layers.Dense(unit, activation = activation))
            #model.add(layers.Dense(46, activation='sigmoid'))
            model.add(layers.Dense(46, activation='softmax'))
            
            #initially ran the model with sigmoid, altough it's ideally used for binary classification
            #and does not provide output probabilities, although also used for MNSIT which is also a multi
            #class problem....anyway...running with sigmoid to check the out and then with softmax which
            #should give out probabilities of the 46 classes resolved by argmax
            
            #with sigmoid the top accuracy is 79.16 with 128 units in a single hidden layer
            
            #with softmax its at 79.34 tanh 1 128
            
            
            model.compile(optimizer='rmsprop',
                          loss = 'categorical_crossentropy',
                          metrics=['accuracy'])
            
            history = model.fit(partial_vector_train_x,
                                partial_vector_train_y,
                                epochs = e,
                                batch_size=512,
                                validation_data=(val_vector_train_x, val_vector_train_y),
                                verbose=0)
            #verbose=0
            
            #model results
            hist_dict = history.history
        
            validation_loss_values = hist_dict['val_loss']
            validation_acc_values = hist_dict['val_acc']
            training_loss_values = hist_dict['loss']
            training_acc_values = hist_dict['acc']
            
            epochs = np.arange(1, e+1)
            
            plt.close()
            fig, ax = plt.subplots(2,1, figsize=(20,10))
            
            plt.subplot(211)
            plt.plot(epochs, training_loss_values, 'r', label='Training Loss')
            plt.plot(epochs, validation_loss_values, 'b', label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(212)
            plt.plot(epochs, training_acc_values, 'r', label='Training Accuracy')
            plt.plot(epochs, validation_acc_values, 'b', label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            
            #plt.show()
            
            #show model evaluation
            results = model.evaluate(vector_test_x, vector_test_y)
            
            fig.savefig('{}/%s %d %d'.format(output_dir) % (activation, lyr, unit))
            plt.clf()
            
            #save the model for later use
            model.save('{}/%s %d %d'.format(output_dir) % (activation, lyr, unit))
            
            s0_stt = '%s %d %d' % (activation, lyr, unit)
            model_master[s0_stt] = results[1]*100
            print('Trained with %s activation %d hidden layers %d hidden units' % (activation, lyr, unit))
            print('Accuracy of this model is {0:.2f}%'.format(results[1]*100))
            
#choose best model and write final summary
#model_master_copy = model_master[:]
model_master = sorted(model_master.items(), key=operator.itemgetter(1))[len(model_master)-1]
final_model = load_model('{}\%s'.format(output_dir) % (model_master[0]))

#get prediction of this model
prediction = final_model.predict(vector_test_x)

#changing prediction to a function for easiness
def pred():
    #get a random number within the range of test samples
    random_number = np.random.choice(len(vector_test_x))

    #get random sample from original test table
    random_sample = test_x[random_number]

    #prediction of the same random sample from model.predict
    pred = prediction[random_number]

    #build sample using reverse index
    news = ' '.join([reverse_word_index.get(i-3, '?') for i in random_sample])

    #prediction
    best_model_acc = model_master[1]
    print('\nThe accuracy of the best model is: %f' % best_model_acc)
    print('\nA sample is as follows: \n %s \n' % news)
    print('\nPredicted Label %d \n' % np.argmax(pred))
    print('\nOriginal Label %d' % test_y[random_number])
    print('\nThe current model is: %s' % model_master[0])

#run prediction 5 times
for i in range(5):
    print('\nPrediction %d' % (i+1))
    print('//////////////////////////////////////////////////////////////////')
    pred()
    print('------------------------------------------------------------------')

#beep at the end
#print('\a')