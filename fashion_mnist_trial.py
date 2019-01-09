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
from keras.utils import to_categorical, plot_model
import operator

(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

def shape(arr):
    x = np.reshape(arr, (len(arr), 784))
    return x

#epochs
e = 40

#prediction array
eval_arr = {}
pred_arr = {}


#label summary
label_list = {0: 'T-shirt/top', 1: 'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

#path
output_dir_model = r'C:\Users\karan.verma\.spyder-py3\deep-learning\models'

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

# run model with several different parameters

#lyrs = [1, 2]
#units_out = [512, 256, 128, 64, 32, 16]
#units_in = [128, 64, 32, 16]

lyrs = [1]
units_out = [512]
units_in = [128]

#make model
for lyr in lyrs:
    for unit_out in units_out:
        for unit_in in units_in:
            
            print('Running model having %d hidden layers & %d units in each hidden layer and %d units in the outer layer' % (lyr, unit_in, unit_out))
            
            model = models.Sequential()
            model.add(layers.Dense(unit_out, activation='relu', input_shape=(784,)))
            model.add(layers.Dropout(0.5))
            
            for i in range(lyr):
                model.add(layers.Dense(unit_in, activation='relu'))
                model.add(layers.Dropout(0.5))
            
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
                                validation_data=(val_x, val_y),
                                verbose = 1)
            
            acc_list = history.history
            fig, ax = plt.subplots(2,1, figsize=(20, 10))
            
            plt.subplot(211)
            plt.plot(np.arange(1, e+1), acc_list['loss'], label='Training Loss')
            plt.plot(np.arange(1, e+1), acc_list['val_loss'], label='Validation Loss')
            plt.title('Loss Graph')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(212)
            plt.plot(np.arange(1, e+1), acc_list['acc'], label='Training Acc')
            plt.plot(np.arange(1, e+1), acc_list['val_acc'], label='Validation Acc')
            plt.title('Accuracy Graph')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            
            #plt.tight_layout()
            fig.savefig('{}/Model having %d hidden layers & %d units in each hidden layer and %d units in the outer layer.png'.format(output_dir_model) % (lyr, unit_in, unit_out))            
            plt.clf()

            eval_arr['Model having %d hidden layers & %d units in each hidden layer and %d units in the outer layer' % (lyr, unit_in, unit_out)] = model.evaluate(test_x_reshaped, test_y)
            pred_arr['Model having %d hidden layers & %d units in each hidden layer and %d units in the outer layer' % (lyr, unit_in, unit_out)]= np.argmax(model.predict(test_x_reshaped), axis=1)


#plot_model(model, to_file='{}/model.png'.format(output_dir_model)) # this is kinda useless ! :|

#get best model from eval_arr
best_model = {}
k = ''

for key, val in eval_arr.items():
    best_model[key] = val[1]

print('The best model is: {}'.format(max(best_model.items(), key=operator.itemgetter(1))))

#should also consider the loss of the best model....? right? high acc ...very high loss...is this good??

#Building Prediction Pipeline
#('Model having 1 hidden layers & 128 units in each hidden layer and 512 units in the outer layer',0.8953) but it varies

prediction = model.predict(test_x_reshaped)

print('Random samples from the test data: ')
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for i in range(1, 7):
        c = np.random.choice(len(test_x_reshaped))
        plt.subplot(2,3,i)
        plt.imshow(test_x[c])
        plt.title('Original {} & Predicted {}'.format(label_list[np.argmax(test_y[c])], label_list[np.argmax(prediction[c])]))        
        #plt.tight_layout()
            

