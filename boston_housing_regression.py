# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 15:28:51 2018

@author: karan.verma
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from timeit import timeit
import matplotlib.pyplot as plt

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

#retrain model with 80 epochs and 16 batch size because its overfitting post 80 epochs


#k fold init
k = 4
num_val_samples = len(train_x) // 4 # for returning int val
num_epochs = 80
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
                        batch_size = 16,
                        validation_data=(val_x, val_y),
                        verbose=0)
    
    mae_history = history.history['val_mean_absolute_error']
    all_scores.append(mae_history)

#testing the stuff after new training

test_mse, test_mae = model.evaluate(test_x, test_y)

# to print the final summary for each fold per epoch
#method1 1. convert to numpy array and then use a for loop
all_scores_np = np.array(all_scores)
final_score_m1 = [np.mean(all_scores_np[:,i]) for i in range(len(all_scores_np[0,:]))]

#method2 use the method in the book of list comprehensions
final_score_m2 = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]
final_score_m2 = np.array(final_score_m2)

final_score_m1 == final_score_m2
          
#plotting the validation score per epoch using all_scores table
plt.plot(range(1, len(final_score_m1)+1), final_score_m1)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#plotting graph with first 10 observation removed from final_score_m1 and using exponential smoothing on all points
'''final_score_m1_revised = final_score_m1[10:]

def smoothit(score, factor = 0.9):
    smooth_curve = []
    for point in score:
        if smooth_curve:
            previous = smooth_curve[-1]
            smooth_curve.append(previous*factor + point*(1-factor))
        else:
            smooth_curve.append(point)
    return smooth_curve

final_score_m1_revised = smoothit(final_score_m1_revised)

plt.clf()
plt.plot(range(1, len(final_score_m1_revised)+1), final_score_m1_revised)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE post smoothening')
plt.show()
#retrain model with 80 epochs and 16 batch size
'''

#making full pipeline
prediction = model.predict(test_x)

def pred():
    random_val = np.random.choice(range(len(test_x)))
    print('\n\n\n********************MEAN ACTUAL ERROR %f *************************' % test_mae)
    print('A random sample from test data is : ', test_x[random_val])
    print('Current value of this sample is : ', test_y[random_val])
    print('Predicted value of this sample is : ', float(prediction[random_val]))
    print('Difference between actual and predicted is : ', float(test_y[random_val]-prediction[random_val]))
    print('//////////////////////////////////////////////////////////////////')

print('5 random predictions from the test data:')
for i in range(5):
    pred()








