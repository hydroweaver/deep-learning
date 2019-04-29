# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:07:53 2019

@author: Karan.Verma
"""

#from what I understand, this is a single shot training example for simple ML demo,
#since the input.dot(w0) takes each row as an input for each neuron, unlike MNIST etc.
#where the first layer essentially takes in the entire first sample and so forth
#as the first input

#WRITING WITHOUT CLASSES....i'll change :) 


import numpy as np
import random

#Input data - each row as a sample will be pushed into a single neuron, e.g. first neuron will take row [0,1,1]
input_x = np.array([[0,1,1],
                    [1,1,1],
                    [1,0,1],
                    [1,1,0]], dtype=float)

#Output data - each row output from above, e.g. [0,1,1] output is [0]
output_y = np.array([[0],
                     [1],
                     [0],
                     [1]], dtype=float)

#First random weight matrix 3x4
random.seed(1)
layer0_weights = 2*np.random.random((3, 4)) - 1
#second random weight matrix 3x1
layer1_weights = 2*np.random.random((4, 1)) - 1

#define sigmoid function - also derived it, makes a lot more sense now...a little more anyway ! haha 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

'''def nonlin(x, deriv=False):
    if deriv==True:
        return sigmoid_derivative(x)
    
    return sigmoid(x)'''

#simple forward pass calculation
#transport input_x and w0 for getting a 3x3 matrix of weights as suggested, I think we can change this too.
# do this 60K times

for i in range(60000):
    layer0 = input_x # first layer with four neurons, since the shape of input_x is 4,3
    layer1 = sigmoid(np.dot(layer0, layer0_weights)) # First hidden weight later 4x3 dot 3x4 = 4x4
    layer2 = sigmoid(np.dot(layer1, layer1_weights)) # Second hidden weight layer 4x4 dot 4x1 = 4x1
    
    #calculate error
    layer2_error = output_y - layer2 # output is a matrix of 4x1
    
    if(i % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(layer2_error))))
        
    #back propagation - just chain rule, error and delta for current layer is based on the values of previous layer
    layer2_delta = layer2_error * sigmoid_derivative(layer2) # error multipled with derivative of the function, 4x1 * 4x1 is 4x1 - element by element multiply
    layer1_error = layer2_delta.dot(layer1_weights.T) # 4x1 dot 1x4 = 4x4
    layer1_delta = layer1_error * sigmoid_derivative(layer1) # 4x4 * 4x4 = 4x4
    
    #gradient descent
    layer1_weights += layer1.T.dot(layer2_delta)
    layer0_weights += layer0.T.dot(layer1_delta)






