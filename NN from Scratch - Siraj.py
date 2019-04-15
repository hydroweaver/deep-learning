# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:51:57 2019

@author: karan.verma
"""

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2*random.random((3, 1)) - 1


    def __sigmoid(self, x):
        return 1 / (1+exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def think(self, inputs):
        '''(4,3) with (3,1) to get outputs of (4,1) which is then subtracted from actual'''
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations+1):
            '''Multiply input matrix (4, 3) with synaptic weight (3, 1) get outputs as (4, 1)'''
            output = self.think(training_set_inputs)
            '''subtract predicted output (4,1) from previous line from actual out (4,1) and get (4,1)'''
            error = training_set_outputs - output
            
            if iteration%2000 == 0:
                print("Iteration %d" % iteration)
                print(output, error)
            ''' multiply error (4,1) with sigmoid derivative value and then make dot product with (4,3).T = (3,4) and get adjustment as (3,1) same as synaptic weights class'''
            adjustment = dot(training_set_inputs.T, error*self._sigmoid_derivative(output))
            '''adjust (3,1) as (3,1) post additon'''
            self.synaptic_weights += adjustment
    
if __name__ == "__main__":
    
    neural_network = NeuralNetwork()
    
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)
    
    training_set_inputs = array([[0,0,1],
                                 [0,1,0],
                                 [1,0,1],
                                 [1,1,1]])
    
    training_set_outputs = array([[0],
                                 [1],
                                 [1],
                                 [0]])
    
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)


