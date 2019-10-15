#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions to generate sequences, count items and transitions 
in the sequence, and plot them.
"""

import random
import numpy as np

# seq = ProbToSequence1(np.random.random(int(1e2)))
# seq = ProbToSequence1(np.hstack((0.25*np.ones(int(1e2)), 0.75*np.ones(int(1e2)))))
# seq = ConvertSequence(seq)['seq']


def ConvertSequence(seq):
    """
    Convert the sequence into a numeric Numpy array, and returns a dictonary 
    for the correspondance between original items and numeric values
    """
    
    # define a dictonary, mapping elements in the input sequence with an integer
    mapping = {}
    for number, element in enumerate(set(seq)):
        mapping[element] = number
        
    # Replace the element in the sequence with the corresponding number
    conv_seq = [mapping[element] for element in seq]
    
    # convert into a numpy array
    seq = np.array(conv_seq, dtype = int)
    
    return {'seq':seq, 'mapping':mapping}
        

def ProbToSequence1(Prob):
    """ 
    Return a random sequence of observations generated based on Prob, a 
    sequence of (0-order) item probability. In other words, the sequence 
    follows a Bernoulli process.
    Prob is a np array.
    """
    length = Prob.size
    seq = np.array([1 if random.random()<Prob[k] 
            else 2 
            for k in range(length)], dtype=int)
    return seq
    
def ProbToSequence2(Prob):
    """ 
    Return a random sequence of observations generated based on Prob, a 
    sequence of first-order transition probability. In other words, the sequence 
    follows a first-order Markov chain.
    Prob is a np array.
    """
    
    length = Prob.shape[1]
    seq = np.zeros(length, dtype=int)
    seq[0] = 1
    for k in range(1, length):
        print(k)
        if random.random()<Prob[seq[k-1]-1,k]:
            seq[k] = 1
        else:
            seq[k] = 2
    return seq

