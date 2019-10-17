#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:50:03 2019

@author: fm239804

Computation: 
    * keep in mind that beta parameter - 1 = event count
      or equivalently: event count = beta parameter + 1
    * when we combine prior and likelihood, the resulting beta distribution has
      parameter: sum of each parameter - 1
    * in the matlab code there is a confusion about opt.priorp1: the doc says 
      that it correspond to the parameters of the beta distribution, but then, 
      reading the code, it seems that it could in fact be the event count 
      themselves.
    
"""
import itertools
import numpy as np
import math

def symetric_prior(order=1, Nitem=None, weight=1):
    """
    symetric_prior(order=1, Nitem=None, weight=1):
        
    Return the prior as a dictionary. Each possible combination of *Nitem* at 
    the specified *order* is a key, and the value (corresponding to the parameter 
    of the beta distribution) is *weight*.
    """
    # create empty dictionary
    prior = {}
    
    for pattern in itertools.product(range(Nitem), repeat=order+1):
        # turn pattern into list
        pattern_list = list(pattern)
        
        # make a (string) key for this pattern
        if order == 0:
            pattern_str = str(pattern_list[0])
        else:
            pattern_str = '|'.join([str(pattern_list.pop()), 
                                    ' '.join(str(item) for item in pattern_list[::-1])])    
        
        # assign parameter of the beta distribution
        prior[pattern_str] = weight
    
    return prior

def count_tuple(seq, order=1, Decay=None, Window=None, Nitem=None):
    """
    count_tuple(seq, Decay=None, Window=None, Nitem=None, order=1)
    
    Returns the cumulative event count in the sequence, searching for tuples 
    of the specified order.
    If specified: use an exponential decay, or sliding window for the count.
    An arbitrary number of items can be specified, by defaut, the number of 
    distinct items found in the sequence is used.
    """
    
    # By default, the number of distinct items is the one found in the sequence.
    if Nitem is None:
        Nitem = len(set(seq))
    
    # initialize count
    count = {}
    
    # Process all possible tuples (= patterns) with the requested order.
    for pattern in itertools.product(range(Nitem), repeat=order+1):
        # turn pattern into list
        pattern_list = list(pattern)
        
        # sliding detection of the pattern in the sequence
        detect_pattern = [1 if list(seq[k:k+order+1]) == pattern_list else 0 
                        for k in range(len(seq)-order)]
        
        # pad the resulting event count to match the sequence length
        detect_pattern = np.hstack((np.zeros(order, dtype=int), 
                                  np.array(detect_pattern, dtype=int)))
        
        # make a (string) key for this pattern
        if order == 0:
            pattern_str = str(pattern_list[0])
        else:
            pattern_str = '|'.join([str(pattern_list.pop()), 
                                    ' '.join(str(item) for item in pattern_list[::-1])])    
        
        # Simple count
        if (Decay is None) & (Window is None):
            count[pattern_str] = np.cumsum(detect_pattern)
        
        # Count within a sliding window
        elif Window is not None:
            # define a uniform kernel for the sliding sum
            kernel = np.ones(Window, dtype=int)
            
            # pad the sequence in order to avoid edge effects
            pad_seq = np.hstack((np.zeros(Window, dtype=int), detect_pattern))
            
            # compute the sliding sum using the kernel
            count[pattern_str] = np.convolve(pad_seq, kernel, 'valid')[1:]
            
        # Count with an exponential filter
        elif Decay is not None:
            # compute exponential decay factor
            decay_factor = math.exp(-1/Decay)
                
            # Initialize count
            count[pattern_str] = np.zeros(seq.shape, dtype=float)
            
            # Accrue the leak count
            leaky_count = 0
            for position, value in enumerate(detect_pattern):
                leaky_count = decay_factor*(value + leaky_count)
                count[pattern_str][position] = leaky_count
                    
    return count

def posterior_no_jump(count, prior):
    """
    posterior_no_jump(count, prior):
        
    Return the posterior inference of (transition) probabilities, for the 
    observed *count* and *prior*.
    """
    
    # Get sequence length
    L = len(count[list(count.keys())[0]])
    
    # Get number of possible items in the sequence
    transitions = list(count.keys())
    if '|' in transitions[0]:
        # select example transition
        trans_type = transitions[0].split('|')[1]
        # count number of trailing items for this transition
        Nitem = 0
        for item in transitions:
            if f"|{trans_type}" in item:
                Nitem += 1
    else:
        Nitem = len(count)
    
    # Initialize containers
    MAP = {}
    mean = {}
    SD = {}

    def get_total(ntuple):
        """
        Get the sum of beta parameter corresponding to this transition 
        type, augmented by the corresponding prior count
        """
        
        # Get this transition type
        if '|' in ntuple:
            trans_type = ntuple.split('|')[1]
        else:
            trans_type = None
        
        # Sum parameters (including the prior) corresponding to this transition
        # >> count + 1 correspond to beta parameters
        # >> the combination of prior and likelihood correspond to adding 
        #    beta parameters and subtracting 1
        if trans_type is None:
            tot = np.zeros(L, dtype=int)
            for item in count.keys():
                tot = tot + (count[item] + 1) + (prior[item] - 1) 
        else:
            tot = np.zeros(L, dtype=int)
            for item in count.keys():
                if f"|{trans_type}" in item:
                    tot = tot + (count[item] + 1) + (prior[item] -1)
        
        return tot
    
    # Compute posterior for each observation type    
    for ntuple in count.keys():
        # get event count (including the prior) for this transition type
        tot_param = get_total(ntuple)
        this_param = (count[ntuple] + 1 + prior[ntuple] - 1)
        this_param_rel = this_param / tot_param
        
        # get MAP, mean and SD
        MAP[ntuple] = (this_param - 1) / (tot_param - Nitem)
        mean[ntuple] = this_param_rel
        SD[ntuple] = np.sqrt(this_param_rel * (1-this_param_rel) / (tot_param + 1))
        
        # of instead used dirichlet.mean, dirichlet.var ... from scipy.stats?
        # however, the MAP is lacking in this class...
        
    return {'MAP':MAP, 'mean':mean, 'SD':SD}

