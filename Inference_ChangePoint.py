#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:56:36 2019

@author: fm239804

In this file, I attempt to compute the inference in the case of coupled 
transition probabilities.
The case of uncoupled transition probabilities is much simpler.

It is not possible to compute the transition matrix between discretized states 
    when order>1 and Nitem>1, because such a matrix becomes far too big. The 
    computation therefore has to be a bit different compared to the previous 
    implementation in Matlab

To be improved: change_marginalize currently corresponds to a flat prior 
distribution. This could be improved (perhaps not to include biases in 
transition probabilities, but at least biases in the base rate of occurence
of items)

to do next: get the predictions and confidence about the estimation underlying
the predictions. In this case, it is less clear how confidence should be 
estimated. When X has one dimension, var[X]=E[(X-E[X])²]. For >1 dimension, we
can interpret X-E[X] as the Euclidian distance, such that var[X] remains a vector. 

seems ok for Nitem=2 but not Nitem=3: the grid changes its size for the last item!!
"""

import numpy as np
import itertools
from scipy.stats import dirichlet
from operator import mul
from functools import reduce 

def likelihood_table(Nitem=2, resol=None, order=0):
    """
    Compute the likelihood of observations on a discretized parameter grid,
    for *Nitem* with *resol* values for each parameter, in the case of 
    transitions with the specified *order*.
    
    If *order*=0, the observation likelihood is determined by Dirichlet 
    parameters (or bernoulli parameters when *Nitem*=1); those parameters (and 
    their combinations) are discretized using a grid. The number of dimensions 
    of the grid is *Nitem*. The function outputs *Dir_grid*, which is a list 
    of tuples, each tuple being a possible combination of dirichlet parameter 
    values.
    
    If *order*>0, one must combine the possible Dirichlet parameters across 
    transitions. 
    
    The function outputs *observation_lik*, the likelihood of observations 
    presented as a dictonary. The keys of this dictionary are the possible 
    sequence of trailing and leading observations of interest given the 
    specified order, and its values correspond to the discretized distribution 
    of likelihoods into states. For instance, the key (0,1,2) corresponds to 
    the sequence 0, then 1, then 2.
    
    """
    
    # compute combination all (discretized) Dirichlet parameter values 
    # (excepted for the last parameter, which is not free)
    grid_param = np.linspace(0,1,resol)
    Dir_grid_init = [list(ntuple) for ntuple in 
                       itertools.product(grid_param, repeat=Nitem-1)]
    
    # only retain possible combinations of those parameter values (keeping in 
    # mind that they must sum to 1), and set the value of the last parameter 
    # given the values of all the others
    Dir_grid = {}
    for item in range(Nitem-1):
        # list combinations and test whether they are possible
        Dir_grid[item] = np.array([combi[item] if sum(combi) <=1 else np.nan
                   for combi in Dir_grid_init])
        
        # remove impossible combinations
        Dir_grid[item] = Dir_grid[item][~np.isnan(Dir_grid[item])]
    # compute the value of the last parameter
    Dir_grid[Nitem-1] = np.ones(Dir_grid[0].shape[0])
    for item in range(Nitem-1):
        Dir_grid[Nitem-1] = Dir_grid[Nitem-1] - Dir_grid[item]
    
    # stack the elements of the dictionary into a matrix
    Dir_grid = np.vstack([Dir_grid[item] for item in Dir_grid])
    
    # convert matrix into a list of tuples
    Dir_grid = [tuple(Dir_grid[:,k]) 
                     for k in range(Dir_grid.shape[1])]
    
    # combine the values of those Dirichlet parameters across higher-order 
    # transition
    if order>0:        
        # get number of patterns used to condition the inference
        n_pattern = Nitem**order
    
        # get likelihood when states are combined across patterns
        Dir_grid_combi = [combi for combi 
                    in itertools.product(Dir_grid, repeat=n_pattern)]
        
    else:        
        # order = 0, we can compute the observation likelihood directly:
        observation_lik = {}
        for item in range(Nitem):
            # Initiliaze
            observation_lik[item] = np.ones(len(Dir_grid), dtype='float')
            
            # Fill
            for k, state_lik in enumerate(Dir_grid):
                observation_lik[item][k] = Dir_grid[k][item]
        return observation_lik, Dir_grid
    
    # get list of trailing pattern in the higher-order transition
    pattern_trail = [combi for combi
                    in itertools.product(range(Nitem), repeat=order)]
    
    # get list of trailing pattern together with the leading item
    pattern_trail_lead = [combi for combi
                    in itertools.product(range(Nitem), repeat=order+1)]
    
    # define a correspondance between trailing pattern and element in Dir_grid_combi
    pattern_index = {}
    for k, pattern in enumerate(pattern_trail):
        pattern_index[pattern] = k
    
    # compute observation likelihood (i.e. likelihood of current observation,
    # given leading ones at the order of interest, on the discretization grid)
    observation_lik = {}
    for pattern in pattern_trail_lead:
        # Initialize
        observation_lik[pattern] = np.ones(len(Dir_grid_combi), dtype='float')
        
        # Fill
        for k, state_lik in enumerate(Dir_grid_combi):
            observation_lik[pattern][k] = \
                state_lik[pattern_index[pattern[:-1]]][pattern[-1]]
    
    return observation_lik, Dir_grid

def change_marginalize(curr_dist):
    """
    Compute the integral: 
        int p(theta_t|y)p(theta_t+1|theta_t)) dtheta_t
        in which the transition matrix has zeros on the diagonal, and 
        1/(n_state-1) elsewhere. In other words, it computes the updated 
        distribution in the case of change point (but does not multiply by 
        the prior probability of change point).
        
        NB: currently, the prior on transition is flat, and the prior on the
        base rate of occurence of item is also flat; we may want to change this
        latter aspect at least.
    """
    return (sum(curr_dist) - curr_dist) / (curr_dist.shape[0]-1)

def init_Alpha(Dir_grid=None, Dirichlet_param=None, order=None):
    """
    Initialize Alpha, which is the joint probability distribution of 
    observations and parameter values.
    This initialization takes into account a bias in the dirchlet parameter 
    (which the constraint that the same bias applies to all transitions).
    Discretized state are sorted as in likelihood_table, such as the output
    of both functions can be combined.
    """
    # get discretized dirichlet distribution at quantitles' location
    dir_dist = [dirichlet.pdf(np.array(grid), Dirichlet_param) 
                for grid in Dir_grid]
    
    # normalize to a probability distribution
    dir_dist = dir_dist / sum(dir_dist)
    
    # combine the values of those Dirichlet parameters across higher-order 
    # transitions
    if order>0:
        
        # get number of patterns used to condition the inference
        n_pattern = len(Dirichlet_param)**order
    
        # get joint likelihood when states are combined across patterns
        Alpha0 = [reduce(mul, combi) for combi 
                    in itertools.product(dir_dist, repeat=n_pattern)]
    else:
        Alpha0 = dir_dist
    
    return np.array(Alpha0)

def convert(s=None):
    """
    Convert the sequence to match the keys used in the likelihood dictionary
    """
    if len(s) == 1:
        return int(s)
    else:
        return tuple(s)

def forward_updating(seq=None, lik=None, order=None, \
                     p_c=None, Alpha0=None, Nitems=None):
    """
    Update iteratively the joint probability of observations and parameters 
    values, moving forward in the sequence
    """
    
    # Initialize containers
    Alpha = np.ndarray((len(Alpha0), len(seq)))
    Alpha_no_change = np.ndarray((len(Alpha0), len(seq)))
    Alpha_change = np.ndarray((len(Alpha0), len(seq)))
    
    # Compute iteratively
    for t in range(len(seq)):
        if t<order or t==0:
            # simply repeat the prior
            Alpha_no_change[:,t] = (1-p_c)*Alpha0
            Alpha_change[:,t] = p_c*Alpha0
            Alpha[:,t] = Alpha0
        else:
            # Update Alpha with the new observation
            Alpha_no_change[:,t] = (1-p_c) * lik[convert(seq[t-order:t+1])] * \
                                    Alpha[:,t-1]
            Alpha_change[:,t] = p_c * lik[convert(seq[t-order:t+1])] * \
                                change_marginalize(Alpha[:,t-1])
            Alpha[:,t] = Alpha_no_change[:,t] + Alpha_change[:,t]
            
            # Normalize
            cst = sum(Alpha[:,t])
            Alpha_no_change[:,t] = Alpha_no_change[:,t]/cst
            Alpha_change[:,t] = Alpha_change[:,t]/cst
            Alpha[:,t] = Alpha[:,t]/cst
        
    return Alpha

def marginal_Alpha(Alpha, lik):
    """
    Compute the marginal distributions for all Dirichlet parameters and 
    transition types
    """
    marg_dist = {}
    for pattern in lik.keys():
        # get grid of values
        grid_val = np.unique(lik[pattern])
        
        # initialize container
        marg_dist[pattern] = np.zeros((len(grid_val), Alpha.shape[1]))
        
        # marginalize over the dimension corresponding to the other patterns
        for k, value in enumerate(grid_val):
            marg_dist[pattern][k,:] = np.sum(Alpha[(lik[pattern] == value),:], axis=0)
    
    return marg_dist

def compute_inference(seq=None, resol=None, order=None, Nitem=None, p_c=None):
    """
    Wrapper function that compute the posterior marginal distribution, starting
    from a sequence
    """
    
    lik, grid = likelihood_table(Nitem=Nitem, resol=resol, order=order)
    Alpha0 = init_Alpha(Dir_grid=grid, order=order, \
                        Dirichlet_param=[1 for k in range(Nitem)])
    Alpha = forward_updating(seq=seq, lik=lik, order=order, \
                     p_c=p_c, Alpha0=Alpha0, Nitems=Nitem)
    marg_Alpha = marginal_Alpha(Alpha, lik)
    return marg_Alpha