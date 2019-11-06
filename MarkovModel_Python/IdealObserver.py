#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for different types of Ideal Observers:
    - The hidden markov model
    - The fixed Bayesian observer model ("window" and "decay")

To do next:
make a wrapper for the case of uncoupled TP

@author: Florent Meyniel
"""

from MarkovModel_Python import Inference_NoChangePoint as IO_fixed
from MarkovModel_Python import Inference_ChangePoint as IO_hmm
import numpy as np

def IdealObserver(seq, ObsType, order=0, Nitem=None, options=None):
    """
    IdealObserver is a wrapper that computes the posterior inference of generative
    probabilities of the sequence seq.
    """

    options = check_options(options)

    if Nitem is None:
        Nitem = len(set(seq))

    if ObsType.lower() == 'fixed':
        prior = IO_fixed.symetric_prior(order=order, Nitem=Nitem, weight=1)
        Decay, Window = parse_options(options, 'fixed_type')
        
        # Get posterior
        count = IO_fixed.count_tuple(seq, order=order, Nitem=Nitem, Decay=Decay, Window=Window)
        post = IO_fixed.posterior_no_jump(count=count, prior=prior, Nitem=Nitem, order=order)
        
        # Fill output
        out = {}
        for item in post['mean'].keys():
            out[item] = {}
            out[item]['mean'] = post['mean'][item]
            out[item]['MAP'] = post['MAP'][item]
            out[item]['SD'] = post['SD'][item]
        out['surprise'] = compute_surprise(seq, out, order)

    if ObsType.lower() == 'hmm':
        resol, p_c = parse_options(options, 'hmm_param')
        
        # Get full posterior
        marg_post = IO_hmm.compute_inference(seq=seq, resol=resol, order=order, Nitem=Nitem, p_c=p_c)
        
        # Fill output
        out = {}
        pgrid = np.linspace(0, 1, resol)
        for item in marg_post.keys():
            out[item] = {}
            out[item]['dist'] = marg_post[item]
            out[item]['mean'] = compute_mean_of_dist(marg_post[item], pgrid)
            out[item]['SD'] = compute_sd_of_dist(marg_post[item], pgrid, Nitem)
        out['surprise'] = compute_surprise(seq, out, order)

    return out

def parse_options(options, key):
    """
    Parse options
    """
    if key == 'fixed_type':
        if 'decay' in options.keys():
            Decay, Window = options['decay'], None
        elif 'window' in options.keys():
            Decay, Window = None, options['window']
        else:
            Decay, Window = None, None
        return Decay, Window

    elif key == 'hmm_param':
        if 'resol' in options.keys():
            resol = options['resol']
        else:
            resol = 10
        if 'p_c' in options.keys():
            p_c = options['p_c']
        else:
            raise ValueError('options should contain a key "p_c"')
        return resol, p_c

    else:
        return None

def check_options(options):
    checked_options = {}

    # use lower case for all options
    for item in options.keys():
        checked_options[item.lower()] = options[item]
    return checked_options

def compute_mean_of_dist(dist, pgrid):
    """ Compute mean of probability distribution"""
    return dist.transpose().dot(pgrid)

def compute_sd_of_dist(dist, pgrid, Nitem):
    """ Compute SD of probability distribution"""
    if Nitem ==2:
        m = compute_mean_of_dist(dist, pgrid)
        v = dist.transpose().dot(pgrid**2) - m**2;
    else:
        # when there are >2 items, we interpret X-E[X] as the Euclidian distance
        # to be done!
        v = np.nan
        
    return np.sqrt(v)

def compute_surprise(seq, out, order):
    """
    Compute surprise, conditional on the specified order
    """
    surprise = np.nan * np.ones(len(seq))
    if order == 0:
        for t in range(1, len(seq)):
            surprise[t] = -np.log2(out[(seq[t],)]['mean'][t-1])
    else:
        for t in range(order, len(seq)):
            surprise[t] = -np.log2(out[tuple(seq[t-order:t+1])]['mean'][t-1])
    return surprise