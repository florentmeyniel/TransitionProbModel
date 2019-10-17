#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:55:35 2019

@author: fm239804
"""

import Inference_NoChangePoint as IO_fixed
import Inference_ChangePoint as IO_hmm

def IdealObserver(seq, ObsType, order=0, Nitem=None, options=None):
    """
    IdealObserver is a wrapper that compute the posterior inference of generative
    probabilities of the sequence seq.
    """
    
    options = check_options(options)
    
    if Nitem is None:
         Nitem = len(set(seq))
    
    if ObsType.lower() == 'fixed':
        prior = IO_fixed.symetric_prior(order=order, Nitem=Nitem, weight=1)
        Decay, Window = parse_options(options, 'fixed_type')
        count = IO_fixed.count_tuple(seq, order=order, Nitem=Nitem, Decay=Decay, Window=Window)
        return IO_fixed.posterior_no_jump(count=count, prior=prior, Nitem=Nitem, order=order) 
    
    if ObsType.lower() == 'hmm':
        resol, p_c = parse_options(options, 'hmm_param')
        return IO_hmm.compute_inference(seq=seq, resol=resol, order=order, Nitem=Nitem, p_c=p_c)

def parse_options(options, key):
    """
    Parse options
    """
    if key == 'fixed_type':
        if 'decay' in options.keys():
            return options['decay'], None
        elif 'window' in options.keys():
            return None, options['window']
        else:
            return None, None
    
    if key == 'hmm_param':
        if 'resol' in options.keys():
            resol = options['resol']
        else:
             resol = 10
        if 'p_c' in options.keys():
            p_c = options['p_c']
        else:
            raise ValueError('options should contain a key "p_c"')
        return resol, p_c

def check_options(options):
    checked_options = {}
    for item in options.keys():
        checked_options[item.lower()] = options[item]
    return checked_options
        