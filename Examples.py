#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:45:56 2019

@author: fm239804
"""
import IdealObserver as IO
import GenerateSequence as sg
import matplotlib.pyplot as plt
import numpy as np

# %% Example with binary sequence and order 0 transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order0(np.hstack((0.25*np.ones(L), 0.75*np.ones(L))))
seq = sg.ConvertSequence(seq)['seq']

# Compute Decay observer and HMM
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# Plot result
plt.subplot(2,1,1)
plt.plot(out_fixed['mean']['0'])
plt.subplot(2,1,2)
plt.imshow(out_hmm[0], origin='lower')

# %% Example with binary sequence and order 0 transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1( \
        np.vstack(( \
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))), \
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))) \
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute Decay observer and HMM
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=1, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)


# Plot result
plt.subplot(3,1,1)
plt.plot(out_fixed['mean']['0|0'], label='p(0|0)')
plt.plot(out_fixed['mean']['0|1'], label='p(0|1)')
plt.legend(loc='best')
plt.subplot(3,1,2)
plt.imshow(out_hmm[(0,0)], origin='lower')
plt.ylabel('p(0|0)')
plt.subplot(3,1,3)
plt.imshow(out_hmm[(1,0)], origin='lower')
plt.ylabel('p(0|1)')

# %% Example with sequence of 3 items and order 0 transition probabilities
L = int(1e2)
Prob = {0: np.hstack((0.10*np.ones(L), 0.75*np.ones(L))), \
    1: np.hstack((0.10*np.ones(L), 0.20*np.ones(L))), \
    2: np.hstack((0.80*np.ones(L), 0.05*np.ones(L)))}
seq = sg.ProbToSequence_Nitem3_Order0(Prob)
seq = sg.ConvertSequence(seq)['seq']

options = {'Decay':10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# Plot result
plt.subplot(4,1,1)
plt.plot(out_fixed['mean']['0'], label='p(0)')
plt.plot(out_fixed['mean']['1'], label='p(1)')
plt.plot(out_fixed['mean']['2'], label='p(2)')
plt.legend(loc='best')
plt.ylim([0,1])
plt.subplot(4,1,2)
vmin, vmax = 0, np.max([np.max(out_hmm[0]), np.max(out_hmm[1]), np.max(out_hmm[2])])
plt.imshow(out_hmm[0], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(0)')
plt.subplot(4,1,3)
plt.imshow(out_hmm[1], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(1)')
plt.subplot(4,1,4)
plt.imshow(out_hmm[2], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(2)')

