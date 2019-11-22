#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script. Present a few example applications for the Markov Model toolbox.

To do next:
    - do the uncoupled case
    - add a user-defined prior bias in the leak & hmm case

@author: Florent Meyniel
"""
# general
import matplotlib.pyplot as plt
import numpy as np

# specific to toolbox
from MarkovModel_Python import IdealObserver as IO
from MarkovModel_Python import GenerateSequence as sg

# %% Binary sequence and order 0 transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order0(np.hstack((0.25*np.ones(L), 0.75*np.ones(L))))
seq = sg.ConvertSequence(seq)['seq']

# Compute HMM observer
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(out_fixed[(0,)]['mean'], label='p(1) mean')
plt.plot(out_fixed[(0,)]['SD'], linestyle='--', label='p(1) sd')
plt.legend(loc='best')
plt.title('Exponential decay')

plt.subplot(3, 1, 2)
plt.imshow(out_hmm[(0,)]['dist'], origin='lower')
plt.title('HMM')

plt.subplot(3, 1, 3)
plt.plot(out_hmm[(0,)]['mean'], label='p(1) mean')
plt.plot(out_hmm[(0,)]['SD'], linestyle='--', label='p(1) sd')
plt.legend(loc='best')
plt.title('HMM -- moments')

# %% Binary sequence and order 1 (coupled) transition probabilities

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
plt.figure()
plt.subplot(3,1,1)
plt.plot(out_fixed[(0,0,)]['mean'], label='p(0|0)')
plt.plot(out_fixed[(1,0,)]['mean'], label='p(0|1)')
plt.legend(loc='best')
plt.subplot(3,1,2)
vmin, vmax = 0, np.max([np.max(out_hmm[(0,0)]['dist']), np.max(out_hmm[(1,0)]['dist'])])
plt.imshow(out_hmm[(0,0)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(0|0)')
plt.subplot(3,1,3)
plt.imshow(out_hmm[(1,0)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(0|1)')

# %% Sequence of 3 items and order 0 transition probabilities

# Generate sequence
L = int(1e2)
Prob = {0: np.hstack((0.10*np.ones(L), 0.50*np.ones(L))), \
    1: np.hstack((0.10*np.ones(L), 0.20*np.ones(L))), \
    2: np.hstack((0.80*np.ones(L), 0.30*np.ones(L)))}
seq = sg.ProbToSequence_Nitem3_Order0(Prob)
seq = sg.ConvertSequence(seq)['seq']

options = {'Decay':10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# Plot result
plt.figure()
plt.subplot(4,1,1)
plt.plot(out_fixed[(0,)]['mean'], label='p(0)')
plt.plot(out_fixed[(1,)]['mean'], label='p(1)')
plt.plot(out_fixed[(2,)]['mean'], label='p(2)')
plt.legend(loc='best')
plt.ylim([0,1])
plt.subplot(4,1,2)
vmin, vmax = 0, np.max([np.max(out_hmm[(0,)]['dist']),
                        np.max(out_hmm[(1,)]['dist']),
                        np.max(out_hmm[(2,)]['dist'])])
plt.imshow(out_hmm[(0,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(0)')
plt.subplot(4,1,3)
plt.imshow(out_hmm[(1,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(1)')
plt.subplot(4,1,4)
plt.imshow(out_hmm[(2,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.ylabel('p(2)')

# %% Binary sequence and order 1 transition probability: coupled vs. uncoupled

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1( \
        np.vstack(( \
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))), \
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))) \
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute HMM observer for coupled and uncoupled case
options = {'p_c': 1/200, 'resol': 20}
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)
out_hmm_unc = IO.IdealObserver(seq, 'hmm_uncoupled', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(2,1,1)
plt.plot(out_hmm[(0,0)]['mean'], 'g', label='p(0|0), coupled')
plt.plot(out_hmm[(1,0)]['mean'], 'b', label='p(0|1), coupled')
plt.plot(out_hmm_unc[(0,0)]['mean'], 'g--', label='p(0|0), unc.')
plt.plot(out_hmm_unc[(1,0)]['mean'], 'b--', label='p(0|1), unc.')
plt.legend(loc='upper left')
plt.ylim([0,1])
plt.title('Comparison of means')

plt.figure()
plt.subplot(2,1,1)
plt.plot(-np.log(out_hmm[(0,0)]['SD']), 'g', label='p(0|0), coupled')
plt.plot(-np.log(out_hmm[(1,0)]['SD']), 'b', label='p(0|1), coupled')
plt.plot(-np.log(out_hmm_unc[(0,0)]['SD']), 'g--', label='p(0|0), unc.')
plt.plot(-np.log(out_hmm_unc[(1,0)]['SD']), 'b--', label='p(0|1), unc.')
plt.legend(loc='upper left')
plt.title('Comparison of confidence')

# %% Estimate volatility of a binary sequence with order 1 (coupled) transition probability

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1( \
        np.vstack(( \
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))), \
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))) \
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute observer HMM with full inference (also estimate volatility)
options = {'resol': 20, 'grid_nu': 1/2 ** np.array([k/2 for k in range(20)]),
           'prior_nu': np.ones(20)/20}
out_hmm_full = IO.IdealObserver(seq, 'hmm+full', order=1, options=options)

# Compute observer HMM with full inference (also estimate volatility)
options = {'resol': 20, 'p_c': 1/L}
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(out_hmm_full['volatility'])
plt.title('Volatility estimate')

plt.subplot(3, 1, 2)
plt.imshow(out_hmm[(0,1)]['dist'])
plt.title('p(1|0), full inference')

plt.subplot(3, 1, 3)
plt.imshow(out_hmm_full[(0,1)]['dist'])
plt.title('p(1|0), assuming vol.=1/L')
