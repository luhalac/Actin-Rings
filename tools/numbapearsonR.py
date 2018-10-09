# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:50:46 2018

@author: CibionPC
"""
import numpy as np
from numba import jit
from scipy.stats.stats import pearsonr
import time

t0 = time.clock()
N = 100000000
P = np.zeros(N)


@jit # Set "nopython" mode for best performance
def pearsonR(N):    
    for i in np.arange(N):
    
        a = np.random.permutation(N)
        b = np.random.permutation(N)
        # first with built in pearson r from scipy
#        P[i] = pearsonr(a,b)[0]
        an = a - np.mean(a)
        bn = b - np.mean(b)
        aa = an * an
        bb = bn * bn
        ab = an * bn
        
        P[i] = np.sum(ab)/np.sqrt(np.sum(aa)*np.sum(bb))
        
    return P

t1 = time.clock()

T = t1 - t0

