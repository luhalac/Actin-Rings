# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:48:05 2018

@author: CibionPC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

K = 20
S = 0.5

if S is not None:
    mask = np.random.binomial(1, S, K)
    Km = mask*np.arange(K)
    Km = Km[np.nonzero(Km)]
    a={}
    for i in Km:
        a[i] =i
    
