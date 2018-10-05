# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:51:08 2018

@author: CibionPC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:58:27 2018

@author: CibionPC
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from numpy import linalg as LA
from scipy.stats import norm, chi2

π = np.pi
M = 2*10**6  # illumination cicles
L = 100 # actin filament lenght in nm
D = 400  # axon diameter in nm
rz = 500  # axial resolution in nm
#rxy =   # radial resolution in nm

def polyvert(L,D):

    K = np.int(np.pi/(np.arcsin(L/D)))
    θ = {}
    pos_nm = {}

    for k in np.arange(K):
        θ[k] = π * 2*k/K
        pos_nm[k]= (D/2)*np.array([np.cos(θ[k]), np.sin(θ[k])])

            
    return K, pos_nm,  θ


[K, pos_nm,  θ] = polyvert(L,D) 
pos_nmx = {}
delta = {}
for i in np.arange(K):
    delta[i] = np.random.uniform(0,10)
    pos_nmx[i] = np.array([pos_nm[i][0]+delta[i], 0])
  
markercolor1 = 'go'
markercolor2 = 'ro'

    
fig7, ax7 = plt.subplots(1, 1)
for i in np.arange(K):
    ax7.plot(pos_nm[i][0], pos_nm[i][1] , markercolor1, markersize=10)
    ax7.plot(pos_nmx[i][0], pos_nmx[i][1], markercolor2, markersize=10, fillstyle='none')
    ax7.add_patch(patches.RegularPolygon([0, 0], K, D/2, fill=False, orientation=θ[0], linestyle='solid'))
ax7.axis('equal')