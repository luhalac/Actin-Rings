# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:06:31 2018

@author: CibionPC
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from scipy.stats import norm
from scipy import signal
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.spatial import distance
import os
import math
import actin_ring



plt.close("all")

class exp_profiles:
    
    def __init__(self, d, *args, **kwargs):
        
        self.rootdir = 'C:\Users\CibionPC\Documents\Lu Lopez\Actin Spectrin Rings\Exp Analysis\STED profiles'
        self.folders = ['DIV8', 'DIV14', 'DIV21', 'DIV28', 'DIV40']
        self.pxSize = 20 # px size in nm
        
        self.π = np.pi
        self.a = d # DIV folder to analize
        
        
        
        self.axon = {}
        self.width = {}
        self.Nrings = {}
        self.pexp = {}
        self.D = {}
        self.center = {}
        self.Npoints = {}
        self.dist = {}
        self.x = {}
        
        self.load_axon()
        self.load_ring()

        
    def load_axon(self):   
        # load axons from each folder
        self.axons = {}  
        filepath = {}
        folderpath = os.path.join(self.rootdir, self.folders[self.a])
        for subdir, dirs, files in os.walk(folderpath):
          Nfiles = len(files)
          for i in np.arange(Nfiles):
              file = files[i]
              filepath[i] = subdir + os.sep + file
              self.axons[i] = filepath[i]
        self.Naxons = len(self.axons)     

        
    def load_ring(self):
        # load rings from each axon
        for i in np.arange(self.Naxons):
            self.axon[i] = np.load(self.axons[i])
            print(self.axons[i])
            self.Nrings[i] = np.shape(self.axon[i])[0]    
            for j in np.arange(self.Nrings[i]):
                self.pexp[i,j] = self.axon[i][j]
                self.width[i,j] = len(self.pexp[i,j][self.pexp[i,j]!=0])
                self.D[i,j] = self.width[i,j]*self.pxSize
                self.center[i,j] = np.argmax(self.pexp[i,j]>0)*self.pxSize + self.D[i,j]/2
            
            self.Npoints[i] = len(self.pexp[i,0])
            self.dist[i] = self.Npoints[i]*self.pxSize
            self.x[i] = np.linspace(0, self.dist[i], self.Npoints[i])
        
        

if __name__ == '__main__':
    
    # load all exp profiles from axons and rings of selected DIV
    # 0 8DIV 1 14DIV 2 21DIV 3 28DIV 4 40DIV
    ring_exp = exp_profiles(4)
    pexp = ring_exp.pexp;
    Naxons = ring_exp.Naxons;
    Nrings = ring_exp.Nrings;
    x = ring_exp.x;
    D = ring_exp.D;
    center = ring_exp.center;
    
    
    # L range to scan in simulated ring
    Lmin = 10; 
    Lmax = 210;
    DL = 10;
    NL = np.int((Lmax-Lmin)/DL);
    L = np.arange(Lmin, Lmax, DL);
    
    # number of rotations φ for each L
    N = 20;
    Nrot = np.arange(N)
    
    # define empty arrays for pearson corr (pcoef)
#    pcoef = {}
    s = np.ones(Naxons);
    for i in np.arange(Naxons):
        s[i] = Nrings[i];
    NringsT = np.int(np.sum(s));
    
#    Nringsmax = np.int(np.max(s))   
    psim = {}
    Npoints = {};
    ang = np.ones((NringsT, NL, N))
    pcoef = np.ones((NringsT, NL, N))
    
    pcoef1 = np.ones((NringsT, NL, N))
    
    Lmaxc = np.zeros((NringsT))
    Kmax = np.zeros((NringsT))
    phimax = np.zeros((NringsT))
    corrmax = np.zeros((NringsT))


    jj = 0
    for i in np.arange(Naxons):
        Npoints[i] = ring_exp.Npoints[i]
        for j in np.arange(Nrings[i]):
#            pexpnorm = pexp[i,j]/np.max(pexp[i,j])

#            fig,ax = plt.subplots(1,2)
#            ax[0].plot(x[i], pexpnorm)
            for k in np.arange(NL):
                for l in np.arange(N):
                    if  (D[i,j]-80)>L[k]: 
                        ring_sim = actin_ring(L[k],Nrot[l], N, D[i,j],Npoints[i],x[i], center[i,j]);

                        psim = ring_sim.psim;
#                        psimnorm = psim[jj,k,l]/np.max(psim[jj,k,l])
#                        ax[1].plot(x[i], psimnorm)
                        phi = ring_sim.φ;
                        theta = ring_sim.θ[1]
                        pcoef[jj,k,l] = pearsonr(psim, pexp[i,j])[0]
#                        pcoef1[jj,k,l] = np.correlate(psimnorm, pexpnorm)[0]
                        ang[jj,k,l] = np.round(phi/theta, decimals = 2)

                        
                    else: 
                      pass
  
#            plt.plot(np.mean(pcoef1[jj,:,:], axis = 1))      
            ind = np.unravel_index(np.argmax(pcoef[jj,:,:], axis=None), pcoef.shape) ;
#            ind1 = np.unravel_index(np.argmax(pcoef1[jj,:,:], axis=None), pcoef1.shape) ;

            
            Lmaxc[jj] = L[ind[1]]
#            Kmax[jj] = np.int(np.pi/(np.arcsin(Lmaxc[jj]/D[i,j])))
#            Lmaxc1[jj] = L[ind1[1]]
#            phimax[jj] = phi[jj, ind[1], ind[2]]
#            phimax[jj] = np.round(phimax[jj], decimals=2)
            corrmax[jj] = pcoef[jj,ind[1], ind[2]]
            corrmax[jj] = np.round(corrmax[jj], decimals=2)
            
#
            jj = jj + 1
#            print(jj)
#            
ang = np.round(np.mean(np.mean(ang, axis=0), axis=0), decimals=2)
#get rid of max correlations under a certain value u
u = 0.7
#indcorr = np.where(corrmax>u) 
indcorr = [(corrmax >= u) & (corrmax < 1)]
#and np.where(corrmax<1)

P1 = np.mean(pcoef, axis = 0)
  
Paux = pcoef[indcorr[0],:,:]
P = np.mean(Paux, axis = 0)
#
fig, ax = plt.subplots(1,1)
im = ax.imshow(np.transpose(P), vmin=np.mean(P), vmax=np.max(P), origin='lower')
ax.set_xticks(np.arange(0, NL, 5))
ax.set_yticks(np.arange(0, N, 5))
ax.set_xticklabels(L[np.arange(0, NL, 5)])
ax.set_yticklabels(ang[np.arange(0, N, 5)])
cbar = fig.colorbar(im)

fig2, ax2 = plt.subplots(1,1)
im2 = ax2.imshow(np.transpose(P1), vmin=np.mean(P1), vmax=np.max(P1), origin='lower')
ax2.set_xticks(np.arange(0, NL, 5))
ax2.set_yticks(np.arange(0, N, 5))
ax2.set_xticklabels(L[np.arange(0, NL, 5)])
ax2.set_yticklabels(ang[np.arange(0, N, 5)])
cbar2 = fig2.colorbar(im2)

#            
#            
##P1 = np.mean(pcoef1, axis =0)
#for jj in np.arange(NringsT):
#    fig, ax = plt.subplots(1,1)
#    u = np.mean(pcoef[jj,:,:]) + 0*np.std(pcoef[jj,:,:])
#    im = ax.imshow(np.transpose(pcoef[jj,:,:]), vmin=u, vmax=np.max(pcoef[jj,:,:]), origin='upper')
#    ax.set_xticks(np.arange(0, NL, 2))
#    ax.set_yticks(np.arange(0, N, 2))
#    ax.set_xticklabels(L[np.arange(0, NL, 2)])
#    ax.set_yticklabels(ang[np.arange(0, N, 2)])
#    cbar = fig.colorbar(im)
#
#fig1, ax1 = plt.subplots(1,1)
#im1 = ax1.imshow(np.transpose(P1), vmin=np.mean(P1), vmax=np.max(P1))
#cbar1 = fig1.colorbar(im1)       
    
