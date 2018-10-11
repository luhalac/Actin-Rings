# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:36:19 2018

@author: Lucia Lopez
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class actin_ring:
    
    def __init__(self, L, N, D, x, center, *args, **kwargs):
        
        self.π = np.pi
        self.L = L # actin filament lenght in nm
        self.N = N #total number of rotations
        self.rxy = 40  # radial resolution in nm
        self.D = D - 3*self.rxy # axon diameter in nm
        self.x = x # distance in nm
        self.Npoints = len(self.x)
        self.center = center     
        self.S = 1  # % of staining
        
        # vertices of the regular polygon (determined by L/D)
        if self.D>self.L:
            self.K = np.int(self.π/(np.arcsin(L/self.D)))
        else:
            self.K = 2
            
                 
        self.polyvert()
        self.simprofile()
        
            
    def polyvert(self):
        
        # creates regular polygon with K vertices (centered in [0,0])
        # rotates by  φ the polygon (N possible rotations for each polygon)
        # project polygon vertices in one direction
       
        self.φ = np.linspace(0, (self.π)/self.K, self.N)
        self.θ = np.ones((self.N,self.K))
        self.posx = np.ones((self.N,self.K))
        for i in np.arange(self.N):
            for k in np.arange(self.K):
                self.θ[i,k] = self.π * 2*k/self.K + self.φ[i]
                self.posx[i,k]= (self.D/2)*np.cos(self.θ[i,k]) + self.center
       
                          
    def simprofile(self):
        
        
        distpdfn = np.ones((self.K,self.Npoints))
        self.psim = np.ones((self.N, self.Npoints))
        for i in np.arange(self.N):
            for k in np.arange(self.K):    
    #             mask = np.random.binomial(1, self.S, self.K)             
                 sigma = self.rxy
                 mu = self.posx[i,k]
                 dist = norm(mu, sigma)
                 distpdfn[k,:] = dist.pdf(self.x)/max(dist.pdf(self.x))
                 
            self.psim[i,:] = np.sum(distpdfn, axis = 0)

if __name__ == '__main__':
    
    N=2
    x = np.linspace(0,1300,60)
    ring = actin_ring(160,N,500,x, 700);
    for i in np.arange(N):
        plt.plot(x,ring.psim[i,:])        

           