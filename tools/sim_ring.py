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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from scipy import signal


#plt.close("all")
N=5

class actin_ring:
    
    def __init__(self, *args, **kwargs):
        
        self.π = np.pi
        self.L = 160 # actin filament lenght in nm
        self.D = 400  # axon diameter in nm
#        rz = 500  # axial resolution in nm
        self.rxy = 40  # radial resolution in nm
        self.S = 1   # % of staining
        # vertices of the regular polygon (determined by L/D)
        self.K = np.int(self.π/(np.arcsin(self.L/self.D)))
        
        self.polyvert()
        self.polyrot()
        self.polyproject()
        self.intensityprofiles()
        self.peaks()
        
        
    def polyvert(self):
        
        # creates regular polygon with K vertices (centered in [0,0])
        
        self.θ = {}
        self.pos = {}
        if (self.K+1) % 2 == 0:  # if K is even
            for k in np.arange(self.K):
                self.θ[k] = self.π * 2*k/self.K
                self.pos[k]= (self.D/2)*np.array([np.cos(self.θ[k]), np.sin(self.θ[k])])
        else:       # if K is odd
            for k in np.arange(self.K):
                self.θ[k] = self.π * (2*k+1)/self.K
                self.pos[k]= (self.D/2)*np.array([np.cos(self.θ[k]), np.sin(self.θ[k])])
            
        
        
    def polyrot(self):
        
        # rotates by  φ the polygon
         
            φ = np.random.uniform(0,self.θ[1])
           
            θrot = {}
            self.posrot = {}
            if (self.K+1) % 2 == 0:  # if K is even
                for k in np.arange(self.K):
                    θrot[k] = self.π * 2*k/self.K + φ
                    self.posrot[k]= (self.D/2)*np.array([np.cos(θrot[k]), np.sin(θrot[k])])
            else:       # if K is odd
                for k in np.arange(self.K):
                    θrot[k] = self.π * (2*k+1)/self.K + φ
                    self.posrot[k]= (self.D/2)*np.array([np.cos(θrot[k]), np.sin(θrot[k])])
        
    def polyproject(self):
        
        # projects vertices in x axis
        
        self.posx = {}
        self.posrotx = {}
        delta = {}
        for i in np.arange(self.K):
            delta[i] = np.random.uniform(0,10)
            self.posx[i] = np.array([self.pos[i][0]+delta[i], 0])
            self.posrotx[i] = np.array([self.posrot[i][0]+delta[i], 0])
      
    
            
    def intensityprofiles(self):      
        
        # assigns intensity profile to projections

        dist = {}
        self.distpdf = np.ones((self.K,1000))
        distpdfn = np.ones((self.K,1000))
        self.x = np.linspace(-300, 300, 1000)
        
        for i in np.arange(self.K):

             mask = np.random.binomial(1, self.S, self.K)
             
             sigma = self.rxy
             mu = self.posrotx[i][0]
             dist[i] = norm(mu, sigma)
             distpdfn[i] = dist[i].pdf(self.x)/max(dist[i].pdf(self.x))
             self.distpdf[i, :] = distpdfn[i] * mask[i]
                 
        self.intensityprofile = np.sum(self.distpdf, axis = 0)
        

        
    def peaks(self):
        
        # counts peaks in intensity profile and their distance
        self.peaks = signal.find_peaks_cwt(self.intensityprofile, np.arange(1,100))
        self.npeaks = len(self.peaks)
        self.distpeaks = np.diff(self.peaks)
        
        

         
         
        
if __name__ == '__main__':

    ring = {}
    numberofpeaks = np.ones(N)
    distpeaks = {}
    
#    fig, ax = plt.subplots(1, 2)

    for i in np.arange(N):
    
        ring[i] = actin_ring()
        numberofpeaks[i] = ring[i].npeaks
        distpeaks[i] = ring[i].distpeaks
        plt.plot(ring[i].intensityprofile)
    