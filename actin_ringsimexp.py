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
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.spatial import distance
import os
import math

#plt.close("all")

class exp_profiles:
    
    def __init__(self, d, *args, **kwargs):
        
#        self.rootdir = r"C:\Users\CibionPC\Documents\Lu Lopez\Actin Spectrin Rings\Exp Analysis\STED profiles"
#        self.folders = ['DIV8', 'DIV14', 'DIV21', 'DIV28', 'DIV40', 'Actin']
        
        self.rootdir = r"C:\Users\CibionPC\Documents\Lu Lopez\Actin Spectrin Rings\Gollum-STED-Results\STED profiles - 3 lines\Spectrin"
        self.folders = ['8DIV', '14DIV', '21DIV', '28DIV', '40DIV']
        
#        self.rootdir = r"C:\Users\CibionPC\Documents\Lu Lopez\Actin Spectrin Rings\Gollum-STORM-Results\STORM Profiles"
#        self.folders = ['8DIV', '14DIV', '21DIV', '28DIV', '40DIV']
        
#        self.rootdir = r"C:\Users\CibionPC\Documents\Lu Lopez\
#        Actin Spectrin Rings\Simulated Images"
#        self.folders = ['nostruct']
        
#        self.rootdir = r"C:\Users\CibionPC\Documents\Lu Lopez\
#         Actin Spectrin Rings\Simulated Images\Labelling efficiency\high\Profiles"
#        self.folders = ['L20', 'L30', 'L40', 'L50', 'L60', 'L70', 'L80', 'L90', 'L100','L110', 'L120', 'L130', 'L140','L150', 'L160', 'L170', 'L180', 'L190', 'L200', '210']
                
        self.pxSize = 20 # px size in nm
        self.π = np.pi
        self.a = d # DIV folder to analize
        
        
        self.axon = {}
        self.width = {}
        self.Nrings = {}
        self.pexp = {}
        self.pexpn = {}
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
#                self.pexpn[i,j] = self.pexp[i,j]/np.max(self.pexp[i,j])
                self.width[i,j] = len(self.pexp[i,j][self.pexp[i,j]!=0])
                self.D[i,j] = self.width[i,j]*self.pxSize
                self.center[i,j] = np.argmax(self.pexp[i,j]>0)*self.pxSize + self.D[i,j]/2
            
            self.Npoints[i] = len(self.pexp[i,0])
            self.dist[i] = self.Npoints[i]*self.pxSize
            self.x[i] = np.linspace(0, self.dist[i], self.Npoints[i])
        
class actin_ring:
    
    def __init__(self, L, l, N, D, x, center, *args, **kwargs):
        
        self.π = np.pi
        self.L = L # actin filament lenght in nm
        self.l = l # index of rotation
        self.N = N
        self.rxy = 40  # radial sigma in nm
        self.r = self.rxy/2.35
        self.D = np.int(D - 4*self.rxy)# axon diameter in nm
        self.x = x # distance in nm
        self.Npoints = len(self.x)
        self.center = center     
        self.S = 1  # % of staining         
        # vertices of the regular polygon (determined by L/D)
        if self.D>self.L:
            self.K = np.int(self.π/(np.arcsin(L/self.D)))
        else:
            self.K = 2
        self.φ = (self.l/(self.N-1))*self.π/self.K                 
        self.polyvert()
        self.simprofile()
        
            
    def polyvert(self):
        
        # creates regular polygon with K vertices (centered in [0,0])
        # rotates by  φ the polygon (N possible rotations for each polygon)
        # project polygon vertices in one direction
       
        self.θ = np.ones((self.K))
        self.posx = np.ones((self.K))
        
        for k in np.arange(self.K):
            self.θ[k] = self.π * 2*k/self.K + self.φ
            self.posx[k]= (self.D/2)*np.cos(self.θ[k]) + self.center
       
                          
    def simprofile(self):
        
        
        distpdfn = np.ones((self.K,self.Npoints))
#        self.psim = np.ones((self.Npoints))

        for k in np.arange(self.K):    
#             mask = np.random.binomial(1, self.S, self.K)             
             sigma = self.rxy
             mu = self.posx[k]
             dist = norm(mu, sigma)
             distpdfn[k,:] = dist.pdf(self.x)/max(dist.pdf(self.x))
             
        self.psim = np.sum(distpdfn, axis = 0)
#        self.psim = np.round(self.psim, decimals = 3)
#            self.psimn[i,:] = self.psim[i,:]/np.max(self.psim[i,:])
        

if __name__ == '__main__':
    
    # load all exp profiles from axons and rings of selected DIV
    # 0 8DIV 1 14DIV 2 21DIV 3 28DIV 4 40DIV
    #    Lmins = 160
    #    Lmaxs = 170
    #    Lsim = np.arange(Lmins, Lmaxs, 10)
    #    M = len(Lsim)
    #    
    ##    Lm = np.zeros(M)
    ##    sigmaL = np.zeros(M)
    ##    corrm = np.zeros(M)
    #    
    #    for m in np.arange(14,15,1):
        ring_exp = exp_profiles(1);
        pexp = ring_exp.pexp;
        Naxons = ring_exp.Naxons;
        Nrings = ring_exp.Nrings;
        x = ring_exp.x;
        D = ring_exp.D;
        center = ring_exp.center;
            
        # L range to scan in simulated ring
        Lmin = 50; 
        Lmax = 200;
        DL = 10;
        NL = np.int((Lmax-Lmin)/DL);
        L = np.arange(Lmin, Lmax, DL);

        # number of rotations φ for each L
        N = 10;    
        # define empty arrays for pearson corr (pcoef)
        pcoef = {}
        s = np.ones(Naxons);
        for i in np.arange(Naxons):
            s[i] = Nrings[i];
        NringsT = np.int(np.sum(s));
        
    
        Npoints = {};
        pcoef = np.ones((NringsT, NL, N))
        pcoefrec = np.ones((NringsT, NL, N))
        phi = np.ones((NL, N))
        theta = np.ones((NL, N))
        ang = np.ones((N))
            
        Lmaxc = np.zeros((NringsT))
        angmax = np.zeros((NringsT))

        Kmaxc = np.zeros((NringsT))
        Kmax2 = np.zeros((NringsT))

        K = np.ones((NL))
        corrmax = np.zeros((NringsT))

    
        psim = {}
        jj = 0
        for i in np.arange(Naxons):
            Npoints[i] = ring_exp.Npoints[i]
            for j in np.arange(Nrings[i]):
#                fig,ax = plt.subplots(1,2)
#                ax[0].plot(x[i], pexp[i,j])
                centerind = center[i,j]/20
                Dind = D[i,j]/20;
                minind = np.int(centerind-(Dind/2 + 1) - 1);
                maxind = np.int(centerind+(Dind/2 + 1) + 1);
                pexprec = pexp[i,j][minind:maxind];
#                ax[0].plot(x[i][minind:maxind], pexprec)
                
                for k in np.arange(NL):
                    for l in np.arange(N):
                        ring_sim = actin_ring(L[k], l, N, D[i,j],x[i], center[i,j]);
                        psim[jj,k,l] = ring_sim.psim  
                        ang[l] = np.round(l/(2*(N-1)), decimals = 2)
                        psim1 = psim[jj,k,l]
                        psimrec = psim1[minind:maxind];
                        pcoefrec[jj, k, l] = pearsonr(psimrec, pexprec)[0]
                        pcoef[jj, k, l] = pearsonr(psim[jj,k,l], pexp[i,j])[0]

                              
                ind = np.unravel_index(np.argmax(pcoef[jj,:,:], axis=None), pcoef.shape) ; 
                indrec = np.unravel_index(np.argmax(pcoefrec[jj,:,:], axis=None), pcoef.shape) ; 
#                ax[1].plot(x[i], psim[jj,ind[1],ind[2]])
#                ax[1].plot(x[i][minind:maxind], psim[jj,ind[1],ind[2]][minind:maxind])
#                ax[1].plot(x[i], psim[jj,indrec[1],indrec[2]])
#                ax[1].plot(x[i][minind:maxind], psim[jj,indrec[1],indrec[2]][minind:maxind])

                Lmaxc[jj] = L[indrec[1]]
                angmax[jj] = ang[indrec[2]]


                corrmax[jj] = pcoefrec[jj,ind[1], ind[2]]
                jj = jj + 1
         
#        corrm[m] = np.mean(corrmax)
#        Lm[m] = np.mean(Lmaxc)
#        sigmaL[m] = np.std(Lmaxc)
#        
#np.savez('nostruct',corrm, Lm, sigmaL)

#
#plt.figure()
#plt.errorbar(Lsim, Lm, yerr=sigmaL, fmt='o')
#plt.plot(Lsim,Lsim,'k-') # identity line
#plt.xlabel('L sim [nm]')
#plt.ylabel('L corr [nm]')

        
##  get rid of max correlations under a certain value u
#u = 0.6
#indcorr = np.where(corrmax>u) 
#Lcorr = Lmaxc[indcorr]
#Paux = pcoefrec[indcorr[0],:,:]
#P = np.mean(Paux, axis = 0)

##
                
# Mean correlation map  (using complete profiles)              
#P = np.mean(pcoef, axis = 0)
#fig, ax = plt.subplots(1,1)
#im = ax.imshow(np.transpose(P), vmin=np.mean(P), vmax=np.max(P), origin='lower')
#ax.set_xticks(np.arange(0, NL, 2*np.int(NL/DL)))
#ax.set_xticklabels(L[np.arange(0, NL, 2*np.int(NL/DL))])
#ax.set_yticks(np.arange(0, N, 1))
#ax.set_yticklabels(ang[np.arange(0, N, 1)])
#ax.set_xlabel('L [nm]')
#ax.set_ylabel('Rotation')
#ax.set_title('Correlation Map')
#cbar = fig.colorbar(im)
#
#
## Mean correlation map  (using cropped profiles)  
#Prec = np.mean(pcoefrec, axis = 0)
#fig, ax = plt.subplots(1,1)
#im = ax.imshow(np.transpose(Prec), vmin=np.mean(Prec), vmax=np.max(Prec), origin='lower')
#ax.set_xticks(np.arange(0, NL, 2*np.int(NL/DL)))
#ax.set_xticklabels(L[np.arange(0, NL, 2*np.int(NL/DL))])
#ax.set_yticks(np.arange(0, N, 1))
#ax.set_yticklabels(ang[np.arange(0, N, 1)])
#ax.set_xlabel('L [nm]')
#ax.set_ylabel('Rotation')
#ax.set_title('Correlation Map')
#cbar = fig.colorbar(im)


## Mean correlation map  (using cropped profiles)  
#Prec = np.mean(pcoefrec, axis = 0)
for jj in np.arange(20,30,1):
    Prec = pcoefrec[jj,:,:]
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(np.transpose(Prec), vmin=np.mean(Prec), vmax=np.max(Prec), origin='lower')
    ax.set_xticks(np.arange(0, NL, 2*np.int(NL/DL)))
    ax.set_xticklabels(L[np.arange(0, NL, 2*np.int(NL/DL))])
    ax.set_yticks(np.arange(0, N, 1))
    ax.set_yticklabels(ang[np.arange(0, N, 1)])
    ax.set_xlabel('L [nm]')
    ax.set_ylabel('Rotation')
    ax.set_title('Correlation Map')
    cbar = fig.colorbar(im)
     
    
