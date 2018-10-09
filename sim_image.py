# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:41:17 2018

@author: L Masullo & L Lopez
"""


import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.stats import norm

class sim_image:
    
    def __init__(self, L=100, N=1, D=400, *args, **kwargs):
        
        self.π = np.pi
        self.px_size = 20 # px size in nm
        self.im_size = 2000 # im size in nm
        self.rxy = 40 # radial resolution in nm
        
        # axon parameters
        self.r = 10  # ring widht CHEQUEAR ESTO 
        self.D = D - 3*self.r # axon diameter in nm
        self.axon_start=.2
        self.axon_end=  self.axon_start + self.D/1000
        
        # ring parameters
        self.L = L # actin filament lenght in nm
        self.N = N #total number of rotations
        
        self.K = np.int(self.π/(np.arcsin(L/self.D)))
        
        self.x = np.linspace(0,self.im_size, self.im_size) # distance in nm
        self.Npoints = len(self.x)
        self.center = self.im_size/2     

        
        self.wvlen = 2*190 # longitudinal wvlen in nm 
        self.phase = 0.25
        self.b = 6
        self.g_noise=0.25
        self.c = 0.8 # contrast
        self.theta = 0 # axon orientation
        
        self.label = 0.5 # labeling efficency. It is very low because of how we define
                          # the spectrin rings in a continous way
            
        self.i = 45        # angle is fixed at 30 deg since it was rather irrelevant for
                          # this simulations, although I tried to avoid 0 or 90 since
                          # they could be regarded as a particular case
        
        self.n_molecules = 1000  # this variable will keep track of how many "fluorophores
                             # you end up putting in your simulated image, it should
                             # be a reasonable number according to inmunostaining
                             # stochiometry
        
        self.signal= 100
        self.poisson = True
        self.w1 = 0.2
        
        self.actin_ring()
        self.simAxon()
        self.imageAxon()
                
    def actin_ring(self):
       
        self.φ = np.linspace(0, (self.π)/self.K, self.N)
        self.posx = np.ones((self.N,self.K))
        self.posxround = np.ones((self.N,self.K))
        for i in np.arange(self.N):
            for k in np.arange(self.K):
                θ = self.π * 2*k/self.K + self.φ[i]
                self.posx[i,k]= (self.D/2)*np.cos(θ) + self.center
                self.posxround[i,k]= np.round(self.posx[i,k], decimals=0)
        
        distpdfn = np.ones((self.K,self.Npoints))
        self.psim = np.ones((self.N, self.Npoints))
        
        for i in np.arange(self.N):
            for k in np.arange(self.K):                
                 sigma = self.r
                 mu = self.posx[i,k]
                 dist = norm(mu, sigma)
                 distpdfn[k,:] = dist.pdf(self.x)/max(dist.pdf(self.x))
                 
            self.psim[i,:] = np.sum(distpdfn, axis = 0)
                               

    def simAxon(self):

        # X is a vector from 1 to imageSize
        self.X = np.arange(1, self.im_size + 1)
        # rescale X -> -.5 to .5
        self.X0 = (self.X / self.im_size) - .5
        # compute frequency from wavelength
        self.freq = self.im_size/self.wvlen
        # convert to radians: 0 -> 2*pi
        self.phaseRad = (self.phase * 2*self.π)       
        [self.Xm, self.Ym] = np.meshgrid(self.X0, self.X0)     # 2D matrices
        # convert theta (orientation) to radians
        self.thetaRad = (self.theta / 360) * 2*self.π
        # compute proportion of Xm for given orientation
        self.Xt = self.Xm * np.cos(self.thetaRad)
        # compute proportion of Ym for given orientation
        self.Yt = self.Ym * np.sin(self.thetaRad)
        # sum X and Y components
        self.XYt = np.array(self.Xt + self.Yt)
        # convert to radians and scale by frequency
        self.XYf = self.XYt * self.freq * 2*self.π 
        # make 2D sinewave
        self.sin2d = np.sin(self.XYf + self.phaseRad)
        self.ringpattern = self.sin2d**self.b

        # creates the start and end point of the axon contour 
        a0 = np.int(self.axon_start * self.im_size)
        a1 = np.int(self.axon_end * self.im_size)

        # intensity offset given that contrast is c
        Ioffset = (1-self.c)*np.max(self.ringpattern)/(2*self.c)  # Ibkg depends on the contrast
        
        
        # create ring profile mask
        ring_mask = np.zeros(self.im_size*self.im_size)
        psim = self.psim[0,:]
        for i in np.arange(self.im_size):
            ring_mask[i*self.im_size:(i+1)*self.im_size] = psim.astype(int)
            
        self.ring_mask = ring_mask.reshape(self.im_size, self.im_size)
        self.ring_mask = np.transpose(self.ring_mask)
        
        # creates the axon_mask and the ring_mask
        axon_mask = np.zeros(self.im_size*self.im_size)        
        axon_mask[a0:a1] = 1
        self.axon_mask = axon_mask.reshape(self.im_size, self.im_size)


        ring_offset = np.ones(self.im_size*self.im_size)
        ring_offset = Ioffset*ring_offset.reshape(self.im_size, self.im_size)
        
        # normalizes and creates the final simulated axon with spectrin rings

        norm_f = np.max(self.ringpattern+ring_offset)

        axon_res = (self.ring_mask*(self.ringpattern + ring_offset) + Ioffset*self.axon_mask)/norm_f
        axon_res = axon_res + self.g_noise*np.random.rand(self.im_size, self.im_size)
#        axon_res = ndi.rotate(axon_res, angle, reshape=False)


        self.data = axon_res.astype(np.float32) # result to be used in simulations
        
#        fig,ax= plt.subplots()
#        ax.imshow(self.data)



    def imageAxon(self):

        image = self.data # actual simulated image 

        result = image.reshape(np.size(image)) # image is reshaped into a 1D array
                                                   # in order to be able to loop on it
            
        # NOTE: result has values between 0 and 1
        
        # the following is a loop that "labels" the axon according to the labeling
        # density and the weight parameters. 
        
        for j in np.arange(np.size(result)):
            
            # if this condition is accomplished (it is random) then the simulated 
            # pixel of the image is kept as it was (non zero)
            if np.random.rand(1) < self.label*(self.w1*result[j]):
                self.n_molecules = self.n_molecules+1
            else:
                result[j] = 0
                
            # image is reshaped to a 2D array
            # TO DO: changes this numbers (2000, 2000) to a variable, as it is now, you
            # get a "digital" simulated resolution of 1 nm        
        
        image = result.reshape(self.im_size, self.im_size)   
            
        # convolute the 1 nm molecules with the resolution of the microscope    
        image = ndi.gaussian_filter(image, self.rxy/2.35)            
        image_nobkg = image/np.max(image)  # normalization
        j = 0 
        
        # different backgrounds are added to the image. This loop was done to test
        # the different SBR for Gollum, but this can be changed, for example you can
        # use a single bkg value    
        
        # bkg is simulated as a gaussian noise around a certain value
        
        bkg_array = np.arange(0, 4, 1)
        for bkg in [1]:
            # addition of bkg + normalization
            image1 = image_nobkg + bkg *(1 + 0.1*np.random.rand(self.im_size, self.im_size))
            image1 = image1/np.max(image1)
            fig,ax = plt.subplots()
            plt.imshow(image1)
            # resample of the image to get 20 nm pixel
            # TO DO: change 0.05 to a variable
    
            image1 = ndi.interpolation.zoom(image1, 1/self.px_size)
            fig,ax = plt.subplots()
            plt.imshow(image1)
            # get rid of possible negative values produced by the sigma of the bkg
            # they are usually just a few pixels, no big deal
            image1[image1 < 0] = 0
            
            # adds poisson noise, simulating an APD/CCD detection
            if self.poisson == True:
                image1 = np.random.poisson(self.signal*image1)
            else:
                pass
            fig,ax = plt.subplots()
            plt.imshow(image1)
            # here you take a 50 x 50 pixels (1 μm x 1 μm) from the 
            # 100 x 100 pixels (2 μm x 2 μm) simulation, this was done to remove
            # boundary effects
            subimage = image1[25:75, 25:75]
            subimage = subimage.astype(np.float32)
            fig,ax = plt.subplots()
            plt.imshow(subimage)
            # saves the image with data in the title
            # TO DO: make a metadata file instead of saving in the title
    
            tiff.imsave('{}testAxon_angle{}_bkg{}_label{}_contrast{}_w1{}_poisson{}.tiff'.format(j, self.i, np.around(bkg, 1), self.label, self.c, self.w1, self.poisson), subimage)
            j = j + 1
            
        # prints the number of molecules to check if it is a reasnoable number
        print(self.n_molecules)







