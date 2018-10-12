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
    
    def __init__(self, L, D=400, *args, **kwargs):
        
        self.π = np.pi
        self.px_size = 20 # px size in nm
        self.im_size = 2000 # im size in nm
        self.rxy = 40 # radial resolution in nm (FWHM)
        self.sigma = self.rxy/2.35 # sigma of the PSF 
        
        # axon model parameters    
        self.D0 = D  # axon diameter in nm
        self.axon_start =.4 #axon position relative to image border
        self.axon_end =  self.axon_start + self.D0/self.im_size
        self.wvlen = 190 # longitudinal wvlen in nm 
        self.phase = 0.25
        self.b = 6
        self.theta = 90 # axon orientation (if 90, vertical axon)
        self.c = 0.7 # contrast
        
        # ring model parameters
        self.r = 20  # ring widht CHEQUEAR ESTO 
        self.D = self.D0 - 2*self.r
        self.L = L # actin filament lenght in nm        
        self.K = np.int(self.π/(np.arcsin(L/self.D))) # number of vertices  
        self.x = np.linspace(0,self.im_size, self.im_size) # distance in nm
        self.Npoints = len(self.x)
        self.center = self.im_size/2     
        self.φ = np.random.uniform(0,self.π/self.K)
        
        
        self.g_noise=0.2

        self.label = 0.02 # labeling efficency

        self.ringstrc = True
        self.signal= 50
        self.poisson = True

        
        
        self.actin_ring()
        self.simAxon()
        self.imageAxon()
        
                
    def actin_ring(self):

        self.posx = np.ones((self.K))
        self.posxround = np.ones((self.K))
    
        for k in np.arange(self.K):
            θ = self.π * 2*k/self.K + self.φ
            posdelta = np.random.uniform(0,self.r)
            self.posx[k]= (self.D/2)*np.cos(θ) + self.center + posdelta
            self.posxround[k]= np.round(self.posx[k], decimals=0)
    
        distpdfn = np.ones((self.K,self.Npoints))
        self.psim = np.ones((self.Npoints))
        
        for k in np.arange(self.K):                
             sigma = self.r
             mu = self.posx[k]
             dist = norm(mu, sigma)
             distpdfn[k,:] = dist.pdf(self.x)/max(dist.pdf(self.x))
             
        self.psim = np.sum(distpdfn, axis = 0)
                                   

    def simAxon(self):

        # X is a vector from 1 to imageSize
        self.X = np.arange(1, self.im_size + 1)
        # rescale X -> -.5 to .5
        self.X0 = (self.X / self.im_size) - .5
        [self.Xm, self.Ym] = np.meshgrid(self.X0, self.X0)     # 2D matrices
       
        # compute frequency from wavelength
        self.freq = self.im_size/self.wvlen
        # convert to radians: 0 -> 2*pi
        self.phaseRad = (self.phase * 2*self.π)
        self.thetaRad = (self.theta / 360) * 2*self.π
        
        # compute proportion of Xm,Ym for given orientation
        self.Xt = self.Xm * np.cos(self.thetaRad)
        self.Yt = self.Ym * np.sin(self.thetaRad)
        # sum X and Y components
        self.XYt = np.array(self.Xt + self.Yt)
        # convert to radians and scale by frequency
        self.XYf = self.XYt * self.freq/2 * 2*self.π 
        # make 2D sinewave and pattern (sin**6)
        self.sin2d = np.sin(self.XYf + self.phaseRad)
        self.ringpattern = self.sin2d**self.b

        # creates the start and end point of the axon contour 
        a0 = np.int(self.axon_start * self.im_size * self.im_size)
        a1 = np.int(self.axon_end * self.im_size * self.im_size)
        
        # creates the axon_mask using axon contour
        axon_mask = np.zeros(self.im_size*self.im_size)        
        axon_mask[a0:a1] = 1
        # vertical axon
        self.axon_mask = np.transpose(axon_mask.reshape(self.im_size, self.im_size))
        
        # if ringstrc is True, add ring profile
        if self.ringstrc == True:
            # create ring profile mask
            ring_mask = np.zeros(self.im_size*self.im_size)
            psim = self.psim        
            for i in np.arange(self.im_size):
                ring_mask[i*self.im_size:(i+1)*self.im_size] = psim            
            self.ring_mask = ring_mask.reshape(self.im_size, self.im_size)
            
            self.offset = np.max(self.ring_mask)*(1.5-self.c)/2*self.c
            #normalizes and creates the final simulated axon with spectrin rings
            norm_f = np.max(self.ring_mask+ self.offset*self.axon_mask)        
            self.axon_res = (self.ringpattern*self.axon_mask*self.ring_mask + self.offset*self.axon_mask)/norm_f
            self.axon_res = self.axon_res + self.g_noise*np.random.rand(self.im_size, self.im_size)
#           axon_res = ndi.rotate(axon_res, angle, reshape=False)
        
        # if ringstrc is False, no ring profile
        if self.ringstrc == False:
            # for axon without ring structure
            self.offset = np.max(self.ring_pattern)*(1.5-self.c)/2*self.c
            norm_f = np.max(self.ringpattern+ self.offset*self.axon_mask)        
            self.axon_res = (self.ringpattern*self.axon_mask+ self.offset*self.axon_mask)/norm_f
            self.axon_res = self.axon_res + self.g_noise*np.random.rand(self.im_size, self.im_size)
        

        self.data = self.axon_res.astype(np.float32) # result to be used in simulations


    def imageAxon(self):

        image = self.data # actual simulated image 
        result = image.reshape(np.size(image)) # image is reshaped into a 1D array
        
        # the following is a loop that "labels" the axon according to the labeling
        # density and the weight parameter. 
        
        for j in np.arange(np.size(result)):            
            # if this condition is accomplished (it is random) then the simulated 
            # pixel of the image is kept as it was (non zero)
            if np.random.rand(1) > self.label*result[j]:
                result[j] = 0
                
        # image is reshaped to a 2D array        
        image = result.reshape(self.im_size, self.im_size)   
            
        # convolute the 1 nm molecules with the resolution of the microscope    
        image = ndi.gaussian_filter(image, self.sigma)            
        image_nobkg = image/np.max(image)  # normalization

        
        # bkg is simulated as a gaussian noise around a certain value    
        bkg = 0.1
        # addition of bkg + normalization
        self.image0 = image_nobkg + bkg *(1 + 0.1*np.random.rand(self.im_size, self.im_size))
        self.image1 = self.image0/np.max(self.image0)

        # resample of the image to get 20 nm pixel
        # TO DO: change 0.05 to a variable 

        image1 = ndi.interpolation.zoom(self.image1, 1/self.px_size)

        # get rid of possible negative values produced by the sigma of the bkg
        # they are usually just a few pixels, no big deal
        image1[image1 < 0] = 0
        
        # adds poisson noise, simulating an APD/CCD detection
        if self.poisson == True:
            self.image2 = np.random.poisson(self.signal*image1)
        else:
            pass

        # here you take a 50 x 50 pixels (1 μm x 1 μm) from the 
        # 100 x 100 pixels (2 μm x 2 μm) simulation, this was done to remove
        # boundary effects
        subimage = self.image2[25:75, 25:75]
        self.subimage = subimage.astype(np.float32)



if __name__ == '__main__':
    
    
  L =100
  for i in np.arange(10):
      im = sim_image(L)
      tiff.imsave('L{}i{}.tiff'.format(L,i), im.subimage)





