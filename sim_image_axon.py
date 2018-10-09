# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:41:17 2018

@author: CibionPC
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:51:00 2016

@author: Luciano Masullo

"""

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.stats import norm

class actin_ring:
    
    def __init__(self, L=50, N=1, D=400, x=np.linspace(0,1000,1000), center=400, *args, **kwargs):
        
        self.π = np.pi
        self.L = L # actin filament lenght in nm
        self.N = N #total number of rotations
        self.rxy = 10  # radial resolution in nm
        self.D = D - 3*self.rxy # axon diameter in nm
        self.x = x # distance in nm
        self.Npoints = len(self.x)
        self.center = center     
        self.S = 1  # % of staining

        self.K = np.int(self.π/(np.arcsin(L/self.D)))


       
        self.φ = np.linspace(0, (self.π)/self.K, self.N)
        self.θ = np.ones((self.N,self.K))
        self.posx = np.ones((self.N,self.K))
        self.posxround = np.ones((self.N,self.K))
        for i in np.arange(self.N):
            for k in np.arange(self.K):
                self.θ[i,k] = self.π * 2*k/self.K + self.φ[i]
                self.posx[i,k]= (self.D/2)*np.cos(self.θ[i,k]) + self.center
                self.posxround[i,k]= np.round(self.posx[i,k], decimals=0)
        
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
                               

class simAxon:

    """
    This class generates a 2D sinusoidal pattern, stored in sin2D().sin2d
    This can be replaced by any other function/class that generates a sinusoidal 
    2D pattern.
    """
    
    
    def __init__(self, imSize=1000, wvlen=2*190, theta=0, phase=.25, b = 6):

        self.imSize = imSize    # image size: n X n
        self.wvlen = wvlen      # wavelength (number of pixels per cycle)
        self.theta = theta      # grating orientation
        self.phase = phase      # phase (0 -> 1)
        self.b = b
        self.pi = np.pi
        g_noise=0.2
        c = 0.8
        axon_start=.2
        axon_end=.6 
        ring_start=.2 
        ring_end=.6
        # X is a vector from 1 to imageSize
        self.X = np.arange(1, self.imSize + 1)
        # rescale X -> -.5 to .5
        self.X0 = (self.X / self.imSize) - .5

        
        # compute frequency from wavelength
        self.freq = self.imSize/self.wvlen
        # convert to radians: 0 -> 2*pi
        self.phaseRad = (self.phase * 2*self.pi)
        
        [self.Xm, self.Ym] = np.meshgrid(self.X0, self.X0)     # 2D matrices

        # convert theta (orientation) to radians
        self.thetaRad = (self.theta / 360) * 2*self.pi
        # compute proportion of Xm for given orientation
        self.Xt = self.Xm * np.cos(self.thetaRad)
        # compute proportion of Ym for given orientation
        self.Yt = self.Ym * np.sin(self.thetaRad)
        # sum X and Y components
        self.XYt = np.array(self.Xt + self.Yt)
        # convert to radians and scale by frequency
        self.XYf = self.XYt * self.freq * 2*self.pi

        # make 2D sinewave
        self.sin2d = np.sin(self.XYf + self.phaseRad)
        self.ringpattern = self.sin2d**self.b

        # creates the start and end point of the axon contour and the rings in
        # pixels (they input is in proportion from 0 to 1)
        # axon and ring start/end can be made different but usually it works to
        # keep them the same. They were used to test a "badly chosen" neuronal
        # discrimination

        axon_start = axon_start * self.imSize
        axon_end = axon_end * self.imSize
        ring_start = axon_start
        ring_end = axon_end
        
        Iringbkg = (1-c)*np.max(self.ringpattern)/(2*c)  # Iringbkg depends on the contrast
        Iaxonbkg = Iringbkg  # bkg in ringed zone set as the same of non-ringed
        
        # create ring profile mask
        ring_mask = np.zeros(self.imSize*self.imSize)
        psim = actin_ring().psim[0,:]
        for i in np.arange(self.imSize):
            ring_mask[i*self.imSize:(i+1)*self.imSize] = psim.astype(int)
            
        self.ring_mask = ring_mask.reshape(self.imSize, self.imSize)
        self.ring_mask = np.transpose(self.ring_mask)
        
        # creates the axon_mask and the ring_mask

        axon_mask = np.zeros(self.imSize*self.imSize)
#        ring_mask = np.zeros(self.imSize*self.imSize)
        
        axon_mask[np.int(axon_start*self.imSize):np.int((ring_start)*self.imSize)] = 1
        axon_mask[np.int((ring_end)*self.imSize):np.int(axon_end*self.imSize)] = 1
        self.axon_mask = axon_mask.reshape(self.imSize, self.imSize)
        
#        ring_mask[np.int(ring_start*self.imSize):np.int(ring_end*self.imSize)] = 1
#        ring_mask = ring_mask.reshape(self.imSize, self.imSize)

        ring_bkg = np.ones(self.imSize*self.imSize)
        ring_bkg = Iringbkg*ring_bkg.reshape(self.imSize, self.imSize)
        
        # normalizes and creates the final simulated axon with spectrin rings

        norm_f = np.max(self.ringpattern+ring_bkg)

        axon_res = (self.ring_mask*(self.ringpattern + ring_bkg) + Iaxonbkg*self.axon_mask)/norm_f
        axon_res = axon_res + g_noise*np.random.rand(self.imSize, self.imSize)
#        axon_res = ndi.rotate(axon_res, angle, reshape=False)


        self.data = axon_res.astype(np.float32) # result to be used in simulations




class imageAxon():

    '''
    Function to generate many different simulated (pieces of) axons with different
    SBR, contrast, labeling efficency
    
    poisson: wether you add poisson noise or not, for default is true
    c: contrast of the image (before adding any noise)
    
    contrast is calculated as in
    https://en.wikipedia.org/wiki/Contrast_(vision)#Michelson_contrast
    
    w1, w2: weights used to simulate the labeling in the axon, w2 accounts for
            inespecific labeling on parts which are not the rings
            
    '''
    def __init__(self, size=1000, fwhm= 45, sigma= 45/2.35, signal= 100,
                 poisson=True, c=0.5, w1=0.2, w2=0.0, *args, **kwargs):
            
            label = 0.1 # labeling efficency. It is very low because of how we define
                          # the spectrin rings in a continous way
            
            i = 90        # angle is fixed at 30 deg since it was rather irrelevant for
                          # this simulations, although I tried to avoid 0 or 90 since
                          # they could be regarded as a particular case
        
            n_molecules = 1000  # this variable will keep track of how many "fluorophores
                             # you end up putting in your simulated image, it should
                             # be a reasonable number according to inmunostaining
                             # stochiometry
            
            axon = simAxon()
        
            image = axon.data # actual simulated image
#            image[image>0.995] = 1
#            image[image<0.995] = 0
            
            
            tiff.imsave('testAxon{}.tiff'.format(0), image)
        
            result = image.reshape(np.size(image)) # image is reshaped into a 1D array
                                                   # in order to be able to loop on it
            
            # NOTE: result has values between 0 and 1
        
            # the following is a loop that "labels" the axon according to the labeling
            # density and the weight parameters. 
        
            for j in np.arange(np.size(result)):
                
                # if this condition is accomplished (it is random) then the simulated 
                # pixel of the image is kept as it was (non zero)
                if np.random.rand(1) < label*(w1*result[j]):
                    n_molecules = n_molecules+1
                else:
                    result[j] = 0
                    
                # this part accounts for unspecific labeling if w2=0 this part is ignored
                if result[j] < 0.5 and result[j] > 0 and np.random.rand(1) < w2:
                    result[j] = 1
                else:
                    pass
                
            # image is reshaped to a 2D array
            # TO DO: changes this numbers (2000, 2000) to a variable, as it is now, you
            # get a "digital" simulated resolution of 1 nm        
        
            image = result.reshape(1000, 1000)   
            
            # convolute the 1 nm molecules with the resolution of the microscope    
            image = ndi.gaussian_filter(image, sigma)
            image_nobkg = image/np.max(image)  # normalization
            j = 0 
            
            # different backgrounds are added to the image. This loop was done to test
            # the different SBR for Gollum, but this can be changed, for example you can
            # use a single bkg value    
            
            # bkg is simulated as a gaussian noise around a certain value
            
            bkg_array = np.arange(0, 4, 0.1)
            for bkg in [1]:
                size = 1000
                # addition of bkg + normalization
                image1 = image_nobkg + bkg *(1 + 0.1*np.random.rand(size, size))
                image1 = image1/np.max(image1)
        
                # resample of the image to get 20 nm pixel
                # TO DO: change 0.05 to a variable
        
                image1 = ndi.interpolation.zoom(image1, 0.05)
                
                # get rid of possible negative values produced by the sigma of the bkg
                # they are usually just a few pixels, no big deal
                image1[image1 < 0] = 0
                
                # adds poisson noise, simulating an APD/CCD detection
                if poisson == True:
                    image1 = np.random.poisson(signal*image1)
                else:
                    pass
        
                # here you take a 50 x 50 pixels (1 μm x 1 μm) from the 
                # 100 x 100 pixels (2 μm x 2 μm) simulation, this was done to remove
                # boundary effects
                subimage = image1[25:75, 25:75]
                subimage = subimage.astype(np.float32)
                
                # saves the image with data in the title
                # TO DO: make a metadata file instead of saving in the title
        
                tiff.imsave('{}testAxon_angle{}_bkg{}_label{}_contrast{}_w1{}_w2{}_poisson{}.tiff'.format(j, i, np.around(bkg, 1), label, c, w1, w2, poisson), image1)
                j = j + 1
                
            # prints the number of molecules to check if it is a reasnoable number
            print(n_molecules)







