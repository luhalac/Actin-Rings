# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:51:00 2016

@author: Luciano Masullo

"""

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from scipy.stats import norm
       

class sin2D:

    """
    This class generates a 2D sinusoidal pattern, stored in sin2D().sin2d
    This can be replaced by any other function/class that generates a sinusoidal 
    2D pattern.
    """
    
    
    def __init__(self, imSize=1000, wvlen=190, theta=0, phase=.25):

        self.imSize = imSize    # image size: n X n
        self.wvlen = wvlen      # wavelength (number of pixels per cycle)
        self.theta = theta      # grating orientation
        self.phase = phase      # phase (0 -> 1)

        self.pi = np.pi

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

        # display
#        plt.figure()
#        plt.imshow(self.grating, cmap='gray')


class simAxon(sin2D):
    
    """
    This class basically only squares a sin2D object, so it could be totally 
    merged with sin2D or an equivalent.
    """
    

    def __init__(self, imSize=50, wvlen=10, theta=90, phase=.25, a=0, b=2,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.imSize = imSize       # image size: n X n
        self.wvlen = wvlen         # wavelength (number of pixels per cycle)
        self.theta = theta         # grating orientation
        self.phase = phase         # phase (0 -> 1)


        # sin2D.sin2d squared in order to always get positive values
        self.grating2 = sin2D(self.imSize, self.wvlen, 90 - self.theta,
                              self.phase).sin2d**b

        # Make simulated axon data
        self.data = self.grating2


class testAxon():
    
    """
    This class is very useful since it deals with the shape of the contour 
    of the axon. It takes a simAxon as an input basically and returns an ideal
    simulated axon with its countour, ready to be used in the testAxon_maker 
    script
    """

    def __init__(self, size, c=0.8, g_noise=0.2, angle=30, axon_start=.2,
                 axon_end=.8, ring_start=.2, ring_end=.8, *args, **kwargs):
        
        # generates the simAxon base data (sinusoidal pattern that fills the whole image)
        axon = simAxon(imSize=size, b=6, wvlen=190).data

        # creates the start and end point of the axon contour and the rings in
        # pixels (they input is in proportion from 0 to 1)
        # axon and ring start/end can be made different but usually it works to
        # keep them the same. They were used to test a "badly chosen" neuronal
        # discrimination

        axon_start = axon_start * size
        axon_end = axon_end * size
        ring_start = ring_start * size
        ring_end = ring_end * size

        # contrast is calculated as in
        # https://en.wikipedia.org/wiki/Contrast_(vision)#Michelson_contrast

        # this part generates the contrast, this bkg is not to be confused with
        # bkg that will be added in the other simulation, this one will account
        # for image contrast the other one for a kind of "autofluorescence"
        # bkg of some kind

        Iringbkg = (1-c)*np.max(axon)/(2*c)  # Iringbkg depends on the contrast
        Iaxonbkg = Iringbkg  # bkg in ringed zone set as the same of non-ringed
        
        # creates the ring_mask

        ring_mask = np.zeros(size*size)
        ring_mask[np.int(ring_start*size):np.int(ring_end*size)] = 1
        ring_mask = ring_mask.reshape(size, size)

        # creates the axon_mask and the ring_mask

        axon_mask = np.zeros(size*size)
        axon_mask[np.int(axon_start*size):np.int((ring_start)*size)] = 1
        axon_mask[np.int((ring_end)*size):np.int(axon_end*size)] = 1
        axon_mask = axon_mask.reshape(size, size)

        ring_bkg = np.ones(size*size)
        ring_bkg = Iringbkg*ring_bkg.reshape(size, size)
        
        # normalizes and creates the final simulated axon with spectrin rings

        norm_f = np.max(axon+ring_bkg)

        axon_res = (ring_mask*(axon + ring_bkg) + Iaxonbkg*axon_mask)/norm_f
        axon_res = axon_res + g_noise*np.random.rand(size, size)
        axon_res = ndi.rotate(axon_res, angle, reshape=False)
#        plt.imshow(ndi.rotate(axon_res, angle, reshape=False),
#                   cmap='hot', clim=(0.0, 2))

        self.data = axon_res.astype(np.float32) # result to be used in simulations


class testMask():
    
    """
    This is an auxiliary class that creates masks without axon/rings
    """    
    
    def __init__(self, size, angle=30, axon_start=.2,
                 axon_end=.8, ring_start=.2, ring_end=.8, *args, **kwargs):

        axon_start = axon_start * size
        axon_end = axon_end * size
        ring_start = ring_start * size
        ring_end = ring_end * size

        ring_mask = np.zeros(size*size)
        ring_mask[np.int(ring_start*size):np.int(ring_end*size)] = 1
        ring_mask = ring_mask.reshape(size, size)

        ring_mask = ndi.rotate(ring_mask, angle, reshape=False)

        self.data = ring_mask.astype(np.float32)



