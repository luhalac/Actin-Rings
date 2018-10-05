
"""
Created on Tue Jul 12 10:51:00 2016

@author: Luciano Masullo
@pep8: Lucía Lopez
"""

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from neurosimulations import testAxon, testMask
import tifffile as tiff
from scipy.stats import norm
from scipy import signal
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
import os



        

class axon:
    
    def __init__(self, imSize=100, wvlen=190, theta=15, phase=.25):

        self.imSize = imSize    # image size: n X n
        self.wvlen = wvlen      # wavelength (number of pixels per cycle)
        self.theta = theta      # grating orientation
        self.phase = phase      # phase (0 -> 1)
        a=0 
        b=6
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
        # sin2D.sin2d squared in order to always get positive values
        self.grating2 = sin2D(self.imSize, self.wvlen, 90 - self.theta,
                                  self.phase).sin2d**b
                              

        
class testAxon():

    def __init__(self, size, c=0.8, g_noise=0.2, angle=30, axon_start=.2,
                 axon_end=.8, ring_start=.2, ring_end=.8, *args, **kwargs):

        axon = simAxon(imSize=size, b=6, wvlen=190).grating2

        axon_start = axon_start * size
        axon_end = axon_end * size
        ring_start = ring_start * size
        ring_end = ring_end * size
#        ring_start, ring_end = [10, 40]
#
#        size = 50
#        c = 0.8

#        g_noise = 0.0
#        prob = 0.1 # for salt and pepper noise
#        angle = 30

        # contrast is calculated as in
        # https://en.wikipedia.org/wiki/Contrast_(vision)#Michelson_contrast

        Iringbkg = (1-c)*np.max(axon)/(2*c)  # Iringbkg depends on the contrast
        Iaxonbkg = Iringbkg  # bkg in ringed zone set as the same of non-ringed

        ring_mask = np.zeros(size*size)
        ring_mask[np.int(ring_start*size):np.int(ring_end*size)] = 1
        ring_mask = ring_mask.reshape(size, size)

        axon_mask = np.zeros(size*size)
        axon_mask[np.int(axon_start*size):np.int((ring_start)*size)] = 1
        axon_mask[np.int((ring_end)*size):np.int(axon_end*size)] = 1
        axon_mask = axon_mask.reshape(size, size)

        ring_bkg = np.ones(size*size)
        ring_bkg = Iringbkg*ring_bkg.reshape(size, size)

        norm_f = np.max(axon+ring_bkg)

        axon_res = (ring_mask*(axon + ring_bkg) + Iaxonbkg*axon_mask)/norm_f
        axon_res = axon_res + g_noise*np.random.rand(size, size)
        axon_res = ndi.rotate(axon_res, angle, reshape=False)
#        plt.imshow(ndi.rotate(axon_res, angle, reshape=False),
#                   cmap='hot', clim=(0.0, 2))

        self.data = axon_res.astype(np.float32)


class testMask():
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

#####
class axon_img:
    
    def __init__(self,  *args, **kwargs):
        
        size = 2000
        fwhm = 45
        sigma = 45/2.35
        signal = 100 # max photon count

    def makeAxons(poisson=True, c=0.5, w1=0.2, w2=0.0): 
    
        label = 0.005
        i = 30
    
        n_molecules = 0
        axon = testAxon(size=2000, c=c, g_noise=0, angle=i,
                        axon_start=.4, axon_end=.6, ring_start=.4,
                        ring_end=.6)
    
        image = axon.data
        # tiff.imsave('testAxon{}.tiff'.format(30), image)
    
        result = image.reshape(np.size(image))
    
        for j in np.arange(np.size(result)):
    
            if np.random.rand(1) < label*(w1*result[j]):
                n_molecules = n_molecules+1
            else:
                result[j] = 0
                
            if result[j] < 0.5 and result[j] > 0 and np.random.rand(1) < w2:
                result[j] = 1
            else:
                pass
    
        image = result.reshape(2000, 2000)
        image = ndi.gaussian_filter(image, sigma)
        image_nobkg = image/np.max(image)  # normalization
        j = 0 
        
        bkg_array = np.arange(0, 4, 0.1)
        for bkg in bkg_array:
    
            image1 = image_nobkg + bkg *(1 + 0.1*np.random.rand(size, size))
            image1 = image1/np.max(image1)
    
            image1 = ndi.interpolation.zoom(image1, 0.05)
            image1[image1 < 0] = 0
            if poisson == True:
                image1 = np.random.poisson(signal*image1)
            else:
                pass
    
            subimage = image1[25:75, 25:75]
            subimage = subimage.astype(np.float32)
    
            tiff.imsave('{}testAxon_angle{}_bkg{}_label{}_contrast{}_w1{}_w2{}_poisson{}.tiff'.format(j, i, np.around(bkg, 1), label, c, w1, w2, poisson), subimage)
            j = j + 1
        print(n_molecules)
    
    def makeMasks():
        for i in [30, 30, 30, 30, 30, 30]:
            mask = testMask(size=2000, angle=i, axon_start=.4,
                            axon_end=.6, ring_start=.4, ring_end=.6)
    
            image = mask.data
            image = ndi.interpolation.zoom(image, 0.05)
            subimage = image[25:75, 25:75]
            tiff.imsave('testMask_angle_{}_50x50.tiff'.format(i), subimage)

makeAxons()
