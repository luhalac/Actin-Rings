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



class actin_ring:
    
    def __init__(self, *args, **kwargs):
        
        π = np.pi
        L = 150 # actin filament lenght in nm
        D = 400  # axon diameter in nm
        #        rz = 500  # axial resolution in nm
        #rxy =   # radial resolution in nm
        N = 10 # trials
        K = np.int(π/(np.arcsin(L/D)))

    def polyvert(self):

        self.θ = {}
        self.pos = {}
        if (K+1) % 2 == 0:  # if K is even
            for k in np.arange(K):
                self.θ[k] = π * 2*k/K
                self.pos[k]= (D/2)*np.array([np.cos(self.θ[k]), np.sin(self.θ[k])])
        else:       # if K is odd
            for k in np.arange(K):
                self.θ[k] = π * (2*k+1)/K
                self.pos[k]= (D/2)*np.array([np.cos(self.θ[k]), np.sin(self.θ[k])])
    
    def polyrot(self):        
            
        φ = np.random.uniform(0,self.θ[1])
        print(φ)
        θrot = {}
        self.posx = {}
        if (K+1) % 2 == 0:  # if K is even
            for k in np.arange(K):
                θrot[k] = self.θ[k] + φ
                self.posx[k]= (D/2)*np.array([np.cos(θrot[k]), 0])
        else:       # if K is odd
            for k in np.arange(K):
                θrot[k] = self.θ[k] + φ
                self.posx[k]= (D/2)*np.array([np.cos(θrot[k]), 0])
                    
    def intensity(self):
        
        print('falta')
        
        
        
        
  
    def polyplots(self):
        
        markercolor1 = 'go'
        markercolor2 = 'ro'
        
        fig7, ax7 = plt.subplots(1, 1)
        for i in np.arange(K):
#            ax7.plot(self.pos[i][0], self.pos[i][1] , markercolor1, markersize=20)
#            ax7.plot(self.posrot[i][0], self.posrot[i][1] , markercolor2, markersize=20)
            ax7.plot(self.posx[i][0], self.posx[i][1], markercolor1, markersize=20, fillstyle='none')
#            ax7.plot(self.posrotx[i][0], self.posrotx[i][1], markercolor2, markersize=20, fillstyle='none')
            ax7.add_patch(patches.RegularPolygon([0, 0], K, D/2, fill=False, orientation=self.θ[0], linestyle='solid'))
        ax7.axis('equal')
        
#    def intensityprofile(self):
        
        
        
if __name__ == '__main__':

    ring = actin_ring()
    ring.polyvert()
    ring.polyrot()
    ring.polyplots()