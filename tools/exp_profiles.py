# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:01:45 2018

@author: CibionPC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
from scipy import signal

data_dir = 'C:\Users\CibionPC\Documents\Lu Lopez\Actin Spectrin Rings\Exp Data\Gollum\STED\'
folder = '17.01.17 DIV 8\'
subfolder = 'DIV 8 STED' 

filename = data_dir + folder + subfolder