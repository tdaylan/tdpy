import numpy as np
import tdpy
import os

import matplotlib
matplotlib.use('agg')
#import matplotlib.pyplot as plt

numbsamp = 100

numbfeat = 5
numbdata = numbfeat * numbsamp

listpara = [[], []]
# class 1
listpara[0] = np.random.randn(numbdata).reshape((numbsamp, numbfeat))

# class 2
listpara[1] = np.random.randn(numbdata).reshape((numbsamp, numbfeat)) + np.random.randn(numbfeat)[None, :]

pathbase = os.environ['TDPY_DATA_PATH'] + '/imag/grid_plot/'
os.system('mkdir -p %s' % pathbase)

listlablpopl = ['Positive', 'Negative']

listlablpara = []
for k in range(numbfeat):
    listlablpara.append(['$p_{%d}$' % k, ''])

strgplot = ''
tdpy.plot_grid(pathbase, strgplot, listpara, listlablpara, listlablpopl=listlablpopl)
              

