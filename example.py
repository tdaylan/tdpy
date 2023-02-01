import numpy as np
import tdpy
import os

import matplotlib
matplotlib.use('agg')

# number of samples
numbsamp = [1000, 1000]

# number of features
numbfeat = 5

numbpopl = 2

listpara = [[] for k in range(numbpopl)]
for k in range(numbpopl):

    # number of points
    numbdata = numbfeat * numbsamp[k]

    listpara[k] = np.random.randn(numbdata).reshape((numbsamp[k], numbfeat)) + 5. * np.random.randn(numbfeat)[None, :]

pathbase = os.environ['TDPY_DATA_PATH'] + '/imag/grid_plot/'
os.system('mkdir -p %s' % pathbase)

listlablpopl = ['Positive', 'Negative']

listlablpara = []
for k in range(numbfeat):
    listlablpara.append(['$p_{%d}$' % k, ''])

strgplot = 'PositivesNegatives'
tdpy.plot_grid(pathbase, strgplot, listpara, listlablpara, listlablpopl=listlablpopl)
              

