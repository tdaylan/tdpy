# numerics
from numpy import *
from numpy.random import *
from numpy.random import choice
from scipy.integrate import *
from scipy.interpolate import *
import scipy as sp

# plotting
import matplotlib as mpl
mpl.rc('image', interpolation='none', origin='lower')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (6., 5.)
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

# FITS
import pyfits as pf

# pixelization
import healpy as hp

# utilities
import sh, os, functools, time

# multiprocessing
import multiprocessing as mp

# warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)



