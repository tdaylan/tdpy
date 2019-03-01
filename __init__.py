# numerics
import numpy as np
import scipy as sp
from scipy.special import erfi

# plotting
import matplotlib as mpl
mpl.rc('image', interpolation='none', origin='lower')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (6., 5.)
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)

# pixelization
import healpy as hp
from healpy.rotator import angdist

# utilities
import psutil, sys, sh, os, functools, time, datetime, cPickle, fnmatch

# multiprocessing
import multiprocessing as mp

# warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# astropy
import astropy.coordinates, astropy.units
import astropy.io
