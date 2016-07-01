from numpy import *
import scipy as sp
from scipy import interpolate
import tdpy.mcmc

# define the likelihood function
def retr_llik(sampvarb):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]

    llik = sp.interpolate.interp2d(xgrd, ygrd, pdfn)(xpos, ypos)

    return llik

    
# define the target PDF
xgrd = linspace(0., 2. * pi, 100)
ygrd = linspace(0., 2. * pi, 100)
pdfn = exp(sin(xgrd[:, None]) * sin(ygrd[None, :]))

# construct the parameter object
datapara = tdpy.util.datapara(2)
datapara.defn_para('xpos', 0., 1., 'self', r'$x$', '', None, None)
datapara.defn_para('ypos', 0., 1., 'self', r'$y$', '', None, None)

# run MCMC
sampbund = tdpy.mcmc.init(retr_llik, datapara)

