from numpy import *
import scipy as sp
from scipy import ndimage
from scipy import interpolate
import matplotlib.pyplot as plt
import tdpy.mcmc
import os

def retr_llik(sampvarb):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]

    llik = sp.interpolate.interp2d(xgrd, ygrd, pdfn)(xpos, ypos)

    return llik, []

def retr_datapara():
    
    numbpara = 2

    dictpara = dict()
    minmpara = zeros(numbpara)
    maxmpara = zeros(numbpara)
    namepara = empty(numbpara, dtype=object)
    scalpara = empty(numbpara, dtype=object)
    lablpara = empty(numbpara, dtype=object)
    unitpara = empty(numbpara, dtype=object)
    varipara = zeros(numbpara)
    
    dictpara['xpos'] = 0
    dictpara['ypos'] = 0
    namepara[0] = 'xpos'
    namepara[1] = 'ypos'
    minmpara[:] = 0.
    maxmpara[:] = 1.
    scalpara[:] = 'self'
    lablpara[0] = r'$x$'
    lablpara[1] = r'$y$'
    unitpara[:] = ''
    varipara[:] = 1e-1

    strgpara = lablpara + ' ' + unitpara

    datapara = namepara, strgpara, minmpara, maxmpara, scalpara, lablpara, unitpara, varipara, dictpara

    return datapara               
                                
# target PDF
imag = sp.ndimage.imread('turkflag.png')
rati = float(imag.shape[0]) / imag.shape[1]
xinp = linspace(0., 1., imag.shape[1])
yinp = linspace(0., 1., imag.shape[0])
numbxgrd = 200
numbygrd = int(200 * rati)
xgrd = linspace(0., 1., numbxgrd)
ygrd = linspace(0., 1., numbygrd)
imag = sp.ndimage.gaussian_filter(imag, sigma=[xinp.size / 100, yinp.size / 100, 0])
pdfn = zeros((numbygrd, numbxgrd, 3))
for k in range(3):
    pdfn[:, :, k] = sp.interpolate.interp2d(xinp, yinp, imag[:, :, k])(xgrd, ygrd)
pdfn = 0.3 * pdfn[:, :, 0]

figr, axis = plt.subplots()
axis.imshow(imag, extent=[0., 1., 0., 1.], interpolation='none', aspect=rati)
plt.savefig('imag.png')
plt.close(figr) 

figr, axis = plt.subplots()
imag = axis.imshow(pdfn, extent=[0., 1., 0., 1.], interpolation='none', aspect=rati)
plt.colorbar(imag)
plt.savefig('targ.png')
plt.close(figr) 

# MCMC setup
verbtype = 1
numbproc = 1
numbswep = 1000000
datapara = retr_datapara()
optiprop = True
pathbase = '/Users/tansu/Desktop/'

# run MCMC
sampbund = tdpy.mcmc.init(numbproc, numbswep, retr_llik, datapara, pathbase=pathbase, optiprop=True, verbtype=verbtype, rtag='flag')
listxpos = sampbund[0][:, 0]
listypos = sampbund[0][:, 1]
numbsamp = listxpos.size
pathtemp = pathbase + 'post'
numbfram = 10
for k in range(numbfram):
    indxsamp = int(numbsamp * float(k) / numbfram)
    print indxsamp
    figr, axis = plt.subplots()
    axis.scatter(listxpos[:indxsamp], listypos[:indxsamp], s=3)
    axis.set_xlim([0., 1.])
    axis.set_ylim([0., 1.])
    path = pathtemp + '%04d.png' % k
    plt.savefig(path)
    plt.close(figr) 

cmnd = 'convert -delay 20 %s*.png post.gif' % pathtemp
os.system(cmnd)
