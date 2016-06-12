from numpy import *
from numpy.random import *
from numpy.random import choice
from scipy.integrate import *
from scipy.interpolate import *
import scipy as sp

import pyfits as pf

# astro & healpix
import healpy as hp

# utilities
import sh, os

def retr_nfwp(nfwg, numbside, norm=None):
    
    edenlocl = 0.3 # [GeV/cm^3]
    radilocl = 8.5 # [kpc]
    rscl = 23.1 # [kpc]
    
    nradi = 100
    minmradi = 1e-2
    maxmradi = 1e2
    radi = logspace(log10(minmradi), log10(maxmradi), nradi)
    
    nsadi = 100
    minmsadi = 0.
    maxmsadi = 2. * radilocl
    sadi = linspace(minmsadi, maxmsadi, nsadi)
    

    lghp, bghp, numbside, numbpixl, apix = retr_healgrid(numbside)
    
    cosigahp = cos(deg2rad(lghp)) * cos(deg2rad(bghp))
    gahp = rad2deg(arccos(cosigahp))
    
    eden = 1. / (radi / rscl)**nfwg / (1. + radi / rscl)**(3. - nfwg)
    eden *= edenlocl / interp1d(radi, eden)(radilocl)
    
    edengrid = zeros((nsadi, numbpixl))
    for i in range(nsadi):
        radigrid = sqrt(radilocl**2 + sadi[i]**2 - 2 * radilocl * sadi[i] * cosigahp)
        edengrid[i, :] = interp1d(radi, eden)(radigrid)


    edengridtotl = sum(edengrid**2, axis=0)

    if norm != None:
        jgahp = argsort(gahp)
        edengridtotl /= interp1d(gahp[jgahp], edengridtotl[jgahp])(5.)
        
        
    return edengridtotl


def mexp(numb):
    logn = log10(numb)
    expo = floor(logn)
    mant = 10**(logn - expo)
    
    if numb > 1e2 or numb < 1e-2:
        if mant == 1.:
            strg = r'$10^{%d}$' % expo
        else:
            strg = r'$%.3g \times 10^{%d}$' % (mant, expo)
    else:
        strg = '%.3g' % numb

    return strg


def retr_p4dm_spec(anch, part='el'):
    
    if part == 'el':
        strg = 'AtProduction_positrons'
    if part == 'ph':
        strg = 'AtProduction_gammas'

    name = os.environ["TDPY_DATA_PATH"] + '/p4dm/' + strg + '.dat'
    p4dm = loadtxt(name)
    
    p4dm[:, 0] *= 1e3 # [MeV]
    
    mass = unique(p4dm[:, 0])
    nmass = mass.size
    numbener = p4dm.shape[0] / nmass
    
    mult = zeros((numbener, nmass))
    for k in range(nmass):
        jp4dm = where(abs(p4dm[:, 0] - mass[k]) == 0)[0]

        if anch == 'e':
            mult[:, k] = p4dm[jp4dm, 4]
        if anch == 'mu':
            mult[:, k] = p4dm[jp4dm, 7]
        if anch == 'tau':
            mult[:, k] = p4dm[jp4dm, 10]
        if anch == 'b':
            mult[:, k] = p4dm[jp4dm, 13]
        
    enerscal = 10**p4dm[jp4dm, 1]

    return mult, enerscal, mass


def show_prog(cntr, maxmcntr, thiscntr, nprog=20, indxprocwork=None):

    nextcntr = int(nprog * float(cntr + 1) / maxmcntr) * 100 / nprog
    if nextcntr > thiscntr:
        if indxprocwork != None:
            print 'Process %d is %3d%% completed.' % (indxprocwork, nextcntr)
        else:
            print '%3d%% completed.' % nextcntr
        thiscntr = nextcntr
        
    return thiscntr            


def show_memo():
    
    memo = float(sh.awk(sh.ps('u', '-p', os.getpid()),'{sum=sum+$6}; END {print sum/1024}'))
    
    print '%.3g MB is being used.' % memo
    

def minm(thissamp, func, verbtype=1, varipara=None, maxmswep=None, limtpara=None, tolrfunc=1e-6, optiprop=False, pathbase='./', rtag=''):

    print 'TDMN launched...'
    numbpara = thissamp.size
    
    if maxmswep == None:
        maxmswep = 1000 * numbpara
    if varipara == None:
        varipara = ones(numbpara)

    thisfunc = func(thissamp)
    thisvaripara = varipara

    # proposal scale optimization
    pathvaripara = pathbase + '/varipara_' + rtag + '.fits'
    perdpropeffi = 1 * numbpara
    if optiprop:
        if not os.path.isfile(pathvaripara): 
            if verbtype > 0:
                print 'Optimizing proposal scale...'
            targpropeffi = 0.25
        else:
            if verbtype > 0:
                print 'Retrieving the optimal proposal scale from %s...' % pathvaripara
            thisvaripara = pf.getdata(pathvaripara)
    else:
        if verbtype > 0:
            print 'Skipping proposal scale optimization...'
        optipropdone = True

    boolconv = zeros(numbpara, dtype=bool)
    rollaccp = zeros((perdpropeffi, numbpara))
    global cntrswep
    cntrswep = 0
    while True:
        
        if verbtype == 1 and cntrswep % 1000 == 0:
            print 'Sweep %d' % cntrswep

        if verbtype > 1:
            print
            print '-' * 10
            print 'Sweep %d' % cntrswep
            print 'thissamp: '
            print thissamp
            print 'thisvaripara'
            print thisvaripara
            print
            
        # propose a sample
        indxparamodi = cntrswep % numbpara
        nextsamp = copy(thissamp)
        nextsamp[indxparamodi] = randn() * thisvaripara[indxparamodi] + thissamp[indxparamodi]
        
        if verbtype > 1:
            print 'indxparamodi'
            print indxparamodi
            print 'nextsamp: '
            print nextsamp

        if limtpara != None:
            if nextsamp[indxparamodi] > limtpara[0, indxparamodi] and nextsamp[indxparamodi] < limtpara[1, indxparamodi]:
                boollimt = True
            else:
                boollimt = False
        else:
            boollimt = True

        if boollimt:
            # evaluate the log-likelihood
            nextfunc = func(nextsamp)
            errrfunc = fabs(nextfunc / thisfunc - 1.)

            if verbtype > 1:
                print 'thisfunc: '
                print thisfunc
                print 'nextfunc: '
                print nextfunc
                print 'errrfunc'
                print errrfunc
                print 
            
        # accept
        if boollimt and nextfunc < thisfunc:

            if verbtype > 1:
                print 'Accepted.'

            # update the minimizer state
            thisfunc = nextfunc
            thispara[indxparamodi] = nextsamp[indxparamodi]
            
            rollaccp[0, indxparamodi] = 1.
        
        else:

            if verbtype > 1:
                print 'Rejected.'

            rollaccp[0, indxparamodi] = 0.
        
        rollaccp = roll(rollaccp, 1, axis=0)
         
        if cntrswep % (numbpara * perdpropeffi) == 0 and optiprop:
            
            thispropeffi = mean(rollaccp, axis=0)
            factcorr = 2**(thispropeffi / targpropeffi - 1.)
            thisvaripara *= factcorr
           
            if verbtype > 1:
                print 'Proposal scale optimization step %d' % (cntrswep / (numbpara * perdpropeffi))
                print 'Current proposal efficiency'
                print thispropeffi
                print 'Correction factor'
                print factcorr
                print 'Current variance: '
                print thisvaripara
                print
   
        if boollimt and errrfunc < tolrfunc:
            nextsamp = randn(numbpara) * thisvaripara + thissamp
        
            if verbtype > 1:
                print 'Checking convergence...'
                print 'nextsamp: '
                print nextsamp

            # evaluate the log-likelihood
            nextfunc = func(nextsamp)
            errrfunctotl = fabs(nextfunc / thisfunc - 1.)
            thisbool = errrfunctotl < tolrfunc
        else:
            thisbool = False

        if thisbool or cntrswep > maxmswep:
            minmsamp = thissamp
            minmfunc = thisfunc
            break
        else:
            cntrswep += 1

    if verbtype > 0:
        print 'Parameter vector at the minimum'
        print minmsamp
        print 'Minimum value of the function'
        print minmfunc
        print 'Final error'
        print errrfunc
        print 'Total number of sweeps'
        print cntrswep
        print

    return minmsamp, minmfunc

    
def cart_heal(cart, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, numbside=256):
    
    numbbgcr = cart.shape[0]
    numblgcr = cart.shape[1]
    lghp, bghp, numbside, numbpixl, apix = retr_healgrid(numbside)
    
    indxlgcr = (numblgcr * (lghp - minmlgal) / (maxmlgal - minmlgal)).astype(int)
    indxbgcr = (numbbgcr * (bghp - minmbgal) / (maxmbgal - minmbgal)).astype(int)
    
    indxpixlrofi = where((minmlgal <= lghp) & (lghp <= maxmlgal) & (minmbgal <= bghp) & (bghp <= maxmbgal))[0]
    
    heal = zeros(numbpixl)
    heal[indxpixlrofi] = fliplr(cart)[indxbgcr[indxpixlrofi], indxlgcr[indxpixlrofi]]
    
    return heal


class cntr():
    def incr(self, valu=1):
        temp = self.cntr
        self.cntr += valu
        return temp
    def __init__(self):
        self.cntr = 0


def cart_heal_depr(cart, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, numbside=256):
    
    nbgcr = cart.shape[0]
    nlgcr = cart.shape[1]
    lghp, bghp, numbside, numbpixl, apix = retr_healgrid(numbside)
    heal = zeros(numbpixl)
    jpixl = where((minmlgal < lghp) & (lghp < maxmlgal) & (minmbgal < bghp) & (bghp < maxmbgal))[0]
    jlgcr = (nlgcr * (lghp[jpixl] - minmlgal) / (maxmlgal - minmlgal)).astype(int)
    jbgcr = (nbgcr * (bghp[jpixl] - minmbgal) / (maxmbgal - minmbgal)).astype(int)
    
    heal[jpixl] = fliplr(cart)[jbgcr, jlgcr]
    
    return heal


def retr_healgrid(numbside):
    
    numbpixl = 12 * numbside**2
    apix = 4. * pi / numbpixl # [sr]
    thhp, phhp = hp.pixelfunc.pix2ang(numbside, arange(numbpixl), nest=False) # [rad]
    lghp = ((rad2deg(phhp) - 180.) % 360.) - 180. # [deg]
    bghp = 90. - rad2deg(thhp)

    return lghp, bghp, numbside, numbpixl, apix


def retr_cart(hmap, indxpixlrofi=None, numbsideinpt=None, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, reso=0.1):
    
    if indxpixlrofi == None:
        numbpixlinpt = hmap.size
        numbsideinpt = int(sqrt(numbpixlinpt / 12.))
    else:
        numbpixlinpt = numbsideinpt**2 * 12
    
    deltlgcr = maxmlgal - minmlgal
    numbbinslgcr = int(deltlgcr / reso)
    
    deltbgcr = maxmbgal - minmbgal
    numbbinsbgcr = int(deltbgcr / reso)
    
    lgcr = linspace(minmlgal, maxmlgal, numbbinslgcr)
    ilgcr = arange(numbbinslgcr)
    
    bgcr = linspace(minmbgal, maxmbgal, numbbinsbgcr)
    ibgcr = arange(numbbinsbgcr)
    
    lghp, bghp, numbside, numbpixl, apix = retr_healgrid(numbsideinpt)

    bgcrmesh, lgcrmesh = meshgrid(bgcr, lgcr)
    
    jpixl = hp.ang2pix(numbsideinpt, pi / 2. - deg2rad(bgcrmesh), deg2rad(lgcrmesh))
    
    if indxpixlrofi == None:
        kpixl = jpixl
    else:
        pixlcnvt = zeros(numbpixlinpt, dtype=int)
        for k in range(indxpixlrofi.size):
            pixlcnvt[indxpixlrofi[k]] = k
        kpixl = pixlcnvt[jpixl]

    hmapcart = zeros((numbbinsbgcr, numbbinslgcr))
    hmapcart[meshgrid(ibgcr, ilgcr)] = hmap[kpixl]

    return hmapcart


def retr_fdfm(binsener, numbside=256, vfdm=7):                    
    
    diffener = binsener[1:] - binsener[0:-1]
    numbener = diffener.size

    numbpixl = numbside**2 * 12
    
    path = os.environ["PCAT_DATA_PATH"] + '/'
    if vfdm == 2:
        path += 'gll_iem_v02.fit'
    if vfdm == 3:
        path += 'gll_iem_v02_P6_V11_DIFFUSE.fit'
    if vfdm == 4:
        path += 'gal_2yearp7v6_v0.fits'
    if vfdm == 5:
        path += 'gll_iem_v05.fit'
    if vfdm == 6:
        path += 'gll_iem_v05_rev1.fit'
    if vfdm == 7:
        path += 'gll_iem_v06.fits'
   
    fluxcart = pf.getdata(path, 0) * 1e3 # [1/cm^2/s/sr/GeV]
    enerfdfm = array(pf.getdata(path, 1).tolist()).flatten() * 1e-3 # [GeV]
    fdfmheal = zeros((enerfdfm.size, numbpixl))
    for i in range(enerfdfm.size):
        fdfmheal[i, :] = cart_heal(fliplr(fluxcart[i, :, :]), numbside=numbside)
    
    fdfm = empty((numbener, numbpixl))
    numbsampbins = 10
    enersamp = logspace(log10(amin(binsener)), log10(amax(binsener)), numbsampbins * numbener)
    
    fdfmheal = interpolate.interp1d(enerfdfm, fdfmheal, axis=0)(enersamp)
    for i in range(numbener):
        fdfm[i, :] = trapz(fdfmheal[i*numbsampbins:(i+1)*numbsampbins, :], enersamp[i*numbsampbins:(i+1)*numbsampbins], axis=0) / diffener[i]

    return fdfm


def plot_braz(ax, xdat, ydat, numbsampdraw=0, lcol='yellow', dcol='green', mcol='black', labl=None, alpha=None):

    if numbsampdraw > 0:
        jsampdraw = choice(arange(ydat.shape[0]), size=numbsampdraw)
        axis.plot(xdat, ydat[jsampdraw[0], :], alpha=0.1, color='b', label='Samples')
        for k in range(1, numbsampdraw):
            axis.plot(xdat, ydat[jsampdraw[k], :], alpha=0.1, color='b')
    axis.plot(xdat, percentile(ydat, 2.5, 0), color=lcol, alpha=alpha)
    axis.plot(xdat, percentile(ydat, 16., 0), color=dcol, alpha=alpha)
    axis.plot(xdat, percentile(ydat, 84., 0), color=dcol, alpha=alpha)
    axis.plot(xdat, percentile(ydat, 97.5, 0), color=lcol, alpha=alpha)
    axis.plot(xdat, percentile(ydat, 50., 0), color=mcol, label=labl, alpha=alpha)
    axis.fill_between(xdat, percentile(ydat, 2.5, 0), percentile(ydat, 97.5, 0), color=lcol, alpha=alpha)#, label='95% C.L.')
    axis.fill_between(xdat, percentile(ydat, 16., 0), percentile(ydat, 84., 0), color=dcol, alpha=alpha)#, label='68% C.L.')
