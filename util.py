import matplotlib.pyplot as plt
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

class gdatstrt(object):
    
    def __init__(self):
        pass


class datapara(object):

    def __init__(self, numbpara)
        
        self.numbpara = numbpara
        self.minm = zeros(numbpara)
        self.maxm = zeros(numbpara)
        self.true = zeros(numbpara)
        self.name = empty(numbpara, dtype=object)
        self.scal = empty(numbpara, dtype=object)
        self.labl = empty(numbpara, dtype=object)
        self.unit = empty(numbpara, dtype=object)
        self.vari = zeros(numbpara)
        self.cntr = 0
    
    def defn_para(name, minm, maxm, scal, labl, unit, vari, true)
        
        datapara.indx[name] = 0
        datapara.name[cntr] = name
        datapara.minm[cntr] = minm
        datapara.maxm[cntr] = maxm
        datapara.scal[cntr] = scal
        datapara.labl[cntr] = labl
        datapara.unit[cntr] = unit
        datapara.vari[cntr] = vari
        datapara.true[cntr] = true
        datapara.strg[cntr] = datapara.labl + ' ' + datapara.unit
        cntr += 1


def retr_postvarb(listvarb):

    shap = zeros(len(listvarb.shape), dtype=int)
    shap[0] = 3
    shap[1:] = listvarb.shape[1:]
    shap = list(shap)
    postvarb = zeros(shap)
    
    postvarb[0, :] = percentile(listvarb, 50., axis=0)
    postvarb[1, :] = percentile(listvarb, 16., axis=0)
    postvarb[2, :] = percentile(listvarb, 84., axis=0)

    return postvarb


def retr_errrvarb(postvarb):

    errr = fabs(postvarb[0, :] - postvarb[1:3, :])

    return errr


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
    

    lghp, bghp, numbpixl, apix = retr_healgrid(numbside)
    
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


def show(*listargs):

    print 'hey'
    for args in listargs:
        print args
    print 


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
    

def minm(thissamp, func, verbtype=1, stdvpara=None, factcorrscal=10., maxmswep=None, limtpara=None, tolrfunc=1e-6, optiprop=True, pathbase='./', rtag=''):

    print 'TDMN launched...'
    numbpara = thissamp.size
    indxpara = arange(numbpara)

    if maxmswep == None:
        maxmswep = 1000 * numbpara
    if stdvpara == None:
        stdvpara = ones(numbpara)

    # matrices needed for computing the Hessian
    matrhess = empty((numbpara, numbpara))
    matriden = zeros((numbpara, numbpara))
    matriden[indxpara, indxpara] = 1.

    thisfunc = func(thissamp)
    thisstdvpara = stdvpara

    thiserrr = 1e10
    cntrswep = 0
    while True:
        
        #if verbtype == 1 and cntrswep % 1000 == 0:
        #    print 'Sweep %d' % cntrswep

        if verbtype > 1:
            print
            print '-' * 10
            print 'Sweep %d' % cntrswep
            print 'Current sample'
            print thissamp
            print 'Current variance'
            print thisstdvpara
            print
            
        # propose a sample
        indxparamodi = cntrswep % numbpara
        nextsamp = copy(thissamp)
        nextsamp[indxparamodi] = randn() * thisstdvpara[indxparamodi] + thissamp[indxparamodi]
        
        if verbtype > 1:
            print 'Index of the parameter to be modified'
            print indxparamodi
            print 'Next sample'
            print nextsamp
            print

        # check if the new sample is within the allowed boundaries
        if limtpara != None:
            if nextsamp[indxparamodi] > limtpara[0, indxparamodi] and nextsamp[indxparamodi] < limtpara[1, indxparamodi]:
                boollimt = True
            else:
                boollimt = False
        else:
            boollimt = True

        # evaluate the function
        if boollimt:
            nextfunc = func(nextsamp)

            if verbtype > 1:
                print 'thisfunc: '
                print thisfunc
                print 'nextfunc: '
                print nextfunc
                print 
            
        # check if the new sample is better
        if boollimt and nextfunc < thisfunc:
            if verbtype > 1:
                print 'Accepted.'
            boolaccp = True
            # update the minimizer state
            thisfunc = nextfunc
            thissamp[indxparamodi] = nextsamp[indxparamodi]
        else:
            if verbtype > 1:
                print 'Rejected.'
            boolaccp = False
  
        # update the proposal scale
        if optiprop:
            if boolaccp:
                factcorr = factcorrscal
            else:
                factcorr = 1. / factcorrscal
            thisstdvpara[indxparamodi] *= factcorr
   
        # check convergence
        ## compute the Hessian
        #diffpara = matriden * thissamp[:, None] + thisstdvpara[:, None]
        #for k in indxpara:
        #    funcplus = funcdiff(thissamp + diffpara[:, k]) / 2.
        #    funcmins = funcdiff(thissamp - diffpara[:, k]) / 2.
        #    matrinfo[:, k] = (funcplus - funcmins) / 2. / diffpara[k, k]
        #    matrvari = invert(matrinfo)
        #    detrvari = determinant(matrvari)
        
        nextsampconv = randn(numbpara) * thisstdvpara + thissamp
        nextfuncconv = func(nextsampconv)
        nexterrr = fabs(nextfuncconv / thisfunc - 1.)
        if nexterrr < thiserrr:
            thiserrr = nexterrr
        nextbool = nexterrr < tolrfunc
        
        if verbtype > 1:
            print 'Checking convergence...'
            print 'nextsampconv'
            print nextsampconv
            print 'nextfuncconv'
            print nextfuncconv
            print 'nexterrr'
            print nexterrr

        if nextbool or cntrswep == maxmswep:
            minmsamp = thissamp
            minmfunc = thisfunc
            break
        else:
            cntrswep += 1

    if verbtype > 0:
        print 'Parameter vector at the minimum'
        print minmsamp
        print 'Parameter proposal scale at the minimum'
        print stdvpara
        print 'Minimum value of the function'
        print minmfunc
        print 'Current error'
        print thiserrr
        print 'Total number of sweeps'
        print cntrswep
        print

    return minmsamp, minmfunc


def test_minm():

    def func_test(samp):
        return sum((samp / 0.2 - 1.)**2)
    numbpara = 10
    stdvpara = ones(numbpara)
    thissamp = rand(numbpara)
    minm(thissamp, func_test, verbtype=1, factcorrscal=100., stdvpara=stdvpara, maxmswep=None, limtpara=None, tolrfunc=1e-6, pathbase='./', rtag='')

#test_minm()

def cart_heal(cart, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, numbside=256):
    
    numbbgcr = cart.shape[0]
    numblgcr = cart.shape[1]
    lghp, bghp, numbpixl, apix = retr_healgrid(numbside)
    
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


def plot_heal(path, heal, indxpixlrofi=None, numbpixl=None, titl='', minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., resi=False, satu=False):
    
    if indxpixlrofi != None:
        healtemp = zeros(numbpixl)
        healtemp[indxpixlrofi] = heal
        heal = healtemp

    # saturate the map
    if satu:
        healtemp = copy(heal)
        heal = healtemp
        if not resi:
            satu = 0.1 * amax(heal)
        else:
            satu = 0.1 * min(fabs(amin(heal)), amax(heal))
            heal[where(heal < -satu)] = -satu
        heal[where(heal > satu)] = satu

    exttrofi = [minmlgal, maxmlgal, minmbgal, maxmbgal]

    cart = retr_cart(heal, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)
    
    figr, axis = plt.subplots(figsize=(8, 8))
    if resi:
        cmap = 'RdBu'
    else:
        cmap = 'Reds'
    imag = plt.imshow(cart, origin='lower', cmap=cmap, extent=exttrofi)
    plt.colorbar(imag, fraction=0.05)
    plt.title(titl)

    plt.savefig(path)
    plt.close(figr)
    

def cart_heal_depr(cart, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, numbside=256):
    
    nbgcr = cart.shape[0]
    nlgcr = cart.shape[1]
    lghp, bghp, numbpixl, apix = retr_healgrid(numbside)
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

    return lghp, bghp, numbpixl, apix


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
    
    lghp, bghp, numbpixl, apix = retr_healgrid(numbsideinpt)

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


def retr_fermpsfn(enerthis, thisangl):
   
    parastrg = ['ntail', 'score', 'gcore', 'stail', 'gtail']
    
    path = os.environ["PCAT_DATA_PATH"] + '/irfn/psf_P8R2_SOURCE_V6_PSF.fits'
    irfn = pf.getdata(path, 1)
    minmenerirfn = irfn['energ_lo'].squeeze() * 1e-3 # [GeV]
    maxmenerirfn = irfn['energ_hi'].squeeze() * 1e-3 # [GeV]
    meanenerirfn = sqrt(minmenerirfn * maxmenerirfn)

    numbenerthis = enerthis.size
    indxenerthis = arange(numbenerthis)
    numbenerirfn = meanenerirfn.size
    numbevtt = 4
    indxevtt = arange(numbevtt)
    numbfermscalpara = 3
    numbfermformpara = 5
    
    fermscal = zeros((numbevtt, numbfermscalpara))
    fermform = zeros((numbenerthis, numbevtt, numbfermformpara))
    
    for m in indxevtt:
        fermscal[m, :] = pf.getdata(path, 2 + 3 * indxevtt[m])['PSFSCALE']
        irfn = pf.getdata(path, 1 + 3 * indxevtt[m])
        for k in range(numbfermformpara):
            fermform[:, m, k] = interp1d(meanenerirfn, mean(irfn[parastrg[k]].squeeze(), axis=0))(enerthis)

    factener = (10. * enerthis[:, None])**fermscal[None, :, 2]
    fermscalfact = sqrt((fermscal[None, :, 0] * factener)**2 + fermscal[None, :, 1]**2)
    
    # convert N_tail to f_core
    for m in indxevtt:
        for i in indxenerthis:
            fermform[i, m, 0] = 1. / (1. + fermform[i, m, 0] * fermform[i, m, 3]**2 / fermform[i, m, 1]**2)

    #temp = sqrt(2. - 2. * cos(thisangl[None, :, None]))
    #scalangl = 2. * arcsin(temp / 2.) / fermscalfact[:, None, :]
    
    fermform[:, :, 1] = fermscalfact * fermform[:, :, 1]
    fermform[:, :, 3] = fermscalfact * fermform[:, :, 3]

    frac = fermform[:, :, 0]
    sigc = fermform[:, :, 1]
    gamc = fermform[:, :, 2]
    sigt = fermform[:, :, 3]
    gamt = fermform[:, :, 4]
   
    # temp
    thisangl = thisangl[None, :, None]
    #thisangl = scalangl
    fermpsfn = retr_doubking(thisangl, frac[:, None, :], sigc[:, None, :], gamc[:, None, :], sigt[:, None, :], gamt[:, None, :])

    return fermpsfn


def retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt):

    psfn = frac / 2. / pi / sigc**2 * (1. - 1. / gamc) * (1. + scaldevi**2 / 2. / gamc / sigc**2)**(-gamc) + \
        (1. - frac) / 2. / pi / sigt**2 * (1. - 1. / gamt) * (1. + scaldevi**2 / 2. / gamt / sigt**2)**(-gamt)
    
    return psfn


def retr_beam(enerthis, indxevttthis, numbside, maxmmpol, fulloutp=False):
   
    numbpixl = 12 * numbside**2
    apix = 4. * pi / numbpixl

    numbener = enerthis.size
    numbevtt = indxevttthis.size

    # alm of the delta function at the North Pole
    mapsinpt = zeros(numbpixl)
    mapsinpt[:4] = 1.
    mapsinpt /= sum(mapsinpt) * apix
    almcinpt = real(hp.map2alm(mapsinpt, lmax=maxmmpol)[:maxmmpol+1])
    
    # alm of the point source at the North Pole
    lgalgrid, bgalgrid, numbpixl, apix = retr_healgrid(numbside)
    dir1 = array([lgalgrid, bgalgrid])
    dir2 = array([0., 90.])
    thisangl = hp.rotator.angdist(dir1, dir2, lonlat=True)
    mapsoutp = retr_fermpsfn(enerthis, thisangl)
    almcoutp = empty((numbener, maxmmpol+1, numbevtt))
    for i in range(numbener):
        for m in range(numbevtt):
            almcoutp[i, :, m] = real(hp.map2alm(mapsoutp[i, :, m], lmax=maxmmpol)[:maxmmpol+1])
    
    tranfunc = almcoutp / almcinpt[None, :, None]

    if fulloutp:
        return tranfunc, almcinpt, almcoutp
    else:
        return tranfunc


def smth_ferm(mapsinpt, enerthis, indxevttthis, maxmmpol=None, makeplot=False, gaus=False):
    
    numbpixl = mapsinpt.shape[1]

    numbside = int(sqrt(numbpixl / 12))
    if maxmmpol == None:
        maxmmpol = 3 * numbside - 1

    numbener = enerthis.size
    numbevtt = indxevttthis.size
    
    numbalmc = (maxmmpol + 1) * (maxmmpol + 2) / 2
    
    # get the beam
    beam = retr_beam(enerthis, indxevttthis, numbside, maxmmpol)
    
    # construct the transfer function
    tranfunc = ones((numbener, numbalmc, numbevtt))
    cntr = 0
    for n in arange(maxmmpol+1)[::-1]:
        tranfunc[:, cntr:cntr+n+1, :] = beam[:, maxmmpol-n:, :]
        cntr += n + 1

    mapsoutp = empty_like(mapsinpt)

    for i in arange(enerthis.size):
        for m in arange(indxevttthis.size):
            almc = hp.map2alm(mapsinpt[i, :, m], lmax=maxmmpol)
            almc *= tranfunc[i, :, m]
            mapsoutp[i, :, m] = hp.alm2map(almc, numbside, lmax=maxmmpol)
    
    return mapsoutp


def plot_fermsmth():

    numbside = 256
    numbpixl = 12 * numbside**2
    maxmmpol = 3 * numbside - 1
    mpol = arange(maxmmpol + 1)
    
    binsenerplot = array([0.3, 1., 3., 10.])
    meanenerplot = sqrt(binsenerplot[1:] * binsenerplot[:-1])
    numbenerplot = meanenerplot.size
    indxevttplot = arange(2, 4)
    numbevttplot = indxevttplot.size
    tranfunc, almcinpt, almcoutp = retr_beam(meanenerplot, indxevttplot, numbside, maxmmpol, fulloutp=True)

    figr, axis = plt.subplots()

    plt.loglog(mpol, almcinpt, label='HealPix')
    plt.loglog(mpol, sqrt((2. * mpol + 1.) / 4. / pi), label='Analytic')
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/almcinpt.pdf'
    plt.legend()
    figr.savefig(path)
    plt.close(figr)
    
    figr, axis = plt.subplots()
    for i in arange(meanenerplot.size):
        for m in arange(indxevttplot.size):
            plt.loglog(mpol, almcoutp[i, :, m], label='$E=%.3g$, PSF%d' % (meanenerplot[i], indxevttplot[m]))
    plt.legend(loc=3)
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/almcoutp.pdf'
    figr.savefig(path)
    plt.close(figr)
        
    figr, axis = plt.subplots()
    for i in arange(meanenerplot.size):
        for m in arange(indxevttplot.size):
            plt.loglog(mpol, tranfunc[i, :, m], label='$E=%.3g$, PSF%d' % (meanenerplot[i], indxevttplot[m]))
    plt.legend(loc=3)
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/tranfunc.pdf'
    figr.savefig(path)
    plt.close(figr)
    
    maxmgang = 20.
    minmlgal = -maxmgang
    maxmlgal = maxmgang
    minmbgal = -maxmgang
    maxmbgal = maxmgang
        
    # get the Planck radiance map
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/HFI_CompMap_ThermalDustModel_2048_R1.20.fits'
    maps = pf.getdata(path, 1)['RADIANCE']
    mapstemp = hp.ud_grade(maps, numbside, order_in='NESTED', order_out='RING')
    maps = empty((numbenerplot, numbpixl, numbevttplot))
    for i in arange(numbenerplot):
        for m in arange(numbevttplot):
             maps[i, :, m] = mapstemp

    # smooth the map with the Fermi-LAT kernel
    mapssmthferm = smth_ferm(maps, meanenerplot, indxevttplot)
    
    # smooth the map with the Gaussian kernel
    mapssmthgaus =  hp.sphtfunc.smoothing(mapstemp, sigma=deg2rad(0.5))

    # plot the maps
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/maps.pdf'
    plot_heal(path, mapstemp, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)

    for i in arange(meanenerplot.size):
        for m in arange(indxevttplot.size):
            path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/mapssmthferm%d%d.pdf' % (i, m)
            plot_heal(path, mapssmthferm[i, :, m], minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)
            
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/mapssmthgaus.pdf'
    plot_heal(path, mapssmthgaus, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)


