# numerics
import numpy as np

import scipy as sp
from scipy.special import erfi
import scipy.fftpack
import scipy.stats

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import multiprocessing

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# utilities
import os

# astropy
import astropy.coordinates, astropy.units
import astropy.io

from .util import *

def icdf_self(paraunit, minmpara, maxmpara):
    para = (maxmpara - minmpara) * paraunit + minmpara
    return para


def icdf_logt(paraunit, minmpara, maxmpara):
    para = minmpara * np.exp(paraunit * np.log(maxmpara / minmpara))
    return para


def icdf_atan(paraunit, minmpara, maxmpara):
    para = tan((arctan(maxmpara) - arctan(minmpara)) * paraunit + arctan(minmpara))
    return para


def icdf_gaus(cdfn, meanpara, stdvpara):
    
    para = meanpara + stdvpara * np.sqrt(2) * sp.special.erfinv(2. * cdfn - 1.)

    return para


def cdfn_self(para, minmpara, maxmpara):
    paraunit = (para - minmpara) / (maxmpara - minmpara)
    return paraunit


def cdfn_logt(para, minmpara, maxmpara):
    paraunit = np.log(para / minmpara) / np.log(maxmpara / minmpara)
    return paraunit


def cdfn_atan(para, minmpara, maxmpara):
    paraunit = (arctan(para) - arctan(minmpara)) / (arctan(maxmpara) - arctan(minmpara))
    return paraunit


def cdfn_samp(sampvarb, datapara, k=None):
    
    if k is None:
        samp = empty_like(sampvarb)
        for k in range(sampvarb.size):
            samp[k] = cdfn_samp_sing(sampvarb[k], k, datapara)
    else:
        samp = cdfn_samp_sing(sampvarb[k], k, datapara)
    return samp


def cdfn_samp_sing(sampvarb, k, datapara):
    
    if datapara.scal[k] == 'self':
        samp = cdfn_self(sampvarb, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'logt':
        samp = cdfn_logt(sampvarb, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'atan':
        samp = cdfn_atan(sampvarb, datapara.minm[k], datapara.maxm[k])
        
    return samp


def icdf_samp(samp, datapara, k=None):
    
    if k is None:
        sampvarb = empty_like(samp)
        for k in range(sampvarb.size):
            sampvarb[k] = icdf_samp_sing(samp[k], k, datapara)
    else:
        sampvarb = icdf_samp_sing(samp[k], k, datapara)
    return sampvarb


def icdf_samp_sing(samp, k, datapara):

    if datapara.scal[k] == 'self':
        sampvarb = icdf_self(samp, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'logt':
        sampvarb = icdf_logt(samp, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'atan':
        sampvarb = icdf_atan(samp, datapara.minm[k], datapara.maxm[k])
        
    return sampvarb


def gmrb_test(griddata):
    
    withvari = np.mean(var(griddata, 0))
    btwnvari = griddata.shape[0] * var(np.mean(griddata, 0))
    wgthvari = (1. - 1. / griddata.shape[0]) * withvari + btwnvari / griddata.shape[0]
    psrf = sqrt(wgthvari / withvari)

    return psrf


def retr_atcr_neww(listpara):

    numbsamp = listpara.shape[0]
    four = sp.fftpack.fft(listpara - np.mean(listpara, axis=0), axis=0)
    atcr = sp.fftpack.ifft(four * np.conjugate(four), axis=0).real
    atcr /= np.amax(atcr, 0)
    
    return atcr[:int(numbsamp/2), ...]


def retr_timeatcr(listpara, verbtype=1, atcrtype='maxm'):

    numbsamp = listpara.shape[0]
    listpara = listpara.reshape((numbsamp, -1))
    numbpara = listpara.shape[1]

    boolfail = False
    if listpara.shape[0] == 1:
        boolfail = True

    atcr = retr_atcr_neww(listpara)
    indxatcr = np.where(atcr > 0.2)
     
    if indxatcr[0].size == 0:
        boolfail = True
        timeatcr = 0
    else:
        if atcrtype == 'nomi':
            timeatcr = np.argmax(indxatcr[0], axis=0)
        if atcrtype == 'maxm':
            indx = np.argmax(indxatcr[0])
            indxtimemaxm = indxatcr[0][indx]
            indxparamaxm = indxatcr[1][indx]
            atcr = atcr[:, indxparamaxm]
            timeatcr = indxtimemaxm
   
    if boolfail:
        if atcrtype == 'maxm':
            return np.zeros((1, 1)), 0.
        else:
            return np.zeros((1, numbpara)), 0.
    else:
        return atcr, timeatcr


def retr_numbsamp(numbswep, numbburn, factthin):
    
    numbsamp = int((numbswep - numbburn) / factthin)
    
    return numbsamp


def plot_gmrb(path, gmrbstat):

    numbbinsplot = 40
    bins = np.linspace(1., np.amax(gmrbstat), numbbinsplot + 1)
    figr, axis = plt.subplots()
    axis.hist(gmrbstat, bins=bins)
    axis.set_title('Gelman-Rubin Convergence Test')
    axis.set_xlabel('PSRF')
    axis.set_ylabel('$N_p$')
    figr.savefig(path + 'gmrb.pdf')
    plt.close(figr)


def plot_atcr(path, atcr, timeatcr, strgextn=''):

    numbsampatcr = atcr.size
    
    figr, axis = plt.subplots(figsize=(6, 4))
    axis.plot(np.arange(numbsampatcr), atcr)
    axis.set_xlabel(r'$\tau$')
    axis.set_ylabel(r'$\xi(\tau)$')
    axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % timeatcr, ha='center', va='center', transform=axis.transAxes)
    axis.axhline(0., ls='--', alpha=0.5)
    plt.tight_layout()
    pathplot = path + 'atcr%s.pdf' % strgextn
    figr.savefig(pathplot)
    plt.close(figr)
    
        
def plot_propeffi(path, numbswep, numbpara, listaccp, listindxparamodi, strgpara):

    indxlistaccp = np.where(listaccp == True)[0]
    binstime = np.linspace(0., numbswep - 1., 10)
    
    numbcols = 2
    numbrows = (numbpara + 1) / 2
    figr, axgr = plt.subplots(numbrows, numbcols, figsize=(16, 4 * (numbpara + 1)))
    if numbrows == 1:
        axgr = [axgr]
    for a, axrw in enumerate(axgr):
        for b, axis in enumerate(axrw):
            k = 2 * a + b
            if k == numbpara:
                axis.axis('off')
                break
            indxlistpara = np.where(listindxparamodi == k)[0]
            indxlistintc = intersect1d(indxlistaccp, indxlistpara, assume_unique=True)
            histotl = axis.hist(indxlistpara, binstime, color='b')
            histaccp = axis.hist(indxlistintc, binstime, color='g')
            axis.set_title(strgpara[k])
    figr.subplots_adjust(hspace=0.3)
    figr.savefig(path + 'propeffi.pdf')
    plt.close(figr)


def plot_trac(path, listpara, labl, truepara=None, scalpara='self', titl=None, \
                        boolquan=True, listvarbdraw=None, listlabldraw=None, numbbinsplot=20, logthist=False, listcolrdraw=None):
    
    if not np.isfinite(listpara).all():
        return
    
    if not np.isfinite(listpara).all():
        raise Exception('')
    
    if listpara.size == 0:
        return

    maxmpara = np.amax(listpara)
    if scalpara == 'logt':
        minmpara = np.amin(listpara[np.where(listpara > 0.)])
        bins = icdf_logt(np.linspace(0., 1., numbbinsplot + 1), minmpara, maxmpara)
    else:
        minmpara = np.amin(listpara)
        bins = icdf_self(np.linspace(0., 1., numbbinsplot + 1), minmpara, maxmpara)
    limspara = np.array([minmpara, maxmpara])
        
    if boolquan:
        quanarry = sp.stats.mstats.mquantiles(listpara, prob=[0.025, 0.16, 0.84, 0.975])

    if scalpara == 'logt':
        numbtick = 5
        listtick = np.logspace(np.log10(minmpara), np.log10(maxmpara), numbtick)
        listlabltick = ['%.3g' % tick for tick in listtick]
    
    figr, axrw = plt.subplots(1, 2, figsize=(14, 7))
    if titl is not None:
        figr.suptitle(titl, fontsize=18)
    for n, axis in enumerate(axrw):
        if n == 0:
            axis.plot(listpara, lw=0.5)
            axis.set_xlabel('$i_{samp}$')
            axis.set_ylabel(labl)
            if truepara is not None and not np.isnan(truepara):
                axis.axhline(y=truepara, color='g', lw=4)
            if scalpara == 'logt':
                axis.set_yscale('log')
                axis.set_yticks(listtick)
                axis.set_yticklabels(listlabltick)
            axis.set_ylim(limspara)
            if listvarbdraw is not None:
                for k in range(len(listvarbdraw)):
                    axis.axhline(listvarbdraw[k], label=listlabldraw[k], color=listcolrdraw[k], lw=3)
            if boolquan:
                axis.axhline(quanarry[0], color='b', ls='--', lw=2)
                axis.axhline(quanarry[1], color='b', ls='-.', lw=2)
                axis.axhline(quanarry[2], color='b', ls='-.', lw=2)
                axis.axhline(quanarry[3], color='b', ls='--', lw=2)
        else:
            axis.hist(listpara, bins=bins)
            axis.set_xlabel(labl)
            if logthist:
                axis.set_yscale('log')
            axis.set_ylabel('$N_{samp}$')
            if truepara is not None and not np.isnan(truepara):
                axis.axvline(truepara, color='g', lw=4)
            if scalpara == 'logt':
                axis.set_xscale('log')
                axis.set_xticks(listtick)
                axis.set_xticklabels(listlabltick)
            axis.set_xlim(limspara)
            if listvarbdraw is not None:
                for k in range(len(listvarbdraw)):
                    axis.axvline(listvarbdraw[k], label=listlabldraw[k], color=listcolrdraw[k], lw=3)
            if boolquan:
                axis.axvline(quanarry[0], color='b', ls='--', lw=2)
                axis.axvline(quanarry[1], color='b', ls='-.', lw=2)
                axis.axvline(quanarry[2], color='b', ls='-.', lw=2)
                axis.axvline(quanarry[3], color='b', ls='--', lw=2)
                
    figr.subplots_adjust()#top=0.9, wspace=0.4, bottom=0.2)

    figr.savefig(path + '_trac.pdf')
    plt.close(figr)


def plot_plot(path, xdat, ydat, lablxdat, lablydat, scalxaxi, titl=None, linestyl=[None], colr=[None], legd=[None], **args):
    
    if not isinstance(ydat, list):
        listydat = [ydat]
    else:
        listydat = ydat

    figr, axis = plt.subplots(figsize=(6, 6))
    for k, ydat in enumerate(listydat):
        if k == 0:
            linestyl = '-'
        else:
            linestyl = '--'
        axis.plot(xdat, ydat, ls=linestyl, color='k', **args)
        # temp
        #axis.plot(xdat, ydat, ls=linestyl[k], color=colr[k], label=legd[k], **args)
    axis.set_ylabel(lablydat)
    axis.set_xlabel(lablxdat)
    if scalxaxi == 'logt':
        axis.set_xscale('log')
    if titl is not None:
        axis.set_title(titl)
    plt.tight_layout()
    figr.savefig(path + '.pdf')
    plt.close(figr)


def plot_hist(path, listvarb, strg, titl=None, numbbins=20, truepara=None, boolquan=True, typefileplot='pdf', \
                                            scalpara='self', listvarbdraw=None, listlabldraw=None, listcolrdraw=None):

    minmvarb = np.amin(listvarb)
    maxmvarb = np.amax(listvarb)
    if scalpara == 'logt':
        bins = icdf_logt(np.linspace(0., 1., numbbins + 1), minmvarb, maxmvarb)
    else:
        bins = icdf_self(np.linspace(0., 1., numbbins + 1), minmvarb, maxmvarb)
    figr, axis = plt.subplots(figsize=(6, 6))
    axis.hist(listvarb, bins=bins)
    axis.set_ylabel(r'$N_{samp}$')
    axis.set_xlabel(strg)
    if truepara is not None:
        axis.axvline(truepara, color='g', lw=4)
    if listvarbdraw is not None:
        for k in range(len(listvarbdraw)):
            axis.axvline(listvarbdraw[k], label=listlabldraw[k], color=listcolrdraw[k], lw=3)
    if boolquan:
        quanarry = sp.stats.mstats.mquantiles(listvarb, prob=[0.025, 0.16, 0.84, 0.975])
        axis.axvline(quanarry[0], color='b', ls='--', lw=2)
        axis.axvline(quanarry[1], color='b', ls='-.', lw=2)
        axis.axvline(quanarry[2], color='b', ls='-.', lw=2)
        axis.axvline(quanarry[3], color='b', ls='--', lw=2)
    if titl is not None:
        axis.set_title(titl)
    plt.tight_layout()
    figr.savefig(path + '_hist.%s' % typefileplot)
    plt.close(figr)


def retr_limtpara(scalpara, minmpara, maxmpara, meanpara, stdvpara):
    
    numbpara = len(scalpara)
    limtpara = np.empty((2, numbpara))
    indxpara = np.arange(numbpara)
    for n in indxpara:
        if scalpara[n] == 'self':
            limtpara[0, n] = minmpara[n]
            limtpara[1, n] = maxmpara[n]
        if scalpara[n] == 'gaus':
            limtpara[0, n] = meanpara[n] - 10 * stdvpara[n]
            limtpara[1, n] = meanpara[n] + 10 * stdvpara[n]
    
    return limtpara


def retr_lpos(para, *dictlpos):
     
    gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik, retr_lpri = dictlpos
    
    boolreje = False
    for k in indxpara:
        if scalpara[k] != 'gaus':
            if para[k] < minmpara[k] or para[k] > maxmpara[k]:
                lpos = -np.inf
                boolreje = True
    
    if not boolreje:
        llik = retr_llik(para, gdat)
        lpri = 0.
        if retr_lpri is None:
            for k in indxpara:
                if scalpara[k] == 'gaus':
                    lpri += (para[k] - meangauspara[k]) / stdvgauspara[k]**2
        else:
            lpri = retr_lpri(para, gdat)
        lpos = llik + lpri
    
    #print('lpos')
    #print(lpos)
    #print('')
    
    return lpos


def retr_icdfunif(cdfn, minm, maxm):

    icdf = minm + cdfn * (maxm - minm)
    
    return icdf


def retr_icdf(cdfn, scalpara, minm, maxm):

    numbpara = len(scalpara)
    indxpara = np.arange(numbpara)
    icdf = np.empty(numbpara)
    for k in indxpara:
        if scalpara[k] == 'self':
            icdf[k] = minm[k] + cdfn[k] * (maxm[k] - minm[k])
    
    return icdf


def opti(pathimag, retr_llik, minmpara, maxmpara, numbtopp=3, numbiter=5):

    numbsamp = 4
    indxsamp = np.arange(numbsamp)
    numbpara = minmpara.size
    indxiter = np.arange(numbiter)
    indxtopp = np.arange(numbtopp)

    # seeds
    listfact = []
    listparacent = []
    listopen = []
    listllikmaxmseed = []
    # all samples
    #listllik = np.empty(0)
    #listpara = np.empty((0, numbpara))
    
    for i in indxiter:
        print('i')
        print(i)
        #print('listllik')
        #print(listllik)
        #print('listpara')
        #print(listpara)
        
        print('listfact')
        print(listfact)
        print('listparacent')
        print(listparacent)
        print('listllikmaxmseed')
        print(listllikmaxmseed)
        print('listopen')
        print(listopen)
        if i == 0:
            minmparatemp = minmpara
            maxmparatemp = maxmpara
            thisindxseed = 0
            paramidi = (maxmpara + minmpara) / 2.
            listfact.append(1.)
            listparacent.append(paramidi)
            listopen.append(True)
            listllikmaxmseed.append([])
        else:
            indxopen = np.where(listopen)[0]
            thisindxseed = np.random.choice(indxopen)
            print('thisindxseed')
            print(thisindxseed)
            maxmparatemp = listparacent[thisindxseed] + listfact[thisindxseed] * (maxmpara - listparacent[thisindxseed])
            minmparatemp = listparacent[thisindxseed] - listfact[thisindxseed] * (listparacent[thisindxseed] - minmpara)
        print('thisindxseed')
        print(thisindxseed)
        para = np.random.rand(numbpara * numbsamp).reshape((numbsamp, numbpara)) * (maxmpara[None, :] - minmpara[None, :]) + minmpara[None, :]
        print('para')
        print(para)
        
        print('Evaluating samples...')
        llik = np.empty(numbsamp)
        for k in indxsamp:
            llik[k] = retr_llik(para[k, :])
        print('llik')
        print(llik)
        listllikmaxmseed[thisindxseed] = np.amax(llik)
        
        # add new seeds
        if i == 0:
            for k in indxtopp:
                # factor
                listfact.append(0.5 * listfact[thisindxseed])
                
                # parameters of the seeds
                indxsampsort = np.argsort(llik)[::-1]
                listparacent.append(para[indxsampsort, :])
                
                # determine if still open
                boolopen = np.amax(llik) >= listllikmaxmseed[thisindxseed]
                listopen.append(boolopen)
                
                listllikmaxmseed.append([])

        #listllik = np.concatenate((listllik, llik))
        #listpara = np.concatenate((listpara, para), 0)
        
        #print('listllik')
        #print(listllik)
        #print('listpara')
        #print(listpara)
        print('listfact')
        print(listfact)
        print('listparacent')
        print(listparacent)
        print('listllikmaxmseed')
        print(listllikmaxmseed)
        print('listopen')
        print(listopen)
        print('')
        print('')
        print('')
        if not np.array(listopen).any():
            break
    
    return listparatopp


def samp(gdat, pathimag, numbsampwalk, numbsampburnwalk, numbsampburnwalkseco, retr_llik, listlablpara, scalpara, \
              minmpara, maxmpara, meangauspara, stdvgauspara, numbdata, retr_lpri=None, boolpool=True, \
              retr_dictderi=None, listlablparaderi=None, \
              numbsamp=None, \
              diagmode=True, strgextn='', samptype='emce', typefileplot='pdf', verbtype=1, strgsaveextn=None):
        
    numbpara = len(listlablpara)
   
    if numbsampwalk <= numbsampburnwalkseco:
        raise Exception('Burn-in samples cannot outnumber samples.')
        
    if numbpara != minmpara.size:
        raise Exception('')
    if numbpara != maxmpara.size:
        raise Exception('')

    indxpara = np.arange(numbpara)
    numbdoff = numbdata - numbpara
    
    if samptype == 'emce':
        numbwalk = max(20, 2 * numbpara)
        indxwalk = np.arange(numbwalk)
        numbsamptotl = numbsampwalk * numbwalk
        if numbsamp is None:
            numbsamp = max(10000, numbsamptotl)

    # plotting
    ## plot limits 
    limtpara = retr_limtpara(scalpara, minmpara, maxmpara, meangauspara, stdvgauspara)

    ## plot bins
    numbbins = 20
    indxbins = np.arange(numbbins)
    binspara = np.empty((numbbins + 1, numbpara)) 
    for k in indxpara:
        binspara[:, k] = np.linspace(limtpara[0, k], limtpara[1, k], numbbins + 1)
    meanpara = (binspara[1:, :] + binspara[:-1, :]) / 2.
    
    for k in indxpara:
        if minmpara[k] >= maxmpara[k]:
            raise Exception('')
    
    dictlpos = [gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik, retr_lpri]
    

    if verbtype == 2:
        print('scalpara')
        print(scalpara)
        print('minmpara')
        print(minmpara)
        print('maxmpara')
        print(maxmpara)
        print('meangauspara')
        print(meangauspara)
        print('stdvgauspara')
        print(stdvgauspara)
        print('limtpara')
        print(limtpara)
    
    # initialize
    if strgsaveextn is None or not os.path.exists(strgsaveextn):
        parainitcent = np.empty(numbpara)
        for m in indxpara:
            if scalpara[m] == 'self':
                parainitcent[m]  = limtpara[0, m] + 0.5 * (limtpara[1, m] - limtpara[0, m])
    else:
        print('Reading the initial state from %s...' % strgsaveextn)
        parainitcent = np.loadtxt(strgsaveextn)
    parainit = [np.empty(numbpara) for k in indxwalk]
    for m in indxpara:
        for k in indxwalk:
            if scalpara[m] == 'self':
                stdvinit = 10.
                parainit[k][m] = 0.5 / stdvinit * scipy.stats.truncnorm.rvs(-stdvinit, stdvinit) * (limtpara[1, m] - limtpara[0, m]) + parainitcent[m]
            if scalpara[m] == 'gaus':
                parainit[k][m] = np.random.rand() * stdvpara[m] + meanpara[m] + parainitcent[m]
        
    if verbtype > 0:
        progress = True
    else:
        progress = False
    
    if samptype == 'emce':
        import emcee

        indxsampwalk = np.arange(numbsampwalk)
        indxsamp = np.arange(numbsamp)
        numbsampburn = numbsampburnwalk * numbwalk
        if diagmode:
            if numbsampwalk == 0:
                raise Exception('')
    
        if boolpool:
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        else:
            pool = None
        objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictlpos, pool=pool)
        if numbsampburnwalk > 0:
            parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampburnwalk, progress=progress)
            if verbtype == 1:
                print('Parameter states from the burn-in:')
                print('parainitburn')
                print(parainitburn)
            parainit = np.array(parainitburn)
            indxwalkmpos = np.argmax(objtsamp.lnprobability[:, -1], 0)
            parainittemp = parainit[indxwalkmpos, :]
            parainitburn = [[[] for m in indxpara] for k in indxwalk]
            for m in indxpara:
                for k in indxwalk:
                    parainitburn[k][m] = parainittemp[m] * (1. + 1e-5 * np.random.randn())
            objtsamp.reset()
        else:
            parainitburn = parainit
        objtsamp.run_mcmc(parainitburn, numbsampwalk, progress=progress)
        listlposwalk = objtsamp.lnprobability
        listparafittwalk = objtsamp.chain
        
        # get rid of burn-in and thin
        indxsampwalkkeep = np.linspace(numbsampburnwalkseco, numbsampwalk - 1, int(numbsamp / numbwalk)).astype(int)
        listparafitt = listparafittwalk[:, indxsampwalkkeep, :].reshape((-1, numbpara))
        
        listparaderi = None
        if retr_dictderi is not None:
            listdictparaderi = [[] for n in indxsamp]
            listdictvarbderi = [[] for n in indxsamp]
            for n in indxsamp:
                listdictparaderi[n], listdictvarbderi[n] = retr_dictderi(listparafitt[n, :], gdat)

            dictparaderi = dict()
            for strg, valu in listdictparaderi[0].items():
                dictparaderi[strg] = np.empty([numbsamp] + list(valu.shape))
                for n in indxsamp:
                    dictparaderi[strg][n, ...] = listdictparaderi[n][strg]
            numbparaderi = len(listdictparaderi[0])
            listparaderi = np.empty((numbsamp, numbparaderi)) 
            k = 0
            for strg, valu in listdictparaderi[0].items():
                listparaderi[:, k] = dictparaderi[strg][:, 0]
                k += 1
                for n in indxsamp:
                    dictparaderi[strg][n, ...] = listdictparaderi[n][strg]
            
        indxsampwalk = np.arange(numbsampwalk)
        
        # plot the posterior
        ### trace
        figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
        for i in indxwalk:
            axis[0].plot(indxsampwalk, listlposwalk[i, :])
        axis[0].axvline(numbsampburnwalkseco, color='k')
        axis[0].set_ylabel('log P')
        for k in indxpara:
            for i in indxwalk:
                axis[k+1].plot(indxsampwalk, listparafittwalk[i, :, k])
            labl = listlablpara[k][0]
            if listlablpara[k][1] != '':
                labl += ' [%s]' % listlablpara[k][1]
            axis[k+1].axvline(numbsampburnwalkseco, color='k')
            axis[k+1].set_ylabel(labl)
        path = pathimag + 'trac%s.%s' % (strgextn, typefileplot)
        if verbtype == 1:
            print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        # plot the posterior
        ### trace
        if numbsampburnwalkseco > 0:
            figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
            for i in indxwalk:
                axis[0].plot(indxsampwalk[numbsampburnwalkseco:], listlposwalk[i, numbsampburnwalkseco:])
            axis[0].set_ylabel('log P')
            for k in indxpara:
                for i in indxwalk:
                    axis[k+1].plot(indxsampwalk[numbsampburnwalkseco:], listparafittwalk[i, numbsampburnwalkseco:, k])
                labl = listlablpara[k][0]
                if listlablpara[k][1] != '':
                    labl += ' [%s]' % listlablpara[k][1]
                axis[k+1].set_ylabel(labl)
            path = pathimag + 'tracgood%s.%s' % (strgextn, typefileplot)
            if verbtype == 1:
                print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
    
    if samptype == 'nest':
        import dynesty
        from dynesty import plotting as dyplot
        from dynesty import utils as dyutils
        
        dictllik = [gdat]
        dicticdf = [scalpara, minmpara, maxmpara]
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)

        sampler = dynesty.NestedSampler(retr_llik, retr_icdf, numbpara, logl_args=dictllik, ptform_args=dicticdf, \
                                                                    pool=pool, queue_size=multiprocessing.cpu_count(), \
        #bound='single', \
        #                                                                                        nlive=10, dlogz=1000. \
        )
        sampler.run_nested()
        results = sampler.results
        results.summary()
        objtsamp = results
        numbsamp = objtsamp['samples'].shape[0]
        
        # resample the nested posterior
        weights = np.exp(results['logwt'] - results['logz'][-1])
        listpara = dyutils.resample_equal(results.samples, weights)
        assert listpara.size == results.samples.size
        
        numbsamp = listpara.shape[0]
        indxsamp = np.arange(numbsamp)

        pathbase = pathimag + '%s/' % samptype
        os.system('mkdir -p %s' % pathbase)
        for keys in objtsamp:
            if isinstance(objtsamp[keys], np.ndarray) and objtsamp[keys].size == numbsamp:
                figr, axis = plt.subplots()
                axis.plot(indxsamp, objtsamp[keys])
                path = pathimag + '%s/%s%s.%s' % (samptype, keys, strgextn, typefileplot)
                if verbtype == 1:
                    print('Writing to %s...' % path)
                plt.savefig(path)
    
        rfig, raxes = dyplot.runplot(results)
        path = pathimag + '%s/dyne_runs%s.%s' % (samptype, strgextn, typefileplot)
        if verbtype == 1:
            print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        tfig, taxes = dyplot.traceplot(results)
        path = pathimag + '%s/dyne_trac%s.%s' % (samptype, strgextn, typefileplot)
        if verbtype == 1:
            print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        cfig, caxes = dyplot.cornerplot(results)
        path = pathimag + '%s/dyne_corn%s.%s' % (samptype, strgextn, typefileplot)
        if verbtype == 1:
            print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
    
    ## joint PDF
    strgplot = 'postparafitt' + strgextn
    plot_grid(pathimag, strgplot, listparafitt, listlablpara, numbbinsplot=numbbins)
    
    # derived
    if retr_dictderi is not None:
        listlablparatotl = listlablpara + listlablparaderi
        listparatotl = np.concatenate([listparafitt, listparaderi], 1)
        strgplot = 'postparaderi' + strgextn
        plot_grid(pathimag, strgplot, listparaderi, listlablparaderi, numbbinsplot=numbbins)
        strgplot = 'postparatotl' + strgextn
        plot_grid(pathimag, strgplot, listparatotl, listlablparatotl, numbbinsplot=numbbins)
    
    if strgsaveextn is not None:
        print('Writing to the initial state from %s...' % strgsaveextn)
        np.savetxt(strgsaveextn, np.median(listpara, 0))
    
    return listparafitt, listparaderi


def plot_grid_diag(k, axis, listpara, bins, truepara, listvarbdraw, boolquan, listlablpara):
                    
    axis.hist(listpara[:, k], bins=bins[:, k])
    if truepara is not None and truepara[k] is not None and not np.isnan(truepara[k]):
        axis.axvline(truepara[k], color='g', lw=4)
    # draw the provided reference values
    if listvarbdraw is not None:
        for m in indxdraw:
            axis.axvline(listvarbdraw[m][k], color='r', lw=3)
    if boolquan:
        quan = np.empty(4)
        quan[0] = np.nanpercentile(listpara[:, k], 2.5)
        quan[1] = np.nanpercentile(listpara[:, k], 16.)
        quan[2] = np.nanpercentile(listpara[:, k], 84.)
        quan[3] = np.nanpercentile(listpara[:, k], 97.5)
        axis.axvline(quan[0], color='r', ls='--', lw=2)
        axis.axvline(quan[1], color='r', ls='-.', lw=2)
        axis.axvline(quan[2], color='r', ls='-.', lw=2)
        axis.axvline(quan[3], color='r', ls='--', lw=2)
        medivarb = np.nanmedian(listpara[:, k])
    if listlablpara[k][1] != '':
        strgunit = ' ' + listlablpara[k][1]
    else:
        strgunit = ''
    axis.set_title(r'%s = %.3g $\substack{+%.2g \\\\ -%.2g}$ %s' % (listlablpara[k][0], medivarb, quan[2] - medivarb, medivarb - quan[1], strgunit))
                

def plot_grid_pair(k, l, axis, limt, listmantlabl, listpara, bins, truepara, listvarbdraw, boolquan, listlablpara, listscalpara, boolsqua, listvectplot):
    
    binstemp = [bins[:, l], bins[:, k]]
    hist = np.histogram2d(listpara[:, l], listpara[:, k], bins=binstemp)[0]
    if np.isfinite(bins[:, l]).all() and np.isfinite(bins[:, k]).all():
        axis.pcolor(bins[:, l], bins[:, k], hist.T, cmap='Blues')
        axis.set_xlim([np.amin(bins[:, l]), np.amax(bins[:, l])])
        axis.set_ylim([np.amin(bins[:, k]), np.amax(bins[:, k])])
    if truepara is not None and truepara[l] is not None and not np.isnan(truepara[l]) and truepara[k] is not None and not np.isnan(truepara[k]):
        axis.scatter(truepara[l], truepara[k], color='g', marker='x', s=500)
    # draw the provided reference values
    if listvarbdraw is not None:
        for m in indxdraw:
            axis.scatter(listvarbdraw[m][l], listvarbdraw[m][k], color='r', marker='x', s=350)
    if listvectplot is not None:
        for vectplot in listvectplot:
            print('vectplot')
            print(vectplot)
            axis.arrow(vectplot[0], vectplot[1], vectplot[2], vectplot[3])
    
    if boolsqua:
        axis.set_aspect('equal')

    if listscalpara[l] == 'logt':
        setp_axislogt(axis, limt[:, l], 'x', listmantlabl)
    
    if listscalpara[k] == 'logt':
        setp_axislogt(axis, limt[:, k], 'y', listmantlabl)
    
    axis.set_xlim(limt[:, l])
    axis.set_ylim(limt[:, k])
                

def retr_listvalutickmajr(minm, maxm):
    
    minmlogt = np.log10(minm)
    maxmlogt = np.log10(maxm)

    listvalutickmajr = 10**(np.arange(np.ceil(minmlogt), np.floor(maxmlogt) + 1))
    
    return listvalutickmajr


def retr_valulablticklogt(minm, maxm, listmantlabl=None):
    
    minmlogt = np.log10(minm)
    maxmlogt = np.log10(maxm)
    
    # determine major ticks and labels
    listvalutickmajr = retr_listvalutickmajr(minm, maxm)
    listlabltickmajr = [retr_lablmexp(listvalutickmajr[a]) for a in range(len(listvalutickmajr))]
    
    # determine minor ticks and labels
    ## minor tick mantissa
    if listmantlabl is None:
        if len(listvalutickmajr) > 2:
            listmantlabl = []
        elif len(listvalutickmajr) == 2:
            listmantlabl = [3.]
        else:
            listmantlabl = [3., 5.]
    listmantminr = np.arange(2., 10.)
    listexpominr = np.arange(np.floor(minmlogt), np.ceil(maxmlogt) + 1)
    listvalutickminr = []
    listvalutickminrlabl = []
    indxlablmant = []
    indxlablexpo = []
    for kk in range(len(listmantminr)):
        for mm in range(len(listexpominr)):
            valu = listmantminr[kk] * 10**(listexpominr[mm])
            listvalutickminr.append(valu)
            if listmantminr[kk] in listmantlabl:
                listvalutickminrlabl.append(valu)
                indxlablmant.append(kk)
                indxlablexpo.append(mm)
    indxlabl = np.array(indxlablexpo, dtype=int) * len(listmantminr) + np.array(indxlablmant, dtype=int)
    listvalutickminr = np.array(listvalutickminr)
    listvalutickminr = np.sort(listvalutickminr)
    listlabltickminr = np.empty_like(listvalutickminr, dtype=object)
    listlabltickminr[:] = ''
    for a in range(len(listvalutickminrlabl)):
        listlabltickminr[indxlabl[a]] = retr_lablmexp(listvalutickminrlabl[a])
    
    return listvalutickmajr, listlabltickmajr, listvalutickminr, listlabltickminr


def setp_axislogt(axis, limt, typeaxis, listmantlabl):
    
    listvalutickmajr, listlabltickmajr, listvalutickminr, listlabltickminr = retr_valulablticklogt(minm, maxm, listmantlabl=listmantlabl)

    if typeaxis == 'x':
        axis.set_xscale('log', basex=10)
        axis.set_xticks(listvalutickmajr)
        axis.set_xticklabels(listlabltickmajr)
        axis.set_xticks(listvalutickminr, minor=True)
        axis.set_xticklabels(listlabltickminr, minor=True)
    else:
        axis.set_yscale('log', basey=10)
        axis.set_yticks(listvalutickmajr)
        axis.set_yticklabels(listlabltickmajr)
        axis.set_yticks(listvalutickminr, minor=True)
        axis.set_yticklabels(listlabltickminr, minor=True)


def plot_grid(pathbase, strgplot, listpara, listlablpara, \
                       limt=None, listscalpara=None, plotsize=2.5, typefileplot='pdf', \
                       
                       # Boolean flag to generate individual histograms
                       boolplotindi=False, \
                       
                       # Boolean flag to generate individual pair-wise correlation plots
                       boolplotpair=False, \
                       
                       # Boolean flag to generate the lower-triangle plot
                       boolplottria=True, \
                       
                       # list of base file names for the individual histograms
                       liststrgvarb=None, \
                           
                       # Boolean flag to make the two-dimensional plots square
                       boolsqua=False, \

                       # list of vectors to overplot
                       listvectplot=None, \

                       truepara=None, \
                       
                       # list of tick mantices (other than 1) to show in the label when the axis is log-streched
                       listmantlabl=None, \
                       
                       numbbinsplot=20, boolquan=True, listvarbdraw=None, verbtype=0, boolscat=False):

    # check inputs
    if (boolplotpair or boolplotindi) and liststrgvarb is None:
        raise Exception('You should define liststrgvarb for individual and pairwise plots.')
    
    numbpara = listpara.shape[1]
    indxpara = np.arange(numbpara)
    listlablparatotl = retr_listlablparatotl(listlablpara)
    
    if not np.isfinite(listpara).all():
        print('plot_grid(): Not all samples are finite!')
    
    if listscalpara is None:
        listscalpara = ['self'] * numbpara

    if len(listscalpara) != len(listlablpara):
        raise Exception('')
    if len(listscalpara) != numbpara:
        print('listscalpara')
        print(listscalpara)
        print('len(listscalpara)')
        print(len(listscalpara))
        print('listpara')
        summgene(listpara)
        print('numbpara')
        print(numbpara)
        raise Exception('len(listscalpara) != numbpara')

    if limt is None:
        limt = np.zeros((2, numbpara))
        
        for k in indxpara:
            if listscalpara[k] == 'logt':
                indxgood = np.where(listpara[:, k] > 0)[0]
                if indxgood.size < listpara[:, k].size:
                    print('Parameter %d (%s) has a log scaling but also nonpositive elements!' % (k, listlablpara[k]))
            else:
                indxgood = np.arange(listpara[:, k].size)
            if indxgood.size > 0:
                limt[0, k] = np.nanmin(listpara[indxgood, k], 0)
                limt[1, k] = np.nanmax(listpara[indxgood, k], 0)
            else:
                limt[0, k] = 0.
                limt[1, k] = 0.
                print('limt[:, k]')
                print(limt[:, k])

        for k in indxpara:
            if not np.isfinite(listpara[:, k]).all():
                print('Warning! Parameter %d (%s) is not all finite.' % (k, listlablpara[k][0]))
                summgene(listpara[:, k])
    
    print('limt')
    print(limt)

    for k in indxpara:
        if limt[0, k] == limt[1, k]:
            print('WARNING! Lower and upper limits are the same for the following parameter.')
            print('k')
            print(k)
            print('listlablpara[k]')
            print(listlablpara[k])
            #return
    
    if truepara is not None:
        for k in indxpara:
            if truepara[k] is not None:
                if truepara[k] < limt[0, k]:
                    limt[0, k] = truepara[k] - 0.1 * (limt[1, k] - truepara[k]) 
                if truepara[k] > limt[1, k]:
                    limt[1, k] = truepara[k] + 0.1 * (truepara[k] - limt[0, k])
    
    if listvarbdraw is not None:
        numbdraw = len(listvarbdraw)
        indxdraw = np.arange(numbdraw)

    boolparagood = np.ones(numbpara, dtype=bool)
    boolparagood[np.where(limt[0, :] == limt[1, :])] = False
    
    if not boolscat:
        bins = np.zeros((numbbinsplot + 1, numbpara))
        for k in indxpara:
            if not boolparagood[k]:
                continue
            if listscalpara[k] == 'self' or listscalpara[k] == 'gaus':
                bins[:, k] = icdf_self(np.linspace(0., 1., numbbinsplot + 1), limt[0, k], limt[1, k])
            elif listscalpara[k] == 'logt':
                bins[:, k] = icdf_logt(np.linspace(0., 1., numbbinsplot + 1), limt[0, k], limt[1, k])
            elif listscalpara[k] == 'atan':
                bins[:, k] = icdf_atan(np.linspace(0., 1., numbbinsplot + 1), limt[0, k], limt[1, k])
            else:
                raise Exception('Unrecognized scaling: %s' % listscalpara[k])
            if not np.isfinite(bins[:, k]).all():
                print('k')
                print(k)
                print('numbbinsplot')
                print(numbbinsplot)
                print('listlablpara[k]')
                print(listlablpara[k])
                print('listscalpara[k]')
                print(listscalpara[k])
                print('limt[:, k]')
                print(limt[:, k])
                print('bins[:, k]')
                print(bins[:, k])
                print('listpara[:, k]')
                summgene(listpara[:, k])
                #raise Exception('')
            if np.amin(bins[:, k]) == 0 and np.amax(bins[:, k]) == 0:
                print('Lower and upper limits of the bins are the same for %s. Grid plot failed.' % listlablpara[k][0])
                print('k')
                print(k)
                print('bins[:, k]')
                print(bins[:, k])
                print('limt[:, k]')
                print(limt[:, k])
                print('listscalpara[k]')
                print(listscalpara[k])
                print('listlablpara[k]')
                print(listlablpara[k])
                #return
    
    if boolplotindi:
        # histogram
        for k in indxpara:
            if not boolparagood[k]:
                continue
            figr, axis = plt.subplots(figsize=(6, 4))
            axis.hist(listpara[:, k])
            if listvarbdraw is not None:
                for m in indxdraw:
                    axis.axvline(listvarbdraw[m][k], color='orange', lw=3)
            axis.set_xlabel(listlablvarb[k])
            axis.set_ylabel('N')
            path = pathimag + 'histgridindi_%s.%s' % (liststrgvarb[k], typefileplot) 
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

    if boolplotpair:
        for k in indxpara:
            for l in indxpara:
                if not boolparagood[k] or not boolparagood[l]:
                    continue
                if k <= l:
                    continue
                figr, axis = plt.subplots(figsize=(8, 6))
                
                plot_grid_pair(k, l, axis, limt, listmantlabl, listpara, bins, truepara, listvarbdraw, boolquan, listlablpara, listscalpara, boolsqua, listvectplot)
                
                axis.set_xlabel(listlablparatotl[l])
                axis.set_ylabel(listlablparatotl[k])
                
                path = pathbase + 'pmar_%s_%s_%s.%s' % (liststrgvarb[k], liststrgvarb[l], strgplot, typefileplot)
                print('Writing to %s...' % path)
                figr.savefig(path)
                plt.close(figr)

    if boolplottria:
        figr, axgr = plt.subplots(numbpara, numbpara, figsize=(plotsize*numbpara, plotsize*numbpara))
        if numbpara == 1:
            axgr = [[axgr]]
        for k, axrw in enumerate(axgr):
            for l, axis in enumerate(axrw):
                if not boolparagood[k] or not boolparagood[l]:
                    continue
                if k < l:
                    axis.axis('off')
                    continue

                if k == l:
                    plot_grid_diag(k, axis, listpara, bins, truepara, listvarbdraw, boolquan, listlablpara)
                else:
                    plot_grid_pair(k, l, axis, limt, listmantlabl, listpara, bins, truepara, listvarbdraw, boolquan, listlablpara, listscalpara, boolsqua, listvectplot)
                    
                if k == numbpara - 1:
                    axis.set_xlabel(listlablparatotl[l])
                else:
                    axis.set_xticklabels([])
                
                if (l == 0 and k != 0):
                    axis.set_ylabel(listlablparatotl[k])
                else:
                    if k != 0:
                        axis.set_yticklabels([])
        figr.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        path = pathbase + 'pmar_' + strgplot + '.%s' % typefileplot
        print('Writing to %s...' % path)
        figr.savefig(path)
        plt.close(figr)
   
