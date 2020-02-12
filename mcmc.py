# numerics
import numpy as np
import scipy as sp
from scipy.special import erfi
import scipy.fftpack
import scipy.stats

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt

import emcee

#import seaborn as sns
#sns.set(context='poster', style='ticks', color_codes=True)

mpl.rc('text', usetex=True)

# pixelization
#import healpy as hp
#from healpy.rotator import angdist

# utilities
import psutil, sys, sh, os, functools, time, datetime, fnmatch

# multiprocessing
import multiprocessing as mp

# warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# astropy
import astropy.coordinates, astropy.units
import astropy.io

from .util import summgene


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


def retr_atcr_neww(listsamp):

    numbsamp = listsamp.shape[0]
    four = sp.fftpack.fft(listsamp - np.mean(listsamp, axis=0), axis=0)
    atcr = sp.fftpack.ifft(four * np.conjugate(four), axis=0).real
    atcr /= np.amax(atcr, 0)
    
    return atcr[:numbsamp/2, ...]


def retr_timeatcr(listsamp, verbtype=1, atcrtype='maxm'):

    numbsamp = listsamp.shape[0]
    listsamp = listsamp.reshape((numbsamp, -1))
    numbpara = listsamp.shape[1]

    boolfail = False
    if listsamp.shape[0] == 1:
        boolfail = True

    atcr = retr_atcr_neww(listsamp)
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


def plot_hist(path, listvarb, strg, titl=None, numbbins=20, truepara=None, boolquan=True, strgplotextn='pdf', \
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
    figr.savefig(path + '_hist.%s' % strgplotextn)
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
     
    gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik = dictlpos

    for k in indxpara:
        if scalpara[k] != 'gaus':
            if para[k] < minmpara[k] or para[k] > maxmpara[k]:
                lpos = -np.inf
    else:
        llik = retr_llik(gdat, para)
        lpri = 0.
        for k in indxpara:
            if scalpara[k] == 'gaus':
                lpri = (para[k] - meangauspara[k]) / stdvgauspara[k]**2
        
        lpos = llik + lpri

    return lpos


def samp(gdat, pathimag, numbsampwalk, numbsampburnwalk, retr_llik, listlablpara, scalpara, \
                                   minmpara, maxmpara, meangauspara, stdvgauspara, numbdata, diagmode=True, strgmodl=None, samptype='emce'):
        
    if strgmodl is None:
        strgmodl = ''
    else:
        strgmodl = '_' + strgmodl

    numbpara = len(listlablpara)
    
    indxpara = np.arange(numbpara)
    numbdoff = numbdata - numbpara
    
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
    
    dictlpos = [gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik]
    
    # initialize
    numbwalk = 2 * numbpara
    indxwalk = np.arange(numbwalk)
    parainit = [np.empty(numbpara) for k in indxwalk]
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
    for k in indxwalk:
        for m in indxpara:
            if scalpara[m] == 'self':
                parainit[k][m]  = scipy.stats.truncnorm.rvs(-100., 100.) * (limtpara[1, m] - limtpara[0, m]) + limtpara[0, m]
            if scalpara[m] == 'gaus':
                parainit[k][m]  = np.random.rand() * stdvpara[m] + meanpara[m]

    numbsamp = numbsampwalk * numbwalk
    indxsamp = np.arange(numbsamp)
    numbsampburn = numbsampburnwalk * numbwalk
    if diagmode:
        if numbsampwalk == 0:
            raise Exception('')
    # temp
    #objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictllik, pool=multiprocessing.Pool())
    objtsamp = emcee.EnsembleSampler(numbwalk, numbpara, retr_lpos, args=dictlpos)
    if numbsampburnwalk > 0:
        parainitburn, prob, state = objtsamp.run_mcmc(parainit, numbsampburnwalk, progress=True)
        objtsamp.reset()
    else:
        parainitburn = parainit
    objtsamp.run_mcmc(parainitburn, numbsampwalk, progress=True)
    objtsave = objtsamp
    
    parapost = objtsave.flatchain
    indxsampwalk = np.arange(numbsampwalk)

    if numbsamp != numbsampwalk * numbwalk:
        raise Exception('')

    listsamp = objtsave.flatchain
    listllik = objtsave.flatlnprobability
    
    listlpos = objtsave.lnprobability
    chi2 = -2. * listlpos
    
    # plot the posterior
    ## parameter
    ### trace
    figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
    print('numbwalk')
    print(numbwalk)
    print('objtsave.lnprobability')
    summgene(objtsave.lnprobability)
    print('indxsampwalk')
    summgene(indxsampwalk)
    for i in indxwalk:
        axis[0].plot(indxsampwalk, objtsave.lnprobability[:, i])
    axis[0].set_ylabel('logL')
    listlablparafull = []
    for k in indxpara:
        for i in indxwalk:
            axis[k+1].plot(indxsampwalk, objtsave.chain[i, :, k])
        labl = listlablpara[k][0]
        if listlablpara[k][1] != '':
            labl += ' [%s]' % listlablpara[k][1]
        listlablparafull.append(labl)
        axis[k+1].set_ylabel(labl)
    path = pathimag + 'trac%s.png' % (strgmodl)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()
        
    indxsampmlik = np.argmax(listllik)
    listparamlik = listsamp[indxsampmlik, :]
    #print('Saving the maximum likelihood to %s...' % pathmlik)
    #np.savetxt(pathmlik, listparamlik, delimiter=',')
    
    strgplot = 'post' + strgmodl
    print('path')
    print(path)
    plot_grid(pathimag, strgplot, listsamp, listlablparafull, listvarbdraw=[meanpara.flatten()], numbbinsplot=numbbins)
    
    print('Minimum chi2: ')
    print(np.amin(chi2))
    print('Minimum chi2 per dof: ')
    print(np.amin(chi2) / numbdoff)
    print('Maximum aposterior: ')
    print(np.amax(listlpos))
    
    if samptype == 'nest':
        import dynesty
        from dynesty import plotting as dyplot
        from dynesty import utils as dyutils
        # resample the nested posterior

        weights = np.exp(results['logwt'] - results['logz'][-1])
        samppara = dyutils.resample_equal(results.samples, weights)
        assert samppara.size == results.samples.size
    
        if samptype == 'nest':
            pass
            #numbsamp = objtsave['samples'].shape[0]
        
        # resample the nested posterior
        if samptype == 'nest':
            weights = np.exp(results['logwt'] - results['logz'][-1])
            samppara = dyutils.resample_equal(results.samples, weights)
            assert samppara.size == results.samples.size
        
        if samptype == 'emce':
            pass
        else:
            sampler = dynesty.NestedSampler(retr_llik, icdf, numbpara, logl_args=dictllik, ptform_args=dictllik, bound='single', dlogz=1000.)
            sampler.run_nested()
            results = sampler.results
            results.summary()
            objtsave = results
        
            for keys in objtsave:
                if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
                    figr, axis = plt.subplots()
                    axis.plot(indxsamp, objtsave[keys])
                    path = pathdata + '%s/%s_%s.pdf' % (samptype, keys, ttvrtype)
                    print('Writing to %s...' % path)
                    plt.savefig(path)
            
        for keys in objtsave:
            if isinstance(objtsave[keys], np.ndarray) and objtsave[keys].size == numbsamp:
                figr, axis = plt.subplots()
                axis.plot(indxsamp, objtsave[keys])
                path = gdat.pathimag + '%s/%s_%s.pdf' % (samptype, keys, ttvrtype)
                print('Writing to %s...' % path)
                plt.savefig(path)
    
        ### nested sampling specific
        rfig, raxes = dyplot.runplot(results)
        path = gdat.pathimag + '%s/dyne_runs_%s.pdf' % (samptype, ttvrtype)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        tfig, taxes = dyplot.traceplot(results)
        path = gdat.pathimag + '%s/dyne_trac_%s.pdf' % (samptype, ttvrtype)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()
        
        cfig, caxes = dyplot.cornerplot(results)
        path = gdat.pathimag + '%s/dyne_corn_%s.pdf' % (samptype, ttvrtype)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

    return parapost


def plot_grid(pathbase, strgplot, listsamp, strgpara, join=False, limt=None, scalpara=None, plotsize=3.5, strgplotextn='pdf', \
                                    truepara=None, numbtickbins=3, numbbinsplot=20, boolquan=True, listvarbdraw=None):

    numbpara = listsamp.shape[1]
    
    if numbpara != 2 and join:
        raise Exception('Joint probability density can only be plotted for two parameters.')

    if scalpara is None:
        scalpara = ['self'] * numbpara

    if limt is None:
        limt = np.zeros((2, numbpara))
        limt[0, :] = np.amin(listsamp, 0)
        limt[1, :] = np.amax(listsamp, 0)
    
    for k in range(numbpara):
        if limt[0, k] == limt[1, k]:
            print('Lower and upper limits are the same. Grid plot failed.')
            return
    
    if truepara is not None:
        for k in range(numbpara):
            if truepara[k] is not None:
                if truepara[k] < limt[0, k]:
                    limt[0, k] = truepara[k] - 0.1 * (limt[1, k] - truepara[k]) 
                if truepara[k] > limt[1, k]:
                    limt[1, k] = truepara[k] + 0.1 * (truepara[k] - limt[0, k])
    
    if listvarbdraw is not None:
        numbdraw = len(listvarbdraw)
        indxdraw = np.arange(numbdraw)

    indxparagood = np.ones(numbpara, dtype=bool)
    indxparagood[np.where(limt[0, :] == limt[1, :])] = False
    
    bins = np.zeros((numbbinsplot + 1, numbpara))
    for k in range(numbpara):
        if scalpara[k] == 'self' or scalpara[k] == 'gaus':
            bins[:, k] = icdf_self(np.linspace(0., 1., numbbinsplot + 1), limt[0, k], limt[1, k])
        if scalpara[k] == 'logt':
            bins[:, k] = icdf_logt(np.linspace(0., 1., numbbinsplot + 1), limt[0, k], limt[1, k])
        if scalpara[k] == 'atan':
            bins[:, k] = icdf_atan(np.linspace(0., 1., numbbinsplot + 1), limt[0, k], limt[1, k])
        if not np.isfinite(bins[:, k]).all():
            print('k')
            print(k)
            print('listsamp[:, k]')
            summgene(listsamp[:, k])
            raise Exception('')
        if np.amin(bins[:, k]) == 0 and np.amax(bins[:, k]) == 0:
            print('Lower and upper limits of the bins are the same. Grid plot failed.')
            return

    figr, axgr = plt.subplots(numbpara, numbpara, figsize=(plotsize*numbpara, plotsize*numbpara))
    if numbpara == 1:
        axgr = [[axgr]]
    for k, axrw in enumerate(axgr):
        for l, axis in enumerate(axrw):
            if k < l or indxparagood[k] == False or indxparagood[l] == False:
                axis.axis('off')
                continue

            if k == l and not join:
                try:
                    axis.hist(listsamp[:, k], bins=bins[:, k])
                except:
                    pass
                if truepara is not None and truepara[k] is not None and not np.isnan(truepara[k]):
                    axis.axvline(truepara[k], color='g', lw=4)
                # draw the provided reference values
                if listvarbdraw is not None:
                    for m in indxdraw:
                        axis.axvline(listvarbdraw[m][k], color='r', lw=3)
                if boolquan:
                    quan = sp.stats.mstats.mquantiles(listsamp[:, k], prob=[0.025, 0.16, 0.84, 0.975])
                    axis.axvline(quan[0], color='b', ls='--', lw=2)
                    axis.axvline(quan[1], color='b', ls='-.', lw=2)
                    axis.axvline(quan[2], color='b', ls='-.', lw=2)
                    axis.axvline(quan[3], color='b', ls='--', lw=2)
                    medivarb = np.median(listsamp[:, k])
                axis.set_title(r'%.3g $substack{+%.2g \\ -%.2g}$' % (medivarb, quan[2] - medivarb, medivarb - quan[1]))
            else:
                if join:
                    k = 0
                    l = 1
                
                binstemp = [bins[:, l], bins[:, k]]
                hist = np.histogram2d(listsamp[:, l], listsamp[:, k], bins=binstemp)[0]
                axis.pcolor(bins[:, l], bins[:, k], hist.T, cmap='Blues')
                axis.set_xlim([np.amin(bins[:, l]), np.amax(bins[:, l])])
                axis.set_ylim([np.amin(bins[:, k]), np.amax(bins[:, k])])
                if truepara is not None and truepara[l] is not None and not np.isnan(truepara[l]) and truepara[k] is not None and not np.isnan(truepara[k]):
                    axis.scatter(truepara[l], truepara[k], color='g', marker='x', s=500)
                # draw the provided reference values
                if listvarbdraw is not None:
                    for m in indxdraw:
                        axis.scatter(listvarbdraw[m][l], listvarbdraw[m][k], color='r', marker='x', s=350)
                if scalpara[k] == 'logt':
                    axis.set_yscale('log', basey=10)
                    arry = np.logspace(np.log10(limt[0, k]), np.log10(limt[1, k]), numbtickbins)
                    strgarry = [util.mexp(arry[a]) for a in range(numbtickbins)]
                    axis.set_yticks(arry)
                    axis.set_yticklabels(strgarry)
                
            if scalpara[l] == 'logt':
                axis.set_xscale('log', basex=10)
                arry = np.logspace(np.log10(limt[0, l]), np.log10(limt[1, l]), numbtickbins)
                if not np.isfinite(arry).all():
                    raise Exception('')
                strgarry = [util.mexp(arry[a]) for a in range(numbtickbins)]
                axis.set_xticks(arry)
                axis.set_xticklabels(strgarry)
            axis.set_xlim(limt[:, l])
            if k == numbpara - 1:
                axis.set_xlabel(strgpara[l])
            else:
                axis.set_xticklabels([])
            if l == 0 and k != 0 or join:
                axis.set_ylabel(strgpara[k])
            else:
                if k != 0:
                    axis.set_yticklabels([])
    figr.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    figr.savefig(pathbase + 'pmar_' + strgplot + '.%s' % strgplotextn)
    plt.close(figr)
   
