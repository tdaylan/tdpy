# math
from numpy import *
from numpy.random import *
from numpy.random import choice
from scipy.integrate import *
from scipy.interpolate import *
import scipy as sp

import time

import multiprocessing as mp, functools

import util

# plotting
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)


def icdf_self(paraunit, minmpara, maxmpara):
    para = (maxmpara - minmpara) * paraunit + minmpara
    return para


def icdf_logt(paraunit, minmpara, maxmpara):
    para = minmpara * exp(paraunit * log(maxmpara / minmpara))
    return para


def icdf_atan(paraunit, minmpara, maxmpara):
    para = tan((arctan(maxmpara) - arctan(minmpara)) * paraunit + arctan(minmpara))
    return para


def cdfn_self(para, minmpara, maxmpara):
    paraunit = (para - minmpara) / (maxmpara - minmpara)
    return paraunit


def cdfn_logt(para, minmpara, maxmpara):
    paraunit = log(para / minmpara) / log(maxmpara / minmpara)
    return paraunit


def cdfn_atan(para, minmpara, maxmpara):
    paraunit = (arctan(para) - arctan(minmpara)) / (arctan(maxmpara) - arctan(minmpara))
    return paraunit


def cdfn_samp(sampvarb, datapara, k=None):
    
    if k == None:
        samp = empty_like(sampvarb)
        for k in range(sampvarb.size):
            samp[k] = cdfn_samp_sing(sampvarb[k], k, datapara)
    else:
        samp = cdfn_samp_sing(sampvarb[k], k, datapara)
    return samp


def cdfn_samp_sing(sampvarb, k, datapara):
    
    if datapara[4][k] == 'self':
        samp = cdfn_self(sampvarb, datapara[2][k], datapara[3][k])
    if datapara[4][k] == 'logt':
        samp = cdfn_logt(sampvarb, datapara[2][k], datapara[3][k])
    if datapara[4][k] == 'atan':
        samp = cdfn_atan(sampvarb, datapara[2][k], datapara[3][k])
        
    return samp


def icdf_samp(samp, datapara, k=None):
    
    if k == None:
        sampvarb = empty_like(samp)
        for k in range(sampvarb.size):
            sampvarb[k] = icdf_samp_sing(samp[k], k, datapara)
    else:
        sampvarb = icdf_samp_sing(samp[k], k, datapara)
    return sampvarb


def icdf_samp_sing(samp, k, datapara):

    if datapara[4][k] == 'self':
        sampvarb = icdf_self(samp, datapara[2][k], datapara[3][k])
    if datapara[4][k] == 'logt':
        sampvarb = icdf_logt(samp, datapara[2][k], datapara[3][k])
    if datapara[4][k] == 'atan':
        sampvarb = icdf_atan(samp, datapara[2][k], datapara[3][k])
        
    return sampvarb


def gmrb_test(griddata):
    
    withvari = mean(var(griddata, 0))
    btwnvari = griddata.shape[0] * var(mean(griddata, 0))
    wgthvari = (1. - 1. / griddata.shape[0]) * withvari + btwnvari / griddata.shape[0]
    psrf = sqrt(wgthvari / withvari)

    return psrf


def mcmc_wrap(numbproc, numbswep, llikfunc, datapara, thissamp=None, optiprop=False, plotpath=None, rtag='', numbburn=None, truepara=None, \
    numbplotside=None, factthin=None, verbtype=0):
    
    if verbtype > 1:
        print 'Forking the sampler...'

    listobjt = numbproc, numbswep, llikfunc, datapara, thissamp, optiprop, plotpath, rtag, numbburn, truepara, numbplotside, factthin, verbtype

    if numbproc == 1:
        listchan = [mcmc(listobjt, 0)]
    else:
        # spawn the processes
        pool = mp.Pool(numbproc)
        workpart = functools.partial(mcmc, listobjt)
        listchan = pool.map(workpart, arange(numbproc))
    
        pool.close()
        pool.join()


    if verbtype > 0:
        print 'Accumulating samples from all processes...'
        tim0 = time.time()


    numbsamp = retr_numbsamp(numbswep, numbburn, factthin)

    # parse the sample chain
    listsampvarb = zeros((numbsamp, numbproc, numbpara))
    listsamp = zeros((numbsamp, numbproc, numbpara)) 
    listsampcalc = zeros((numbsamp, numbproc, numbsampcalc))
    listllik = zeros((numbsamp, numbproc))
    listaccp = zeros((numbsamp, numbproc, numbpara))
    listindxparamodi = zeros((numbsamp, numbproc, numbpara))

    indxproc = arange(numbproc)
    for k in indxproc:
        listsampvarb[:, k, :] = listchan[k][0]
        listsamp[:, k, :] = listchan[k][1]
        listsampcalc[:, k, :] = listchan[k][2]
        listllik[:, k] = listchan[k][3]
        listaccp[:, k, :] = listchan[k][4]
        listindxparamodi[:, k, :] = listchan[k][5]


    indxlistaccp = where(listaccp == True)[0]
    propeffi = zeros(numbpara)
    for k in range(numbpara):
        indxlistpara = where(listindxparamodi == k)[0]
        indxlistintc = intersect1d(indxlistaccp, indxlistpara, assume_unique=True)
        if indxlistpara.size != 0:
            propeffi[k] = float(indxlistintc.size) / indxlistpara.size    
    
    minmlistllik = amin(listllik)
    levi = -log(mean(1. / exp(listllik - minmlistllik))) + minmlistllik
    info = mean(listllik) - levi
    
    strgpara = lablpara + ' ' + unitpara

    if plotpath != None:
        
        if verbtype > 1:
            print 'Making plots...'
            timeinit = time.time()
            
        path = plotpath + 'propeffi' + rtag + '.png'
        plot_propeffi(path, numbswep, numbpara, listaccp, listindxparamodi, strgpara)

        path = plotpath + 'llik' + rtag + '.png'
        plot_trac(path, listllik, '$P(D|y)$', titl='log P(D) = %.3g' % levi)
        
        if numbplotside != 0:
            path = plotpath + 'grid' + rtag + '.png'
            plot_grid(path, listsampvarb, strgpara, truepara=truepara,                                 scalpara=scalpara, numbplotside=numbplotside)
            
        for k in indxpara:
            path = plotpath + 'trac_' + namepara[k] + rtag + '.png'
            plot_trac(path, listsampvarb[:, k], strgpara[k], scalpara=scalpara[k],                                 truepara=truepara[k])
            
        if verbtype > 1:
            timefinl = time.time()
            print 'Done in %.3g seconds' % (timefinl - timeinit)

        gmrbstat = zeros(numbpara)
        if numbproc > 1:
            if verbtype > 1:
                print 'Performing Gelman-Rubin convergence test...'
                tim0 = time.time()
            for k in indxpara:
                gmrbstat[k] = gmrb_test(listsampvarb[:, :, k])
            path = plotpath + 'gmrb' + rtag + '.png'
            plot_gmrb(path, gmrbstat)
            if verbtype > 1:
                timefinl = time.time()
                print 'Done in %.3g seconds' % (timefinl - timeinit)
                
    listsampvarb = listsampvarb.reshape((numbsamp * numbproc, numbpara))
    listsamp = listsamp.reshape((numbsamp * numbproc, numbpara))
    for k in range(numbsampcalc):
        listsampcalc[k] = listsampcalc[k].reshape((numbsamp * numbproc, -1))
    listllik = listsamp.reshape((numbsamp * numbproc))
    listaccp = listsamp.reshape((numbsamp * numbproc, numbpara))
    listindxparamodi = listindxparamodi.reshape((numbsamp * numbproc))

    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listindxparamodi, propeffi, levi, info, gmrbstat]
    
    return chan


def mcmc(listobjt, indxprocwork):
    
    numbproc, numbswep, llikfunc, datapara, thissamp, optiprop, plotpath, rtag, numbburn, truepara, numbplotside, factthin, verbtype = listobjt
 
    global namepara, minmpara, maxmpara, scalpara, lablpara, unitpara, varipara, numbpara
    
    namepara, strgpara, minmpara, maxmpara, scalpara, lablpara, unitpara, varipara, dictpara = datapara
    
    numbpara = len(namepara)
    indxpara = arange(numbpara)
    
    # Defaults
    if truepara == None:
        truepara = array([None] * numbpara)
        
    if numbplotside == None:
        numbplotside = numbpara

    # Sampler settings
    if numbburn == None:
        numbburn = numbswep / 10
    if factthin == None:
        factthin = (numbswep - numbburn) / numbpara
    
    # sweeps to be saved
    save = zeros(numbswep, dtype=bool)
    jswep = arange(numbburn, numbswep, factthin)
    save[jswep] = True
    
    if thissamp == None:
        thissamp = rand(numbpara) 

    sampindx = zeros(numbswep, dtype=int)
    numbsamp = (numbswep - numbburn) / factthin
    sampindx[jswep] = arange(numbsamp)

    listsamp = zeros((numbsamp, numbpara)) + -1.
    listsampvarb = zeros((numbsamp, numbpara))
    listllik = zeros(numbsamp)
    
    listaccp = empty(numbswep, dtype=bool)
    listindxparamodi = empty(numbswep, dtype=int)
    
    indxpara = arange(numbpara)
        
    global j
    j = 0
    
    # initialize the chain
    thissampvarb = icdf_samp(thissamp, datapara)
    thisllik, thissampcalc = llikfunc(thissampvarb)
    numbsampcalc = len(thissampcalc)
    listsampcalc = [[] for l in range(numbsampcalc)]
    
    # current sample index
    thiscntr = -1

    # proposal scale optimization
    if optiprop:
        perditer = 5
        targpropeffi = 0.3
        perdpropeffi = 100 * numbpara
        propefficntr = zeros(numbpara)
        propefficntrtotl = zeros(numbpara)
        rollvaripara = empty((perditer, numbpara))
        optipropdone = False
        cntroptisamp = 0
        cntroptimean = 0
        thissamptemp = copy(thissamp)
        if verbtype > 0:
            print 'Optimizing proposal scale...'
    else:
        optipropdone = True
        
    while j < numbswep:
        
        if verbtype > 0:
            thiscntr = util.show_prog(j, numbswep, thiscntr)     

        if verbtype > 1:
            print
            print '-' * 10
            print 'Sweep %d' % j    
            print 'thissamp: '
            print thissamp
            print 'thissampvarb: '
            print thissampvarb
            print 'Proposing...'
            print
            
        # propose a sample
        indxparamodi = choice(indxpara)
        nextsamp = copy(thissamp)
        nextsamp[indxparamodi] = randn() * varipara[indxparamodi] + thissamp[indxparamodi]
        
        if verbtype > 1:
            print 'indxparamodi'
            print namepara[indxparamodi]
            print 'nextsamp: '
            print nextsamp

        if where((nextsamp < 0.) | (nextsamp > 1.))[0].size == 0:

            nextsampvarb = icdf_samp(nextsamp, datapara)

            if verbtype > 1:
                print 'nextsampvarb: '
                print nextsampvarb

            # evaluate the log-likelihood
            nextllik, nextsampcalc = llikfunc(nextsampvarb)
            
            accpprob = exp(nextllik - thisllik)

            if verbtype > 1:
                print 'thisllik: '
                print thisllik
                print 'nextllik: '
                print nextllik
        else:
            accpprob = 0.
            
        # accept
        if accpprob >= rand():

            if verbtype > 1:
                print 'Accepted.'

            # store utility variables
            listaccp[j] = True
            
            # update the sampler state
            thisllik = nextllik
            thissamp[indxparamodi] = nextsamp[indxparamodi]
            thissampvarb[indxparamodi] = nextsampvarb[indxparamodi]
            thissampcalc = nextsampcalc
        
        else:

            if verbtype > 1:
                print 'Rejected.'

            # store the utility variables
            listaccp[j] = False
         
        listindxparamodi[j] = indxparamodi
        
        if save[j]:
            listllik[sampindx[j]] = thisllik
            listsamp[sampindx[j], :] = thissamp
            listsampvarb[sampindx[j], :] = thissampvarb
            for l in range(numbsampcalc):
                listsampcalc[l].append(thissampcalc[l])
        
        if optipropdone:
            j += 1
        else:
            propefficntrtotl[indxparamodi] += 1.
            if listaccp[j]:
                propefficntr[indxparamodi] += 1.

            if cntroptisamp % perdpropeffi == 0 and (propefficntrtotl > 0).all():
                
                varipara *= 2**(propefficntr / propefficntrtotl / targpropeffi - 1.)
                
                propefficntr[:] = 0.
                propefficntrtotl[:] = 0.
                
                fracopti = std(rollvaripara, 0) / mean(rollvaripara, 0)
                
                if verbtype > 1:
                    print 'Proposal scale step %d' % cntroptimean
                    print 'fracopti: ', fracopti
                    
                if (fracopti < 0.2).all() and cntroptisamp >= perditer:
                    optipropdone = True
                    thissamp = thissamptemp
                    if verbtype > 1:
                        print 'Optimized variance vector: '
                        print varipara
                    
                rollvaripara[0, :] = copy(varipara)
                rollvaripara = roll(rollvaripara, 1, axis=0)
            
                cntroptimean += 1
                
            cntroptisamp += 1

    
    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listindxparamodi]

    return chan


def retr_atcr(listsamp, ndela=10):
    
    numbvarb = listsamp.shape[1]
    corr = empty(numbvarb)
    for k in range(numbvarb):
        sp.signal.correlate(listsamp[:, k], listsamp[:, k])
        
    meansgnlsqrd = mean(sgnl)**2

    atcr = empty(ndela)
    sgnllist = zeros((2, nsgnl))
    sgnllist[0,:] = sgnl
    for t in range(ndela):
        sgnllist[1,0:nsgnl-t] = sgnl[t:nsgnl]
        atcr[t] = mean(roll(sgnl, t) * sgnl) - meansgnlsqrd
        
    # normalize the autocorrelation
    vari = var(sgnl)
    atcr /= vari
         
    iact = 1. + 2. * sum(atcr[1:-1])
    return atcr, iact


def retr_numbsamp(numbswep, numbburn, factthin):
    
    numbsamp = (numbswep - numbburn) / factthin
    
    return numbsamp


def plot_gmrb(path, gmrbstat):

    numbbins = 40
    bins = linspace(1., amax(gmrbstat), numbbins + 1)
    figr, axis = plt.subplots()
    axis.hist(gmrbstat, bins=bins)
    axis.set_title('Gelman-Rubin Convergence Test')
    axis.set_xlabel('PSRF')
    axis.set_ylabel('$N_p$')
    figr.savefig(path)
    plt.close(figr)

        
def plot_propeffi(path, numbswep, numbpara, listaccp, listindxparamodi, strgpara):

    indxlistaccp = where(listaccp == True)[0]

    binstime = linspace(0., numbswep - 1., 10)
    
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
            indxlistpara = where(listindxparamodi == k)[0]
            indxlistintc = intersect1d(indxlistaccp, indxlistpara, assume_unique=True)
            histotl = axis.hist(indxlistpara, binstime, color='b')
            histaccp = axis.hist(indxlistintc, binstime, color='g')
            axis.set_title(strgpara[k])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(path)
    plt.close(figr)


def plot_trac(path, listpara, labl, truepara=None, scalpara='self', titl=None, quan=True):
    
    numbbins = 20
    
    minmpara = amin(listpara)
    maxmpara = amax(listpara)
    limspara = array([minmpara, maxmpara])
    if scalpara == 'self':
        bins = icdf_self(linspace(0., 1., numbbins + 1), minmpara, maxmpara)
    if scalpara == 'logt':
        bins = icdf_logt(linspace(0., 1., numbbins + 1), minmpara, maxmpara)
    if scalpara == 'atan':
        bins = icdf_atan(linspace(0., 1., numbbins + 1), minmpara, maxmpara)
        
    if quan:
        quanarry = sp.stats.mstats.mquantiles(listpara, prob=[0.025, 0.16, 0.84, 0.975])

    figr, axrw = plt.subplots(1, 2, figsize=(14, 7))
    if titl != None:
        figr.suptitle(titl, fontsize=18)
    for n, axis in enumerate(axrw):
        if n == 0:
            axis.plot(listpara)
            axis.set_xlabel('$i_{samp}$')
            axis.set_ylabel(labl)
            if truepara != None:
                axis.axhline(y=truepara, color='g')
            if scalpara == 'logt':
                axis.set_yscale('log')
            axis.set_ylim(limspara)
            if quan:
                axis.axhline(quanarry[0], color='b', ls='--')
                axis.axhline(quanarry[1], color='b', ls='-.')
                axis.axhline(quanarry[2], color='b', ls='-.')
                axis.axhline(quanarry[3], color='b', ls='--')
        else:
            axis.hist(listpara, bins=bins)
            axis.set_xlabel(labl)
            axis.set_ylabel('$N_{samp}$')
            if truepara != None:
                axis.axvline(truepara, color='g')
            if scalpara == 'logt':
                axis.set_xscale('log')
            axis.set_xlim(limspara)
            if quan:
                axis.axvline(quanarry[0], color='b', ls='--')
                axis.axvline(quanarry[1], color='b', ls='-.')
                axis.axvline(quanarry[2], color='b', ls='-.')
                axis.axvline(quanarry[3], color='b', ls='--')
                
    figr.subplots_adjust(top=0.9, wspace=0.4, bottom=0.2)

    if path != None:
        figr.savefig(path)
        plt.close(figr)
    else:
        plt.show()


def plot_grid(path, listsamp, strgpara, lims=None, scalpara=None,               plotsize=6, numbbins=30, numbplotside=None,               truepara=None, ntickbins=3, quan=True):
    
    numbpara = listsamp.shape[1]
    
    if numbplotside == None:
        numbplotside = numbpara
        
    if truepara == None:
        truepara = array([None] * numbpara)
        
    if scalpara == None:
        scalpara = ['self'] * numbpara
        
    if lims == None:
        lims = zeros((2, numbpara))
        lims[0, :] = amin(listsamp, 0)
        lims[1, :] = amax(listsamp, 0)
        
    indxparagood = ones(numbpara, dtype=bool)
    indxparagood[where(lims[0, :] == lims[1, :])] = False
        
    bins = zeros((numbbins + 1, numbpara))
    for k in range(numbpara):
        if scalpara[k] == 'self':
            bins[:, k] = icdf_self(linspace(0., 1., numbbins + 1), lims[0, k], lims[1, k])
        if scalpara[k] == 'logt':
            bins[:, k] = icdf_logt(linspace(0., 1., numbbins + 1), lims[0, k], lims[1, k])
        if scalpara[k] == 'atan':
            bins[:, k] = icdf_atan(linspace(0., 1., numbbins + 1), lims[0, k], lims[1, k])
            
    numbfram = numbpara // numbplotside
    numbplotsidelast = numbpara % numbplotside
    if numbplotsidelast != 0:
        numbfram += 1
        
    for n in range(numbfram):

        if n == numbfram - 1 and numbplotsidelast != 0:
            thisnumbpara = numbplotsidelast
            thislistsamp = listsamp[:, n*numbplotside:]
            thisparastrg = strgpara[n*numbplotside:]
            thisscalpara = scalpara[n*numbplotside:]
            thistruepara = truepara[n*numbplotside:]
            thisbins = bins[:, n*numbplotside:]
            thisindxparagood = indxparagood[n*numbplotside:]
            thislims = lims[:, n*numbplotside:]
            
        else:
            thisnumbpara = numbplotside
            thislistsamp = listsamp[:, n*numbplotside:(n+1)*numbplotside]
            thisparastrg = strgpara[n*numbplotside:(n+1)*numbplotside]
            thisscalpara = scalpara[n*numbplotside:(n+1)*numbplotside]
            thistruepara = truepara[n*numbplotside:(n+1)*numbplotside]
            thisbins = bins[:, n*numbplotside:(n+1)*numbplotside]
            thisindxparagood = indxparagood[n*numbplotside:(n+1)*numbplotside]
            thislims = lims[:, n*numbplotside:(n+1)*numbplotside]
            
        figr, axgr = plt.subplots(thisnumbpara, thisnumbpara, figsize=(plotsize*thisnumbpara, plotsize*thisnumbpara))
        if thisnumbpara == 1:
            axgr = [[axgr]]
        for k, axrw in enumerate(axgr):
            for l, axis in enumerate(axrw):
                if k < l or thisindxparagood[k] == False or  thisindxparagood[l] == False:
                    axis.axis('off')
                    continue
                if k == l:
                    axis.hist(thislistsamp[:, k], bins=thisbins[:, k])
                    if thistruepara[k] != None:
                        axis.axvline(thistruepara[k], color='r')
                    if quan:
                        thisquan = sp.stats.mstats.mquantiles(thislistsamp[:, k], prob=[0.025, 0.16, 0.84, 0.975])
                        axis.axvline(thisquan[0], color='b', ls='--')
                        axis.axvline(thisquan[1], color='b', ls='-.')
                        axis.axvline(thisquan[2], color='b', ls='-.')
                        axis.axvline(thisquan[3], color='b', ls='--')
                else:
                    h = axis.hist2d(thislistsamp[:, l], thislistsamp[:, k], bins=[thisbins[:, l], thisbins[:, k]], cmap='Blues')
                    if thistruepara[l] != None and thistruepara[k] != None:
                        axis.scatter(thistruepara[l], thistruepara[k], color='r', marker='o')
                    if thisscalpara[k] == 'logt':
                        axis.set_yscale('log', basey=10)
                        arry = logspace(log10(thislims[0, k]), log10(thislims[1, k]), ntickbins)
                        strgarry = [util.mexp(arry[a]) for a in range(ntickbins)]
                        axis.set_yticks(arry)
                        axis.set_yticklabels(strgarry)
                if thisscalpara[l] == 'logt':
                    axis.set_xscale('log', basex=10)
                    arry = logspace(log10(thislims[0, l]), log10(thislims[1, l]), ntickbins)
                    strgarry = [util.mexp(arry[a]) for a in range(ntickbins)]
                    axis.set_xticks(arry)
                    axis.set_xticklabels(strgarry)
                axis.set_xlim(thislims[:, l])
                if k == thisnumbpara - 1:
                    axis.set_xlabel(thisparastrg[l])  
                if l == 0 and k != 0:
                    axis.set_ylabel(thisparastrg[k])
        figr.subplots_adjust(bottom=0.2)
        if numbfram == 1:
            strg = ''
        else:
            strg = '_fram%d' % n
        if path == None:
            plt.show()
        else:
            plt.savefig(path + strg + '.png')
            plt.close(figr)
    

