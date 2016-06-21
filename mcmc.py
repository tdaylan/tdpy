# math
from numpy import *
from numpy.random import *
from numpy.random import choice
from scipy.integrate import *
from scipy.interpolate import *
import scipy as sp
import pyfits as pf
import time, os
import copy as copytemp

import multiprocessing as mp, functools

import util


import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

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
    
    if datapara.scal[k] == 'self':
        samp = cdfn_self(sampvarb, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'logt':
        samp = cdfn_logt(sampvarb, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'atan':
        samp = cdfn_atan(sampvarb, datapara.minm[k], datapara.maxm[k])
        
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

    if datapara.scal[k] == 'self':
        sampvarb = icdf_self(samp, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'logt':
        sampvarb = icdf_logt(samp, datapara.minm[k], datapara.maxm[k])
    if datapara.scal[k] == 'atan':
        sampvarb = icdf_atan(samp, datapara.minm[k], datapara.maxm[k])
        
    return sampvarb


def gmrb_test(griddata):
    
    withvari = mean(var(griddata, 0))
    btwnvari = griddata.shape[0] * var(mean(griddata, 0))
    wgthvari = (1. - 1. / griddata.shape[0]) * withvari + btwnvari / griddata.shape[0]
    psrf = sqrt(wgthvari / withvari)

    return psrf


def init(numbproc, numbswep, llikfunc, datapara, initsamp=None, optiprop=False, pathbase='./', rtag='', numbburn=None, truepara=None, \
    numbplotside=None, factthin=None, verbtype=0, factpropeffi=2.):
   
    # construct the global object
    gdat = util.gdatstrt()
    gdat.numbproc = numbproc
    gdat.numbswep = numbswep
    gdat.llikfunc = llikfunc
    gdat.datapara = datapara
    gdat.initsamp = initsamp
    gdat.optiprop = optiprop
    gdat.pathbase = pathbase
    gdat.rtag = rtag
    gdat.numbburn = numbburn
    gdat.numbplotside = numbplotside
    gdat.factthin = factthin
    gdat.verbtype = verbtype
    gdat.factpropeffi = factpropeffi
    
    if gdat.verbtype > 1:
        print 'TDMC initialized.'
    
    gdat.numbpara = len(datapara.name)
    gdat.indxpara = arange(gdat.numbpara)
        
    # Defaults
    if truepara == None:
        truepara = array([None] * gdat.numbpara)
        
    if numbplotside == None:
        numbplotside = gdat.numbpara

    # Sampler settings
    if gdat.numbburn == None:
        gdat.numbburn = int(floor(0.1 * gdat.numbswep))
    if gdat.factthin == None:
        gdat.factthin = min(gdat.numbswep - gdat.numbburn, 4 * gdat.numbpara)
   
    gdat.indxproc = arange(gdat.numbproc)

    # sweeps to be saved
    gdat.boolsave = zeros(gdat.numbswep, dtype=bool)
    gdat.indxswepsave = arange(gdat.numbburn, gdat.numbswep, gdat.factthin)
    gdat.boolsave[gdat.indxswepsave] = True
    
    gdat.indxsampsave = zeros(gdat.numbswep, dtype=int)
    gdat.numbsamp = retr_numbsamp(gdat.numbswep, gdat.numbburn, gdat.factthin)
    gdat.indxsamp = arange(gdat.numbsamp)
    gdat.numbsamptotl = gdat.numbsamp * gdat.numbproc
    gdat.indxsamptotl = arange(gdat.numbsamptotl)
    gdat.indxsampsave[gdat.indxswepsave] = arange(gdat.numbsamp)

    # initialize the chain
    if gdat.verbtype > 1:
        print 'Forking the sampler...'
        print 'datapara'
        print datapara.indx
        print datapara.minm
        print datapara.maxm
        print datapara.name
        print datapara.scal
        print datapara.labl
        print datapara.unit
        print datapara.vari

    # get the number of auxiliary variables to be saved for each sample
    tempsamp = rand(gdat.numbpara)
    templlik, tempsampcalc = gdat.llikfunc(icdf_samp(tempsamp, gdat.datapara))
    gdat.numbsampcalc = len(tempsampcalc)
    gdat.indxsampcalc = arange(gdat.numbsampcalc)

    if numbproc == 1:
        listchan = [work(gdat, 0)]
    else:
        pool = mp.Pool(numbproc)
        workpart = functools.partial(work, gdat)
        listchan = pool.map(workpart, gdat.indxproc)
   
        pool.close()
        pool.join()

    if gdat.verbtype > 0:
        print 'Accumulating samples from all processes...'
        tim0 = time.time()

    # parse the sample chain
    listsampvarb = zeros((gdat.numbsamp, gdat.numbproc, gdat.numbpara))
    listsamp = zeros((gdat.numbsamp, gdat.numbproc, gdat.numbpara))
    listsampcalc = []
    for n in gdat.indxsampcalc:
        size = listchan[0][2][0][n].size
        listsampcalc.append(empty((gdat.numbsamp, gdat.numbproc, size)))
    listllik = zeros((gdat.numbsamp, gdat.numbproc))
    listaccp = zeros((gdat.numbswep, gdat.numbproc))
    listindxparamodi = zeros((gdat.numbswep, gdat.numbproc))

    for k in gdat.indxproc:
        listsampvarb[:, k, :] = listchan[k][0]
        listsamp[:, k, :] = listchan[k][1]
        for j in gdat.indxsamp:
            for n in gdat.indxsampcalc:
                listsampcalc[n][j, k, :] = listchan[k][2][j][n]
        listllik[:, k] = listchan[k][3]
        listaccp[:, k] = listchan[k][4]
        listindxparamodi[:, k] = listchan[k][5]

    indxlistaccp = where(listaccp == True)[0]
    propeffi = zeros(gdat.numbpara)
    for k in gdat.indxpara:
        indxlistpara = where(listindxparamodi == k)[0]
        indxlistintc = intersect1d(indxlistaccp, indxlistpara, assume_unique=True)
        if indxlistpara.size != 0:
            propeffi[k] = float(indxlistintc.size) / indxlistpara.size    
    
    minmlistllik = amin(listllik)
    levi = -log(mean(1. / exp(listllik - minmlistllik))) + minmlistllik
    info = mean(listllik) - levi
    
    gmrbstat = zeros(gdat.numbpara)
    if numbproc > 1:
        if gdat.verbtype > 1:
            print 'Performing Gelman-Rubin convergence test...'
            tim0 = time.time()
        for k in gdat.indxpara:
            gmrbstat[k] = gmrb_test(listsampvarb[:, :, k])
        if gdat.verbtype > 1:
            timefinl = time.time()
            print 'Done in %.3g seconds' % (timefinl - timeinit)

    listsampvarb = listsampvarb.reshape((gdat.numbsamptotl, gdat.numbpara))
    listsamp = listsamp.reshape((gdat.numbsamptotl, gdat.numbpara))
    for n in gdat.indxsampcalc:
        listsampcalc[n] = listsampcalc[n].reshape((gdat.numbsamptotl, -1))
    listllik = listllik.flatten()
    listaccp = listaccp.flatten()
    listindxparamodi = listindxparamodi.flatten()

    gdat.pathplot = gdat.pathbase + '/imag/%s/' % gdat.rtag
    os.system('mkdir -p %s' % gdat.pathplot)

    if gdat.verbtype > 1:
        print 'Making plots...'
        timeinit = time.time()
    
    path = gdat.pathplot
    plot_propeffi(path, gdat.numbswep, gdat.numbpara, listaccp, listindxparamodi, gdat.datapara.strg)

    path = gdat.pathplot + 'llik'
    plot_trac(path, listllik, '$P(D|y)$', titl='log P(D) = %.3g' % levi)
    
    if numbplotside != 0:
        path = gdat.pathplot
        plot_grid(path, listsampvarb, gdat.datapara.strg, truepara=gdat.datapara.true, scalpara=gdat.datapara.scal, numbplotside=numbplotside)
        
    for k in gdat.indxpara:
        path = gdat.pathplot + gdat.datapara.name[k]
        plot_trac(path, listsampvarb[:, k], gdat.datapara.strg[k], scalpara=gdat.datapara.scal[k], truepara=gdat.datapara.true[k])
        
    if gdat.numbproc > 1 and gdat.numbsamp > 1:
        path = gdat.pathplot + 'gmrb'
        plot_gmrb(path, gmrbstat)
            
    if gdat.verbtype > 1:
        timefinl = time.time()
        print 'Done in %.3g seconds' % (timefinl - timeinit)

    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listindxparamodi, propeffi, levi, info, gmrbstat]
    
    return chan


def work(gdat, indxprocwork):
    
    # re-seed the random number generator for the process
    seed()
    
    listsampvarb = zeros((gdat.numbsamp, gdat.numbpara))
    listsamp = zeros((gdat.numbsamp, gdat.numbpara)) + -1.
    listsampcalc = []
    listllik = zeros(gdat.numbsamp)
    listaccp = empty(gdat.numbswep, dtype=bool)
    listindxparamodi = empty(gdat.numbswep, dtype=int)
    
    if gdat.initsamp == None:
        thissamp = rand(gdat.numbpara)
    else:
        thissamp = copy(gdat.initsamp[indxprocwork, :])

    thissampvarb = icdf_samp(thissamp, gdat.datapara)
    thisllik, thissampcalc = gdat.llikfunc(thissampvarb)
    
    if gdat.verbtype > 1:
        print 'Process %d' % indxprocwork
        print 'thissamp'
        print thissamp
        print 'thissampvarb'
        print thissampvarb
        print

    datapara = copytemp.deepcopy(gdat.datapara)
    
    # proposal scale optimization
    pathvaripara = gdat.pathbase + '/varipara_' + gdat.rtag + '.fits'
    if gdat.optiprop:
        if not os.path.isfile(pathvaripara): 
            if gdat.verbtype > 0 and indxprocwork == 0:
                print 'Optimizing proposal scale...'
            targpropeffi = 0.25
            minmpropeffi = targpropeffi / gdat.factpropeffi
            maxmpropeffi = targpropeffi * gdat.factpropeffi
            perdpropeffi = 400 * gdat.numbpara
            cntrprop = zeros(gdat.numbpara)
            cntrproptotl = zeros(gdat.numbpara)
            gdat.optipropdone = False
            cntroptisamp = 0
            cntroptimean = 0
            thissamptemp = copy(thissamp)
        else:
            if gdat.verbtype > 0 and indxprocwork == 0:
                print 'Retrieving the optimal proposal scale from %s...' % pathvaripara
            gdat.optipropdone = True
            varipara = pf.getdata(pathvaripara)
    else:
        if gdat.verbtype > 0 and indxprocwork == 0:
            print 'Skipping proposal scale optimization...'
        gdat.optipropdone = True

    cntrprog = -1
    cntrswep = 0
    while cntrswep < gdat.numbswep:
        
        if gdat.verbtype > 0:
            cntrprog = util.show_prog(cntrswep, gdat.numbswep, cntrprog, indxprocwork=indxprocwork) 

        if gdat.verbtype > 1:
            print
            print '-' * 10
            print 'Sweep %d' % cntrswep
            print 'thissamp: '
            print thissamp
            print 'thissampvarb: '
            print thissampvarb
            print 'Proposing...'
            print
            
        # propose a sample
        indxparamodi = choice(gdat.indxpara)
        nextsamp = copy(thissamp)
        nextsamp[indxparamodi] = randn() * gdat.datapara.vari[indxparamodi] + thissamp[indxparamodi]
        
        if gdat.verbtype > 1:
            print 'indxparamodi'
            print datapara.name[indxparamodi]
            print 'nextsamp: '
            print nextsamp

        if where((nextsamp < 0.) | (nextsamp > 1.))[0].size == 0:

            nextsampvarb = icdf_samp(nextsamp, gdat.datapara)

            if gdat.verbtype > 1:
                print 'nextsampvarb: '
                print nextsampvarb

            # evaluate the log-likelihood
            nextllik, nextsampcalc = gdat.llikfunc(nextsampvarb)
            accpprob = exp(nextllik - thisllik)

            if gdat.verbtype > 1:
                print 'thisllik: '
                print thisllik
                print 'nextllik: '
                print nextllik
        else:
            accpprob = 0.
            
        # accept
        if accpprob >= rand():

            if gdat.verbtype > 1:
                print 'Accepted.'

            # store utility variables
            listaccp[cntrswep] = True
            
            # update the sampler state
            thisllik = nextllik
            thissamp[indxparamodi] = nextsamp[indxparamodi]
            thissampvarb[indxparamodi] = nextsampvarb[indxparamodi]
            thissampcalc = nextsampcalc
        
        else:

            if gdat.verbtype > 1:
                print 'Rejected.'

            # store the utility variables
            listaccp[cntrswep] = False
         
        listindxparamodi[cntrswep] = indxparamodi
        
        if gdat.boolsave[cntrswep]:
            listllik[gdat.indxsampsave[cntrswep]] = copy(thisllik)
            listsamp[gdat.indxsampsave[cntrswep], :] = copy(thissamp)
            listsampvarb[gdat.indxsampsave[cntrswep], :] = copy(thissampvarb)
            listsampcalc.append(copy(thissampcalc))
        
        if gdat.optipropdone:
            cntrswep += 1
        else:
            cntrproptotl[indxparamodi] += 1.
            if listaccp[cntrswep]:
                cntrprop[indxparamodi] += 1.

            if cntroptisamp % perdpropeffi == 0 and (cntrproptotl > 0).all():
                
                thispropeffi = cntrprop / cntrproptotl
                print 'Proposal scale optimization step %d' % cntroptimean
                print 'Current proposal efficiency'
                print thispropeffi
                print 'Mean of the current proposal efficiency'
                print mean(thispropeffi)
                print 'Standard deviation of the current proposal efficiency'
                print std(thispropeffi)
                if (thispropeffi > minmpropeffi).all() and (thispropeffi < maxmpropeffi).all():
                    print 'Optimized variance: '
                    print datapara.vari
                    print 'Writing the optimized variance to %s...' % pathvaripara
                    gdat.optipropdone = True
                    thissamp = thissamptemp
                    pf.writeto(pathvaripara, datapara.vari, clobber=True)
                else:
                    factcorr = 2**(thispropeffi / targpropeffi - 1.)
                    gdat.datapara.vari *= factcorr
                    cntrprop[:] = 0.
                    cntrproptotl[:] = 0.
                    print 'Current sample'
                    print thissampvarb
                    print 'Correction factor'
                    print factcorr
                    print 'Current variance: '
                    print datapara.vari
                    print
                cntroptimean += 1
            cntroptisamp += 1
    
    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listindxparamodi]

    return chan


def retr_atcr(listsamp, numbdela=10):
    
    numbparatemp = listsamp.shape[1]
    
    # mean square signal
    meansqrd = mean(listsamp, axis=0)**2

    # autocorrelation
    atcr = empty((numbdela, numbparatemp))
    for t in range(numbdela):
        atcr[t, :] = mean(roll(listsamp, t, axis=0) * listsamp - meansqrd[None, :], axis=0)
        
    # normalize the autocorrelation
    atcr /= amax(atcr, axis=0)
         
    return atcr


def retr_numbsamp(numbswep, numbburn, factthin):
    
    numbsamp = int((numbswep - numbburn) / factthin)
    
    return numbsamp


def plot_gmrb(path, gmrbstat):

    numbbins = 40
    bins = linspace(1., amax(gmrbstat), numbbins + 1)
    figr, axis = plt.subplots()
    axis.hist(gmrbstat, bins=bins)
    axis.set_title('Gelman-Rubin Convergence Test')
    axis.set_xlabel('PSRF')
    axis.set_ylabel('$N_p$')
    figr.savefig(path + '_gmrb.pdf')
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
    plt.savefig(path + 'propeffi.pdf')
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
        figr.savefig(path + '_trac.pdf')
        plt.close(figr)
    else:
        plt.show()


def plot_grid(path, listsamp, strgpara, lims=None, scalpara=None, plotsize=6, numbbins=20, numbplotside=None, truepara=None, numbtickbins=3, quan=True):
    
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
                        arry = logspace(log10(thislims[0, k]), log10(thislims[1, k]), numbtickbins)
                        strgarry = [util.mexp(arry[a]) for a in range(numbtickbins)]
                        axis.set_yticks(arry)
                        axis.set_yticklabels(strgarry)
                if thisscalpara[l] == 'logt':
                    axis.set_xscale('log', basex=10)
                    arry = logspace(log10(thislims[0, l]), log10(thislims[1, l]), numbtickbins)
                    strgarry = [util.mexp(arry[a]) for a in range(numbtickbins)]
                    axis.set_xticks(arry)
                    axis.set_xticklabels(strgarry)
                axis.set_xlim(thislims[:, l])
                if k == thisnumbpara - 1:
                    axis.set_xlabel(thisparastrg[l])  
                if l == 0 and k != 0:
                    axis.set_ylabel(thisparastrg[k])
        figr.subplots_adjust(bottom=0.2)
        strg = 'grid'
        if numbfram != 1:
            strg += '%04d' % n
        if path == None:
            plt.show()
        else:
            plt.savefig(path + strg + '.pdf')
            plt.close(figr)
    

