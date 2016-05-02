
# coding: utf-8

# In[ ]:

# math
from numpy import *
from numpy.random import *
from numpy.random import choice
from scipy.integrate import *
from scipy.interpolate import *
import scipy as sp

# plotting
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)


# In[ ]:

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
    
    if datapara[3][k] == 'self':
        samp = cdfn_self(sampvarb, datapara[1][k], datapara[2][k])
    if datapara[3][k] == 'logt':
        samp = cdfn_logt(sampvarb, datapara[1][k], datapara[2][k])

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

    if datapara[3][k] == 'self':
        sampvarb = icdf_self(samp, datapara[1][k], datapara[2][k])
    if datapara[3][k] == 'logt':
        sampvarb = icdf_logt(samp, datapara[1][k], datapara[2][k])
            
    return sampvarb



# In[ ]:

def mcmc(numbswep, llikfunc, datapara, thissamp=None, optiprop=False,          plotpath=None, plotextn='', numbburn=None, factthin=None, verbtype=0):
    
    global strgpara, minmpara, maxmpara, scalpara, lablpara, unitpara, varipara, numbpara
    
    strgpara, minmpara, maxmpara, scalpara, lablpara, unitpara, varipara, dictpara = datapara
    
    numbpara = len(strgpara)
    
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
    listjsampvari = empty(numbswep, dtype=int)
    
    isamp = arange(numbpara)
        
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
            thiscntr = show_prog(j, numbswep, thiscntr)     

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
        jsampvari = choice(isamp)
        nextsamp = copy(thissamp)
        nextsamp[jsampvari] = randn() * varipara[jsampvari] + thissamp[jsampvari]
        
        if verbtype > 1:
            print 'jsampvari'
            print strgpara[jsampvari]
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
            thissamp[jsampvari] = nextsamp[jsampvari]
            thissampvarb[jsampvari] = nextsampvarb[jsampvari]
            thissampcalc = nextsampcalc
        
        else:

            if verbtype > 1:
                print 'Rejected.'

            # store the utility variables
            listaccp[j] = False
         
        listjsampvari[j] = jsampvari
        
        if save[j]:
            listllik[sampindx[j]] = thisllik
            listsamp[sampindx[j], :] = thissamp
            listsampvarb[sampindx[j], :] = thissampvarb
            for l in range(numbsampcalc):
                listsampcalc[l].append(thissampcalc[l])
        
            
        if optipropdone:
            j += 1
        else:
            propefficntrtotl[jsampvari] += 1.
            if listaccp[j]:
                propefficntr[jsampvari] += 1.

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

    jlistaccp = where(listaccp == True)[0]
    propeffi = zeros(numbpara)
    for k in range(numbpara):
        jlistpara = where(listjsampvari == k)[0]
        jlistintc = intersect1d(jlistaccp, jlistpara, assume_unique=True)
        if jlistpara.size != 0:
            propeffi[k] = float(jlistintc.size) / jlistpara.size    
    
    minmlistllik = amin(listllik)
    levi = -log(mean(1. / exp(listllik - minmlistllik))) + minmlistllik
    info = mean(listllik) - levi
    
    if plotpath != None:
        plot_propeffi(plotpath, plotextn, numbswep, numbpara, listaccp, listjsampvari, strgpara)

        path = plotpath + 'llik' + plotextn + '.png'
        plot_trac(listllik, '$P(D|y)$', path=path)


    sampbund = [listsampvarb, listsamp, listsampcalc,                 listllik, listaccp, listjsampvari, propeffi, levi, info]

    return sampbund


# In[ ]:

def plot_trac(listpara, labl, truepara=None, scalpara='self', path=None, titl=None, quan=False):
    
    numbbins = 20
    
    minmpara = amin(listpara)
    maxmpara = amax(listpara)
    limspara = array([minmpara, maxmpara])
    if scalpara == 'self':
        bins = icdf_self(linspace(0., 1., numbbins + 1), minmpara, maxmpara)
    if scalpara == 'logt':
        bins = icdf_logt(linspace(0., 1., numbbins + 1), minmpara, maxmpara)

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


def plot_propeffi(plotpath, plotextn, numbswep, numbpara, listaccp, listjsampvari, strgpara):
    
    jlistaccp = where(listaccp == True)[0]

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
            jlistpara = where(listjsampvari == k)[0]
            jlistintc = intersect1d(jlistaccp, jlistpara, assume_unique=True)
            histotl = axis.hist(jlistpara, binstime, color='b')
            histaccp = axis.hist(jlistintc, binstime, color='g')
            axis.set_title(strgpara[k])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(plotpath + '/propeffi' + plotextn + '.png')
    plt.close(figr)
    
    
def retr_atcr(sgnl, ndela=10):
    
    nsgnl = sgnl.size
    
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


def plot_mcmc(samp, strgpara, lims=None, scalpara=None,               plotsize=6, numbbins=30, path=None, numbplot=4,               truepara=None, ntickbins=3, quan=False):
    
    numbpara = samp.shape[1]
    
    if truepara == None:
        truepara = array([None] * numbpara)
        
    if scalpara == None:
        scalpara = ['self'] * numbpara
        
    if lims == None:
        lims = zeros((2, numbpara))
        lims[0, :] = amin(samp, 0)
        lims[1, :] = amax(samp, 0)
        
    jparagood = ones(numbpara, dtype=bool)
    jparagood[where(lims[0, :] == lims[1, :])] = False
        
    bins = zeros((numbbins, numbpara))
    for k in range(numbpara):
        if scalpara[k] == 'self':
            bins[:, k] = linspace(lims[0, k], lims[1, k], numbbins)
        if scalpara[k] == 'logt':
            bins[:, k] = logspace(log10(lims[0, k]), log10(lims[1, k]), numbbins)

            
    numbfram = numbpara // numbplot
    numbplotlast = numbpara % numbplot
    if numbplotlast != 0:
        numbfram += 1
        

    for n in range(numbfram):

        if n == numbfram - 1 and numbplotlast != 0:
            thisnumbpara = numbplotlast
            thissamp = samp[:, n*numbplot:]
            thisparastrg = strgpara[n*numbplot:]
            thisscalpara = scalpara[n*numbplot:]
            thistruepara = truepara[n*numbplot:]
            thisbins = bins[:, n*numbplot:]
            thisjparagood = jparagood[n*numbplot:]
            thislims = lims[:, n*numbplot:]
            
        else:
            thisnumbpara = numbplot
            thissamp = samp[:, n*numbplot:(n+1)*numbplot]
            thisparastrg = strgpara[n*numbplot:(n+1)*numbplot]
            thisscalpara = scalpara[n*numbplot:(n+1)*numbplot]
            thistruepara = truepara[n*numbplot:(n+1)*numbplot]
            thisbins = bins[:, n*numbplot:(n+1)*numbplot]
            thisjparagood = jparagood[n*numbplot:(n+1)*numbplot]
            thislims = lims[:, n*numbplot:(n+1)*numbplot]
            
        figr, axgr = plt.subplots(thisnumbpara, thisnumbpara, figsize=(plotsize*thisnumbpara, plotsize*thisnumbpara))
        if thisnumbpara == 1:
            axgr = [[axgr]]
        for k, axrw in enumerate(axgr):
            for l, axis in enumerate(axrw):
                if k < l or thisjparagood[k] == False or  thisjparagood[l] == False:
                    axis.axis('off')
                    continue
                if k == l:

                    axis.hist(thissamp[:, k], bins=thisbins[:, k])
                    #axis.set_yticks([])
                    if thistruepara[k] != None:
                        axis.axvline(thistruepara[k], color='r')
                    if quan:
                        thisquan = sp.stats.mstats.mquantiles(thissamp[:, k], prob=[0.025, 0.16, 0.84, 0.975])
                        axis.axvline(thisquan[0], color='b', ls='--')
                        axis.axvline(thisquan[1], color='b', ls='-.')
                        axis.axvline(thisquan[2], color='b', ls='-.')
                        axis.axvline(thisquan[3], color='b', ls='--')
    
                else:
            
                    h = axis.hist2d(thissamp[:, l], thissamp[:, k], bins=[thisbins[:, l], thisbins[:, k]], cmap='Blues')

                    if thistruepara[l] != None and thistruepara[k] != None:
                        axis.scatter(thistruepara[l], thistruepara[k], color='r', marker='o')
                    if thisscalpara[k] == 'logt':
                        axis.set_yscale('log', basey=10)
                        arry = logspace(log10(thislims[0, k]), log10(thislims[1, k]), ntickbins)
                        strgarry = [mexp(arry[a]) for a in range(ntickbins)]
                        axis.set_yticks(arry)
                        axis.set_yticklabels(strgarry)
                            
                
                if thisscalpara[l] == 'logt':
                    axis.set_xscale('log', basex=10)
                    arry = logspace(log10(thislims[0, l]), log10(thislims[1, l]), ntickbins)
                    strgarry = [mexp(arry[a]) for a in range(ntickbins)]
                    axis.set_xticks(arry)
                    axis.set_xticklabels(strgarry)
                
                axis.set_xlim(thislims[:, l])
                
                if k == thisnumbpara - 1:
                    axis.set_xlabel(thisparastrg[l])
                #else:
                #    axis.set_xticklabels([])
                    
                if l == 0 and k != 0:
                    axis.set_ylabel(thisparastrg[k])
                #else:
                #    axis.set_yticklabels([])
                
                #if ntickbins != None:
                    #axis.locator_params(ntickbins)
                
        figr.subplots_adjust(bottom=0.2)
        
        if path == None:
            plt.show()
        else:
            plt.savefig(path + '_fram%d.png' % n)
            plt.close(figr)
    

    #q = sp.stats.mstats.mquantiles(hist[0], prob=[0.68, 0.95])
    #axis.imshow(hist[0].T, origin='lower', interpolation='none', cmap='Reds', \
    #          extent=[minmtimedeca, maxmtimedeca, minmampl, maxmampl])
    #cont = axis.contour(meantimedeca, meanampl, hist[0].T, origin='lower', color='b', levels=q)
    #fmt = {}
    #strs = ['68 % CL', '95 % CL']
    #for l, s in zip(q, strs):
    #    fmt[l] = s
    #plt.clabel(cont, q, fmt=fmt, fontsize=12)
    

