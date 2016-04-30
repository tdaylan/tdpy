
# coding: utf-8

# In[4]:

# math
from numpy import *
from numpy.random import *
#from numpy.random import choice


from scipy import integrate
from scipy import interpolate

# astro & healpix
import healpy as hp

# plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(context='poster', color_codes=True)

# animation
import matplotlib.animation as animation

# utilities
import time, sh, os

# tdpy
import tdpy_util


# ## RJ/DR MCMC Sampler

# In[5]:

def rjmc(nswep, nburn, nthin, initsamp, llikfunc, lprifunc,                propfunc, jpropfunc, verbtype=0,                updtfunc=None, delafunc=None,                propstdv=0.02, propbias=0.14,                fracdrmc=0., nstag=1, heavy=False, fracrand=0.05, maxmsampsize=None):
        
    nsamp = (nswep - nburn) / nthin

    # sweeps to be saved
    save = zeros(nswep, dtype=bool)
    iswep = arange(nswep)
    jswep = arange(nburn, nswep, nthin)
    
    jsave = intersect1d(iswep, jswep, assume_unique=True)
    save[jsave] = True
    sampindx = zeros(nswep, dtype=int)
    sampindx[jswep] = arange(nsamp)

    nchro = 5
    if maxmsampsize == None:
        maxmsampsize = initsamp.size
        
    listsamp = zeros((nsamp, maxmsampsize)) + -1.
    listncalc = zeros(nswep)
    listjprop = zeros(nswep)
    listchro = zeros((nswep, nchro))
    listltpr = zeros((nswep, nstag + 1, nstag + 1))
    listacpr = zeros((nswep, nstag + 1, nstag + 1))
    listthis = zeros((nsamp, maxmsampsize)) + -1.
    listllik = zeros(nswep)
    listlpri = zeros(nswep)
    listaccp = zeros(nswep, dtype=bool)
    
    drmcltpr = zeros((nstag + 1, nstag + 1))
    drmcacpr = zeros((nstag + 1, nstag + 1))
    drmclpos = zeros(nstag + 1)
    drmcsamp = zeros((maxmsampsize, nstag + 1))
    
    # initialize the chain
    drmcsamp[:, 0] = initsamp
    thisllik = llikfunc(initsamp, None)
    thislpri = lprifunc(initsamp, None)
    drmclpos[0] = thisllik + thislpri
    thissamp = initsamp

    
    # current sample index
    j = 0
    
    thiscntr = -1
    ncalc = 0
    j = 0
    while j < nswep:
        
        if verbtype > 1:
            print
            print
            print '-' * 10
            print 'Sweep %d' % j
            tdpy_util.print_mem_usage()
            
        ts = time.time()
        
        # choose a proposal type
        jprop = jpropfunc(drmcsamp[:, 0])
        listjprop[j] = jprop
        if verbtype > 1:
            print 'jprop: ', jprop
                
        # log the progress
        if verbtype > 0:
            thiscntr = tdpy_util.show_prog(j, nswep, thiscntr)
                
        # decide whether to delay rejection
        if rand() < fracdrmc and nstag > 1 and jprop < 4:
            drmc = True
            thispropbias = propbias
        else:
            drmc = False
            thispropbias = 0. 
    
        n = 0
        while n < nstag:
                
            if verbtype > 1:        
                print
                print '-----'
                if drmc:
                    print 'Stage %d' % n
                print 'Proposing...'
                print

            # propose a sample
            t0 = time.time()
            pprofrac, auxiprdffrac, jcbn, modijsamp, reje = propfunc(drmcsamp[:, 0:n+2],                                                                jprop, propstdv, thispropbias, heavy, fracrand)
            listchro[j, 0] = time.time() - t0
            
            if reje:
                break
             
            # evaluate the log-prior
            t0 = time.time()
            nextlpri = lprifunc(drmcsamp[:, n+1], jprop)
            listchro[j, 1] = time.time() - t0

            # evaluate the log-likelihood
            t0 = time.time()
            nextllik = llikfunc(drmcsamp[:, n+1], jprop)
            listchro[j, 2] = time.time() - t0               

            if verbtype > 1:
                print 'nextllik: ', nextllik

            ncalc += 1
            drmclpos[n+1] = nextllik + nextlpri

            # determine the acceptance probability
            t0 = time.time()
            if drmc:
                drmc_acpr(n, drmcsamp, drmcacpr, drmcltpr, drmclpos,                           propstdv, thispropbias, heavy, fracrand, verbtype)
            else:
                drmcacpr[0,1] = exp(drmclpos[1] - drmclpos[0]) * pprofrac * auxiprdffrac * jcbn

            listchro[j, 3] = time.time() - t0
                    
            if verbtype > 1:
                print 'drmcsamp: '
                print drmcsamp
                print 'nextlpri: ', nextlpri
                print 'nextlpos: ', drmclpos[n + 1]
                print 'thislpos: ', drmclpos[0]
                    
            n += 1
            
            # accept
            if drmcacpr[0, n] >= rand():
                
                # update the current state
                drmclpos[0] = drmclpos[n]
                if modijsamp.size > 0:
                    drmcsamp[modijsamp, 0] = drmcsamp[modijsamp, n]
                
                # save the state
                listltpr[j,:,:] = drmcltpr
                listacpr[j,:,:] = drmcacpr
                listncalc[j] = ncalc
                listllik[j] = nextllik
                listlpri[j] = nextlpri
                listaccp[j] = True
                if save[j]:
                    listsamp[sampindx[j], :] = drmcsamp[:, 0]
                
                if verbtype > 1:
                    print 'Accepted in stage number %d.' % n
                    
                # call the update routine, if any
                if updtfunc != None:
                    updtfunc(jprop, )
                    
                listchro[j, 4] = time.time() - ts
                
                j += 1
                
            # delay
            elif drmc and (n < nstag):

                # call the delay routine, if any
                if delafunc != None:
                    delafunc(jprop)
                    
                if verbtype > 1:
                    print 'Delaying rejection!'
                
            # reject
            else:
                
                # save the state
                listltpr[j,:,:] = drmcltpr
                listacpr[j,:,:] = drmcacpr
                listncalc[j] = ncalc
                listllik[j] = nextllik
                listlpri[j] = nextlpri
                listaccp[j] = False
                if save[j]:
                    listsamp[sampindx[j], :] = drmcsamp[:, 0]
                    
                if verbtype > 1:
                    print 'Rejected in stage number %d.' % n
                        
                listchro[j, 4] = time.time() - ts
                
                j += 1
    
    if verbtype > 1:
        print 'listsamp: '
        print listsamp
            
    grid = [listsamp, listncalc, listjprop, listchro, listltpr, listacpr, listthis, listllik, listlpri, listaccp]
    
    return grid



def drmc_acpr(n, drmcsamp, drmcacpr, drmcltpr, drmclpos, propstdv, propbias, heavy, fracrand, verbtypef):
    
    if verbtype > 2:
        print 'Evaluating forward transition probabilities'
        
    l = n + 1
    
    for k in range(n+1):
        if verbtype > 2:
            print 'k,l: ', k, l
    
        if abs(k - l) == 1:
    
            drmcltpr[k,l] = sum(retr_gaus(drmcsamp[:,k:l+1], jsamp, propstdv=propstdv, propbias=propbias, heavy=heavy, fracrand=fracrand, getltpr=True))
            if verbtype > 2:
                print 'bold proposal'
                print 'drmcltpr[k,l]: ', drmcltpr[k,l]
        else:
    
            drmcltpr[k,l] = sum(retr_gaus(drmcsamp[:,k:l+1], jsamp, propstdv=propstdv, propbias=propbias, heavy=heavy, fracrand=fracrand, getltpr=True))
            if verbtype > 2:
                print 'timid proposal'
                print 'drmcltpr[k,l]: ', drmcltpr[k,l]
    
        
    if verbtype > 2:
        print
        print 'Evaluating reverse transition probabilities'
        
    k = n + 1
    for l in range(n+1):
        if verbtype > 2:
            print 'k,l: ', k, l
        if abs(k - l) == 1:
            drmcltpr[k,l] = sum(retr_gaus(fliplr(drmcsamp[:,l:k+1]), jsamp, propstdv=propstdv, propbias=propbias, heavy=heavy, fracrand=fracrand, getltpr=True))
            if verbtype > 2:
                print 'bold proposal'
                print 'drmcltpr[k,l]: ', drmcltpr[k,l]
        else:
            drmcltpr[k,l] = sum(retr_gaus(fliplr(drmcsamp[:,l:k+1]), jsamp, propstdv=propstdv, propbias=propbias, heavy=heavy, fracrand=fracrand, getltpr=True))
            if verbtype > 2:
                print 'timid proposal'
                print 'drmcltpr[k,l]: ', drmcltpr[k,l]
                
    if verbtype > 2:
        print 'drmcltpr: '
        print drmcltpr
        
    p = n + 1
    for m in range(p)[::-1]:
        
        if verbtype > 2:
            print
            print 'Constructing drmcacpr[%d,%d]... ' % (m,p)
        
        drmcacpr[m,p] = drmclpos[p] - drmclpos[m]
    
        if verbtype > 2:
            print 'Taking posterior difference...'
            print 'drmcacpr[m,p]: ', drmcacpr[m,p]
            print 
            print 'Traversing forward transition probabilities...'
                             
        for r in range(m,p)[::-1]:
            k = r
            l = p
            drmcacpr[m,p] -= drmcltpr[k,l]
            if verbtype > 2:
                print 'Subtracting drmcltpr[%d,%d]: ' % (k, l), drmcltpr[k,l]
                
        if verbtype > 2:
            print 
            print 'Traversing reverse transition probabilities...'
        
        for r in range(m,p):
            k = p
            l = r
            drmcacpr[m,p] += drmcltpr[k,l]
            if verbtype > 2:
                print 'Adding drmcltpr[%d,%d]: ' % (k, l), drmcltpr[k,l]
        if verbtype > 2:
            print 'drmcacpr[%d,%d]: ' % (m, p), drmcacpr[m,p]
        
        
        if verbtype > 2:
            print
            print 'Exponentiating...'
        drmcacpr[m,p] = exp(drmcacpr[m,p])
        if verbtype > 2:
            print 'drmcacpr[%d,%d]: ' % (m, p), drmcacpr[m,p]
            
        if verbtype > 2:
            print
            print 'Traversing failure probabilities...'
        for k in range(p-m-1):
            drmcacpr[m,p] *= (1. - drmcacpr[p,m+1]) / (1. - drmcacpr[m,p-1])
            if verbtype > 2:
                print 'Multiplying by 1 - drmcacpr[%d,%d] = ' % (p,m+1), drmcacpr[p,m+1]
                print 'Dividing by 1 - drmcacpr[%d,%d] = ' % (m,p-1), drmcacpr[m,p-1]
                
        drmcacpr[p,m] = min(1., 1. / drmcacpr[m,p])
        drmcacpr[m,p] = min(1., drmcacpr[m,p])
        if verbtype > 2:
            print 'drmcacpr[%d,%d]: ' % (m, p), drmcacpr[m,p]
            print 'drmcacpr[%d,%d]: ' % (p, m), drmcacpr[p,m]


# ## Gaussian proposal

# In[6]:

def retr_gaus(samp, jsamp, propstdv=0.02, propbias=0., heavy=False, fracrand=0.05, getltpr=False):
    
    shape = samp.shape
    size = jsamp.size
    lenshape = len(shape)
    
    # MCMC
    if lenshape == 1:
        propsamp = samp[jsamp]

    # DRMCMC
    if lenshape == 2:
        if samp.shape[1] > 2:
            propsamp = samp[jsamp, 1]
        else:
            propsamp = samp[jsamp, 0]

    #print
    #print 'samp:'
    #print samp
    #print 'propsamp: '
    #print propsamp
    
    # sample proposal
    if not getltpr:
        step = normal(scale=propstdv, size=size)
        nextsamp = propsamp + step
        if heavy:
            jrand = where(rand(size) < fracrand)[0]
            nextsamp[jrand] = rand(jrand.size)
            
        if lenshape == 2:
            if samp.shape[1] == 2 and (not heavy):
                randvect = sort(rand(size - 1)) * propbias**2
                randvect = concatenate((array([0.]), randvect, array([propbias**2])))
                randvect = sqrt((roll(randvect, -1) - randvect)[0:size]) * choice([-1.,1.], size=size)
                nextsamp += randvect
            samp[jsamp, -1] = nextsamp % 1.
            
        #print 'nextsamp: '
        #print nextsamp
    
    # determine where the pdf will be evaluated
    else:
        evalsamp = samp[jsamp,shape[1]-1]
        
        # evaluate the pdf
        if samp.shape[1] == 2:
            ltpr = log(1. / 2. / sqrt(2. * pi) / propstdv *             (exp(-((propsamp - evalsamp - propbias) / propstdv)**2 / 2.) + exp(-((propsamp - evalsamp + propbias) / propstdv)**2 / 2.)))
        else:
            ltpr = -((propsamp - evalsamp) / propstdv)**2 / 2. - log(sqrt(2. * pi) * propstdv)
            if heavy:
                ltpr = log(exp(ltpr) * (1. - fracrand) + fracrand)

        #print 'evalsamp: ', evalsamp
    
    if getltpr:
        return ltpr
    elif lenshape == 1:
        return nextsamp
    


# In[7]:

def plot_rand_gaus():
    
    ndata = 100
    binsdata = linspace(0., 1., ndata + 1)
    meandata = (roll(binsdata, -1) + binsdata)[0:ndata] / 2.
    diffdata = (roll(binsdata, -1) - binsdata)[0:ndata]
    datavect = empty(ndata)
    
    nsamp = 10000
    sampvect = empty(nsamp)
    
    jsamp = arange(1)
    
    fig, ax = plt.subplots()
    fig.suptitle(r'HTMCMC proposal, $q_H(\theta_0,\theta_1)$')
    
    drmcsamp = zeros((1, 2))
    drmcsamp[0, 0] = 0.5
    for i in range(nsamp):
        retr_gaus(drmcsamp, jsamp, propstdv=0.02, heavy=True)
        sampvect[i] = drmcsamp[0, 1]
    hist = ax.hist(sampvect, binsdata, log=True)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_title(r'$\theta_0$ = 0.5')
    
    for i in range(ndata):
        drmcsamp[:,1] = meandata[i]
        datavect[i] = exp(retr_gaus(drmcsamp, jsamp, propstdv=0.02, heavy=True, fracrand=0.05, getltpr=True))
    ax.plot(meandata, datavect * nsamp * diffdata, 'r')
    fig.subplots_adjust(hspace=.5, top=0.9)
    ax.set_ylim([0.1,None])
    plt.show()


# In[8]:

def plot_dela_gaus():
    
    ndata = 100
    binsdata = linspace(0., 1., ndata + 1)
    meandata = (roll(binsdata, -1) + binsdata)[0:ndata] / 2.
    diffdata = (roll(binsdata, -1) - binsdata)[0:ndata]
    datavect = empty(ndata)
    
    nsamp = 10000
    sampvect = empty(nsamp)
    
    jsamp = arange(1)
    
    # DRMCMC
    for i in range(nsamp):
        retr_gaus(drmcsamp, jsamp, propstdv=0.02, propbias=0.20)
        sampvect[i] = drmcsamp[0, 1]
    randsamp = sampvect[0]
    
    fig, ax = plt.subplots()
    fig.suptitle(r'DRMCMC bold proposal, $q_B(\theta_i;\theta_{i+1})$')
    hist = ax.hist(sampvect, binsdata, log=True)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_title(r'$\theta_0$ = 0.5')

    for i in range(ndata):
        drmcsamp[:,1] = meandata[i]
        datavect[i] = exp(retr_gaus(drmcsamp, jsamp, propstdv=0.02, propbias=0.20, getltpr=True))
        
    ax.plot(meandata, datavect * nsamp * diffdata, 'r')
    fig.subplots_adjust(hspace=.5, top=0.9)
    ax.set_ylim([0.1,None])
    plt.show()
    
    drmcsamp = zeros((1, 3))
    drmcsamp[:,0] = 0.5
    drmcsamp[:,1] = randsamp
    for i in range(nsamp):
        retr_gaus(drmcsamp, jsamp, propstdv=0.02, propbias=0.20)
        sampvect[i] = drmcsamp[0, 2]
    
    fig, ax = plt.subplots()
    fig.suptitle(r'DRMCMC timid proposal, $q_T(\theta_i,\theta_{i+1},...,\theta_j;\theta_{j+1})$')
    hist = ax.hist(sampvect, binsdata, log=True)
    ax.set_xlabel(r'$\theta_2$')
    ax.set_title(r'$\theta_0$ = 0.5, $\theta_1$ = %.3g' % randsamp)

    for i in range(ndata):
        drmcsamp[:,2] = meandata[i]
        datavect[i] = exp(retr_gaus(drmcsamp, jsamp, propstdv=0.02, propbias=0.20, getltpr=True))
        
    ax.plot(meandata, datavect * nsamp * diffdata, 'r')
    fig.subplots_adjust(hspace=.5, top=0.9)
    ax.set_ylim([0.1,None])
    plt.show()
    
    # four random realizations
    nstag = 4
    fig, axgrd = plt.subplots(2, 2, sharex='all',sharey='all', figsize=(16,10))
    fig.suptitle(r'Random realizations of DRMCMC proposals, $\theta_0$ = 0.5, $\sigma$ = 0.02, $\mu$ = 0.20', fontsize=16)

    lines = []
    labels = []
    npara = 1
    jsamp = arange(npara)
    drmcsamp = zeros((npara, nstag+1))
    drmcsamp[:,0] = 0.5
    for a, axrow in enumerate(axgrd):
        for b, ax in enumerate(axrow):
            for n in range(nstag):
                retr_gaus(drmcsamp[:,0:n+2], jsamp, propstdv=0.02, propbias=0.20)
            title = ''
            for n in range(nstag+1):
                if n != 0:
                    title += r'$\theta_{%d} = %.3g$' % (n, drmcsamp[:,n])
                    if n != nstag:
                        title += ', '
                hist = ax.hist(drmcsamp[:,n], binsdata, log=True)
                if a == b == 0:
                    lines.append(hist[2][0])
                    if n == 0:
                        labels.append('Initial')
                    labels.append('Stage %d' % n)
            ax.set_xlabel(r'$\theta_i$')
            ax.set_title(title)
    ax.set_ylim([0.1,None])
    fig.subplots_adjust(hspace=0.2, wspace=.5, top=0.9)
    fig.legend(lines, labels, 'center')
    plt.show()
    

