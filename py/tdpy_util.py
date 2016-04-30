
# coding: utf-8

# In[ ]:

# math
from numpy import *
from numpy.random import *
from numpy.random import choice
from scipy.integrate import *
from scipy.interpolate import *
import scipy as sp

import pyfits as pf

# astro & healpix
import healpy as hp

# plotting
import matplotlib.pyplot as plt
import matplotlib

# animation
import matplotlib.animation as animation

from healpy import ang2pix

# utilities
import time, sh, os, glob

# animations
import IPython

import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)


# In[2]:

def retr_heal(nside=256):
    
    npixl = 12 * nside**2
    apix = 4. * pi / npixl # [sr]
    thhp, phhp = hp.pixelfunc.pix2ang(nside, arange(npixl), nest=False) # [rad]
    lghp = ((rad2deg(phhp) - 180.) % 360.) - 180. # [deg]
    bghp = 90. - rad2deg(thhp) # [deg]

    return lghp, bghp, nside, npixl, apix


# In[3]:

def show_prog(cntr, maxmcntr, thiscntr, nprog=20, jproc=None):

    nextcntr = int(nprog * float(cntr + 1) / maxmcntr) * 100 / nprog
    if nextcntr > thiscntr:
        if jproc != None:
            print 'Process %d is %3d%% completed.' % (jproc, nextcntr)
        else:
            print '%3d%% completed.' % nextcntr
        thiscntr = nextcntr
        
    return thiscntr            


# In[4]:

def print_mem_usage():
    mem = float(sh.awk(sh.ps('u','-p',os.getpid()),'{sum=sum+$6}; END {print sum/1024}'))
    print '%.3g MB is being used.' % mem


# In[5]:

def cart_heal(cart, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, nside=256):
    
    nbgcr = cart.shape[0]
    nlgcr = cart.shape[1]
    lghp, bghp, nside, npixl, apix = retr_heal(nside)
    heal = zeros(npixl)
    jpixl = where((minmlgal < lghp) & (lghp < maxmlgal) & (minmbgal < bghp) & (bghp < maxmbgal))[0]
    jlgcr = (nlgcr * (lghp[jpixl] - minmlgal) / (maxmlgal - minmlgal)).astype(int)
    jbgcr = (nbgcr * (bghp[jpixl] - minmbgal) / (maxmbgal - minmbgal)).astype(int)
    
    heal[jpixl] = fliplr(cart)[jbgcr, jlgcr]
    
    return heal


# In[6]:

def retr_fdfm(binsener, nside=256, vfdm=7):                    
    
    diffener = binsener[1:] - binsener[0:-1]
    nener = diffener.size
    
    path = os.environ["PNTS_TRAN_DATA_PATH"] + '/'

    npixl = nside**2 * 12
    
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
    fdfmheal = zeros((enerfdfm.size, npixl))
    for i in range(enerfdfm.size):
        fdfmheal[i, :] = cart_heal(fliplr(fluxcart[i, :, :]), nside=nside)
        
    
    fdfm = empty((nener, npixl))
    numbsampbins = 10
    enersamp = logspace(log10(amin(binsener)), log10(amax(binsener)), numbsampbins * nener)
    fdfmheal = interpolate.interp1d(enerfdfm, fdfmheal, axis=0)(enersamp)
    for i in range(nener):
        fdfm[i, :] = trapz(fdfmheal[i*numbsampbins:(i+1)*numbsampbins, :],                            enersamp[i*numbsampbins:(i+1)*numbsampbins], axis=0) / diffener[i]


    return fdfm


# In[7]:

def retr_cart(hmap, jpixlrofi=None, nsideinpt=None,               minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, reso=0.1):
    
    if jpixlrofi == None:
        npixlinpt = hmap.size
        nsideinpt = int(sqrt(npixlinpt / 12.))
    else:
        npixlinpt = nsideinpt**2 * 12
    
    deltlgcr = maxmlgal - minmlgal
    numbbinslgcr = int(deltlgcr / reso)
    
    deltbgcr = maxmbgal - minmbgal
    numbbinsbgcr = int(deltbgcr / reso)
    
    lgcr = linspace(minmlgal, maxmlgal, numbbinslgcr)
    ilgcr = arange(numbbinslgcr)
    
    bgcr = linspace(minmbgal, maxmbgal, numbbinsbgcr)
    ibgcr = arange(numbbinsbgcr)
    
    lghp, bghp, nside, npixl, apix = retr_heal(nsideinpt)

    bgcrmesh, lgcrmesh = meshgrid(bgcr, lgcr)
    
    jpixl = hp.ang2pix(nsideinpt, pi / 2. - deg2rad(bgcrmesh), deg2rad(lgcrmesh))
    
    if jpixlrofi == None:
        kpixl = jpixl
    else:
        pixlcnvt = zeros(npixlinpt, dtype=int)
        for k in range(jpixlrofi.size):
            pixlcnvt[jpixlrofi[k]] = k
        kpixl = pixlcnvt[jpixl]

    hmapcart = zeros((numbbinsbgcr, numbbinslgcr))
    hmapcart[meshgrid(ibgcr, ilgcr)] = hmap[kpixl]

    return hmapcart


# In[8]:

# temp
if False:
    get_ipython().magic(u'matplotlib inline')

    nsideinpt = 256

    lghp, bghp, nside, npixl, apix = retr_heal(nsideinpt)

    minmlgal=-20.
    maxmlgal=20.
    minmbgal=-10.
    maxmbgal=10.

    jpixlrofi = where((lghp > minmlgal) & (lghp < maxmlgal) & (bghp > minmbgal) & (bghp < maxmbgal))[0]

    hmap = exp(-lghp**2 / 100.) * exp(-bghp**2 / 10.)

    hmapcart = retr_cart(hmap[jpixlrofi], jpixlrofi=jpixlrofi, nsideinpt=nsideinpt,                          minmlgal=minmlgal, maxmlgal=maxmlgal,                          minmbgal=minmbgal, maxmbgal=maxmbgal)
    plt.imshow(hmapcart)
    plt.show()


# In[9]:

def plot_mcmc(samp, strgpara, lims=None, scalpara=None,               plotsize=6, numbbins=30, path=None, nplot=4,               truepara=None, ntickbins=3, quan=False):
    
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

            
    nfram = numbpara // nplot
    nplotlast = numbpara % nplot
    if nplotlast != 0:
        nfram += 1
        

    for n in range(nfram):

        if n == nfram - 1 and nplotlast != 0:
            thisnumbpara = nplotlast
            thissamp = samp[:, n*nplot:]
            thisparastrg = strgpara[n*nplot:]
            thisscalpara = scalpara[n*nplot:]
            thistruepara = truepara[n*nplot:]
            thisbins = bins[:, n*nplot:]
            thisjparagood = jparagood[n*nplot:]
            thislims = lims[:, n*nplot:]
            
        else:
            thisnumbpara = nplot
            thissamp = samp[:, n*nplot:(n+1)*nplot]
            thisparastrg = strgpara[n*nplot:(n+1)*nplot]
            thisscalpara = scalpara[n*nplot:(n+1)*nplot]
            thistruepara = truepara[n*nplot:(n+1)*nplot]
            thisbins = bins[:, n*nplot:(n+1)*nplot]
            thisjparagood = jparagood[n*nplot:(n+1)*nplot]
            thislims = lims[:, n*nplot:(n+1)*nplot]
            
        fig, axgr = plt.subplots(thisnumbpara, thisnumbpara, figsize=(plotsize*thisnumbpara, plotsize*thisnumbpara))
        if thisnumbpara == 1:
            axgr = [[axgr]]
        for k, axrw in enumerate(axgr):
            for l, ax in enumerate(axrw):
                if k < l or thisjparagood[k] == False or  thisjparagood[l] == False:
                    ax.axis('off')
                    continue
                if k == l:

                    ax.hist(thissamp[:, k], bins=thisbins[:, k])
                    #ax.set_yticks([])
                    if thistruepara[k] != None:
                        ax.axvline(thistruepara[k], color='r')
                    if quan:
                        thisquan = sp.stats.mstats.mquantiles(thissamp[:, k], prob=[0.025, 0.16, 0.84, 0.975])
                        ax.axvline(thisquan[0], color='b', ls='--')
                        ax.axvline(thisquan[1], color='b', ls='-.')
                        ax.axvline(thisquan[2], color='b', ls='-.')
                        ax.axvline(thisquan[3], color='b', ls='--')
    
                else:
            
                    h = ax.hist2d(thissamp[:, l], thissamp[:, k], bins=[thisbins[:, l], thisbins[:, k]], cmap='Blues')

                    if thistruepara[l] != None and thistruepara[k] != None:
                        ax.scatter(thistruepara[l], thistruepara[k], color='r', marker='o')
                    if thisscalpara[k] == 'logt':
                        ax.set_yscale('log', basey=10)
                        arry = logspace(log10(thislims[0, k]), log10(thislims[1, k]), ntickbins)
                        strgarry = [mexp(arry[a]) for a in range(ntickbins)]
                        ax.set_yticks(arry)
                        ax.set_yticklabels(strgarry)
                            
                
                if thisscalpara[l] == 'logt':
                    ax.set_xscale('log', basex=10)
                    arry = logspace(log10(thislims[0, l]), log10(thislims[1, l]), ntickbins)
                    strgarry = [mexp(arry[a]) for a in range(ntickbins)]
                    ax.set_xticks(arry)
                    ax.set_xticklabels(strgarry)
                
                ax.set_xlim(thislims[:, l])
                
                if k == thisnumbpara - 1:
                    ax.set_xlabel(thisparastrg[l])
                #else:
                #    ax.set_xticklabels([])
                    
                if l == 0 and k != 0:
                    ax.set_ylabel(thisparastrg[k])
                #else:
                #    ax.set_yticklabels([])
                
                #if ntickbins != None:
                    #ax.locator_params(ntickbins)
                
        fig.subplots_adjust(bottom=0.2)
        
        if path == None:
            plt.show()
        else:
            plt.savefig(path + '_fram%d.png' % n)
            plt.close(fig)
    

    #q = sp.stats.mstats.mquantiles(hist[0], prob=[0.68, 0.95])
    #ax.imshow(hist[0].T, origin='lower', interpolation='none', cmap='Reds', \
    #          extent=[minmtimedeca, maxmtimedeca, minmampl, maxmampl])
    #cont = ax.contour(meantimedeca, meanampl, hist[0].T, origin='lower', color='b', levels=q)
    #fmt = {}
    #strs = ['68 % CL', '95 % CL']
    #for l, s in zip(q, strs):
    #    fmt[l] = s
    #plt.clabel(cont, q, fmt=fmt, fontsize=12)
    


# In[10]:

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

    fig, axrd = plt.subplots(1, 2, figsize=(14, 7))
    if titl != None:
        fig.suptitle(titl, fontsize=18)
    for n, ax in enumerate(axrd):
        if n == 0:
            ax.plot(listpara)
            ax.set_xlabel('$i_{samp}$')
            ax.set_ylabel(labl)
            if truepara != None:
                ax.axhline(y=truepara, color='g')
            if scalpara == 'logt':
                ax.set_yscale('log')
            ax.set_ylim(limspara)
            if quan:
                ax.axhline(quanarry[0], color='b', ls='--')
                ax.axhline(quanarry[1], color='b', ls='-.')
                ax.axhline(quanarry[2], color='b', ls='-.')
                ax.axhline(quanarry[3], color='b', ls='--')
        else:
            ax.hist(listpara, bins=bins)
            ax.set_xlabel(labl)
            ax.set_ylabel('$N_{samp}$')
            if truepara != None:
                ax.axvline(truepara, color='g')
            if scalpara == 'logt':
                ax.set_xscale('log')
            ax.set_xlim(limspara)
            if quan:
                ax.axvline(quanarry[0], color='b', ls='--')
                ax.axvline(quanarry[1], color='b', ls='-.')
                ax.axvline(quanarry[2], color='b', ls='-.')
                ax.axvline(quanarry[3], color='b', ls='--')
                
    fig.subplots_adjust(top=0.9, wspace=0.4, bottom=0.2)

    if path != None:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()


# In[11]:

def plot_braz(ax, xdat, ydat, numbsampdraw=0, lcol='yellow', dcol='green', mcol='black', labl=None, alpha=None):

    if numbsampdraw > 0:
        jsampdraw = choice(arange(ydat.shape[0]), size=numbsampdraw)
        ax.plot(xdat, ydat[jsampdraw[0], :], alpha=0.1, color='b', label='Samples')
        for k in range(1, numbsampdraw):
            ax.plot(xdat, ydat[jsampdraw[k], :], alpha=0.1, color='b')
    ax.plot(xdat, percentile(ydat, 2.5, 0), color=lcol, alpha=alpha)
    ax.plot(xdat, percentile(ydat, 16., 0), color=dcol, alpha=alpha)
    ax.plot(xdat, percentile(ydat, 84., 0), color=dcol, alpha=alpha)
    ax.plot(xdat, percentile(ydat, 97.5, 0), color=lcol, alpha=alpha)
    ax.plot(xdat, percentile(ydat, 50., 0), color=mcol, label=labl, alpha=alpha)
    ax.fill_between(xdat, percentile(ydat, 2.5, 0), percentile(ydat, 97.5, 0), color=lcol, alpha=alpha)#, label='95% C.L.')
    ax.fill_between(xdat, percentile(ydat, 16., 0), percentile(ydat, 84., 0), color=dcol, alpha=alpha)#, label='68% C.L.')
    


# In[12]:

def sample(pdfvar, arrvar, numbsamp, axis=0, getcdf=False, binned=False, rej=False):
    
    if rej:
        pdfvar -= amin(pdfvar, axis=axis)
        pdfvar /= amax(pdfvar, axis=axis)
        numbsamp_ = numbsamp
        samvar = []
        while numbsamp_ > 0:
            if not binned:
                rands = rand(numbsamp_ * 2)
                maxvar = max(arrvar)
                minvar = min(arrvar)
                samvar_ = rands[0:thisnumbsamp] * (maxvar - minvar) + minvar
                jsam = where(rands[numbsamp_:numbsamp_*2] - interpolate.interp1d(arrvar, pdfvar)(samvar_) < 0)[0]
                samvar.extend(samvar_[jsamp])
                thisnumbsamp -= jsam.size
            else:
                rands = rand(numbsamp_)
                samjvar = random_integers(0, arrvar.size-1, size=thisnumbsamp)
                jsamp = where(rands < pdfvar[samjvar])[0]
                jvar = samjvar[jsam]
                samvar.extend(arrvar[jvar])
                thisnumbsamp -= jvar.size
        samvar = array(samvar)
    else:
        pdfvar /= trapz(pdfvar, arrvar)
        cdfvar = cumtrapz(pdfvar, arrvar, axis=axis, initial=0.)
        rands = rand(numbsamp)
        if not binned:
            samvar = interpolate.interp1d(cdfvar, arrvar)(rands)
        else:
            jvar = argmin(fabs(cdfvar[:,None] - rands[None,:]), axis=0)
            samvar = arrvar[jvar]
            
    if getcdf and (not rej):
        return samvar, cdfvar
    else:
        return samvar


# In[13]:

def test_sample():
    
    numbsamp = 1
    numbbins = 1000
    npara = 10
    
    # random.choice
    t0 = time.time()
    limvar = linspace(0., 100., numbbins + 1)
    arrvar = (limvar[1:] + limvar[0:-1]) / 2.
    delvar = limvar[1:] - limvar[0:-1]
    pdfvar = exp(-(arrvar - 10.)**2 / 10**2) + exp(-(arrvar - 70.)**2 / 20.**2)
    pdfvar /= sum(pdfvar * delvar)
    provar = delvar * pdfvar
    samvar0 = choice(arrvar, p=provar, size=numbsamp)
    t1 = time.time()
    print 'choice: %d seconds' % (t1 - t0)
    
    # its
    t0 = time.time()
    limvar = linspace(0.,100.,numbbins+1)
    arrvar = (limvar[1:numbbins+1] + limvar[0:numbbins]) / 2.
    pdfvar = exp(-(arrvar - 10.)**2 / 10**2) + exp(-(arrvar - 70.)**2 / 20.**2)
    samvar1, cdfvar = sample(pdfvar, arrvar, numbsamp, getcdf=True)
    t1 = time.time()
    print 'Unbinned ITS: %d seconds' % (t1 - t0)
    
    t0 = time.time()
    limvar = linspace(0.,100.,numbbins+1)
    arrvar = (limvar[1:numbbins+1] + limvar[0:numbbins]) / 2.
    pdfvar = exp(-(arrvar - 10.)**2 / 10**2) + exp(-(arrvar - 70.)**2 / 20.**2)
    samvar3 = sample(pdfvar, arrvar, numbsamp, rej=True)
    t1 = time.time()
    print 'Unbinned RS: %d seconds' % (t1 - t0)
    
    #plt.plot(arrvar, cdfvar)
    
    fig, ax = plt.subplots()
    ax.hist(samvar0, 50, alpha=0.2)   
    ax.hist(samvar1, 50, alpha=0.2)
    ax.hist(samvar2, 50, alpha=0.2) 
    ax.hist(samvar3, 50, alpha=0.2)
    ax.hist(samvar4, 50, alpha=0.2)
    plt.show()
    


# In[14]:

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


# In[15]:

def anim(fig, fname, animfunc, indxvect, args=None, fps=3, dpi=50):
    
    ani = animation.FuncAnimation(fig, animfunc, indxvect, fargs=args, blit=True)
    ani.save('/n/pan/www/tansu/' + fname, writer='imagemagick', fps=fps, dpi=dpi)
    plt.close(fig)
    path = 'http://faun.rc.fas.harvard.edu/tansu/' + fname
    IPython.display.Image(url=path)


# In[16]:

def retr_p4dm_spec(anch, part='el'):
    
    if part == 'el':
        strg = 'AtProduction_positrons'
    if part == 'ph':
        strg = 'AtProduction_gammas'

    name = os.environ["TDPY_UTIL_DATA_PATH"] + '/p4dm/' + strg + '.dat'
    p4dm = loadtxt(name)
    
    p4dm[:, 0] *= 1e3 # [MeV]
    
    mass = unique(p4dm[:, 0])
    nmass = mass.size
    nener = p4dm.shape[0] / nmass
    
    mult = zeros((nener, nmass))
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


# In[17]:

def retr_nfwp(nfwg, nside, norm=None):
    
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
    

    lghp, bghp, nside, npixl, apix = retr_heal(nside)
    
    cosigahp = cos(deg2rad(lghp)) * cos(deg2rad(bghp))
    gahp = rad2deg(arccos(cosigahp))
    
    eden = 1. / (radi / rscl)**nfwg / (1. + radi / rscl)**(3. - nfwg)
    eden *= edenlocl / interp1d(radi, eden)(radilocl)
    
    edengrid = zeros((nsadi, npixl))
    for i in range(nsadi):
        radigrid = sqrt(radilocl**2 + sadi[i]**2 - 2 * radilocl * sadi[i] * cosigahp)
        edengrid[i, :] = interp1d(radi, eden)(radigrid)


    edengridtotl = sum(edengrid**2, axis=0)

    #plt.loglog(radi, eden)
    #plt.show()
    #test = retr_cart(edengridtotl, latra=[-90., 90.], lonra=[-90.,90.])
    #plt.imshow(test, origin='lower', cmap='Reds', extent=[-90.,90.,-90.,90.], norm=matplotlib.colors.LogNorm())
    #plt.show()
    
    if norm != None:
        jgahp = argsort(gahp)
        edengridtotl /= interp1d(gahp[jgahp], edengridtotl[jgahp])(5.)
        
        
    return edengridtotl


# In[18]:

def retr_gang(lghl, bghl):
    
    gang = rad2deg(arccos(cos(deg2rad(lghl)) * cos(deg2rad(bghl))))
    
    return gang


# In[19]:

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


# In[20]:

def retr_filelist(version, nside, evclass, evtype):

    tag = os.environ["PNTS_TRAN_DATA_PATH"] + "/" + version + "/flux_*_%4.4d_evc%3.3d_evt%2.2d.fits" % (nside, evclass, evtype)
    filelist = sorted(glob.glob(tag))

    return filelist


# In[21]:

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


def plot_propeffi(plotpath, plotextn, numbswep, numbpara, listaccp, listjsampvari, strgpara):
    
    jlistaccp = where(listaccp == True)[0]

    binstime = linspace(0., numbswep - 1., 10)
    
    numbcols = 2
    numbrows = (numbpara + 1) / 2
    fig, axgr = plt.subplots(numbrows, numbcols, figsize=(16, 4 * (numbpara + 1)))
    if numbrows == 1:
        axgr = [axgr]
    for a, axrw in enumerate(axgr):
        for b, ax in enumerate(axrw):
            k = 2 * a + b
            if k == numbpara:
                ax.axis('off')
            jlistpara = where(listjsampvari == k)[0]
            jlistintc = intersect1d(jlistaccp, jlistpara, assume_unique=True)
            histotl = ax.hist(jlistpara, binstime, color='b')
            histaccp = ax.hist(jlistintc, binstime, color='g')
            ax.set_title(strgpara[k])
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(plotpath + '/propeffi' + plotextn + '.png')
    plt.close(fig)
    


    
def retr_numbsamp(numbswep, numbburn, factthin):
    
    numbsamp = (numbswep - numbburn) / factthin
    
    return numbsamp



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

