from __init__ import *
import util

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


def init(llikfunc, datapara, numbproc=1, numbswep=1000, initsamp=None, optiprop=True, loadchan=False, loadvaripara=True, fracrand=0., \
                            gdatextr=None, pathdata=None, pathimag=None, rtag='', numbburn=None, truepara=None, numbbinsplot=20, \
                            numbplotside=None, factthin=None, verbtype=1, factpropeffi=1.2):
  
    timeinit = time.time()

    if pathimag == None:
        try:
            pathimag = os.environ["TDPY_DATA_PATH"] + '/'
        except:
            pathimag = './'
        pathimag += 'imag/'
    
    if pathdata == None:
        try:
            pathdata = os.environ["TDPY_DATA_PATH"] + '/'
        except:
            pathdata = './'
        pathdata += 'data/'
    
    # construct the global object
    gdat = util.gdatstrt()
    gdat.numbproc = numbproc
    gdat.numbswep = numbswep
    gdat.llikfunc = llikfunc
    gdat.datapara = datapara
    gdat.initsamp = initsamp
    gdat.optiprop = optiprop
    gdat.pathimag = pathimag
    gdat.pathdata = pathdata
    gdat.rtag = rtag
    gdat.numbburn = numbburn
    gdat.numbplotside = numbplotside
    gdat.factthin = factthin
    gdat.verbtype = verbtype
    gdat.factpropeffi = factpropeffi
    gdat.gdatextr = gdatextr
    gdat.loadvaripara = loadvaripara
    gdat.fracrand = fracrand

    if gdat.verbtype > 0:
        print 'TDMC initialized.'
    
    if gdat.rtag == '':
        strg = 'tdmcchan.p'
    else:
        strg = 'tdmcchan_%s.p' % gdat.rtag
    pathchan = gdat.pathdata + strg
    if os.path.isfile(pathchan) and loadchan:
        if gdat.verbtype > 0:
            print 'Reading the previously computed chain from %s...' % pathchan
        fobj = open(pathchan, 'rb')
        chan = cPickle.load(fobj)
        fobj.close()
        return chan
        
    gdat.numbpara = len(datapara.name)
    gdat.indxpara = arange(gdat.numbpara)
        
    # Defaults
    if truepara == None:
        truepara = array([None] * gdat.numbpara)
        
    if numbplotside == None:
        numbplotside = min(gdat.numbpara, 4)

    # Sampler settings
    if gdat.numbburn == None:
        if gdat.optiprop:
            gdat.numbburn = 0
        else:
            gdat.numbburn = int(floor(0.1 * gdat.numbswep))
    if gdat.factthin == None:
        gdat.factthin = min(gdat.numbswep - gdat.numbburn, 1000 * gdat.numbpara)
   
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
        timeinit = time.time()

    # parse the sample chain
    listsampvarb = zeros((gdat.numbsamp, gdat.numbproc, gdat.numbpara))
    listsamp = zeros((gdat.numbsamp, gdat.numbproc, gdat.numbpara))
    listsampcalc = []
    gdat.numbsampcalc = len(listchan[0][2][0])
    gdat.indxsampcalc = arange(gdat.numbsampcalc)
    for n in gdat.indxsampcalc:
        size = listchan[0][2][0][n].size
        listsampcalc.append(empty((gdat.numbsamp, gdat.numbproc, size)))
    listllik = zeros((gdat.numbsamp, gdat.numbproc))
    listaccp = zeros((gdat.numbswep, gdat.numbproc))
    listchro = zeros((gdat.numbswep, gdat.numbproc))
    listindxparamodi = zeros((gdat.numbswep, gdat.numbproc))

    for k in gdat.indxproc:
        listsampvarb[:, k, :] = listchan[k][0]
        listsamp[:, k, :] = listchan[k][1]
        for j in gdat.indxsamp:
            for n in gdat.indxsampcalc:
                listsampcalc[n][j, k, :] = listchan[k][2][j][n]
        listllik[:, k] = listchan[k][3]
        listaccp[:, k] = listchan[k][4]
        listchro[:, k] = listchan[k][5]
        listindxparamodi[:, k] = listchan[k][6]

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
    
    gdat.strgtimestmp = util.retr_strgtimestmp()

    gdat.pathimag += '%s_%s/' % (gdat.strgtimestmp, gdat.rtag)
       
    os.system('mkdir -p %s' % gdat.pathimag)

    gmrbstat = zeros(gdat.numbpara)
    if gdat.numbsamp > 1:
    
        if gdat.verbtype > 1:
            print 'Calculating autocorrelation...'
            timeinit = time.time()
        atcr, timeatcr = retr_timeatcr(listsamp, atcrtype='maxm')
        if gdat.verbtype > 1:
            timefinl = time.time()
            print 'Done in %.3g seconds' % (timefinl - timeinit)
        path = gdat.pathimag
        plot_atcr(path, atcr, timeatcr)
    
        if gdat.numbproc > 1:
            if gdat.verbtype > 1:
                print 'Performing Gelman-Rubin convergence test...'
                timeinit = time.time()
            for k in gdat.indxpara:
                gmrbstat[k] = gmrb_test(listsampvarb[:, :, k])
            if gdat.verbtype > 1:
                timefinl = time.time()
                print 'Done in %.3g seconds' % (timefinl - timeinit)
            path = gdat.pathimag
            plot_gmrb(path, gmrbstat)

    listsampvarb = listsampvarb.reshape((gdat.numbsamptotl, gdat.numbpara))
    listsamp = listsamp.reshape((gdat.numbsamptotl, gdat.numbpara))
    for n in gdat.indxsampcalc:
        listsampcalc[n] = listsampcalc[n].reshape((gdat.numbsamptotl, -1))
    listllik = listllik.flatten()
    listaccp = listaccp.flatten()
    listchro = listchro.flatten()
    listindxparamodi = listindxparamodi.flatten()

    if gdat.verbtype > 1:
        print 'Making plots...'
        timeinit = time.time()
    
    # make plots
    ## proposal efficiency
    path = gdat.pathimag
    plot_propeffi(path, gdat.numbswep, gdat.numbpara, listaccp, listindxparamodi, gdat.datapara.strg)

    ## processing time per sample
    figr, axis = plt.subplots()
    binstime = logspace(log10(amin(listchro * 1e3)), log10(amax(listchro * 1e3)), 50)
    axis.hist(listchro * 1e3, binstime, log=True)
    axis.set_ylabel('$N_{samp}$')
    axis.set_xlabel('$t$ [ms]')
    axis.set_xscale('log')
    axis.set_xlim([amin(binstime), amax(binstime)])
    axis.set_ylim([0.5, None])
    plt.tight_layout()
    path = gdat.pathimag + 'chro.pdf'
    figr.savefig(gdat.pathimag + 'chro.pdf')
    plt.close(figr)

    ## likelihood
    path = gdat.pathimag + 'llik'
    plot_trac(path, listllik, '$P(D|y)$', titl='log P(D) = %.3g' % levi)
    
    if numbplotside != 0:
        path = gdat.pathimag + 'fixp'
        plot_grid(path, listsampvarb, gdat.datapara.strg, truepara=gdat.datapara.true, scalpara=gdat.datapara.scal, numbplotside=numbplotside, numbbinsplot=numbbinsplot)
        
    for k in gdat.indxpara:
        path = gdat.pathimag + gdat.datapara.name[k]
        plot_trac(path, listsampvarb[:, k], gdat.datapara.strg[k], scalpara=gdat.datapara.scal[k], truepara=gdat.datapara.true[k], numbbinsplot=numbbinsplot)
        
    if gdat.verbtype > 1:
        timefinl = time.time()
        print 'Done in %.3g seconds' % (timefinl - timeinit)

    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listchro, listindxparamodi, propeffi, levi, info, gmrbstat]
    
    # save the chain if the run was long enough
    timefinl = time.time()
    # temp
    if timefinl - timeinit > 1.:
        if not os.path.isfile(pathchan):
            print 'Writing the chain to %s...' % pathchan
            fobj = open(pathchan, 'wb')
            cPickle.dump(chan, fobj, protocol=cPickle.HIGHEST_PROTOCOL)
            fobj.close()
        
    return chan


def work(gdat, indxprocwork):
    
    # re-seed the random number generator for the process
    seed()
    
    listsampvarb = zeros((gdat.numbsamp, gdat.numbpara))
    listsamp = zeros((gdat.numbsamp, gdat.numbpara)) + -1.
    listsampcalc = []
    listllik = zeros(gdat.numbsamp)
    listaccp = empty(gdat.numbswep, dtype=bool)
    listchro = empty(gdat.numbswep)
    listindxparamodi = empty(gdat.numbswep, dtype=int)
    
    gdat.cntrswep = 0
    
    if gdat.initsamp == None:
        thissamp = rand(gdat.numbpara)
    else:
        thissamp = copy(gdat.initsamp[indxprocwork, :])

    thissampvarb = icdf_samp(thissamp, gdat.datapara)
    thisllik, thissampcalc = gdat.llikfunc(thissampvarb, gdat.gdatextr)
    
    if gdat.verbtype > 1:
        print 'Process %d' % indxprocwork
        print 'thissamp'
        print thissamp
        print 'thissampvarb'
        print thissampvarb
        print

    varipara = copy(gdat.datapara.vari)
    
    # proposal scale optimization
    if gdat.optiprop:
        if gdat.rtag == '':
            strg = 'tdmcvaripara.fits'
        else:
            strg = 'tdmcvaripara_%s.fits' % gdat.rtag
        pathvaripara = gdat.pathdata + strg
        if os.path.isfile(pathvaripara) and gdat.loadvaripara:
            if gdat.verbtype > 0 and indxprocwork == 0:
                print 'Reading the previously computed proposal scale from %s...' % pathvaripara
            gdat.optipropdone = True
            varipara = pf.getdata(pathvaripara)
        else:
            if gdat.verbtype > 0 and indxprocwork == 0:
                print 'Optimizing proposal scale...'
            targpropeffi = 0.25
            minmpropeffi = targpropeffi / gdat.factpropeffi
            maxmpropeffi = targpropeffi * gdat.factpropeffi
            perdpropeffi = 4000 * gdat.numbpara
            cntrprop = zeros(gdat.numbpara)
            cntrproptotl = zeros(gdat.numbpara)
            gdat.optipropdone = False
            cntroptisamp = 0
            cntroptimean = 0
    else:
        if gdat.verbtype > 0 and indxprocwork == 0:
            print 'Skipping proposal scale optimization...'
        gdat.optipropdone = True

    cntrprog = -1
    while gdat.cntrswep < gdat.numbswep:
        
        timeinit = time.time()

        if gdat.verbtype > 0:
            cntrprog = util.show_prog(gdat.cntrswep, gdat.numbswep, cntrprog, indxprocwork=indxprocwork) 

        if gdat.verbtype > 1:
            print
            print '-' * 10
            print 'Sweep %d' % gdat.cntrswep
            print 'thissamp: '
            print thissamp
            print 'thissampvarb: '
            print thissampvarb
            print 'Proposing...'
            print
            
        # propose a sample
        indxparamodi = choice(gdat.indxpara)
        nextsamp = copy(thissamp)
        if gdat.fracrand > rand():
            nextsamp[indxparamodi] = rand()
        else:
            nextsamp[indxparamodi] = randn() * varipara[indxparamodi] + thissamp[indxparamodi]
        
        if gdat.verbtype > 1:
            print 'indxparamodi'
            print gdat.datapara.name[indxparamodi]
            print 'nextsamp: '
            print nextsamp

        if where((nextsamp < 0.) | (nextsamp > 1.))[0].size == 0:

            nextsampvarb = icdf_samp(nextsamp, gdat.datapara)

            if gdat.verbtype > 1:
                print 'nextsampvarb: '
                print nextsampvarb

            # evaluate the log-likelihood
            nextllik, nextsampcalc = gdat.llikfunc(nextsampvarb, gdat.gdatextr)
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
            listaccp[gdat.cntrswep] = True
            
            # update the sampler state
            thisllik = nextllik
            thissamp[indxparamodi] = nextsamp[indxparamodi]
            thissampvarb[indxparamodi] = nextsampvarb[indxparamodi]
            thissampcalc = nextsampcalc
        
        else:

            if gdat.verbtype > 1:
                print 'Rejected.'

            # store the utility variables
            listaccp[gdat.cntrswep] = False
         
        listindxparamodi[gdat.cntrswep] = indxparamodi
        
        if gdat.boolsave[gdat.cntrswep]:
            listllik[gdat.indxsampsave[gdat.cntrswep]] = copy(thisllik)
            listsamp[gdat.indxsampsave[gdat.cntrswep], :] = copy(thissamp)
            listsampvarb[gdat.indxsampsave[gdat.cntrswep], :] = copy(thissampvarb)
            listsampcalc.append(copy(thissampcalc))
        
        timefinl = time.time()

        listchro[gdat.cntrswep] = timefinl - timeinit
        
        if gdat.optipropdone:
            gdat.cntrswep += 1
        else:

            cntrproptotl[indxparamodi] += 1.
            if listaccp[gdat.cntrswep]:
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
                    print varipara
                    print 'Writing the optimized variance to %s...' % pathvaripara
                    gdat.optipropdone = True
                    pf.writeto(pathvaripara, varipara, clobber=True)
                else:
                    factcorr = 2**(thispropeffi / targpropeffi - 1.)
                    varipara *= factcorr
                    cntrprop[:] = 0.
                    cntrproptotl[:] = 0.
                    print 'Current sample'
                    print thissampvarb
                    print 'Correction factor'
                    print factcorr
                    print 'Current variance: '
                    print varipara
                    print
                cntroptimean += 1
            cntroptisamp += 1
            
    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listchro, listindxparamodi]

    return chan


def retr_atcr_neww(listsamp):

    numbsamp = listsamp.shape[0]
    four = fft.fft(listsamp - mean(listsamp, axis=0), axis=0)
    atcr = fft.ifft(four * conjugate(four), axis=0).real
    atcr /= amax(atcr, 0)
    
    return atcr[:numbsamp/2, ...]


def retr_timeatcr(listsamp, verbtype=1, atcrtype='maxm'):

    numbsamp = listsamp.shape[0]
    listsamp = listsamp.reshape((numbsamp, -1))
    numbpara = listsamp.shape[1]

    boolfail = False
    if listsamp.shape[0] == 1:
        boolfail = True

    atcr = retr_atcr_neww(listsamp)
    indxatcr = where(atcr > 0.2)
     
    if indxatcr[0].size == 0:
        boolfail = True
        timeatcr = 0
    else:
        if atcrtype == 'nomi':
            timeatcr = argmax(indxatcr[0], axis=0)
        if atcrtype == 'maxm':
            indx = argmax(indxatcr[0])
            indxtimemaxm = indxatcr[0][indx]
            indxparamaxm = indxatcr[1][indx]
            atcr = atcr[:, indxparamaxm]
            timeatcr = indxtimemaxm
   
    if boolfail:
        print 'Autocorrelation time could not be estimated.'
        if atcrtype == 'maxm':
            return zeros((1, 1)), 0.
        else:
            return zeros((1, numbpara)), 0.
    else:
        return atcr, timeatcr


def retr_numbsamp(numbswep, numbburn, factthin):
    
    numbsamp = int((numbswep - numbburn) / factthin)
    
    return numbsamp


def plot_gmrb(path, gmrbstat):

    numbbinsplot = 40
    bins = linspace(1., amax(gmrbstat), numbbinsplot + 1)
    figr, axis = plt.subplots()
    axis.hist(gmrbstat, bins=bins)
    axis.set_title('Gelman-Rubin Convergence Test')
    axis.set_xlabel('PSRF')
    axis.set_ylabel('$N_p$')
    figr.savefig(path + 'gmrb.pdf')
    plt.close(figr)


def plot_atcr(path, atcr, timeatcr, strgextn=''):

    numbsampatcr = atcr.size
    
    figr, axis = plt.subplots(figsize=(6, 6))
    axis.plot(arange(numbsampatcr), atcr)
    axis.set_xlabel(r'$\tau$')
    axis.set_ylabel(r'$\xi(\tau)$')
    axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % timeatcr, ha='center', va='center', transform=axis.transAxes)
    axis.axhline(0., ls='--', alpha=0.5)
    plt.tight_layout()
    pathplot = path + 'atcr%s.pdf' % strgextn
    figr.savefig(pathplot)
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
    figr.subplots_adjust(hspace=0.3)
    figr.savefig(path + 'propeffi.pdf')
    plt.close(figr)


def plot_trac(path, listpara, labl, truepara=None, scalpara='self', titl=None, quan=True, varbdraw=None, labldraw=None, numbbinsplot=20, logthist=False, colrdraw=None):
    
    if not isfinite(listpara).all():
        print 'plot_trac encountered infinite input. Returning...'
        print 'path'
        print path
        return
    
    if not isfinite(listpara).all():
        print 'labl'
        print labl
        print 'listpara'
        print listpara
        summgene(listpara)
        raise Exception('')
    
    if listpara.size == 0:
        return

    maxmpara = amax(listpara)
    if scalpara == 'logt':
        minmpara = amin(listpara[where(listpara > 0.)])
        bins = icdf_logt(linspace(0., 1., numbbinsplot + 1), minmpara, maxmpara)
    else:
        minmpara = amin(listpara)
        bins = icdf_self(linspace(0., 1., numbbinsplot + 1), minmpara, maxmpara)
    limspara = array([minmpara, maxmpara])
        
    if quan:
        quanarry = sp.stats.mstats.mquantiles(listpara, prob=[0.025, 0.16, 0.84, 0.975])

    if scalpara == 'logt':
        numbtick = 5
        listtick = logspace(log10(minmpara), log10(maxmpara), numbtick)
        listlabltick = ['%.3g' % tick for tick in listtick]
    
    figr, axrw = plt.subplots(1, 2, figsize=(14, 7))
    if titl != None:
        figr.suptitle(titl, fontsize=18)
    for n, axis in enumerate(axrw):
        if n == 0:
            axis.plot(listpara, lw=0.5)
            axis.set_xlabel('$i_{samp}$')
            axis.set_ylabel(labl)
            if truepara != None and not isnan(truepara):
                axis.axhline(y=truepara, color='g', lw=4)
            if scalpara == 'logt':
                axis.set_yscale('log')
                axis.set_yticks(listtick)
                axis.set_yticklabels(listlabltick)
            axis.set_ylim(limspara)
            if varbdraw != None:
                for k in range(len(varbdraw)):
                    axis.axhline(varbdraw[k], label=labldraw[k], color=colrdraw[k], lw=3)
            if quan:
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
            if truepara != None and not isnan(truepara):
                axis.axvline(truepara, color='g', lw=4)
            if scalpara == 'logt':
                axis.set_xscale('log')
                axis.set_xticks(listtick)
                axis.set_xticklabels(listlabltick)
            axis.set_xlim(limspara)
            if varbdraw != None:
                for k in range(len(varbdraw)):
                    axis.axvline(varbdraw[k], label=labldraw[k], color=colrdraw[k], lw=3)
            if quan:
                axis.axvline(quanarry[0], color='b', ls='--', lw=2)
                axis.axvline(quanarry[1], color='b', ls='-.', lw=2)
                axis.axvline(quanarry[2], color='b', ls='-.', lw=2)
                axis.axvline(quanarry[3], color='b', ls='--', lw=2)
                
    figr.subplots_adjust(top=0.9, wspace=0.4, bottom=0.2)

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
    if titl != None:
        axis.set_title(titl)
    plt.tight_layout()
    figr.savefig(path + '.pdf')
    plt.close(figr)


def plot_hist(path, listvarb, strg, titl=None, numbbins=20, truepara=None, quan=True, scalpara='self', varbdraw=None, labldraw=None, colrdraw=None):

    minmvarb = amin(listvarb)
    maxmvarb = amax(listvarb)
    if scalpara == 'logt':
        bins = icdf_logt(linspace(0., 1., numbbins + 1), minmvarb, maxmvarb)
    else:
        bins = icdf_self(linspace(0., 1., numbbins + 1), minmvarb, maxmvarb)
    figr, axis = plt.subplots(figsize=(6, 6))
    axis.hist(listvarb, bins=bins)
    axis.set_ylabel(r'$N_{samp}$')
    axis.set_xlabel(strg)
    if truepara != None:
        axis.axvline(truepara, color='g', lw=4)
    if varbdraw != None:
        for k in range(len(varbdraw)):
            axis.axvline(varbdraw[k], label=labldraw[k], color=colrdraw[k], lw=3)
    if quan:
        quanarry = sp.stats.mstats.mquantiles(listvarb, prob=[0.025, 0.16, 0.84, 0.975])
        axis.axvline(quanarry[0], color='b', ls='--', lw=2)
        axis.axvline(quanarry[1], color='b', ls='-.', lw=2)
        axis.axvline(quanarry[2], color='b', ls='-.', lw=2)
        axis.axvline(quanarry[3], color='b', ls='--', lw=2)
    if titl != None:
        axis.set_title(titl)
    plt.tight_layout()
    figr.savefig(path + '_hist.pdf')
    plt.close(figr)


def plot_grid(path, listsamp, strgpara, join=False, lims=None, scalpara=None, plotsize=6, numbplotside=6, \
                                                truepara=None, numbtickbins=3, numbbinsplot=20, quan=True, varbdraw=None):

    numbpara = listsamp.shape[1]
    
    if numbpara != 2 and join:
        raise Exception('Joint probability density can only be plotted for two parameters.')

    if join:
        numbplotside = 1

    if join:
        numbfram = 1
        numbplotsidelast = 1
    else:
        numbfram = numbpara // numbplotside
        numbplotsidelast = numbpara % numbplotside
        if numbplotsidelast != 0:
            numbfram += 1
      
    if truepara == None:
        truepara = zeros(numbpara) + nan
        
    if scalpara == None:
        scalpara = ['self'] * numbpara
        
    if lims == None:
        lims = zeros((2, numbpara))
        lims[0, :] = amin(listsamp, 0)
        lims[1, :] = amax(listsamp, 0)
        for k in range(numbpara):
            if truepara[k] != None:
                if isfinite(truepara[k]):
                    lims[0, k] = min(lims[0, k], truepara[k])
                    lims[1, k] = max(lims[1, k], truepara[k])

            if lims[0, k] == lims[1, k]:
                lims[0, k] /= 2.
                lims[1, k] *= 2.
    
    for k in range(numbpara):
        if lims[0, k] == 0 and scalpara[k] == 'logt':
            print 'Increasing the lower limit of the plotting region because the variable is log-stretched and has a minimum of 0...'
            lims[0, k] = amin(listsamp[where(listsamp[:, k] > 0.), k])
    
    indxparagood = ones(numbpara, dtype=bool)
    indxparagood[where(lims[0, :] == lims[1, :])] = False
    
    bins = zeros((numbbinsplot + 1, numbpara))
    for k in range(numbpara):
        if scalpara[k] == 'self':
            bins[:, k] = icdf_self(linspace(0., 1., numbbinsplot + 1), lims[0, k], lims[1, k])
        if scalpara[k] == 'logt':
            bins[:, k] = icdf_logt(linspace(0., 1., numbbinsplot + 1), lims[0, k], lims[1, k])
        if scalpara[k] == 'atan':
            bins[:, k] = icdf_atan(linspace(0., 1., numbbinsplot + 1), lims[0, k], lims[1, k])
         
    for n in range(numbfram):
        
        if n == numbfram - 1 and numbplotsidelast != 0:
            thisnumbpara = numbplotsidelast
            thislistsamp = listsamp[:, n*numbplotside:]
            thisstrgpara = strgpara[n*numbplotside:]
            thisscalpara = scalpara[n*numbplotside:]
            thistruepara = truepara[n*numbplotside:]
            if varbdraw != None:
                thisvarbdraw = varbdraw[n*numbplotside:]
            thisbins = bins[:, n*numbplotside:]
            thisindxparagood = indxparagood[n*numbplotside:]
            thislims = lims[:, n*numbplotside:]
            
        else:
            thisnumbpara = numbplotside
            thislistsamp = listsamp[:, n*numbplotside:(n+1)*numbplotside]
            thisstrgpara = strgpara[n*numbplotside:(n+1)*numbplotside]
            thisscalpara = scalpara[n*numbplotside:(n+1)*numbplotside]
            thistruepara = truepara[n*numbplotside:(n+1)*numbplotside]
            if varbdraw != None:
                thisvarbdraw = varbdraw[n*numbplotside:(n+1)*numbplotside]
            thisbins = bins[:, n*numbplotside:(n+1)*numbplotside]
            thisindxparagood = indxparagood[n*numbplotside:(n+1)*numbplotside]
            thislims = lims[:, n*numbplotside:(n+1)*numbplotside]
    
        figr, axgr = plt.subplots(thisnumbpara, thisnumbpara, figsize=(plotsize*thisnumbpara, plotsize*thisnumbpara))
        if thisnumbpara == 1:
            axgr = [[axgr]]
        for k, axrw in enumerate(axgr):
            for l, axis in enumerate(axrw):

                if k < l or thisindxparagood[k] == False or thisindxparagood[l] == False:
                    axis.axis('off')
                    continue

                if k == l and not join:
                    try:
                        axis.hist(thislistsamp[:, k], bins=thisbins[:, k])
                    except:
                        pass
                    if thistruepara[k] != None and not isnan(thistruepara[k]):
                        axis.axvline(thistruepara[k], color='g', lw=4)
                    # draw the provided reference values
                    if varbdraw != None:
                        axis.axvline(thisvarbdraw[k], color='r', lw=3)
                    if quan:
                        thisquan = sp.stats.mstats.mquantiles(thislistsamp[:, k], prob=[0.025, 0.16, 0.84, 0.975])
                        axis.axvline(thisquan[0], color='b', ls='--', lw=2)
                        axis.axvline(thisquan[1], color='b', ls='-.', lw=2)
                        axis.axvline(thisquan[2], color='b', ls='-.', lw=2)
                        axis.axvline(thisquan[3], color='b', ls='--', lw=2)
                    
                else:
                    if join:
                        k = 0
                        l = 1

                    try:
                        hist = histogram2d(thislistsamp[:, l], thislistsamp[:, k], bins=[thisbins[:, l], thisbins[:, k]])[0]
                        axis.pcolor(thisbins[:, l], thisbins[:, k], hist.T, cmap='Blues')
                    except:
                        pass
                    axis.set_xlim([amin(thisbins[:, l]), amax(thisbins[:, l])])
                    axis.set_ylim([amin(thisbins[:, k]), amax(thisbins[:, k])])
                    if thistruepara[l] != None and not isnan(thistruepara[l]) and thistruepara[k] != None and not isnan(thistruepara[k]):
                        axis.scatter(thistruepara[l], thistruepara[k], color='g', marker='x', s=500)
                    # draw the provided reference values
                    if varbdraw != None:
                        axis.scatter(thisvarbdraw[l], thisvarbdraw[k], color='r', marker='x', s=350)
                    if thisscalpara[k] == 'logt':
                        axis.set_yscale('log', basey=10)
                        arry = logspace(log10(thislims[0, k]), log10(thislims[1, k]), numbtickbins)
                        strgarry = [util.mexp(arry[a]) for a in range(numbtickbins)]
                        axis.set_yticks(arry)
                        axis.set_yticklabels(strgarry)
                    
                if thisscalpara[l] == 'logt':
                    axis.set_xscale('log', basex=10)
                    arry = logspace(log10(thislims[0, l]), log10(thislims[1, l]), numbtickbins)
                    if not isfinite(arry).all():
                        print 'l'
                        print l
                        print 'listsamp'
                        print listsamp
                        util.summgene(listsamp)
                        print 'strgpara'
                        print strgpara
                        print 'scalpara'
                        print scalpara
                        print 'lims'
                        print lims
                        print 'thislims'
                        print thislims
                        print 'arry'
                        print arry
                        raise Exception('')
                    strgarry = [util.mexp(arry[a]) for a in range(numbtickbins)]
                    axis.set_xticks(arry)
                    axis.set_xticklabels(strgarry)
                axis.set_xlim(thislims[:, l])
                if k == thisnumbpara - 1:
                    axis.set_xlabel(thisstrgpara[l])
                else:
                    axis.set_xticklabels([])
                if l == 0 and k != 0 or join:
                    axis.set_ylabel(thisstrgpara[k])
                else:
                    if k != 0:
                        axis.set_yticklabels([])
        figr.tight_layout()
        # temp
        #figr.subplots_adjust(bottom=0.2)
        if join:
            strg = '_join'
        else:
            strg = '_grid'
            if numbfram != 1:
                strg += '%04d' % n
        figr.savefig(path + strg + '.pdf')
        plt.close(figr)
    
