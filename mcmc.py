from __init__ import *
import util
from util import summgene

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


#def init(llikfunc, datapara, numbproc=1, numbswep=1000, initsamp=None, optiprop=True, loadchan=False, loadvaripara=True, fracrand=0., \
#                            gdatextr=None, pathdata=None, pathimag=None, rtag='', numbburn=None, truepara=None, numbbinsplot=20, \
#                            numbplotside=None, factthin=None, verbtype=1, factpropeffi=1.2):
#  
#    timeinit = time.time()
#
#    if pathimag is None:
#        try:
#            pathimag = os.environ["TDPY_DATA_PATH"] + '/'
#        except:
#            pathimag = './'
#        pathimag += 'imag/'
#    
#    if pathdata is None:
#        try:
#            pathdata = os.environ["TDPY_DATA_PATH"] + '/'
#        except:
#            pathdata = './'
#        pathdata += 'data/'
#    
#    # construct the global object
#    gdat = util.gdatstrt()
#    gdat.numbproc = numbproc
#    gdat.numbswep = numbswep
#    gdat.llikfunc = llikfunc
#    gdat.datapara = datapara
#    gdat.initsamp = initsamp
#    gdat.optiprop = optiprop
#    gdat.pathimag = pathimag
#    gdat.pathdata = pathdata
#    gdat.rtag = rtag
#    gdat.numbburn = numbburn
#    gdat.numbplotside = numbplotside
#    gdat.factthin = factthin
#    gdat.verbtype = verbtype
#    gdat.factpropeffi = factpropeffi
#    gdat.gdatextr = gdatextr
#    gdat.loadvaripara = loadvaripara
#    gdat.fracrand = fracrand
#
#    if gdat.verbtype > 0:
#        print('TDMC initialized.')
#    
#    if gdat.rtag == '':
#        strg = 'tdmcchan.p'
#    else:
#        strg = 'tdmcchan_%s.p' % gdat.rtag
#    pathchan = gdat.pathdata + strg
#    if os.path.isfile(pathchan) and loadchan:
#        if gdat.verbtype > 0:
#            print 'Reading the previously computed chain from %s...' % pathchan
#        fobj = open(pathchan, 'rb')
#        chan = cPickle.load(fobj)
#        fobj.close()
#        return chan
#        
#    gdat.numbpara = len(datapara.name)
#    gdat.indxpara = np.arange(gdat.numbpara)
#        
#    # Defaults
#    if truepara is None:
#        truepara = np.array([None] * gdat.numbpara)
#        
#    if numbplotside is None:
#        numbplotside = min(gdat.numbpara, 4)
#
#    # Sampler settings
#    if gdat.numbburn is None:
#        if gdat.optiprop:
#            gdat.numbburn = 0
#        else:
#            gdat.numbburn = int(np.floor(0.1 * gdat.numbswep))
#    if gdat.factthin is None:
#        gdat.factthin = min(gdat.numbswep - gdat.numbburn, 1000 * gdat.numbpara)
#   
#    gdat.indxproc = np.arange(gdat.numbproc)
#
#    # sweeps to be saved
#    gdat.boolsave = np.zeros(gdat.numbswep, dtype=bool)
#    gdat.indxswepsave = np.arange(gdat.numbburn, gdat.numbswep, gdat.factthin)
#    gdat.boolsave[gdat.indxswepsave] = True
#    
#    gdat.indxsampsave = np.zeros(gdat.numbswep, dtype=int)
#    gdat.numbsamp = retr_numbsamp(gdat.numbswep, gdat.numbburn, gdat.factthin)
#    gdat.indxsamp = np.arange(gdat.numbsamp)
#    gdat.numbsamptotl = gdat.numbsamp * gdat.numbproc
#    gdat.indxsamptotl = np.arange(gdat.numbsamptotl)
#    gdat.indxsampsave[gdat.indxswepsave] = np.arange(gdat.numbsamp)
#
#    # initialize the chain
#    if gdat.verbtype > 1:
#        print 'Forking the sampler...'
#        print 'datapara'
#        print datapara.indx
#        print datapara.minm
#        print datapara.maxm
#        print datapara.name
#        print datapara.scal
#        print datapara.labl
#        print datapara.unit
#        print datapara.vari
#
#    if numbproc == 1:
#        listchan = [work(gdat, 0)]
#    else:
#        pool = mp.Pool(numbproc)
#        workpart = functools.partial(work, gdat)
#        listchan = pool.map(workpart, gdat.indxproc)
#   
#        pool.close()
#        pool.join()
#
#    if gdat.verbtype > 0:
#        print 'Accumulating samples from all processes...'
#        timeinit = time.time()
#
#    # parse the sample chain
#    listsampvarb = np.zeros((gdat.numbsamp, gdat.numbproc, gdat.numbpara))
#    listsamp = np.zeros((gdat.numbsamp, gdat.numbproc, gdat.numbpara))
#    listsampcalc = []
#    gdat.numbsampcalc = len(listchan[0][2][0])
#    gdat.indxsampcalc = np.arange(gdat.numbsampcalc)
#    for n in gdat.indxsampcalc:
#        size = listchan[0][2][0][n].size
#        listsampcalc.append(empty((gdat.numbsamp, gdat.numbproc, size)))
#    listllik = np.zeros((gdat.numbsamp, gdat.numbproc))
#    listaccp = np.zeros((gdat.numbswep, gdat.numbproc))
#    listchro = np.zeros((gdat.numbswep, gdat.numbproc))
#    listindxparamodi = np.zeros((gdat.numbswep, gdat.numbproc))
#
#    for k in gdat.indxproc:
#        listsampvarb[:, k, :] = listchan[k][0]
#        listsamp[:, k, :] = listchan[k][1]
#        for j in gdat.indxsamp:
#            for n in gdat.indxsampcalc:
#                listsampcalc[n][j, k, :] = listchan[k][2][j][n]
#        listllik[:, k] = listchan[k][3]
#        listaccp[:, k] = listchan[k][4]
#        listchro[:, k] = listchan[k][5]
#        listindxparamodi[:, k] = listchan[k][6]
#
#    indxlistaccp = np.where(listaccp == True)[0]
#    propeffi = np.zeros(gdat.numbpara)
#    for k in gdat.indxpara:
#        indxlistpara = np.where(listindxparamodi == k)[0]
#        indxlistintc = intersect1d(indxlistaccp, indxlistpara, assume_unique=True)
#        if indxlistpara.size != 0:
#            propeffi[k] = float(indxlistintc.size) / indxlistpara.size    
#    
#    minmlistllik = np.amin(listllik)
#    levi = -np.log(np.mean(1. / np.exp(listllik - minmlistllik))) + minmlistllik
#    info = np.mean(listllik) - levi
#    
#    gdat.strgtimestmp = util.retr_strgtimestmp()
#
#    gdat.pathimag += '%s_%s/' % (gdat.strgtimestmp, gdat.rtag)
#       
#    os.system('mkdir -p %s' % gdat.pathimag)
#
#    gmrbstat = np.zeros(gdat.numbpara)
#    if gdat.numbsamp > 1:
#    
#        if gdat.verbtype > 1:
#            print 'Calculating autocorrelation...'
#            timeinit = time.time()
#        atcr, timeatcr = retr_timeatcr(listsamp, atcrtype='maxm')
#        if gdat.verbtype > 1:
#            timefinl = time.time()
#            print 'Done in %.3g seconds' % (timefinl - timeinit)
#        path = gdat.pathimag
#        plot_atcr(path, atcr, timeatcr)
#    
#        if gdat.numbproc > 1:
#            if gdat.verbtype > 1:
#                print 'Performing Gelman-Rubin convergence test...'
#                timeinit = time.time()
#            for k in gdat.indxpara:
#                gmrbstat[k] = gmrb_test(listsampvarb[:, :, k])
#            if gdat.verbtype > 1:
#                timefinl = time.time()
#                print 'Done in %.3g seconds' % (timefinl - timeinit)
#            path = gdat.pathimag
#            plot_gmrb(path, gmrbstat)
#
#    listsampvarb = listsampvarb.reshape((gdat.numbsamptotl, gdat.numbpara))
#    listsamp = listsamp.reshape((gdat.numbsamptotl, gdat.numbpara))
#    for n in gdat.indxsampcalc:
#        listsampcalc[n] = listsampcalc[n].reshape((gdat.numbsamptotl, -1))
#    listllik = listllik.flatten()
#    listaccp = listaccp.flatten()
#    listchro = listchro.flatten()
#    listindxparamodi = listindxparamodi.flatten()
#
#    if gdat.verbtype > 1:
#        print 'Making plots...'
#        timeinit = time.time()
#    
#    # make plots
#    ## proposal efficiency
#    path = gdat.pathimag
#    plot_propeffi(path, gdat.numbswep, gdat.numbpara, listaccp, listindxparamodi, gdat.datapara.strg)
#
#    ## processing time per sample
#    figr, axis = plt.subplots()
#    binstime = np.logspace(np.log10(np.amin(listchro * 1e3)), np.log10(np.amax(listchro * 1e3)), 50)
#    axis.hist(listchro * 1e3, binstime, log=True)
#    axis.set_ylabel('$N_{samp}$')
#    axis.set_xlabel('$t$ [ms]')
#    axis.set_xscale('log')
#    axis.set_xlim([np.amin(binstime), np.amax(binstime)])
#    axis.set_ylim([0.5, None])
#    plt.tight_layout()
#    path = gdat.pathimag + 'chro.pdf'
#    figr.savefig(gdat.pathimag + 'chro.pdf')
#    plt.close(figr)
#
#    ## likelihood
#    path = gdat.pathimag + 'llik'
#    plot_trac(path, listllik, '$P(D|y)$', titl='log P(D) = %.3g' % levi)
#    
#    if numbplotside != 0:
#        path = gdat.pathimag + 'fixp'
#        plot_grid(path, listsampvarb, gdat.datapara.strg, truepara=gdat.datapara.true, scalpara=gdat.datapara.scal, numbplotside=numbplotside, numbbinsplot=numbbinsplot)
#        
#    for k in gdat.indxpara:
#        path = gdat.pathimag + gdat.datapara.name[k]
#        plot_trac(path, listsampvarb[:, k], gdat.datapara.strg[k], scalpara=gdat.datapara.scal[k], truepara=gdat.datapara.true[k], numbbinsplot=numbbinsplot)
#        
#    if gdat.verbtype > 1:
#        timefinl = time.time()
#        print 'Done in %.3g seconds' % (timefinl - timeinit)
#
#    chan = [listsampvarb, listsamp, listsampcalc, listllik, listaccp, listchro, listindxparamodi, propeffi, levi, info, gmrbstat]
#    
#    # save the chain if the run was long enough
#    timefinl = time.time()
#    # temp
#    if timefinl - timeinit > 1.:
#        if not os.path.isfile(pathchan):
#            print 'Writing the chain to %s...' % pathchan
#            fobj = open(pathchan, 'wb')
#            cPickle.dump(chan, fobj, protocol=cPickle.HIGHEST_PROTOCOL)
#            fobj.close()
#        
#    return chan


def work(gdat, indxprocwork):
    
    # re-seed the random number generator for the process
    seed()
    
    listsampvarb = np.zeros((gdat.numbsamp, gdat.numbpara))
    listsamp = np.zeros((gdat.numbsamp, gdat.numbpara)) + -1.
    listsampcalc = []
    listllik = np.zeros(gdat.numbsamp)
    listaccp = empty(gdat.numbswep, dtype=bool)
    listchro = empty(gdat.numbswep)
    listindxparamodi = empty(gdat.numbswep, dtype=int)
    
    gdat.cntrswep = 0
    
    if gdat.initsamp is None:
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
            cntrprop = np.zeros(gdat.numbpara)
            cntrproptotl = np.zeros(gdat.numbpara)
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

        if np.where((nextsamp < 0.) | (nextsamp > 1.))[0].size == 0:

            nextsampvarb = icdf_samp(nextsamp, gdat.datapara)

            if gdat.verbtype > 1:
                print 'nextsampvarb: '
                print nextsampvarb

            # evaluate the log-likelihood
            nextllik, nextsampcalc = gdat.llikfunc(nextsampvarb, gdat.gdatextr)
            accpprob = np.exp(nextllik - thisllik)

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
                print np.mean(thispropeffi)
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
        print 'Autocorrelation time could not be estimated.'
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
        print 'plot_trac encountered infinite input. Returning...'
        print 'path'
        print path
        return
    
    if not np.isfinite(listpara).all():
        print 'labl'
        print labl
        print 'listpara'
        print listpara
        summgene(listpara)
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


def plot_hist(path, listvarb, strg, titl=None, numbbins=20, truepara=None, boolquan=True, scalpara='self', listvarbdraw=None, listlabldraw=None, listcolrdraw=None):

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
    figr.savefig(path + '_hist.pdf')
    plt.close(figr)


def plot_grid(pathbase, strgplot, listsamp, strgpara, join=False, limt=None, scalpara=None, plotsize=3.5, \
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
            print 'k'
            print k
            print 'strgpara[k]'
            print strgpara[k]
            print 
            #raise Exception('Lower and upper limits are zero.')
            print 'Lower and upper limits are zero.'
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
            print 'k'
            print k
            print 'strgpara[k]'
            print strgpara[k]
            print 'limt[:, k]'
            print limt[:, k]
            print 'bins[:, k]'
            print bins[:, k]
            summgene(bins[:, k])
            raise Exception('')
        if np.amin(bins[:, k]) == 0 and np.amax(bins[:, k]) == 0:
            print 'plot_grid() found min=max=0. Returning...'
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
                    print 'l'
                    print l
                    print 'listsamp'
                    print listsamp
                    summgene(listsamp)
                    print 'strgpara'
                    print strgpara
                    print 'scalpara'
                    print scalpara
                    print 'limt'
                    print limt
                    print 'arry'
                    print arry
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
    figr.savefig(pathbase + 'pmar_' + strgplot + '.pdf')
    plt.close(figr)
    
