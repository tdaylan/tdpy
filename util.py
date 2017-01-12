from __init__ import *

class gdatstrt(object):
    
    def __init__(self):
        self.boollockmodi = False
        pass

    def lockmodi(self):
        self.boollockmodi = True

    def unlkmodi(self):
        self.boollockmodi = False

    def __setattr__(self, attr, valu):
        if hasattr(self, attr) and self.boollockmodi and attr != 'boollockmodi':
            raise KeyError('{} has already been set'.format(attr))
        super(gdatstrt, self).__setattr__(attr, valu)


class datapara(object):

    def __init__(self, numbpara):
        
        self.numbpara = numbpara
        
        self.indx = dict()
        self.name = empty(numbpara, dtype=object)
        self.minm = zeros(numbpara)
        self.maxm = zeros(numbpara)
        self.scal = empty(numbpara, dtype=object)
        self.labl = empty(numbpara, dtype=object)
        self.unit = empty(numbpara, dtype=object)
        self.vari = zeros(numbpara)
        self.true = zeros(numbpara)
        self.strg = empty(numbpara, dtype=object)
        
        self.cntr = 0
    
    def defn_para(self, name, minm, maxm, scal, labl, unit, vari, true):
        
        self.indx[name] = self.cntr
        self.name[self.cntr] = name
        self.minm[self.cntr] = minm
        self.maxm[self.cntr] = maxm
        self.scal[self.cntr] = scal
        self.labl[self.cntr] = labl
        self.unit[self.cntr] = unit
        self.vari[self.cntr] = vari
        self.true[self.cntr] = true
        self.strg[self.cntr] = labl + ' ' + unit
        self.cntr += 1


def time_func(func, *args):
    
    numbiter = 100
    timediff = empty(numbiter)
    for k in range(numbiter):
        timeinit = time.time()
        func(*args)
        timediff[k] = time.time() - timeinit
    
    return mean(timediff), std(timediff)


def time_func_verb(func, *args):
    
    meantimediff, stdvtimediff = time_func(func, *args)
    
    print 'Calling %s' % func.__name__ 
    print '%3g pm %3g ms' % (meantimediff * 1e3, stdvtimediff * 1e3)
    

def retr_postvarb(listvarb):

    shap = zeros(len(listvarb.shape), dtype=int)
    shap[0] = 3
    shap[1:] = listvarb.shape[1:]
    shap = list(shap)
    postvarb = zeros(shap)
    
    postvarb[0, ...] = percentile(listvarb, 50., axis=0)
    postvarb[1, ...] = percentile(listvarb, 16., axis=0)
    postvarb[2, ...] = percentile(listvarb, 84., axis=0)

    return postvarb


def retr_errrvarb(inpt, samp=False):

    if samp:
        postvarb = retr_postvarb(inpt)
    else:
        postvarb = inpt

    errr = fabs(postvarb[0, ...] - postvarb[1:3, ...])

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


def show(varb):

    print varb
    print


def summ(gdat, strg, k=None):
    if k == None:
        varb = getattr(gdat, strg)
    else:
        varb = getattr(gdat, strg)[k]
    print strg
    print amin(varb)
    print amax(varb)
    print mean(varb)
    print varb.shape
    print


class varb(object):
    
    def __init__(self, numb=None):
        
        self.name = []
        self.strg = []
        self.para = []
        self.scal = []
        self.strg = []
        if numb != None:
            self.numb = numb

    
    def defn_para(self, name, minm, maxm, numb=None, strg='', scal='self'):
        
        if numb != None:
            numbtemp = numb
        else:
            numbtemp = self.numb

        if scal == 'logt':
            arry = logspace(log10(minm), log10(maxm), numbtemp) 
        if scal == 'self':
            arry = linspace(minm, maxm, numbtemp) 
        
        self.name.append(name)
        self.para.append(arry)
        self.scal.append(scal)
        self.strg.append(strg)
        self.size = len(self.para)

    
def summgene(varb):
    print amin(varb)
    print amax(varb)
    print mean(varb)
    print varb.shape
    print


def retr_p4dm_spec(anch, part='el'):
    
    pathimag, pathdata = retr_path('tdpy')
    if part == 'el':
        strg = 'AtProduction_positrons'
    if part == 'ph':
        strg = 'AtProduction_gammas'
    name = pathdata + 'p4dm/' + strg + '.dat'
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


def show_prog(cntr, maxmcntr, thiscntr, nprog=20, indxprocwork=None, showmemo=False, accp=None):

    nextcntr = int(nprog * float(cntr + 1) / maxmcntr) * 100 / nprog
    if nextcntr > thiscntr:
        if indxprocwork != None:
            print 'Process %d is %3d%% completed.' % (indxprocwork, nextcntr)
        else:
            print '%3d%% completed.' % nextcntr
        if accp != None:
            print 'Acceptance ratio: %.3g%%' % accp
        thiscntr = nextcntr
        if showmemo:
            show_memo_simp()
        
    return thiscntr            


def retr_galcfromequc(rasc, decl):

    icrs = astropy.coordinates.SkyCoord(ra=rasc*astropy.units.degree, dec=decl*astropy.units.degree)

    lgal = icrs.galactic.l.degree
    bgal = icrs.galactic.b.degree
    
    return lgal, bgal


def regr(xaxi, yaxi, ordr):
    
    coef = polyfit(xaxi, yaxi, ordr)
    func = poly1d(coef)
    strg = '$y = '
    if ordr == 0:
        strg += '%.5g$'
    if ordr == 1:
        strg += '%.5g x + %.5g$' % (coef[0], coef[1])
    if ordr == 2:
        strg += '%.5g x^2 + %.5g x + %.5g$' % (coef[0], coef[1], coef[2])

    return coef, func, strg


def corr_catl(lgalseco, bgalseco, lgalfrst, bgalfrst, anglassc=deg2rad(1.), verbtype=1):

    numbfrst = lgalfrst.size

    indxsecoassc = zeros(numbfrst, dtype=int) - 1
    numbassc = zeros(numbfrst, dtype=int)
    distassc = zeros(numbfrst) + 1000.
    lgalbgalfrst = array([lgalfrst, bgalfrst])
    thisfraccomp = -1
    numbseco = lgalseco.size
    for k in range(numbseco):
        lgalbgalseco = array([lgalseco[k], bgalseco[k]])
        dist = angdist(lgalbgalfrst, lgalbgalseco, lonlat=True)
        thisindxfrst = where(dist < anglassc)[0]
        
        if thisindxfrst.size > 0:
            
            # if there are multiple associated true PS, sort them
            indx = argsort(dist[thisindxfrst])
            dist = dist[thisindxfrst][indx]
            thisindxfrst = thisindxfrst[indx]
                
            # store the index of the model PS
            numbassc[thisindxfrst[0]] += 1
            if dist[0] < distassc[thisindxfrst[0]]:
                distassc[thisindxfrst[0]] = dist[0]
                indxsecoassc[thisindxfrst[0]] = k

        nextfraccomp = int(100 * float(k) / numbseco)
        if verbtype > 1 and nextfraccomp > thisfraccomp:
            thisfraccomp = nextfraccomp
            print '%02d%% completed.' % thisfraccomp

    return indxsecoassc


def retr_memoresi():
    
    # temp
    if os.uname()[1] == 'fink1.rc.fas.harvard.edu' or os.uname()[1] == 'fink2.rc.fas.harvard.edu': 
        memoresi = 0.
        memoresiperc = 0.
    else:
        proc = psutil.Process(os.getpid())
        memoinfo = proc.memory_info()
        memoresi = memoinfo.rss
        memoresiperc = proc.memory_percent()

    return memoresi, memoresiperc


def show_memo_simp():
    
    memoresi, memoresiperc = retr_memoresi()

    strgmemo = retr_strgmemo(memoresi)

    print 'Resident memory: %s, %4.3g%%' % (strgmemo, memoresiperc)


def retr_strgmemo(memo):

    if memo >= float(2**30):
        memonorm = memo / float(2**30)
        strg = 'GB'
    elif memo >= float(2**20):
        memonorm = memo / float(2**20)
        strg = 'MB'
    elif memo >= float(2**10):
        memonorm = memo / float(2**10)
        strg = 'KB'
    else:
        memonorm = memo
        strg = 'B'
    strgmemo = '%d %s' % (memonorm, strg)
    return strgmemo


def retr_axis(minm=None, maxm=None, numb=None, bins=None, scal='self'):
    
    if bins == None:
        if scal == 'self':
            bins = linspace(minm, maxm, numb + 1)
            mean = (bins[1:] + bins[:-1]) / 2.
        else:
            bins = logspace(log10(minm), log10(maxm), numb + 1)
            mean = sqrt(bins[1:] * bins[:-1])
    else:
        if scal == 'self':
            mean = (bins[1:] + bins[:-1]) / 2.
        else:
            mean = sqrt(bins[1:] * bins[:-1])
        numb = mean.size
        
    indx = arange(numb)
   
    return bins, mean, diff(bins), numb, indx


def show_memo(objt, name):

    if isinstance(objt, list):
        for k in len(objt):
            size = sys.getsizeof(objt[k]) / 2.**20
    else:
        listsize = []
        listattr = []
        for attr, valu in objt.__dict__.iteritems():
            listsize.append(sys.getsizeof(valu) / 2.**20)
            listattr.append(attr)
        size = array(listsize)
        attr = array(listattr)
        sizetotl = sum(size) 
        print 'Memory budget: %s' % name
        print 'Total size: %.4g MB' % sizetotl
        
        # sort the sizes to get the largest tail
        indxsizetemp = argsort(size)[::-1]
        size = size[indxsizetemp]
        attr = attr[indxsizetemp]
        print 'Largest 5:'
        for k in range(5):
            print '%s: %.4g MB' % (attr[k], size[k])
        print 


def retr_psfngausnorm(angl):

    norm = sqrt(2. / pi**3) / angl / exp(-0.5 * angl**2) / \
                        real(-erfi((angl**2 - pi * 1j) / sqrt(2) / angl) - erfi((angl**2 + pi * 1j) / sqrt(2) / angl) + 2. * erfi(angl / sqrt(2.)))

    return norm


def retr_mapspnts(lgal, bgal, stdv, flux, numbside=256, verbtype=0):
    
    # lgal, bgal and stdv are in degrees
    numbpnts = lgal.size
    lgalheal, bgalheal, numbpixl, apix = retr_healgrid(numbside)
    gridheal = array([lgalheal, bgalheal])
    stdvradi = deg2rad(stdv)
    mapspnts = zeros(numbpixl)
    for n in range(numbpnts):
        gridpnts = array([lgal[n], bgal[n]])
        angl = angdist(gridheal, gridpnts, lonlat=True)
        norm = retr_psfngausnorm(stdvradi)
        print 'norm'
        print norm
        print 'angl'
        print angl.shape
        print 'flux'
        print flux.shape
        print 'stdvradi'
        print stdvradi
        print
        
        mapspnts += apix * norm * flux[n] * exp(-0.5 * angl**2 / stdvradi**2)
        if verbtype > 0:
            print '%d out of %d' % (n, numbpnts)

    return mapspnts


def retr_mapsplnkfreq(indxpixloutprofi=None, numbsideoutp=256, indxfreqrofi=None):

    numbside = 2048
    numbpixl = 12 * numbside**2
    meanfreq = array([30, 44, 70, 100, 143, 217, 353, 545, 857])
    numbfreq = meanfreq.size
    indxfreq = arange(numbfreq)
    strgfreq = ['%04d' % meanfreq[k] for k in indxfreq]
    
    indxpixloutp = arange(numbsideoutp)

    if indxfreqrofi == None:
        indxfreqrofi = indxfreq

    if indxpixloutprofi == None:
        indxpixloutprofi = indxpixloutp

    rtag = '_%04d' % (numbsideoutp)

    path = retr_path('tdpy', onlydata=True)
    pathflux = path + 'plnkflux_%s.fits' % rtag
    pathfluxstdv = path + 'plnkfluxstdv_%s.fits' % rtag
    if os.path.isfile(pathflux) and os.path.isfile(pathfluxstdv):
        print 'Reading %s...' % pathflux
        mapsplnkfreq = pf.getdata(pathflux)
        print 'Reading %s...' % pathfluxstdv
        mapsplnkfreqstdv= pf.getdata(pathfluxstdv)
    else:
        mapsplnkfreq = zeros((numbfreq, numbpixl))
        mapsplnkfreqstdv = zeros((numbfreq, numbpixl))
        for k in indxfreq:
            print 'Processing Planck Map at %d GHz...' % (meanfreq[k])
            # read sky maps
            if strgfreq[k][1] == '0':
                strgfrst = 'plnk/LFI_SkyMap_' 
                strgseco = '-BPassCorrected-field-IQU_0256_R2.01_full.fits'
                strgcols = 'TEMPERATURE'
            elif strgfreq[k][1] == '1' or strgfreq[k][1] == '2' or strgfreq[k][1] == '3':
                strgfrst = 'plnk/HFI_SkyMap_'
                strgseco = '-field-IQU_2048_R2.02_full.fits'
                strgcols = 'I_STOKES'
            else:
                strgfrst = 'plnk/HFI_SkyMap_'
                strgseco = '-field-Int_2048_R2.02_full.fits'
                strgcols = 'I_STOKES'
            strg = strgfrst + '%s' % strgfreq[k][1:] + strgseco
        
            mapsinpt = pf.getdata(path + strg, 1)[strgcols]
            numbpixlinpt = mapsinpt.size
            numbsideinpt = int(sqrt(numbpixlinpt / 12))
            mapsplnkfreq[k, :] = pf.getdata(path + strg, 1)[strgcols]
            mapsplnkfreq[k, :] = hp.reorder(mapsplnkfreq[k, :], n2r=True)
        
            print 'Changing units...'
            # change units of the sky maps to Jy/sr
            if strgfreq[k] != '0545' and strgfreq[k] != '0857':
                ## from Kcmb
                if calcfactconv:
                    # read Planck band transmission data
                    if strgfreq[k][1] == '0':
                        strg = 'LFI_RIMO_R2.50'
                        strgextn = 'BANDPASS_%s' % strgfreq[k][1:]
                        freqband = pf.open(path + '%s.fits' % strg)[strgextn].data['WAVENUMBER'][1:] * 1e9
                    else:
                        strg = 'plnk/HFI_RIMO_R2.00'
                        strgextn = 'BANDPASS_F%s' % strgfreq[k][1:]
                        freqband = 1e2 * velolght * pf.open(path + '%s.fits' % strg)[strgextn].data['WAVENUMBER'][1:]
                    tranband = pf.open(path + '%s.fits' % strg)[strgextn].data['TRANSMISSION'][1:]
                    indxfreqbandgood = where(tranband > 1e-6)[0]
                    indxfreqbandgood = arange(amin(indxfreqbandgood), amax(indxfreqbandgood) + 1)
        
                    # calculate the unit conversion factor
                    freqscal = consplnk * freqband[indxfreqbandgood] / consbolt / tempcmbr
                    freqcntr = float(strgfreq[k]) * 1e9
                    specdipo = 1e26 * 2. * (consplnk * freqband[indxfreqbandgood]**2 / velolght / tempcmbr)**2 / consbolt / (exp(freqscal) - 1.)**2 * exp(freqscal) # [Jy/sr]
                    factconv = trapz(specdipo * tranband[indxfreqbandgood], freqband[indxfreqbandgood]) / \
                                                    trapz(freqcntr * tranband[indxfreqbandgood] / freqband[indxfreqbandgood], freqband[indxfreqbandgood]) # [Jy/sr/Kcmb]
                else:
                    # read the unit conversion factors provided by Planck
                    factconv = factconvplnk[k, 1] * 1e6
            else:
                ## from MJy/sr
                factconv = 1e6
            mapsplnk[k, :] *= factconv
        
        pf.writeto(pathflux, mapsplnkfreq, clobber=True)
        pf.writeto(pathfluxstdv, mapsplnkfreqstdv, clobber=True)
        
    return mapsplnkfreq, mapsplnkfreqstdv


def retr_indximagmaxm(data):

    sizeneig = 10
    cntsthrs = 10
    maxmdata = sp.ndimage.filters.maximum_filter(data, sizeneig)
    
    boolmaxm = (data == maxmdata)
    minmdata = sp.ndimage.filters.minimum_filter(data, sizeneig)
    diff = ((maxmdata - minmdata) > cntsthrs)
    boolmaxm[diff == 0] = 0
    mapslabl, numbobjt = sp.ndimage.label(boolmaxm)
    mapslablones = zeros_like(mapslabl)
    mapslablones[where(mapslabl > 0)] = 1.
    indxmaxm = array(sp.ndimage.center_of_mass(data, mapslabl, range(1, numbobjt+1))).astype(int)
    if len(indxmaxm) == 0:
        indxyaximaxm = array([0])
        indxxaximaxm = array([0])
    else:
        indxyaximaxm = indxmaxm[:, 1]
        indxxaximaxm = indxmaxm[:, 0]
    return indxxaximaxm, indxyaximaxm


def minm(thissamp, func, verbtype=1, stdvpara=None, factcorrscal=2., gdat=None, maxmswep=None, limtpara=None, tolrfunc=1e-6, optiprop=True, pathbase='./', rtag=''):

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

    thisfunc = func(thissamp, gdat)
    thisstdvpara = stdvpara

    numbsccs = 100
    nextbool = zeros(numbsccs, dtype=bool)

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
            nextfunc = func(nextsamp, gdat)

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
        nextfuncconv = func(nextsampconv, gdat)
        nexterrr = fabs(nextfuncconv / thisfunc - 1.)
        if nexterrr < thiserrr:
            thiserrr = nexterrr
        nextbool[0] = nexterrr < tolrfunc

        
        if verbtype > 1:
            print 'Checking convergence...'
            print 'nextsampconv'
            print nextsampconv
            print 'nextfuncconv'
            print nextfuncconv
            print 'nexterrr'
            print nexterrr

        if nextbool.all() or cntrswep == maxmswep:
            minmsamp = thissamp
            minmfunc = thisfunc
            break
        else:
            roll(nextbool, 1)
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

    def func_test(samp, gdat=None):
        
        return sum((samp / 0.2 - 1.)**2)
    
    numbpara = 10
    stdvpara = ones(numbpara)
    thissamp = rand(numbpara)
    minm(thissamp, func_test, verbtype=1, factcorrscal=100., stdvpara=stdvpara, maxmswep=None, limtpara=None, tolrfunc=1e-6, pathbase='./', rtag='')
    

def plot_gene(path, xdat, ydat, scalxdat=None, scalydat=None, lablxdat='', lablydat='', scat=False, hist=False):
    
    figr, axis = plt.subplots(figsize=(6, 6))
    
    if scat:
        axis.scatter(xdat, ydat)
    elif hist:
        deltxdat = xdat[1] - xdat[0]
        axis.bar(xdat - deltxdat / 2., ydat, deltxdat)
    else:
        axis.plot(xdat, ydat)

    if scalxdat == 'logt':
        axis.set_xscale('log')
    if scalydat == 'logt':
        axis.set_yscale('log')
    
    axis.set_xlabel(lablxdat)
    axis.set_ylabel(lablydat)

    plt.tight_layout()
    plt.savefig(path)
    plt.close(figr)


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


def prep_maps(recotype, enertype, regitype, pathdata, numbside, timetype):
    
    if enertype == 'back':
        numbener = 30
        minmener = 0.1
        maxmener = 100.
        binsener = logspace(log10(minmener), log10(maxmener), numbener + 1)
    else:
        binsener = array([0.1, 0.3, 1., 3., 10., 100.])
        numbener = binsener.size - 1
    indxener = arange(numbener)
    diffener = binsener[1:] - binsener[:-1]

    numbside = 256
    evtt = array([4, 8, 16, 32])

    numbpixl = 12 * numbside**2
    numbevtt = evtt.size
    indxevtt = arange(numbevtt)
    apix = 4. * pi / numbpixl

    cnts = zeros((numbener, numbpixl, numbevtt))
    expo = zeros((numbener, numbpixl, numbevtt))
    flux = zeros((numbener, numbpixl, numbevtt))
    
    for m in indxevtt:

        if recotype == 'rec7':
            if m < 2:
                continue
            elif m == 2:
                thisevtt = 2
            elif m == 3:
                thisevtt = 1
        else:
            thisevtt = evtt[m]

        path = pathdata + '/fermexpo_%04d_%s_%s_%04d_%s.fits' % (thisevtt, recotype, enertype, numbside, timetype)
        expoarry = pf.getdata(path, 1)
        for i in indxener:
            expo[i, :, m] = expoarry['ENERGY%d' % (i + 1)]

        path = pathdata + '/fermcnts_%04d_%s_%s_%04d_%s.fits' % (thisevtt, recotype, enertype, numbside, timetype)
        cntsarry = pf.getdata(path)
        for i in indxener:
            cnts[i, :, m] = cntsarry['CHANNEL%d' % (i + 1)]

    indxexpo = where(expo > 0.) 
    flux[indxexpo] = cnts[indxexpo] / expo[indxexpo] / apix
    flux /= diffener[:, None, None]

    if regitype == 'ngal':
        for i in indxener:
            for m in indxevtt:
                
                if recotype == 'rec7':
                    if m < 2:
                        continue
                    elif m == 2:
                        thisevtt = 2
                    elif m == 3:
                        thisevtt = 1
                else:
                    thisevtt = evtt[m]

                almc = hp.map2alm(flux[i, :, m])
                hp.rotate_alm(almc, 0., 0.5 * pi, 0.)
                flux[i, :, m] = hp.alm2map(almc, numbside)

                almc = hp.map2alm(expo[i, :, m])
                hp.rotate_alm(almc, 0., 0.5 * pi, 0.)
                expo[i, :, m] = hp.alm2map(almc, numbside)

    path = pathdata + '/fermexpo_%s_%s_%s_%04d_%s.fits' % (recotype, enertype, regitype, numbside, timetype)
    pf.writeto(path, expo, clobber=True)

    path = pathdata + '/fermflux_%s_%s_%s_%04d_%s.fits' % (recotype, enertype, regitype, numbside, timetype)
    pf.writeto(path, flux, clobber=True)


def prep_fdfm(regitype, enertype, pathdata):
    
    numbside = 256
    indxevtt = arange(4)
    binsener = array([0.1, 0.3, 1., 3., 10., 100.])
   
    meanener = sqrt(binsener[1:] * binsener[:-1])
    numbpixl = 12 * numbside**2
    numbener = binsener.size - 1
    numbevtt = evtt.size

    # get the Fermi-LAT diffuse model
    temp = tdpy.util.retr_fdfm(binsener, numbside)
    
    # rotate if necessary
    fdfmflux = zeros((numbener, numbpixl))
    for m in arange(numbevtt):
        if regitype == 'ngal':
            for i in range(numbener):
                almc = hp.map2alm(fdfmfluxigaltemp[i, :])
                hp.rotate_alm(almc, 0., 0.5 * pi, 0.)
                fdfmflux[i, :, m] = hp.alm2map(almc, numbside)
        else:
            fdfmflux[:, :, m] = temp

    # smooth the model
    fdfmflux = smth_ferm(maps, meanener, indxevtt)

    path = pathdata + '/fdfmflux_%s_%s_%s.fits' % (recotype, enertype, regitype)
    pf.writeto(path, fdfmfluxigal, clobber=True)


def retr_strgtimestmp():

    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    return strgtimestmp


def read_fits(path, pathimag=None):
    
    print 'Reading the header of %s...' % path
        
    if pathimag != None:
        os.system('mkdir -p ' + pathimag)
    
    hdun = pf.open(path)
    numbhead = len(hdun)
    for k in range(numbhead):
        print 'Extension %d' % k
        head = hdun[k].header
        data = hdun[k].data
        arry = array(stack((head.keys(), head.values()), 1))
        listtype = []
        listform = []
        listunit = []
        for n in range(arry.shape[0]):
            if arry[n, 0] == 'EXTNAME':
                print 'Extension name: ', arry[n, 1]
        for n in range(arry.shape[0]):
            if arry[n, 0].startswith('TTYPE') or arry[n, 0].startswith('TFORM') or arry[n, 0].startswith('TUNIT'):
                print arry[n, 0], ': ', arry[n, 1]
            if arry[n, 0].startswith('TTYPE'):
                listtype.append(arry[n, 1])
            if arry[n, 0].startswith('TFORM'):
                listform.append(arry[n, 1])
            if arry[n, 0].startswith('TUNIT'):
                listunit.append(arry[n, 1])
                print

        if pathimag != None:
            for n in range(len(listtype)):
                if not listform[n].endswith('A') and isfinite(data[listtype[n]]).all():
                    figr, axis = plt.subplots()
                    try:
                        bins = linspace(amin(data[listtype[n]]), amax(data[listtype[n]]), 100)
                        axis.hist(data[listtype[n]], bins=bins)
                        #axis.set_xlabel('%s [%s]' % (listtype[n], listunit[n]))
                        axis.set_yscale('log')
                        axis.set_xlabel('%s' % (listtype[n]))
                        plt.tight_layout()
                        path = pathimag + 'readfits_%s.pdf' % listtype[n]
                        plt.savefig(path)
                        plt.close(figr)
                    except:
                        print 'Failed on %s' % listtype[n]
                        print


def plot_maps(path, maps, pixltype='heal', scat=None, indxpixlrofi=None, numbpixl=None, titl='', minmlgal=None, maxmlgal=None, minmbgal=None, maxmbgal=None, \
                                                                                                resi=False, satu=False, numbsidelgal=None, numbsidebgal=None, igal=False):
   
    if minmlgal == None:
        if not igal:
            minmlgal = -180.
            minmbgal = -90.
            maxmlgal = 180.
            maxmbgal = 90.
        else:
            minmlgal = -20.
            minmbgal = -20.
            maxmlgal = 20.
            maxmbgal = 20.
            
    asperati = (maxmbgal - minmbgal) / (maxmlgal - minmlgal)
    
    if indxpixlrofi != None:
        mapstemp = zeros(numbpixl)
        mapstemp[indxpixlrofi] = maps
        maps = mapstemp
    else:
        numbpixl = maps.size
    
    if numbsidelgal == None:
        numbsidelgal = min(4 * int((maxmlgal - minmlgal) / rad2deg(sqrt(4. * pi / numbpixl))), 2000)
    if numbsidebgal == None:
        numbsidebgal = min(4 * int((maxmbgal - minmbgal) / rad2deg(sqrt(4. * pi / numbpixl))), 2000)
    
    # saturate the map
    if satu:
        mapstemp = copy(maps)
        maps = mapstemp
        if not resi:
            satu = 0.1 * amax(maps)
        else:
            satu = 0.1 * min(fabs(amin(maps)), amax(maps))
            maps[where(maps < -satu)] = -satu
        maps[where(maps > satu)] = satu

    exttrofi = [minmlgal, maxmlgal, minmbgal, maxmbgal]

    if pixltype == 'heal':
        cart = retr_cart(maps, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal, numbsidelgal=numbsidelgal, numbsidebgal=numbsidebgal)
    else:
        numbsidetemp = int(sqrt(maps.size))
        cart = maps.reshape((numbsidetemp, numbsidetemp)).T

    sizefigr = 8
    if resi:
        cmap = 'RdBu'
    else:
        cmap = 'Reds'

    if asperati < 1.:   
        factsrnk = 1.5 * asperati
    else:
        factsrnk = 0.8
    figr, axis = plt.subplots(figsize=(sizefigr, asperati * sizefigr))
    if resi:
        valu = max(fabs(amin(cart)), fabs(amax(cart)))
        imag = plt.imshow(cart, origin='lower', cmap=cmap, extent=exttrofi, interpolation='none', vmin=-valu, vmax=valu)
    else:
        imag = plt.imshow(cart, origin='lower', cmap=cmap, extent=exttrofi, interpolation='none')
    
    if scat != None:
        numbscat = len(scat)
        for k in range(numbscat):
            axis.scatter(scat[k][0], scat[k][1])

    cbar = plt.colorbar(imag, shrink=factsrnk) 
    
    plt.title(titl, y=1.08)

    plt.savefig(path)
    plt.close(figr)
    

def rebn(arry, shapoutp, totl=False):
    
    shaptemp = shapoutp[0], arry.shape[0] // shapoutp[0], shapoutp[1], arry.shape[1] // shapoutp[1]
    
    if totl:
        arryoutp = arry.reshape(shaptemp).sum(-1).sum(1)
    else:
        arryoutp = arry.reshape(shaptemp).mean(-1).mean(1)

    return arryoutp


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


def retr_isot(binsener, numbside=256):
    
    diffener = binsener[1:] - binsener[:-1]
    numbpixl = 12 * numbside**2
    numbener = binsener.size - 1
    numbsamp = 10

    # get the best-fit isotropic flux given by the Fermi-LAT collaboration

    pathdata = retr_path('tdpy', onlydata=True)
    path = pathdata + 'iso_P8R2_SOURCE_V6_v06.txt'
    isotdata = loadtxt(path)
    enerisot = isotdata[:, 0] * 1e-3 # [GeV]
    isotfluxtemp = isotdata[:, 1] * 1e3 # [1/cm^2/s/sr/GeV]
    
    # sampling energy grid
    binsenersamp = logspace(log10(amin(binsener)), log10(amax(binsener)), numbsamp * numbener)
    
    # interpolate the flux over the sampling energy grid
    isotfluxtemp = interp(binsenersamp, enerisot, isotfluxtemp)
    
    # take the mean flux in the desired energy bins
    isotflux = empty((numbener, numbpixl))
    for i in range(numbener):
        isotflux[i, :] = trapz(isotfluxtemp[i*numbsamp:(i+1)*numbsamp], binsenersamp[i*numbsamp:(i+1)*numbsamp]) / diffener[i]
        
    return isotflux


def retr_cart(hmap, indxpixlrofi=None, numbsideinpt=None, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, \
                                                                                                            numbsidelgal=None, numbsidebgal=None):
    
    if indxpixlrofi == None:
        numbpixlinpt = hmap.size
        numbsideinpt = int(sqrt(numbpixlinpt / 12.))
    else:
        numbpixlinpt = numbsideinpt**2 * 12
    
    if numbsidelgal == None:
        numbsidelgal = 4 * int((maxmlgal - minmlgal) / rad2deg(sqrt(4. * pi / numbpixlinpt)))
    if numbsidebgal == None:
        numbsidebgal = 4 * int((maxmbgal - minmbgal) / rad2deg(sqrt(4. * pi / numbpixlinpt)))
    
    lgcr = linspace(minmlgal, maxmlgal, numbsidelgal)
    indxlgcr = arange(numbsidelgal)
    
    bgcr = linspace(minmbgal, maxmbgal, numbsidebgal)
    indxbgcr = arange(numbsidebgal)
    
    lghp, bghp, numbpixl, apix = retr_healgrid(numbsideinpt)

    bgcrmesh, lgcrmesh = meshgrid(bgcr, lgcr)
    
    indxpixlmesh = hp.ang2pix(numbsideinpt, pi / 2. - deg2rad(bgcrmesh), deg2rad(lgcrmesh))
    
    if indxpixlrofi == None:
        indxpixltemp = indxpixlmesh
    else:
        pixlcnvt = zeros(numbpixlinpt, dtype=int) - 1
        for k in range(indxpixlrofi.size):
            pixlcnvt[indxpixlrofi[k]] = k
        indxpixltemp = pixlcnvt[indxpixlmesh]
    
    indxbgcrgrid, indxlgcrgrid = meshgrid(indxbgcr, indxlgcr)
    hmapcart = empty((numbsidebgal, numbsidelgal))
    hmapcart[meshgrid(indxbgcr, indxlgcr)] = hmap[indxpixltemp]

    return hmapcart


def smth(maps, scalsmth, mpol=False, retrfull=False, numbsideoutp=None, indxpixlmask=None):

    if mpol:
        mpolsmth = scalsmth
    else:
        mpolsmth = 180. / scalsmth

    numbpixl = maps.size
    numbside = int(sqrt(numbpixl / 12))
    numbmpol = 3 * numbside
    maxmmpol = 3. * numbside - 1.
    mpolgrid, temp = hp.Alm.getlm(lmax=maxmmpol)
    mpol = arange(maxmmpol + 1.)
    
    if numbsideoutp == None:
        numbsideoutp = numbside
    
    if indxpixlmask != None:
        mapsoutp = copy(maps)
        mapsoutp[indxpixlmask] = hp.UNSEEN
        mapsoutp = hp.ma(mapsoutp)
        almctemp = hp.map2alm(maps)
    else:
        mapsoutp = maps
    
    almc = hp.map2alm(mapsoutp)

    wght = exp(-0.5 * (mpolgrid / mpolsmth)**2)
    almc *= wght
    
    mapsoutp = hp.alm2map(almc[where(mpolgrid < 3 * numbsideoutp)], numbsideoutp, verbose=False)

    if retrfull:
        return mapsoutp, almc, mpol, exp(-0.5 * (mpol / mpolsmth)**2)
    else:
        return mapsoutp


def retr_fdfm(binsener, numbside=256, vfdm=7):                    
    
    diffener = binsener[1:] - binsener[0:-1]
    numbener = diffener.size

    numbpixl = numbside**2 * 12
    
    path = os.environ["TDPY_DATA_PATH"] + '/data/'
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


def plot_matr(axis, xdat, ydat, labl, loc=1):
    
    listlinestyl = [':', '--', '-']
    listcolr = ['b', 'r', 'g']
    
    for i in range(3):
        for  j in range(3):
            if len(xdat.shape) == 3:
                axis.plot(xdat[i, j, :], ydat[i, j, :], color=listcolr[j], ls=listlinestyl[i])
            else:
                axis.plot(xdat, ydat[i, j, :], color=c[j], ls=ls[i])

    line = []
    for k in arange(3):
        line.append(plt.Line2D((0,1),(0,0), color='k', ls=listlinestyl[k]))
    for l in range(3):
        line.append(plt.Line2D((0,1),(0,0), color=listcolr[l]))
    axis.legend(line, labl, loc=loc, ncol=2) 


def plot_braz(axis, xdat, ydat, numbsampdraw=0, lcol='yellow', dcol='green', mcol='black', labl=None, alpha=None):
    
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


def retr_fermpsfn(meanenerrofi, indxevttrofi, meanangl, reco=8):
   
    numbenerrofi = meanenerrofi.size
    indxenerrofi = arange(numbenerrofi)
    numbevttrofi = indxevttrofi.size
    indxevttrofi = arange(numbevttrofi)

    strgpara = ['ntail', 'score', 'gcore', 'stail', 'gtail']
    
    path = retr_path('tdpy', onlydata=True) + 'irfn/ferm/'
    if reco == 8:
        path += 'psf_P8R2_SOURCE_V6_PSF.fits'
    else:
        path += 'psf_P7REP_SOURCE_V15_back.fits'
    irfn = pf.getdata(path, 1)
    minmener = irfn['energ_lo'].squeeze() * 1e-3 # [GeV]
    maxmener = irfn['energ_hi'].squeeze() * 1e-3 # [GeV]
    meanener = sqrt(minmener * maxmener)
    
    indxevtt = arange(4)
    numbevtt = indxevtt.size
    
    numbfermscalpara = 3
    numbfermformpara = 5
    
    fermscal = zeros((numbevttrofi, numbfermscalpara))
    fermform = zeros((numbenerrofi, numbevttrofi, numbfermformpara))
    for m in indxevttrofi:
        fermscal[m, :] = pf.getdata(path, 2 + 3 * m)['PSFSCALE']
        irfn = pf.getdata(path, 1 + 3 * m)
        for k in range(numbfermformpara):
            fermform[:, m, k] = interp1d(meanener, mean(irfn[strgpara[k]].squeeze(), axis=0))(meanenerrofi)

    fermscalfact = sqrt((fermscal[None, :, 0] * (10. * meanenerrofi[:, None])**fermscal[None, :, 2])**2 + fermscal[None, :, 1]**2)
    
    # convert N_tail to f_core
    for m in arange(fermform.shape[1]):
        for i in arange(fermform.shape[0]):
            fermform[i, m, 0] = 1. / (1. + fermform[i, m, 0] * fermform[i, m, 3]**2 / fermform[i, m, 1]**2)

    temp = sqrt(2. - 2. * cos(meanangl[None, :, None]))
    scalangl = 2. * arcsin(temp / 2.) / fermscalfact[:, None, :]
    
    fermform[:, :, 1] = fermscalfact * fermform[:, :, 1]
    fermform[:, :, 3] = fermscalfact * fermform[:, :, 3]

    frac = fermform[:, :, 0]
    sigc = fermform[:, :, 1]
    gamc = fermform[:, :, 2]
    sigt = fermform[:, :, 3]
    gamt = fermform[:, :, 4]
   
    fermpsfn = retr_doubking(scalangl, frac[:, None, :], sigc[:, None, :], gamc[:, None, :], sigt[:, None, :], gamt[:, None, :])

    return fermpsfn


def retr_fermpsfn_depr(gdat):
    
    irfn = pf.getdata(path, 1)
    minmener = irfn['energ_lo'].squeeze() * 1e-3 # [GeV]
    maxmener = irfn['energ_hi'].squeeze() * 1e-3 # [GeV]
    enerirfn = sqrt(minmener * maxmener)

    numbfermscalpara = 3
    numbfermformpara = 5
    
    fermscal = zeros((gdat.numbevtt, numbfermscalpara))
    fermform = zeros((gdat.numbener, gdat.numbevtt, numbfermformpara))
    
    parastrg = ['ntail', 'score', 'gcore', 'stail', 'gtail']
    for m in gdat.indxevtt:
        if reco == 8:
            irfn = pf.getdata(path, 1 + 3 * gdat.indxevttincl[m])
            fermscal[m, :] = pf.getdata(path, 2 + 3 * gdat.indxevttincl[m])['PSFSCALE']
        else:
            if m == 1:
                path = gdat.pathdata + 'expr/irfn/psf_P7REP_SOURCE_V15_front.fits'
            elif m == 0:
                path = gdat.pathdata + 'expr/irfn/psf_P7REP_SOURCE_V15_back.fits'
            else:
                continue
            irfn = pf.getdata(path, 1)
            fermscal[m, :] = pf.getdata(path, 2)['PSFSCALE']
        for k in range(numbfermformpara):
            fermform[:, m, k] = interp1d(enerirfn, mean(irfn[parastrg[k]].squeeze(), axis=0))(gdat.meanener)
        
    # convert N_tail to f_core
    for m in gdat.indxevtt:
        for i in gdat.indxener:
            fermform[i, m, 0] = 1. / (1. + fermform[i, m, 0] * fermform[i, m, 3]**2 / fermform[i, m, 1]**2)

    # store the fermi PSF parameters
    gdat.fermpsfipara = zeros((gdat.numbener * numbfermformpara * gdat.numbevtt))
    for m in gdat.indxevtt:
        for k in range(numbfermformpara):
            indxfermpsfiparatemp = m * numbfermformpara * gdat.numbener + gdat.indxener * numbfermformpara + k
            gdat.fermpsfipara[indxfermpsfiparatemp] = fermform[:, m, k]

    # calculate the scale factor
    gdat.fermscalfact = sqrt((fermscal[None, :, 0] * (10. * gdat.meanener[:, None])**fermscal[None, :, 2])**2 + fermscal[None, :, 1]**2)
    
    # evaluate the PSF
    gdat.fermpsfn = retr_psfn(gdat, gdat.fermpsfipara, gdat.indxener, gdat.binsangl, 'doubking')


def retr_fwhmferm(meanener, evtt):

    meanangl = linspace(0., 20., 1000)
    psfn = retr_fermpsfn(meanener, evtt, meanangl)
    fwhm = retr_fwhm(psfn, meanangl) 

    return fwhm


def retr_fwhm(psfn, binsangl):

    if psfn.ndim == 1:
        indxener = arange(1)
        indxevtt = arange(1)
        psfn = psfn[None, :, None]
    else:
        numbener = psfn.shape[0]
        indxener = arange(numbener)
        numbevtt = psfn.shape[2]
        indxevtt = arange(numbevtt)
    wdth = zeros((numbener, numbevtt))
    for i in indxener:
        for m in indxevtt:
            indxanglgood = argsort(psfn[i, :, m])
            intpwdth = max(0.5 * amax(psfn[i, :, m]), amin(psfn[i, :, m]))
            if intpwdth > amin(psfn[i, indxanglgood, m]) and intpwdth < amax(psfn[i, indxanglgood, m]):
                wdth[i, m] = interp1d(psfn[i, indxanglgood, m], binsangl[indxanglgood])(intpwdth)
    return wdth


def retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt):

    psfn = frac / 2. / pi / sigc**2 * (1. - 1. / gamc) * (1. + scaldevi**2 / 2. / gamc / sigc**2)**(-gamc) + \
        (1. - frac) / 2. / pi / sigt**2 * (1. - 1. / gamt) * (1. + scaldevi**2 / 2. / gamt / sigt**2)**(-gamt)
    
    return psfn


def retr_beam(meanener, indxevttthis, numbside, maxmmpol, fulloutp=False):
   
    numbpixl = 12 * numbside**2
    apix = 4. * pi / numbpixl

    numbener = meanener.size
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
    mapsoutp = retr_fermpsfn(meanener, evtt, meanangl)
    almcoutp = empty((numbener, maxmmpol+1, numbevtt))
    for i in range(numbener):
        for m in range(numbevtt):
            almcoutp[i, :, m] = real(hp.map2alm(mapsoutp[i, :, m], lmax=maxmmpol)[:maxmmpol+1])
    
    tranfunc = almcoutp / almcinpt[None, :, None]
    
    # temp
    tranfunc /= tranfunc[:, 0, :][:, None, :]

    if fulloutp:
        return tranfunc, almcinpt, almcoutp
    else:
        return tranfunc


def make_maps_main(gdat, pathdata):
    
    numbproc = len(gdat.recotype)
    
    if not hasattr(gdat, 'timetype'):
        gdat.timetype = ['tim0' for k in range(numbproc)]
    if not hasattr(gdat, 'enertype'):
        gdat.enertype = ['pnts' for k in range(numbproc)]
    if not hasattr(gdat, 'strgtime'):
        gdat.strgtime = ['tmin=INDEF tmax=INDEF' for k in range(numbproc)]
    if not hasattr(gdat, 'timefrac'):
        gdat.timefrac = [1. for k in range(numbproc)]
    if not hasattr(gdat, 'numbside'):
        gdat.numbside = [256 for k in range(numbproc)]
    if not hasattr(gdat, 'test'):
        gdat.test = False

    gdat.evtc = []
    gdat.photpath = []
    gdat.strgtime = []
    gdat.weekinit = []
    gdat.weekfinl = []
    for n in range(numbproc):
        if gdat.recotype[n] == 'rec7':
            gdat.evtc.append(2)
            gdat.photpath.append('p7v6c')
            gdat.strgtime.append('tmin=239155201 tmax=364953603')
            gdat.weekinit.append(9)
            gdat.weekfinl.append(218)
        if gdat.recotype[n] == 'rec8':
            gdat.evtc.append(128)
            gdat.photpath.append('photon')
            gdat.strgtime.append('tmin=INDEF tmax=INDEF')
            gdat.weekinit.append(11)
            gdat.weekfinl.append(420)
    
    gdat.strgener = ['gtbndefn_%s.fits' % gdat.enertype[k] for k in range(numbproc)]
    
    gdat.indxevtt = arange(4)

    gdat.pathdata = pathdata
    
    gdat.evtt = [4, 8, 16, 32]

    indxproc = arange(numbproc)
    
    if numbproc == 1:
        make_maps_work(gdat, 0)
    else:
        # process pool
        pool = mp.Pool(numbproc)

        # spawn the processes
        make_maps_part = functools.partial(make_maps_work, gdat)
        pool.map(make_maps_part, indxproc)
        pool.close()
        pool.join()


def retr_path(strg, pathextndata=None, pathextnimag=None, rtag=None, onlyimag=False, onlydata=False):
    
    pathbase = os.environ[strg.upper() + '_DATA_PATH'] + '/'

    if not onlyimag:
        pathdata = pathbase
        if pathextndata != None:
            pathdata += pathextndata
        pathdata += 'data/'
        os.system('mkdir -p %s' % pathdata)
    if not onlydata:        
        pathimag = pathbase
        if pathextnimag != None:
            pathimag += pathextnimag
        pathimag += 'imag/'
        if rtag != None:
            pathimag += rtag + '/'
        os.system('mkdir -p %s' % pathimag)

    if not onlyimag and not onlydata:
        return pathimag, pathdata
    elif onlyimag:
        return pathimag
    else:
        return pathdata

def conv_rascdecl(args):

    rasc = args[0] * 8 + args[1] / 60. + args[2] / 3600.
    decl = args[3] + args[4] / 60. + args[5] / 3600.

    return rasc, decl


def make_maps_work(gdat, indxprocwork):

    rtag = '%s_%s_%04d_%s' % (gdat.recotype[indxprocwork], gdat.enertype[indxprocwork], gdat.numbside[indxprocwork], gdat.timetype[indxprocwork])
    
    # make file lists
    infl = gdat.pathdata + '/phot_%s.txt' % rtag
    spac = gdat.pathdata + '/spac_%s.txt' % rtag
        
    numbweek = (gdat.weekfinl[indxprocwork] - gdat.weekinit[indxprocwork]) * gdat.timefrac[indxprocwork]
    listweek = floor(linspace(gdat.weekinit[indxprocwork], gdat.weekfinl[indxprocwork] - 1, numbweek)).astype(int)
    cmnd = 'rm -f ' + infl
    os.system(cmnd)
    cmnd = 'rm -f ' + spac
    os.system(cmnd)
    for week in listweek:
        cmnd = 'ls -d -1 $FERMI_DATA/weekly/spacecraft/*_w%03d_* >> ' % week + spac
        os.system(cmnd)
        cmnd = 'ls -d -1 $FERMI_DATA/weekly/%s/*_w%03d_* >> ' % (gdat.photpath[indxprocwork], week) + infl
        os.system(cmnd)
    for m in gdat.indxevtt:

        if gdat.recotype[indxprocwork] == 'rec7':
            strgirfn = 'P7REP_SOURCE_V15'
        if gdat.recotype[indxprocwork] == 'rec8':
            strgirfn = 'P8R2_SOURCE_V6'

        if gdat.recotype[indxprocwork] == 'rec7':
            if m == 3:
                thisevtt = 1
                thisevttdepr = 0
            elif m == 2:
                thisevtt = 2
                thisevttdepr = 1
            else:
                continue
            strgpsfn = 'convtype=%d' % thisevttdepr

        if gdat.recotype[indxprocwork] == 'rec8':
            thisevtt = gdat.evtt[m]
            strgpsfn = 'evtype=%d' % thisevtt
         
        sele = gdat.pathdata + '/fermsele_%04d_%s.fits' % (thisevtt, rtag)
        filt = gdat.pathdata + '/fermfilt_%04d_%s.fits' % (thisevtt, rtag)
        live = gdat.pathdata + '/fermlive_%04d_%s.fits' % (thisevtt, rtag)
        cnts = gdat.pathdata + '/fermcnts_%04d_%s.fits' % (thisevtt, rtag)
        expo = gdat.pathdata + '/fermexpo_%04d_%s.fits' % (thisevtt, rtag)
        psfn = gdat.pathdata + '/fermpsfn_%04d_%s.fits' % (thisevtt, rtag)

        cmnd = 'gtselect infile=' + infl + ' outfile=' + sele + ' ra=INDEF dec=INDEF rad=INDEF ' + \
            gdat.strgtime[indxprocwork] + ' emin=100 emax=100000 zmax=90 evclass=%d %s' % (gdat.evtc[indxprocwork], strgpsfn)
        
        if os.path.isfile(cnts) and os.path.isfile(expo):
            continue
        
        if gdat.test:
            print cmnd
            print ''
        else:
            os.system(cmnd)

        cmnd = 'gtmktime evfile=' + sele + ' scfile=' + spac + ' filter="DATA_QUAL==1 && LAT_CONFIG==1"' + ' outfile=' + filt + ' roicut=no'
        if gdat.test:
            print cmnd
            print ''
        else:
            os.system(cmnd)

        cmnd = 'gtbin evfile=' + filt + ' scfile=NONE outfile=' + cnts + \
            ' ebinalg=FILE ebinfile=$TDPY_DATA_PATH/%s ' % gdat.strgener[indxprocwork] + \
            'algorithm=HEALPIX hpx_ordering_scheme=RING coordsys=GAL hpx_order=%d hpx_ebin=yes' % log2(gdat.numbside[indxprocwork])
        if gdat.test:
            print cmnd
            print ''
        else:
            os.system(cmnd)

        cmnd = 'gtltcube evfile=' + filt + ' scfile=' + spac + ' outfile=' + live + ' dcostheta=0.025 binsz=1'
        if gdat.test:
            print cmnd
            print ''
        else:
            os.system(cmnd)

        cmnd = 'gtexpcube2 infile=' + live + ' cmap=' + cnts + ' outfile=' + expo + ' irfs=CALDB evtype=%03d bincalc=CENTER' % thisevtt
        if gdat.test:
            print cmnd
            print ''
        else:
            os.system(cmnd)

        psfno = gdat.pathdata + '/fermpsfn_%04d_%s.fits' % (thisevtt, rtag)
        cmnd = 'gtpsf %s %s %s %.4g %.4g ebinalg=FILE ebinfile=$TDPY_DATA_PATH/%s 10. 50' % (live, psfn, strgpsfn, rasc, decl) + \
            ' ebinalg=FILE ebinfile=$TDPY_DATA_PATH/%s ' % gdat.strgener[indxprocwork]
        if gdat.test:
            print cmnd
            print ''
        else:
            os.system(cmnd)

    cmnd = 'rm %s %s %s %s %s' % (infl, spac, sele, filt, live)
    os.system(cmnd)


def smth_ferm(mapsinpt, meanener, indxevttthis, maxmmpol=None, makeplot=False, gaus=False):
    
    numbpixl = mapsinpt.shape[1]

    numbside = int(sqrt(numbpixl / 12))
    if maxmmpol == None:
        maxmmpol = 3 * numbside - 1

    numbener = meanener.size
    numbevtt = indxevttthis.size
    
    numbalmc = (maxmmpol + 1) * (maxmmpol + 2) / 2
    
    # get the beam
    beam = retr_beam(meanener, indxevttthis, numbside, maxmmpol)
    
    # construct the transfer function
    tranfunc = ones((numbener, numbalmc, numbevtt))
    cntr = 0
    for n in arange(maxmmpol+1)[::-1]:
        tranfunc[:, cntr:cntr+n+1, :] = beam[:, maxmmpol-n:, :]
        cntr += n + 1

    mapsoutp = empty_like(mapsinpt)

    for i in indxener:
        for m in indxevtt:
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
    plot_maps(path, mapstemp, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)

    for i in arange(meanenerplot.size):
        for m in arange(indxevttplot.size):
            path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/mapssmthferm%d%d.pdf' % (i, m)
            plot_maps(path, mapssmthferm[i, :, m], minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)
            
    path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/mapssmthgaus.pdf'
    plot_maps(path, mapssmthgaus, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)


def prca(matr):
    
    M = (matr - mean(matr.T, 1)).T
    eigl, eigt = linalg.eig(cov(M))
    tranmatr = dot(eigt.T, M).T
    
    return eigl, tranmatr, eigt


def test_prca():

    npara = 2
    nsamp = 1000
    matr = zeros((nsamp, 2))
    matr[:, 0] = randn(nsamp)
    matr[:, 1] = randn(nsamp) * 10.
    eigl, tranmatr, eigt = prca(matr)
    
    plt.scatter(matr[:, 0], matr[:, 1])
    plt.show()
    
    plt.scatter(tranmatr[:, 0], tranmatr[:, 1])
    plt.show()


