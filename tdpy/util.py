## numerics
import numpy as np
import scipy as sp
from scipy.special import erfi
import scipy.fftpack
import scipy.stats

import matplotlib.pyplot as plt

# utilities
import os, time, datetime

# astropy
import astropy.coordinates, astropy.units
import astropy.io


class gdatstrt(object):

    def __init__(self):
        self.boollockmodi = False
        pass
    
    def __setattr__(self, attr, valu):
        super(gdatstrt, self).__setattr__(attr, valu)


class gdatstrtpcat(object):
    
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
        
        if len(attr) % 4 != 0 and not attr.startswith('path'):
            raise Exception('')
        
        #if attr == 'thislliktotl' and hasattr(self, attr) and getattr(self, attr) - 100. > valu:
        #    raise Exception('Trying to decrease lliktotl too much...')
       
        # temp
        #if attr == 'indxsampmodi':
        #    if not isinstance(valu, int64) and len(valu) > 6:
        #        raise Exception('Setting gdatmodi.indxsampmodi to an array!')
        
        super(gdatstrt, self).__setattr__(attr, valu)


class datapara(object):

    def __init__(self, numbpara):
        
        self.numbpara = numbpara
        
        self.indx = dict()
        self.name = np.empty(numbpara, dtype=object)
        self.minm = np.zeros(numbpara)
        self.maxm = np.zeros(numbpara)
        self.scal = np.empty(numbpara, dtype=object)
        self.labl = np.empty(numbpara, dtype=object)
        self.unit = np.empty(numbpara, dtype=object)
        self.vari = np.zeros(numbpara)
        self.true = np.zeros(numbpara)
        self.strg = np.empty(numbpara, dtype=object)
        
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
    timediff = np.empty(numbiter)
    for k in range(numbiter):
        timeinit = time.time()
        func(*args)
        timediff[k] = time.time() - timeinit
    
    return mean(timediff), std(timediff)


def retr_specbbod(tmpt, wlen):
    
    #0.0143877735e6 # [um K]
    spec = 3.742e11 / wlen**5 / (np.exp(0.0143877735e6 / (wlen * tmpt)) - 1.)
    
    return spec


def time_func_verb(func, *args):
    
    meantimediff, stdvtimediff = time_func(func, *args)
    

def retr_pctlvarb(listvarb):

    shap = np.zeros(len(listvarb.shape), dtype=int)
    shap[0] = 3
    shap[1:] = listvarb.shape[1:]
    shap = list(shap)
    postvarb = np.zeros(shap)
    
    postvarb[0, ...] = np.percentile(listvarb, 50., axis=0)
    postvarb[1, ...] = np.percentile(listvarb, 16., axis=0)
    postvarb[2, ...] = np.percentile(listvarb, 84., axis=0)

    return postvarb


def prnt_list(listvarb, strg):
    
    medi = np.median(listvarb)
    lowr = np.percentile(listvarb, 16.)
    uppr = np.percentile(listvarb, 84.)
    print('%s: %.3g +%.3g -%.3g' % (strg, medi, medi - lowr, uppr - medi))


def retr_errrvarb(inpt, samp=False):

    if samp:
        postvarb = retr_pctlvarb(inpt)
    else:
        postvarb = inpt

    errr = np.abs(postvarb[0, ...] - postvarb[1:3, ...])

    return errr


def retr_kdegpdfn(listsamp, binsvarb, stdv):
    
    meanvarb = (binsvarb[1:] + binsvarb[:-1]) / 2.
    deltvarb = (binsvarb[1:] - binsvarb[:-1]) / 2.
    kdeg = retr_kdeg(listsamp, meanvarb, stdv)
    kdegpdfn = kdeg / np.sum(kdeg)
    kdegpdfn /= deltvarb
    
    return kdeg


def retr_kdeg(listsamp, varb, stdv):
    
    if np.isscalar(varb):
        varb = np.array([varb])
    kdeg = np.sum(np.exp(-0.5 * (varb[None, :] - listsamp[:, None])**2 / stdv**2), axis=0)
    
    return kdeg


def retr_listlablparaaugm(listlablpara):
    
    listlablparaaugm = []
    for lablpara in listlablpara:
        if lablpara[1] != '':
            listlablparaaugm.append('%s [%s]' % (lablpara[0], lablpara[1]))
        else:
            listlablparaaugm.append('%s' % lablpara[0])
   
    return listlablparaaugm


def plot_recaprec(pathimag, strgextn, listvarbreca, listvarbprec, \
                                   liststrgvarbreca, liststrgvarbprec, listlablvarbreca, listlablvarbprec, \
                                                    boolposirele, boolreleposi, strgplotextn='pdf', verbtype=2, numbbins=10):
    
    verbtype = 2

    listlablvarbrecaaugm = retr_listlablparaaugm(listlablvarbreca)
    listlablvarbprecaugm = retr_listlablparaaugm(listlablvarbprec)
    
    if verbtype > 1:
        print('numbbins')
        print(numbbins)
    
    if isinstance(boolposirele, list):
        raise Exception('')

    indxbins = np.arange(numbbins)
    for c in range(2):
        if verbtype > 1:
            print('c')
            print(c)
        if c == 0:
            if boolposirele.size != listvarbreca.shape[0]:
                print('listvarbreca')
                summgene(listvarbreca)
                print('boolposirele')
                summgene(boolposirele)
                raise Exception('')
            listvarb = listvarbreca
            liststrgvarb = liststrgvarbreca
            listlablvarbtemp = listlablvarbrecaaugm
            strgmetr = 'reca'
            strgyaxi = 'Recall'
        else:
            listvarb = listvarbprec
            liststrgvarb = liststrgvarbprec
            listlablvarbtemp = listlablvarbprecaugm
            strgmetr = 'prec'
            strgyaxi = 'Precision'
            
        k = 0
        for k, strgvarb in enumerate(liststrgvarb):
        
            if verbtype > 1:
                print('k')
                print(k)
                print('strgvarb')
                print(strgvarb)
                print('listvarb[:, k]')
                summgene(listvarb[:, k])
            bins = np.linspace(np.amin(listvarb[:, k]), np.amax(listvarb[:, k]), numbbins + 1)
            meanvarb = (bins[1:] + bins[:-1]) / 2.
            metr = np.zeros(numbbins) + np.nan
            for a in indxbins:
                indx = np.where((bins[a] < listvarb[:, k]) & (listvarb[:, k] < bins[a+1]))[0]
                numb = indx.size
                if verbtype > 1:
                    print('a')
                    print(a)
                    print('indx')
                    summgene(indx)
                if numb > 0:
                    # recall
                    if c == 0:
                        if verbtype > 1:
                            print('boolposirele')
                            summgene(boolposirele)
                        metr[a] = float(np.sum(boolposirele[indx].astype(float))) / numb
                    # precision
                    if c == 1:
                        if verbtype > 1:
                            print('boolreleposi')
                            print(type(boolreleposi))
                            summgene(boolreleposi)
                            print('indx')
                            summgene(indx)
                        print('boolreleposi')
                        summgene(boolreleposi)
                        print('indx')
                        summgene(indx)
                        metr[a] = float(np.sum(boolreleposi[indx].astype(float))) / numb
                    if metr[a] < 0:
                        raise Exception('')
            figr, axis = plt.subplots(figsize=(6, 4))
            axis.plot(meanvarb, metr)
            axis.set_xlabel(listlablvarbtemp[k])
            axis.set_ylabel(strgyaxi)
            path = pathimag + '%s_%s_%s.%s' % (strgmetr, strgvarb, strgextn, strgplotextn) 
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            k += 1


def prep_mask(data, epoc=None, peri=None, duramask=None, limttime=None):
    '''
    Read the data, mask out the transits
    '''
    
    time = data[:, 0]
    
    if epoc is not None:
        listindxtimethis = []
        for n in range(-2000, 2000):
            timeinit = epoc + n * peri - duramask / 24.
            timefinl = epoc + n * peri + duramask / 24.
            indxtimethis = np.where((time > timeinit) & (time < timefinl))[0]
            listindxtimethis.append(indxtimethis)
        indxtimethis = np.concatenate(listindxtimethis)
    else:
        indxtimethis = np.where((time < limttime[1]) & (time > limttime[0]))[0]

    numbtime = time.size
    indxtime = np.arange(numbtime)
    
    indxtimegood = np.setdiff1d(indxtime, indxtimethis)
    
    dataoutp = data[indxtimegood, :]
    
    return dataoutp


def prep_mask_knwn(pathbase, epoc, peri, duramask, listindxtranmask=None, strgextn=None):
    
    '''
    Read the TESS data, mask out the transits, write csv files to the allesfitter folders
    '''
    
    pathdata = pathbase + 'data_preparation/'
    pathinpt = '%sPDCSAP/TESS.csv' % pathbase
    pathoutp = '%sallesfits/allesfit_TESS_oot/TESS.csv' % pathbase
    pathorig = pathdata + 'original_data/'
    pathtess = pathdata + 'PDCSAP/TESS.csv'
    
    print('Reading the TESS PDCSAP light curve and saving the transit-masked light curve to the OOT allesfit folder...')
    
    # read data
    data = np.loadtxt(pathtess, delimiter=',')
    
    # mask out the transits
    dataoutp = prep_mask(data, epoc, peri, duramask)

    # save to CSV file
    np.savetxt(pathoutp, dataoutp, delimiter=',')


def rbin(arry, shap):
        
    arry = arry[[slice(None, None, thissize / nextsize) for thissize, nextsize in zip(arry.shape, shap)]]
    
    return arry


def retr_nfwp(nfwg, numbside, norm=None):
    
    edenlocl = 0.3 # [GeV/cm^3]
    radilocl = 8.5 # [kpc]
    rscl = 23.1 # [kpc]
    
    nradi = 100
    minmradi = 1e-2
    maxmradi = 1e2
    radi = np.logspace(np.log10(minmradi), np.log10(maxmradi), nradi)
    
    nsadi = 100
    minmsadi = 0.
    maxmsadi = 2. * radilocl
    sadi = np.linspace(minmsadi, maxmsadi, nsadi)
    
    lghp, bghp, numbpixl, apix = retr_healgrid(numbside)
    
    cosigahp = cos(np.deg2rad(lghp)) * cos(np.deg2rad(bghp))
    gahp = np.rad2deg(arccos(cosigahp))
    
    eden = 1. / (radi / rscl)**nfwg / (1. + radi / rscl)**(3. - nfwg)
    eden *= edenlocl / interp1d(radi, eden)(radilocl)
    
    edengrid = np.zeros((nsadi, numbpixl))
    for i in range(nsadi):
        radigrid = np.sqrt(radilocl**2 + sadi[i]**2 - 2 * radilocl * sadi[i] * cosigahp)
        edengrid[i, :] = interp1d(radi, eden)(radigrid)

    edengridtotl = sum(edengrid**2, axis=0)

    if norm != None:
        jgahp = np.argsort(gahp)
        edengridtotl /= interp1d(gahp[jgahp], edengridtotl[jgahp])(5.)
        
    return edengridtotl


def mexp(numb):
    if numb == 0.:
        strg = '0'
    else:
        logn = np.log10(np.fabs(numb))
        expo = np.floor(logn)
        expo = int(expo)
        mant = 10**(logn - expo) * numb / np.fabs(numb)
        
        if np.fabs(numb) > 1e2 or np.fabs(numb) < 1e-2:
            if mant == 1. or mant == -1.:
                strg = r'$10^{%d}$' % expo
            else:
                strg = r'$%.3g \times 10^{%d}$' % (mant, expo)
        else:
            strg = r'%.3g' % numb

    return strg


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
            arry = np.logspace(np.log10(minm), np.log10(maxm), numbtemp) 
        if scal == 'self':
            arry = np.linspace(minm, maxm, numbtemp) 
        
        self.name.append(name)
        self.para.append(arry)
        self.scal.append(scal)
        self.strg.append(strg)
        self.size = len(self.para)

    
def summgene(varb):
    
    try:
        print(np.amin(varb))
        print(np.amax(varb))
        print(np.mean(varb))
        print(varb.shape)
    except:
        print(varb)


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
    
    mult = np.zeros((numbener, nmass))
    for k in range(nmass):
        jp4dm = np.where(abs(p4dm[:, 0] - mass[k]) == 0)[0]

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


def show_prog(cntr, maxmcntr, thiscntr, nprog=20, indxprocwork=None, showmemo=False, accp=None, accpprio=None):

    nextcntr = int(nprog * float(cntr + 1) / maxmcntr) * 100 / nprog
    if nextcntr > thiscntr:
        if indxprocwork != None:
            print('Process %d is %3d%% completed.' % (indxprocwork, nextcntr))
        else:
            print('%3d%% completed.' % nextcntr)
        if accp != None:
            print('Acceptance ratio: %.3g%%' % accp)
            print('Acceptance through prior boundaries: %.3g%%' % accpprio)
        thiscntr = nextcntr
        if showmemo:
            show_memo_simp()
        
    return thiscntr            


def retr_galcfromequc(rasc, decl):

    icrs = astropy.coordinates.SkyCoord(ra=rasc*astropy.units.degree, dec=decl*astropy.units.degree)

    lgal = icrs.galactic.l.degree
    bgal = icrs.galactic.b.degree
    
    return lgal, bgal


def regr(xdat, ydat, ordr):
    
    coef = polyfit(xdat, ydat, ordr)
    func = poly1d(coef)
    strg = '$y = '
    if ordr == 0:
        strg += '%.5g$'
    if ordr == 1:
        strg += '%.5g x + %.5g$' % (coef[0], coef[1])
    if ordr == 2:
        strg += '%.5g x^2 + %.5g x + %.5g$' % (coef[0], coef[1], coef[2])

    return coef, func, strg


def corr_catl(lgalseco, bgalseco, lgalfrst, bgalfrst, anglassc=np.pi/180., verbtype=1):

    numbfrst = lgalfrst.size

    indxsecoassc = np.zeros(numbfrst, dtype=int) - 1
    numbassc = np.zeros(numbfrst, dtype=int)
    distassc = np.zeros(numbfrst) + 1000.
    lgalbgalfrst = np.array([lgalfrst, bgalfrst])
    thisfraccomp = -1
    numbseco = lgalseco.size
    for k in range(numbseco):
        lgalbgalseco = np.array([lgalseco[k], bgalseco[k]])
        dist = angdist(lgalbgalfrst, lgalbgalseco, lonlat=True)
        thisindxfrst = np.where(dist < anglassc)[0]
        
        if thisindxfrst.size > 0:
            
            # if there are multiple associated true PS, sort them
            indx = np.argsort(dist[thisindxfrst])
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
            print('%02d%% completed.' % thisfraccomp)

    return indxsecoassc


def show_memo_simp():
    
    memoresi, memoresiperc = retr_memoresi()

    strgmemo = retr_strgmemo(memoresi)

    print('Resident memory: %s, %4.3g%%' % (strgmemo, memoresiperc))


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
            bins = np.linspace(minm, maxm, numb + 1)
            mean = (bins[1:] + bins[:-1]) / 2.
        else:
            bins = np.logspace(np.log10(minm), np.log10(maxm), numb + 1)
            mean = np.sqrt(bins[1:] * bins[:-1])
    else:
        if scal == 'self':
            mean = (bins[1:] + bins[:-1]) / 2.
        else:
            mean = np.sqrt(bins[1:] * bins[:-1])
        numb = mean.size
        
    indx = np.arange(numb)
   
    return bins, mean, diff(bins), numb, indx


def retr_psfngausnorm(angl):

    norm = np.sqrt(2. / np.pi**3) / angl / exp(-0.5 * angl**2) / \
                        real(-erfi((angl**2 - np.pi * 1j) / np.sqrt(2) / angl) - erfi((angl**2 + np.pi * 1j) / np.sqrt(2) / angl) + 2. * erfi(angl / np.sqrt(2.)))

    return norm


def retr_mapspnts(lgal, bgal, stdv, flux, numbside=256, verbtype=0):
    
    # lgal, bgal and stdv are in degrees
    numbpnts = lgal.size
    lgalheal, bgalheal, numbpixl, apix = retr_healgrid(numbside)
    gridheal = np.array([lgalheal, bgalheal])
    stdvradi = np.deg2rad(stdv)
    mapspnts = np.zeros(numbpixl)
    for n in range(numbpnts):
        gridpnts = np.array([lgal[n], bgal[n]])
        angl = angdist(gridheal, gridpnts, lonlat=True)
        norm = retr_psfngausnorm(stdvradi)
        mapspnts += apix * norm * flux[n] * exp(-0.5 * angl**2 / stdvradi**2)

    return mapspnts


def retr_mapsplnkfreq(indxpixloutprofi=None, numbsideoutp=256, indxfreqrofi=None):

    numbside = 2048
    numbpixl = 12 * numbside**2
    meanfreq = np.array([30, 44, 70, 100, 143, 217, 353, 545, 857])
    numbfreq = meanfreq.size
    indxfreq = np.arange(numbfreq)
    strgfreq = ['%04d' % meanfreq[k] for k in indxfreq]
    
    indxpixloutp = np.arange(numbsideoutp)

    if indxfreqrofi == None:
        indxfreqrofi = indxfreq

    if indxpixloutprofi == None:
        indxpixloutprofi = indxpixloutp

    rtag = '_%04d' % (numbsideoutp)

    path = retr_path('tdpy', onlydata=True)
    pathsbrt = path + 'plnksbrt_%s.fits' % rtag
    pathsbrtstdv = path + 'plnksbrtstdv_%s.fits' % rtag
    if os.path.isfile(pathsbrt) and os.path.isfile(pathsbrtstdv):
        print('Reading %s...' % pathsbrt)
        mapsplnkfreq = pf.getdata(pathsbrt)
        print('Reading %s...' % pathsbrtstdv)
        mapsplnkfreqstdv= pf.getdata(pathsbrtstdv)
    else:
        mapsplnkfreq = np.zeros((numbfreq, numbpixl))
        mapsplnkfreqstdv = np.zeros((numbfreq, numbpixl))
        for k in indxfreq:
            print('Processing Planck Map at %d GHz...' % (meanfreq[k]))
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
            numbsideinpt = int(np.sqrt(numbpixlinpt / 12))
            mapsplnkfreq[k, :] = pf.getdata(path + strg, 1)[strgcols]
            mapsplnkfreq[k, :] = hp.reorder(mapsplnkfreq[k, :], n2r=True)
        
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
                    indxfreqbandgood = np.where(tranband > 1e-6)[0]
                    indxfreqbandgood = np.arange(amin(indxfreqbandgood), amax(indxfreqbandgood) + 1)
        
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
        
        pf.writeto(pathsbrt, mapsplnkfreq, clobber=True)
        pf.writeto(pathsbrtstdv, mapsplnkfreqstdv, clobber=True)
        
    return mapsplnkfreq, mapsplnkfreqstdv


def retr_indximagmaxm(data):

    sizeneig = 10
    cntpthrs = 10
    maxmdata = sp.ndimage.filters.maximum_filter(data, sizeneig)
    
    boolmaxm = (data == maxmdata)
    minmdata = sp.ndimage.filters.minimum_filter(data, sizeneig)
    diff = ((maxmdata - minmdata) > cntpthrs)
    boolmaxm[diff == 0] = 0
    mapslabl, numbobjt = sp.ndimage.label(boolmaxm)
    mapslablones = np.zeros_like(mapslabl)
    mapslablones[np.where(mapslabl > 0)] = 1.
    indxmaxm = np.array(sp.ndimage.center_of_mass(data, mapslabl, range(1, numbobjt+1))).astype(int)
    if len(indxmaxm) == 0:
        indxydatmaxm = np.array([0])
        indxxdatmaxm = np.array([0])
    else:
        indxydatmaxm = indxmaxm[:, 1]
        indxxdatmaxm = indxmaxm[:, 0]
    return indxxdatmaxm, indxydatmaxm


def plot_gene(path, xdat, ydat, yerr=None, scalxdat=None, scalydat=None, \
                                            lablxdat='', lablydat='', plottype=None, limtxdat=None, limtydat=None, colr=None, listlinestyl=None, \
                                            alph=None, listlegd=None, listvlinfrst=None, listvlinseco=None, listhlin=None, drawdiag=False):
    
    if not isinstance(ydat, list):
        listydat = [ydat]
    else:
        listydat = ydat
  
    if yerr != None and not isinstance(ydat, list):
        listyerr = [yerr]
    else:
        listyerr = yerr

    numbelem = len(listydat)
    
    if listlinestyl == None:
        listlinestyl = [None for k in range(numbelem)]

    if listlegd == None:
        listlegd = [None for k in range(numbelem)]

    if not isinstance(xdat, list):
        listxdat = [xdat for k in range(numbelem)]
    else:
        listxdat = xdat
    
    if plottype == None:
        listplottype = ['line' for k in range(numbelem)]
    elif not isinstance(plottype, list):
        listplottype = [plottype]
    else:
        listplottype = plottype
    
    figr, axis = plt.subplots(figsize=(6, 6))
    
    for k in range(numbelem):
        xdat = listxdat[k]
        ydat = listydat[k]
        plottype = listplottype[k]
        legd = listlegd[k]
        if plottype == 'scat':
            axis.scatter(xdat, ydat, color=colr, alpha=alph, label=legd, s=2)
        elif plottype == 'hist':
            deltxdat = xdat[1] - xdat[0]
            axis.bar(xdat - deltxdat / 2., ydat, deltxdat, color=colr, alpha=alph, label=legd)
        else:
            if listyerr != None:
                axis.errorbar(xdat, ydat, yerr=listyerr[k], color=colr, lw=2, alpha=alph, label=legd, ls=listlinestyl[k])
            else:
                axis.plot(xdat, ydat, color=colr, lw=2, alpha=alph, label=legd, ls=listlinestyl[k])
    
    if listlegd != None:
        axis.legend()

    if scalxdat == 'logt':
        axis.set_xscale('log')
    if scalydat == 'logt':
        axis.set_yscale('log')
    
    if limtxdat == None:
        limtxdat = [np.amin(np.concatenate(listxdat)), np.amax(np.concatenate(listxdat))]
    if limtydat == None:
        limtydat = [np.amin(np.concatenate(listydat)), np.amax(np.concatenate(listydat))]
    
    if drawdiag:
        axis.plot(limtxdat, limtxdat, ls='--', alpha=0.3, color='black')
    
    axis.set_xlim(limtxdat)
    axis.set_ylim(limtydat)
    
    if listhlin != None:
        if isscalar(listhlin):
            listhlin = [listhlin]
        for k in range(len(listhlin)):
            axis.axhline(listhlin[k], ls='--', alpha=0.2, color=colr)
    
    if listvlinfrst != None:
        if isscalar(listvlinfrst):
            listvlinfrst = [listvlinfrst]
        for k in range(len(listvlinfrst)):
            axis.axvline(listvlinfrst[k], ls='--', alpha=0.2, color=colr)
    
    if listvlinseco != None:
        if isscalar(listvlinseco):
            listvlinseco = [listvlinseco]
        for k in range(len(listvlinseco)):
            axis.axvline(listvlinseco[k], ls='-.', alpha=0.2, color=colr)
    
    axis.set_xlabel(lablxdat)
    axis.set_ylabel(lablydat)

    figr.tight_layout()
    figr.savefig(path)
    plt.close(figr)


def cart_heal(cart, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, numbside=256):
    
    numbbgcr = cart.shape[0]
    numblgcr = cart.shape[1]
    lghp, bghp, numbpixl, apix = retr_healgrid(numbside)
    
    indxlgcr = (numblgcr * (lghp - minmlgal) / (maxmlgal - minmlgal)).astype(int)
    indxbgcr = (numbbgcr * (bghp - minmbgal) / (maxmbgal - minmbgal)).astype(int)
    
    indxpixlrofi = np.where((minmlgal <= lghp) & (lghp <= maxmlgal) & (minmbgal <= bghp) & (bghp <= maxmbgal))[0]
    
    heal = np.zeros(numbpixl)
    heal[indxpixlrofi] = np.fliplr(cart)[indxbgcr[indxpixlrofi], indxlgcr[indxpixlrofi]]
    
    return heal


class cntr():
    def gets(self):
        return self.cntr

    def incr(self, valu=1):
        temp = self.cntr
        self.cntr += valu
        return temp
    def __init__(self):
        self.cntr = 0


def retr_evttferm(recotype):
    
    if recotype == 'rec7':
        evtt = np.array([1, 2])
    if recotype == 'rec8':
        evtt = np.array([16, 32])
    if recotype == 'manu':
        evtt = np.array([0, 1])
    
    numbevtt = evtt.size
    indxevtt = np.arange(numbevtt)

    return evtt, numbevtt, indxevtt


def writ_sbrtfdfm(numbside=256, regitype='igal', binsenertype='full', recotype='rec7'):
    
    pathdata = os.environ["PCAT_DATA_PATH"] + '/data/'
    
    evtt, numbevtt, indxevtt = retr_evttferm(recotype)
    
    binsener = np.array([0.1, 0.3, 1., 3., 10., 100.])
    
    meanener = np.sqrt(binsener[1:] * binsener[:-1])
    numbpixl = 12 * numbside**2
    numbener = binsener.size - 1

    # get the Fermi-LAT diffuse model
    sbrtfdfm = retr_sbrtfdfm(binsener, numbside)
    
    # rotate if necessary
    for m in indxevtt:
        if regitype == 'ngal':
            for i in range(numbener):
                almc = hp.map2alm(sbrtfdfm[i, :])
                hp.rotate_alm(almc, 0., 0.5 * np.pi, 0.)
                sbrtfdfm[i, :] = hp.alm2map(almc, numbside)

    # smooth the model
    sbrtfdfm = smth_ferm(sbrtfdfm[:, :, None], meanener, recotype)

    path = pathdata + 'sbrtfdfm%04d%s%s%s.fits' % (numbside, recotype, binsenertype, regitype)
    pf.writeto(path, sbrtfdfm, clobber=True)


def retr_strgtimestmp():

    strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    return strgtimestmp


def read_fits(path, pathimag=None, full=False, verbtype=0):
    
    if pathimag != None:
        os.system('mkdir -p ' + pathimag)
    
    hdun = pf.open(path)
    numbhead = len(hdun)
    listdata = []
    for k in range(numbhead):
        head = hdun[k].header
        data = hdun[k].data
        
        listdata.append(data)

        if data == None:
            continue

        arry = np.array(stack((head.keys(), head.values()), 1))
        listtype = []
        listform = []
        listunit = []
       
        for n in range(arry.shape[0]):
            if arry[n, 0].startswith('TTYPE'):
                listtype.append(arry[n, 1])
            if arry[n, 0].startswith('TFORM'):
                listform.append(arry[n, 1])
            if arry[n, 0].startswith('TUNIT'):
                listunit.append(arry[n, 1])

        if pathimag != None:
            cmnd = 'convert -density 300'
    
            for n in range(len(listtype)):
                if not listform[n].endswith('A') and isfinite(data[listtype[n]]).all():
                    figr, axis = plt.subplots()
                    bins = np.linspace(amin(data[listtype[n]]), amax(data[listtype[n]]), 100)
                    axis.hist(data[listtype[n]], bins=bins)
                    #axis.set_xlabel('%s [%s]' % (listtype[n], listunit[n]))
                    axis.set_yscale('log')
                    
                    axis.set_xlabel('%s' % (listtype[n]))
                    plt.tight_layout()
                    path = pathimag + 'readfits_%s.pdf' % listtype[n]
                    cmnd += ' ' + path
                    figr.savefig(path)
                    plt.close(figr)
        
            cmnd += ' ' + pathimag + 'merg.pdf'
            os.system(cmnd)

    return listdata


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
        mapstemp = np.zeros(numbpixl)
        mapstemp[indxpixlrofi] = maps
        maps = mapstemp
    else:
        numbpixl = maps.size
    
    if numbsidelgal == None:
        numbsidelgal = min(4 * int((maxmlgal - minmlgal) / np.rad2deg(np.sqrt(4. * np.pi / numbpixl))), 2000)
    if numbsidebgal == None:
        numbsidebgal = min(4 * int((maxmbgal - minmbgal) / np.rad2deg(np.sqrt(4. * np.pi / numbpixl))), 2000)
    
    # saturate the map
    if satu:
        mapstemp = copy(maps)
        maps = mapstemp
        if not resi:
            satu = 0.1 * amax(maps)
        else:
            satu = 0.1 * min(np.fabs(amin(maps)), amax(maps))
            maps[np.where(maps < -satu)] = -satu
        maps[np.where(maps > satu)] = satu

    exttrofi = [minmlgal, maxmlgal, minmbgal, maxmbgal]

    if pixltype == 'heal':
        cart = retr_cart(maps, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal, numbsidelgal=numbsidelgal, numbsidebgal=numbsidebgal)
    else:
        numbsidetemp = int(np.sqrt(maps.size))
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
        valu = max(np.fabs(amin(cart)), np.fabs(amax(cart)))
        imag = plt.imshow(cart, origin='lower', cmap=cmap, extent=exttrofi, interpolation='none', vmin=-valu, vmax=valu)
    else:
        imag = plt.imshow(cart, origin='lower', cmap=cmap, extent=exttrofi, interpolation='none')
    
    if scat != None:
        numbscat = len(scat)
        for k in range(numbscat):
            axis.scatter(scat[k][0], scat[k][1])

    cbar = plt.colorbar(imag, shrink=factsrnk) 
    
    plt.title(titl, y=1.08)

    figr.savefig(path)
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
    heal = np.zeros(numbpixl)
    jpixl = np.where((minmlgal < lghp) & (lghp < maxmlgal) & (minmbgal < bghp) & (bghp < maxmbgal))[0]
    jlgcr = (nlgcr * (lghp[jpixl] - minmlgal) / (maxmlgal - minmlgal)).astype(int)
    jbgcr = (nbgcr * (bghp[jpixl] - minmbgal) / (maxmbgal - minmbgal)).astype(int)
    
    heal[jpixl] = np.fliplr(cart)[jbgcr, jlgcr]
    
    return heal


def retr_healgrid(numbside):
    
    numbpixl = 12 * numbside**2
    apix = 4. * np.pi / numbpixl # [sr]
    thhp, phhp = hp.pixelfunc.pix2ang(numbside, np.arange(numbpixl), nest=False) # [rad]
    lghp = np.rad2deg(phhp)
    lghp = 180. - ((lghp - 180.) % 360.)# - 180. # [deg]
    bghp = 90. - np.rad2deg(thhp)

    return lghp, bghp, numbpixl, apix


def retr_isot(binsener, numbside=256):
    
    diffener = binsener[1:] - binsener[:-1]
    numbpixl = 12 * numbside**2
    numbener = binsener.size - 1
    numbsamp = 10

    # get the best-fit isotropic surface brightness given by the Fermi-LAT collaboration

    pathdata = retr_path('tdpy', onlydata=True)
    path = pathdata + 'iso_P8R2_SOURCE_V6_v06.txt'
    isotdata = loadtxt(path)
    enerisot = isotdata[:, 0] * 1e-3 # [GeV]
    sbrtisottemp = isotdata[:, 1] * 1e3 # [1/cm^2/s/sr/GeV]
    
    # sampling energy grid
    binsenersamp = np.logspace(np.log10(amin(binsener)), np.log10(amax(binsener)), numbsamp * numbener)
    
    # interpolate the surface brightness over the sampling energy grid
    sbrtisottemp = interp(binsenersamp, enerisot, sbrtisottemp)
    
    # take the mean surface brightness in the desired energy bins
    sbrtisot = np.empty((numbener, numbpixl))
    for i in range(numbener):
        sbrtisot[i, :] = trapz(sbrtisottemp[i*numbsamp:(i+1)*numbsamp], binsenersamp[i*numbsamp:(i+1)*numbsamp]) / diffener[i]
        
    return sbrtisot


def retr_cart(hmap, indxpixlrofi=None, numbsideinpt=None, minmlgal=-180., maxmlgal=180., minmbgal=-90., maxmbgal=90., nest=False, \
                                                                                                            numbsidelgal=None, numbsidebgal=None):
   
    shap = hmap.shape
    if indxpixlrofi == None:
        numbpixlinpt = shap[0]
        numbsideinpt = int(np.sqrt(numbpixlinpt / 12.))
    else:
        numbpixlinpt = numbsideinpt**2 * 12
    
    if numbsidelgal == None:
        numbsidelgal = 4 * int((maxmlgal - minmlgal) / np.rad2deg(np.sqrt(4. * np.pi / numbpixlinpt)))
    if numbsidebgal == None:
        numbsidebgal = 4 * int((maxmbgal - minmbgal) / np.rad2deg(np.sqrt(4. * np.pi / numbpixlinpt)))
    
    lgcr = np.linspace(minmlgal, maxmlgal, numbsidelgal)
    indxlgcr = np.arange(numbsidelgal)
    
    bgcr = np.linspace(minmbgal, maxmbgal, numbsidebgal)
    indxbgcr = np.arange(numbsidebgal)
    
    lghp, bghp, numbpixl, apix = retr_healgrid(numbsideinpt)

    bgcrmesh, lgcrmesh = np.meshgrid(bgcr, lgcr, indexing='ij')
    
    indxpixlmesh = hp.ang2pix(numbsideinpt, np.pi / 2. - np.deg2rad(bgcrmesh), np.deg2rad(lgcrmesh))
    
    if indxpixlrofi == None:
        indxpixltemp = indxpixlmesh
    else:
        pixlcnvt = np.zeros(numbpixlinpt, dtype=int) - 1
        for k in range(indxpixlrofi.size):
            pixlcnvt[indxpixlrofi[k]] = k
        indxpixltemp = pixlcnvt[indxpixlmesh]
    
    indxbgcrgrid, indxlgcrgrid = np.meshgrid(indxbgcr, indxlgcr, indexing='ij')

    if hmap.ndim == 2:
        hmapcart = np.empty((numbsidebgal, numbsidelgal, shap[1]))
        for k in range(shap[1]):
            hmapcart[np.meshgrid(indxbgcr, indxlgcr, k, indexing='ij')] = hmap[indxpixltemp, k][:, :, None]
    else:
        hmapcart = np.empty((numbsidebgal, numbsidelgal))
        hmapcart[np.meshgrid(indxbgcr, indxlgcr, indexing='ij')] = hmap[indxpixltemp]

    return np.fliplr(hmapcart).T


def retr_sbrtfdfm(binsener, numbside=256, vfdm=7):                    
    
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
    
    sbrtcart = pf.getdata(path, 0) * 1e3 # [1/cm^2/s/sr/GeV]
    enerfdfm = np.array(pf.getdata(path, 1).tolist()).flatten() * 1e-3 # [GeV]
    sbrtfdfmheal = np.zeros((enerfdfm.size, numbpixl))
    for i in range(enerfdfm.size):
        sbrtfdfmheal[i, :] = cart_heal(np.fliplr(sbrtcart[i, :, :]), numbside=numbside)
    
    sbrtfdfm = np.empty((numbener, numbpixl))
    numbsampbins = 10
    enersamp = np.logspace(np.log10(amin(binsener)), np.log10(amax(binsener)), numbsampbins * numbener)
    
    sbrtfdfmheal = interpolate.interp1d(enerfdfm, sbrtfdfmheal, axis=0)(enersamp)
    for i in range(numbener):
        sbrtfdfm[i, :] = trapz(sbrtfdfmheal[i*numbsampbins:(i+1)*numbsampbins, :], enersamp[i*numbsampbins:(i+1)*numbsampbins], axis=0) / diffener[i]

    return sbrtfdfm


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
    for k in np.arange(3):
        line.append(plt.Line2D((0,1),(0,0), color='k', ls=listlinestyl[k]))
    for l in range(3):
        line.append(plt.Line2D((0,1),(0,0), color=listcolr[l]))
    axis.legend(line, labl, loc=loc, ncol=2) 


def plot_braz(axis, xdat, ydat, yerr=None, numbsampdraw=0, lcol='yellow', dcol='green', mcol='black', labl=None, alpha=None):
    
    if numbsampdraw > 0:
        jsampdraw = choice(np.arange(ydat.shape[0]), size=numbsampdraw)
        axis.plot(xdat, ydat[jsampdraw[0], :], alpha=0.1, color='b', label='Samples')
        for k in range(1, numbsampdraw):
            axis.plot(xdat, ydat[jsampdraw[k], :], alpha=0.1, color='b')

    if yerr != None:
        axis.plot(xdat, ydat - yerr[0, :], color=dcol, alpha=alpha)
        line, = axis.plot(xdat, ydat, color=mcol, label=labl, alpha=alpha)
        axis.plot(xdat, ydat + yerr[1, :], color=dcol, alpha=alpha)
        ptch = mpl.patches.Patch(facecolor=dcol, alpha=alpha, linewidth=0)
        #ax.legend([(p1,p2)],['points'],scatterpoints=2)
        #plt.legend([(ptch, line)], ["Theory"], handler_map = {line : mpl.legend_handler.HandlerLine2D(marker_pad = 0)} )
        
        axis.fill_between(xdat, ydat - yerr[0, :], ydat + yerr[1, :], color=dcol, alpha=alpha)
        
        hand, labl = axis.get_legend_handles_labels()
        hand[0] = [hand[0], ptch]

        return ptch, line
    else:
        axis.plot(xdat, np.percentile(ydat, 2.5, 0), color=lcol, alpha=alpha)
        axis.plot(xdat, np.percentile(ydat, 16., 0), color=dcol, alpha=alpha)
        axis.plot(xdat, np.percentile(ydat, 50., 0), color=mcol, label=labl, alpha=alpha)
        axis.plot(xdat, np.percentile(ydat, 84., 0), color=dcol, alpha=alpha)
        axis.plot(xdat, np.percentile(ydat, 97.5, 0), color=lcol, alpha=alpha)
        axis.fill_between(xdat, np.percentile(ydat, 2.5, 0), np.percentile(ydat, 97.5, 0), color=lcol, alpha=alpha)#, label='95% C.L.')
        axis.fill_between(xdat, np.percentile(ydat, 16., 0), np.percentile(ydat, 84., 0), color=dcol, alpha=alpha)#, label='68% C.L.')


def retr_psfn(gdat, psfp, indxenertemp, thisangl, psfntype, binsoaxi=None, oaxitype=None, strgmodl='fitt'):

    numbpsfpform = getattr(gdat, strgmodl + 'numbpsfpform')
    numbpsfptotl = getattr(gdat, strgmodl + 'numbpsfptotl')
    
    indxpsfpinit = numbpsfptotl * (indxenertemp[:, None] + gdat.numbener * gdat.indxevtt[None, :])
    if oaxitype:
        indxpsfponor = numbpsfpform + numbpsfptotl * gdat.indxener[indxenertemp]
        indxpsfpoind = numbpsfpform + numbpsfptotl * gdat.indxener[indxenertemp] + 1
    
    if gdat.exprtype == 'ferm':
        scalangl = 2. * arcsin(np.sqrt(2. - 2. * cos(thisangl)) / 2.)[None, :, None] / gdat.fermscalfact[:, None, :]
        scalanglnorm = 2. * arcsin(np.sqrt(2. - 2. * cos(gdat.binsangl)) / 2.)[None, :, None] / gdat.fermscalfact[:, None, :]
    else:
        if oaxitype:
            scalangl = thisangl[None, :, None, None]
        else:
            scalangl = thisangl[None, :, None]
    
    if oaxitype:
        factoaxi = retr_factoaxi(gdat, binsoaxi, psfp[indxpsfponor], psfp[indxpsfpoind])
   
    if psfntype == 'singgaus':
        sigc = psfp[indxpsfpinit]
        if oaxitype:
            sigc = sigc[:, None, :, None] * factoaxi[:, None, :, :]
        else:
            sigc = sigc[:, None, :]
        psfn = retr_singgaus(scalangl, sigc)
    
    elif psfntype == 'singking':
        sigc = psfp[indxpsfpinit]
        gamc = psfp[indxpsfpinit+1]
        sigc = sigc[:, None, :]
        gamc = gamc[:, None, :]
        
        psfn = retr_singking(scalangl, sigc, gamc)
    
    elif psfntype == 'doubking':
        sigc = psfp[indxpsfpinit]
        gamc = psfp[indxpsfpinit+1]
        sigt = psfp[indxpsfpinit+2]
        gamt = psfp[indxpsfpinit+3]
        frac = psfp[indxpsfpinit+4]
        sigc = sigc[:, None, :]
        gamc = gamc[:, None, :]
        sigt = sigt[:, None, :]
        gamt = gamt[:, None, :]
        frac = frac[:, None, :]
        
        psfn = retr_doubking(scalangl, frac, sigc, gamc, sigt, gamt)
        if gdat.exprtype == 'ferm':
            psfnnorm = retr_doubking(scalanglnorm, frac, sigc, gamc, sigt, gamt)
    
    # normalize the PSF
    if gdat.exprtype == 'ferm':
        fact = 2. * np.pi * trapz(psfnnorm * sin(gdat.binsangl[None, :, None]), gdat.binsangl, axis=1)[:, None, :]
        psfn /= fact

    return psfn


def retr_psfpferm(gdat):
   
    gdat.exproaxitype = False
    
    if gdat.recotype == 'rec8':
        path = gdat.pathdata + 'expr/irfn/psf_P8R2_SOURCE_V6_PSF.fits'
    else:
        path = gdat.pathdata + 'expr/irfn/psf_P7REP_SOURCE_V15_back.fits'
    irfn = pf.getdata(path, 1)
    minmener = irfn['energ_lo'].squeeze() * 1e-3 # [GeV]
    maxmener = irfn['energ_hi'].squeeze() * 1e-3 # [GeV]
    enerirfn = np.sqrt(minmener * maxmener)

    numbpsfpscal = 3
    numbpsfpform = 5
    
    fermscal = np.zeros((gdat.numbevtt, numbpsfpscal))
    fermform = np.zeros((gdat.numbener, gdat.numbevtt, numbpsfpform))
    
    parastrg = ['score', 'gcore', 'stail', 'gtail', 'ntail']
    for m in gdat.indxevtt:
        if gdat.recotype == 'rec7':
            if m == 0:
                path = gdat.pathdata + 'expr/irfn/psf_P7REP_SOURCE_V15_front.fits'
            elif m == 1:
                path = gdat.pathdata + 'expr/irfn/psf_P7REP_SOURCE_V15_back.fits'
            irfn = pf.getdata(path, 1)
            fermscal[m, :] = pf.getdata(path, 2)['PSFSCALE']
        if gdat.recotype == 'rec8':
            irfn = pf.getdata(path, 1 + 3 * gdat.indxevtt[m])
            fermscal[m, :] = pf.getdata(path, 2 + 3 * gdat.indxevtt[m])['PSFSCALE']
        for k in range(numbpsfpform):
            fermform[:, m, k] = interp1d(enerirfn, mean(irfn[parastrg[k]].squeeze(), axis=0))(gdat.meanener)
    # convert N_tail to f_core
    for m in gdat.indxevtt:
        for i in gdat.indxener:
            fermform[i, m, 4] = 1. / (1. + fermform[i, m, 4] * fermform[i, m, 2]**2 / fermform[i, m, 0]**2)

    # calculate the scale factor
    gdat.fermscalfact = np.sqrt((fermscal[None, :, 0] * (10. * gdat.meanener[:, None])**fermscal[None, :, 2])**2 + fermscal[None, :, 1]**2)
    
    # store the fermi PSF parameters
    gdat.psfpexpr = np.zeros(gdat.numbener * gdat.numbevtt * numbpsfpform)
    for m in gdat.indxevtt:
        for k in range(numbpsfpform):
            indxfermpsfptemp = m * numbpsfpform * gdat.numbener + gdat.indxener * numbpsfpform + k
            #if k == 0 or k == 2:
            #    gdat.psfpexpr[indxfermpsfptemp] = fermform[:, m, k] * gdat.fermscalfact[:, m]
            #else:
            #    gdat.psfpexpr[indxfermpsfptemp] = fermform[:, m, k]
            gdat.psfpexpr[indxfermpsfptemp] = fermform[:, m, k]
    

def retr_fwhm(psfn, binsangl):

    if psfn.ndim == 1:
        numbener = 1
        numbevtt = 1
        psfn = psfn[None, :, None]
    else:
        numbener = psfn.shape[0]
        numbevtt = psfn.shape[2]
    indxener = np.arange(numbener)
    indxevtt = np.arange(numbevtt)
    wdth = np.zeros((numbener, numbevtt))
    for i in indxener:
        for m in indxevtt:
            indxanglgood = np.argsort(psfn[i, :, m])
            intpwdth = max(0.5 * amax(psfn[i, :, m]), amin(psfn[i, :, m]))
            if intpwdth > amin(psfn[i, indxanglgood, m]) and intpwdth < amax(psfn[i, indxanglgood, m]):
                wdth[i, m] = interp1d(psfn[i, indxanglgood, m], binsangl[indxanglgood])(intpwdth)
    return wdth


def retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt):

    psfn = frac / 2. / np.pi / sigc**2 * (1. - 1. / gamc) * (1. + scaldevi**2 / 2. / gamc / sigc**2)**(-gamc) + \
        (1. - frac) / 2. / np.pi / sigt**2 * (1. - 1. / gamt) * (1. + scaldevi**2 / 2. / gamt / sigt**2)**(-gamt)
    
    return psfn


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


def smth(maps, scalsmth, mpol=False, retrfull=False, numbsideoutp=None, indxpixlmask=None):

    if mpol:
        mpolsmth = scalsmth
    else:
        mpolsmth = 180. / scalsmth

    numbpixl = maps.size
    numbside = int(np.sqrt(numbpixl / 12))
    numbmpol = 3 * numbside
    maxmmpol = 3. * numbside - 1.
    mpolgrid, temp = hp.Alm.getlm(lmax=maxmmpol)
    mpol = np.arange(maxmmpol + 1.)
    
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
    
    mapsoutp = hp.alm2map(almc[np.where(mpolgrid < 3 * numbsideoutp)], numbsideoutp, verbose=False)

    if retrfull:
        return mapsoutp, almc, mpol, exp(-0.5 * (mpol / mpolsmth)**2)
    else:
        return mapsoutp


def smth_ferm(mapsinpt, meanener, recotype, maxmmpol=None, makeplot=False, kerntype='ferm'):

    numbpixl = mapsinpt.shape[1]
    numbside = int(np.sqrt(numbpixl / 12))
    if maxmmpol == None:
        maxmmpol = 3 * numbside - 1

    numbener = meanener.size
    indxener = np.arange(numbener)
    
    evtt, numbevtt, indxevtt = retr_evttferm(recotype)
    numbalmc = (maxmmpol + 1) * (maxmmpol + 2) / 2
    
    mapsoutp = np.empty_like(mapsinpt)
    if kerntype == 'gaus':
        angl = np.pi * np.linspace(0., 10., 100) / 180.
        
        if recotype != 'manu':
            gdat = gdatstrt()
            gdat.pathdata = os.environ["PCAT_DATA_PATH"] + '/data/'
            gdat.numbener = numbener
            gdat.indxener = indxener
            gdat.numbevtt = numbevtt
            gdat.indxevtt = indxevtt
            gdat.meanener = meanener
            gdat.recotype = recotype
            retr_psfpferm(gdat)
            gdat.exprtype = 'ferm'
            gdat.fittnumbpsfpform = 5
            gdat.fittnumbpsfptotl = 5
            gdat.binsangl = angl
            psfn = retr_psfn(gdat, gdat.psfpexpr, gdat.indxener, angl, 'doubking', None, False)
            fwhm = retr_fwhm(psfn, angl) 
        for i in indxener:
            for m in indxevtt:
                if recotype == 'manu':
                    #fwhmfitt = np.array([ \
                    #                  2.2591633256, \
                    #                  1.71342705148, \
                    #                  1.45102416042, \
                    #                  1.10904308863, \
                    #                  0.928844633041, \
                    #                  0.621357720854, \
                    #                  0.510777917886, \
                    #                  0.500998238444, \
                    #                  0.36102878406, \
                    #                  0.246788005029, \
                    #                  0.195373208584, \
                    #                  0.192829849688, \
                    #                  0.155934418827, \
                    #                  0.123744918778, \
                    #                  0.0889795596446, \
                    #                  0.0739099177209, \
                    #                  0.0777070595049, \
                    #                  0.0590032526699, \
                    #                  0.0570066113952, \
                    #                  0.0522932064601, \
                    #                 ])
                    #fwhmtemp = fwhmfitt[4*i+m]
                    if i == 0:
                        if m == 0:
                            sigm = 100.
                        if m == 1:
                            sigm = 100.
                    if i == 1:
                        if m == 0:
                            sigm = 1.05
                        if m == 1:
                            sigm = 0.7
                    if i == 2:
                        if m == 0:
                            sigm = 0.47
                        if m == 1:
                            sigm = 0.35
                    if i == 3:
                        if m == 0:
                            sigm = 0.06
                        if m == 1:
                            sigm = 0.04
                    if i == 4:
                        if m == 0:
                            sigm = 0.04
                        if m == 1:
                            sigm = 0.03
                    fwhmtemp = 2.355 * sigm * np.pi / 180.
                else:
                    fwhmtemp = fwhm[i, m]
                mapsoutp[i, :, m] = hp.smoothing(mapsinpt[i, :, m], fwhm=fwhmtemp)
    
    if kerntype == 'ferm':
        # get the beam
        beam = retr_beam(meanener, evtt, numbside, maxmmpol, recotype)
        # construct the transfer function
        tranfunc = ones((numbener, numbalmc, numbevtt))
        cntr = 0
        for n in np.arange(maxmmpol+1)[::-1]:
            tranfunc[:, cntr:cntr+n+1, :] = beam[:, maxmmpol-n:, :]
            cntr += n + 1


        indxener = np.arange(numbener)
        indxevtt = np.arange(numbevtt)
        for i in indxener:
            for m in indxevtt:
                
                # temp
                if i != 0 or m != 0:
                    continue
                # temp
                #mapsoutp[i, :, m] = hp.smoothing(mapsinpt[i, :, m], fwhm=radians(1.))
                #mapsoutp[i, :, m] = mapsinpt[i, :, m]
                almc = hp.map2alm(mapsinpt[i, :, m], lmax=maxmmpol)
                almc *= tranfunc[i, :, m]
                mapsoutp[i, :, m] = hp.alm2map(almc, numbside, lmax=maxmmpol)
    
    return mapsoutp


def retr_beam(meanener, evtt, numbside, maxmmpol, recotype, fulloutp=False, evaltype='invt'):
    
    numbpixl = 12 * numbside**2
    apix = 4. * np.pi / numbpixl

    numbener = meanener.size
    numbevtt = evtt.size
    indxevtt = np.arange(numbevtt)
    numbmpol = int(maxmmpol) + 1

    # alm of the delta function at the North Pole
    mapsinpt = np.zeros(numbpixl)
    mapsinpt[:4] = 1.
    mapsinpt /= sum(mapsinpt) * apix
    almcinpt = real(hp.map2alm(mapsinpt, lmax=maxmmpol)[:maxmmpol+1])
    
    # alm of the point source at the North Pole
    if evaltype != 'invt':
        lgalgrid, bgalgrid, numbpixl, apix = retr_healgrid(numbside)
        dir1 = np.array([lgalgrid, bgalgrid])
        dir2 = np.array([0., 90.])
        angl = hp.rotator.angdist(dir1, dir2, lonlat=True)
    else:
        angl = np.pi / np.linspace(1., maxmmpol, maxmmpol + 1)
    mapsoutp = retr_psfnferm(meanener, angl)
    if evaltype != 'invt':
        almcoutp = np.empty((numbener, maxmmpol+1, numbevtt))
        for i in range(numbener):
            for m in indxevtt:
                almcoutp[i, :, m] = real(hp.map2alm(mapsoutp[i, :, m], lmax=maxmmpol)[:maxmmpol+1])
        tranfunc = almcoutp / almcinpt[None, :, None]
        # temp
        tranfunc /= tranfunc[:, 0, :][:, None, :]
    else:    
        numbangl = angl.size
        matrdesi = np.empty((numbmpol, numbener, numbangl, numbevtt))
        tranfunc = np.empty((numbener, numbangl, numbevtt))
        for n in range(numbmpol):
            temp = 1. / np.sqrt(2. * n + 1.) * np.sqrt(4. * np.pi) / sp.special.lpmv(0, n, cos(angl))
            matrdesi[n, :, :, :] = temp[None, :, None]
        for i in range(numbener):
            for m in indxevtt:
                # temp
                if i != 0 or m != 0:
                    continue
                tranfunc[i, :, m] = matmul(linalg.inv(matrdesi[:, i, :, m]), mapsoutp[i, :, m])
        tranfunc = tranfunc**2
        for i in range(numbener):
            for m in indxevtt:
                tranfunc[i, :, m] /= tranfunc[i, 0, m]

    if fulloutp:
        return tranfunc, almcinpt, almcoutp
    else:
        return tranfunc


def plot_fermsmth():

    numbside = 256
    numbpixl = 12 * numbside**2
    maxmmpol = 3 * numbside - 1
    mpol = np.arange(maxmmpol + 1)
    
    listrecotype = ['rec7', 'rec8']
    for recotype in listrecotype:
        evtt, numbevtt, indxevtt = retr_evttferm(recotype)
        
        binsenerplot = np.array([0.3, 1., 3., 10.])
        meanenerplot = np.sqrt(binsenerplot[1:] * binsenerplot[:-1])
        numbenerplot = meanenerplot.size
        tranfunc, almcinpt, almcoutp = retr_beam(meanenerplot, evtt, numbside, maxmmpol, recotype, fulloutp=True)

        figr, axis = plt.subplots()

        plt.loglog(mpol, almcinpt, label='HealPix')
        plt.loglog(mpol, np.sqrt((2. * mpol + 1.) / 4. / np.pi), label='Analytic')
        path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/almcinpt.pdf'
        plt.legend()
        figr.savefig(path)
        plt.close(figr)
        
        figr, axis = plt.subplots()
        for i in np.arange(meanenerplot.size):
            for m in indxevtt:
                plt.loglog(mpol, almcoutp[i, :, m], label='$E=%.3g$, PSF%d' % (meanenerplot[i], indxevtt[m]))
        plt.legend(loc=3)
        path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/almcoutp.pdf'
        figr.savefig(path)
        plt.close(figr)
            
        figr, axis = plt.subplots()
        for i in np.arange(meanenerplot.size):
            for m in indxevtt:
                plt.loglog(mpol, tranfunc[i, :, m], label='$E=%.3g$, PSF%d' % (meanenerplot[i], indxevtt[m]))
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
        maps = np.empty((numbenerplot, numbpixl, numbevtt))
        for i in np.arange(numbenerplot):
            for m in indxevtt:
                 maps[i, :, m] = mapstemp

        # smooth the map with the Fermi-LAT kernel
        mapssmthferm = smth_ferm(maps, meanenerplot, recotype)
        
        # smooth the map with the Gaussian kernel
        mapssmthgaus =  hp.sphtfunc.smoothing(mapstemp, sigma=np.deg2rad(0.5))

        # plot the maps
        path = os.environ["FERM_IGAL_DATA_PATH"] + '/imag/maps.pdf'
        plot_maps(path, mapstemp, minmlgal=minmlgal, maxmlgal=maxmlgal, minmbgal=minmbgal, maxmbgal=maxmbgal)

        for i in np.arange(meanenerplot.size):
            for m in indxevtt:
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
    matr = np.zeros((nsamp, 2))
    matr[:, 0] = randn(nsamp)
    matr[:, 1] = randn(nsamp) * 10.
    eigl, tranmatr, eigt = prca(matr)
    
    plt.scatter(matr[:, 0], matr[:, 1])
    plt.show()
    
    plt.scatter(tranmatr[:, 0], tranmatr[:, 1])
    plt.show()


