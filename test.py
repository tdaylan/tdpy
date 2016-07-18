from __init__ import *
from tdpy import mcmc
from tdpy import util

def retr_llik_flag(sampvarb):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]

    llik = sp.interpolate.interp2d(xgrd, ygrd, pdfn)(xpos, ypos)
    sampcalc = [arange(5), arange(6)]

    return llik, sampcalc


def retr_datapara():
    
    numbpara = 2
    
    datapara.indx['xpos'] = 0
    datapara.indx['ypos'] = 0
    datapara.name[0] = 'xpos'
    datapara.name[1] = 'ypos'
    datapara.minm[:] = 0.
    datapara.maxm[:] = 1.
    datapara.scal[:] = 'self'
    datapara.labl[0] = r'$x$'
    datapara.labl[1] = r'$y$'
    datapara.unit[:] = ''
    datapara.vari[:] = 1e-1
    datapara.true[:] = None
    datapara.strg = datapara.labl + ' ' + datapara.unit


def cnfg_flag():

    # target PDF
    path = path = os.environ["TDPY_DATA_PATH"] + '/turkflag.png'
    imag = sp.ndimage.imread(path)
    rati = float(imag.shape[0]) / imag.shape[1]
    xinp = linspace(0., 1., imag.shape[1])
    yinp = linspace(0., 1., imag.shape[0])
    numbxgrd = 200
    numbygrd = int(200 * rati)
    xgrd = linspace(0., 1., numbxgrd)
    ygrd = linspace(0., 1., numbygrd)
    imag = sp.ndimage.gaussian_filter(imag, sigma=[xinp.size / 100, yinp.size / 100, 0])
    pdfn = zeros((numbygrd, numbxgrd, 3))
    for k in range(3):
        pdfn[:, :, k] = sp.interpolate.interp2d(xinp, yinp, imag[:, :, k])(xgrd, ygrd)
    pdfn = 0.3 * pdfn[:, :, 0]
    
    figr, axis = plt.subplots()
    axis.imshow(imag, extent=[0., 1., 0., 1.], interpolation='none', aspect=rati)
    plt.savefig('imag.pdf')
    plt.close(figr) 
    
    figr, axis = plt.subplots()
    imag = axis.imshow(pdfn, extent=[0., 1., 0., 1.], interpolation='none', aspect=rati)
    plt.colorbar(imag)
    plt.savefig('targ.pdf')
    plt.close(figr) 
    
    # MCMC setup
    verbtype = 1
    numbproc = 2
    numbswep = 7
    datapara = retr_datapara()
    optiprop = True
    pathbase = '/Users/tansu/Desktop/'
    
    # run MCMC
    sampbund = mcmc.init(numbproc, numbswep, retr_llik_flag, datapara, pathbase=pathbase, optiprop=True, verbtype=verbtype, rtag='flag')
    listxpos = sampbund[0][:, 0]
    listypos = sampbund[0][:, 1]
    numbsamp = listxpos.size
    pathtemp = pathbase + 'post'
    numbfram = 10
    for k in range(numbfram):
        indxsamp = int(numbsamp * float(k) / numbfram)
        figr, axis = plt.subplots()
        axis.scatter(listxpos[:indxsamp], listypos[:indxsamp], s=3)
        axis.set_xlim([0., 1.])
        axis.set_ylim([0., 1.])
        path = pathtemp + '%04d.pdf' % k
        plt.savefig(path)
        plt.close(figr) 
    
    cmnd = 'convert -delay 20 %s*.pdf post.gif' % pathtemp
    os.system(cmnd)


# define some likelihood functions
def retr_llik_gaus(sampvarb, gdat):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]

    print 'hey'

    print sampvarb
    print
    llik = 0.
    for n in range(gdat.numbgaus):
        print llik
        print gdat.stdvgaus[n]**2
        print 0.5 * ((xpos - gdat.xposgaus[n])**2 + (ypos - gdat.yposgaus[n])**2)
        print
        llik -= 0.5 * ((xpos - gdat.xposgaus[n])**2 + (ypos - gdat.yposgaus[n])**2) / gdat.stdvgaus[n]**2
        
    print

    return llik, []


def retr_llik_ebox(sampvarb, gdat):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]
    
    llik = sin(xpos) * sin(ypos)

    return llik


def cnfg_gaus():

    # define the target PDF
    minmxpos = 0.
    maxmxpos = 10. * pi
    minmypos = 0.
    maxmypos = 10. * pi

    xgrd = linspace(minmxpos, maxmxpos, 100)
    ygrd = linspace(minmypos, maxmypos, 100)
    
    # construct the parameter object
    datapara = util.datapara(2)
    datapara.defn_para('xpos', minmxpos, maxmxpos, 'self', r'$x$', '', None, None)
    datapara.defn_para('ypos', minmypos, maxmypos, 'self', r'$y$', '', None, None)
    
    gdat = util.gdatstrt()
    gdat.numbgaus = 10
    gdat.xposgaus = rand(gdat.numbgaus) * maxmxpos
    gdat.yposgaus = rand(gdat.numbgaus) * maxmypos
    gdat.stdvgaus = 0.5 + rand(gdat.numbgaus)

    # run MCMC
    sampbund = mcmc.init(retr_llik_gaus, datapara, gdatextr=gdat)

cnfg_gaus()

