This Python library contains some of my numerical analysis and visualization routines that I have written for my reasearch. I maintain and improve the library, but the core features are stable. The primary product is a general purpose MCMC sampler with adaptive burn-in, `tdpy.mcmc`.

# MCMC sampler
## Introduction
The module `tdpy.mcmc` is a general purpose Metropolis-Hastings MCMC sampler. Given a likelihood function and prior distribution in the parameter space of interest, it takes Gaussian steps to construct a Markov chain of states, whose stationary distribution is the target probability density. The sampler takes steps in a transformed parameter space where the prior density is uniform. By setting the keyword `optiprop=True`, the sampler takes its initial steps while optimizing its proposal scale until the acceptance ratio is ~ 25%. These samples are later discarded along with the first `numbburn` samples. The chain can also be thinned down by a factor of `factthin`. The algoritm employs multiple workers, which sample independent chains that are brought together at the end. This is accomplished through bypassing the Python global interpreter lock using `multiprocessing`. The module also offers robust visualization tools to analyse the output chain.


## Example usage

```python
import pcat
     
numbener = 5
minmspec = array([3e-11])
maxmspec = array([1e-7])
mockfdfnslop = array([[1.9]])
pcat.init(, \
          psfntype='doubking', \
          randinit=False, \
          trueinfo=True, \
          maxmgang=20., \
          mocknumbpnts=array([300]), \
          minmspec=minmspec, \
          maxmspec=maxmspec, \
          regitype='ngal', \
          maxmnormback=array([2., 2.]), \
          minmnormback=array([0.5, 0.5]), \
          strgexpo='fermexpo_comp_ngal.fits', \
          datatype='mock', \
          numbsideheal=256, \
          mockfdfnslop=mockfdfnslop, \
          mocknormback=ones((2, numbener)), \
         )
```

### Options
---
#### Diagnostics
`diagsamp`
Boolean flag to run the sampler in diagnostic mode

---
#### User interaction
`verbtype`
Verbosity level
- `0`: no standard output
- `1`: print only critical statements along with periodic progress
- `2`: full log of internal variables, to be used only for diagnostic purposes

---
#### Plotting
`numbswepplot`

