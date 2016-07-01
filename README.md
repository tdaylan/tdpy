This Python library contains some of my numerical analysis and visualization routines that I have written for my reasearch. I maintain and improve the library, but the core features are stable. The primary product is a general purpose MCMC sampler with adaptive burn-in, `tdpy.mcmc`.

# MCMC sampler
## Introduction
The module `tdpy.mcmc` is a general purpose Metropolis-Hastings MCMC sampler. Given a likelihood function and prior distribution in the parameter space of interest, it takes Gaussian steps to construct a Markov chain of states, whose stationary distribution is the target probability density. The sampler takes steps in a transformed parameter space where the prior density is uniform. By setting the keyword `optiprop=True`, the sampler takes its initial steps while optimizing its proposal scale until the acceptance ratio is ~ 25%. These samples are later discarded along with the first `numbburn` samples. The chain can also be thinned down by a factor of `factthin`. The algoritm employs multiple workers, which sample independent chains that are brought together at the end. This is accomplished through bypassing the Python global interpreter lock using `multiprocessing`. The module also offers robust visualization tools to analyse the output chain.

## Installation

You can install `tdpy` by running the `setup.py` script.
```
python setup.py install
```

## Usage

```
```


