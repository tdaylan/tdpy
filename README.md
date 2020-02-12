# TDPY (TD's Python Library)

## Introduction

This Python library contains some of the numerical analysis and visualization routines that I have written for my reasearch. The primary product is a general purpose MCMC sampler with adaptive burn-in, `tdpy.mcmc`.

The module `tdpy.mcmc` is a general purpose Metropolis-Hastings MCMC sampler. Given a likelihood function and prior probability distribution in the parameter space of interest, it takes heavy-tailed Gaussian steps to construct a Markovian chain of states, whose stationary distribution is the target probability density. The sampler takes steps in a transformed parameter space where the prior is uniform. Therefore, the prior is accounted for by asymmetric proposals rather than explicitly evaluating the prior ratio between the proposed and current states.

By setting the keyword `optiprop=True`, the sampler takes its initial steps while optimizing its proposal scale until the acceptance ratio is ~ 25%. These samples are later discarded along with the first `numbburn` samples. As a result, the sampler minimizes the autocorrelation time by first **learning** the local likelihood topology. The chain can also be thinned down by a factor of `factthin`. The sampler employs multiple workers that sample independent chains, which are accumulated at the end of the sampling. Parallelism is accomplished through bypassing the Python global interpreter lock. The module also offers robust visualization tools to analyse the output chain.

In inference problems one usually likes to obtain the posterior probability distribution of secondary variables (derived from the parameters of the model) whose calculation is not necessary for sampling, but whose posterior probability distribution is of interest. With many chains running in parallel that need burn-in and thinning, this can require a lot of custom coding. `tdpy.mcmc` makes this a one-liner, by letting the user specify the secondary variables to be calculated. It then returns the accumulated and thinned chain of posterior samples of these variables as well as producing plots of their marginal and joint distributions.

## Installation

You can install `tdpy` by running the `setup.py` script.
```
python setup.py install
```
