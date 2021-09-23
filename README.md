# TDPY (TD's Python Library)

## Introduction

tdpy is a python library of numerical analysis and visualization routines that I have written for my research. The primary product is a general purpose MCMC sampler with adaptive burn-in, `tdpy.mcmc`.

In inference problems the desired object is the posterior probability distribution of fitted and derived parameters. Towards this purpose, `tdpy.mcmc` offers a parallized and easy-to-use Metropolis-Hastings MCMC sampler. Given a likelihood function and prior probability distribution in a parameter space of interest, it makes heavy-tailed multi-variate Gaussian proposals to construct a Markovian chain of states, whose stationary distribution is the target probability density. It then visualizes the marginal posterior. The sampler takes steps in a transformed parameter space where the prior is uniform. Therefore, the prior is accounted for by asymmetric proposals rather than explicitly evaluating the prior ratio between the proposed and current states. Parallelism is accomplished via multiprocessing by gathering chains indepedently and simulataneously-sampled chains.

## Installation

You can install `tdpy` by running the `setup.py` script.
```
python setup.py install
```
