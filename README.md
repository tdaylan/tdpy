# Introduction
This Python library contains some of my numerical analysis and visualization routines that I have written for my reasearch. I maintain and improve the library, but the core features are stable. The primary product is a general purpose MCMC sampler with adaptive burn-in, `tdpy.mcmc`.

## MCMC sampler
The module `tdpy.mcmc` is a general purpose Metropolis-Hastings MCMC sampler. Given a likelihood function and prior distribution in the parameter space of interest, it takes Gaussian steps to construct a Markov chain of states, whose stationary distribution is the target probability density. The sampler takes steps in a transformed parameter space where the prior density is uniform. This sets the prior fraction in the acceptance ratio, to unity. Hence the acceptance ratio is not suppressed by the prior beliefs.

By setting the keyword `optiprop=True`, the sampler takes its initial steps while optimizing its proposal scale until the acceptance ratio is ~ 25%. These samples are later discarded along with the first `numbburn` samples. As a result, the sampler minimizes the autocorrelation time by first **learning** the local likelihood topology. This is useful as long as the learned covarience structure is global, i.e., does not change significantly.

The chain can also be thinned down by a factor of `factthin`. The algoritm employs multiple workers, which sample independent chains that are brought together at the end. This is accomplished through bypassing the Python global interpreter lock using `multiprocessing`. The module also offers robust visualization tools to analyse the output chain.

During a typical MCMC run, it happens frequently that you like to save samples of intermediate variables. These variables can be the model or residual emission map of a Poisson regression or any diagnostic variables, whose posterior distribution might be of interest. With many chains in parallel that need burn-in and thinning, this can create a lot of overhead. `tdpy.mcmc` makes this a one-liner, by letting the likelihood function return a list of numpy array, whose posterior chain is to be returned.


### Installation

You can install `tdpy` by running the `setup.py` script.
```
python setup.py install
```


