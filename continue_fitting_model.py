import pdb
import emcee
from fit_model import *

new_backend = emcee.backends.HDFBackend("model_chain_wampf.h5")
print("Initial size: {0}".format(new_backend.iteration))

new_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), backend=new_backend)

n_steps = 350
print("Ready to run {0} more steps?".format(n_steps))
pdb.set_trace()
new_sampler.run_mcmc(None, n_steps, progress=True)