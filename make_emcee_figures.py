"""
Make the figures from a given emcee output.

"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from fit_model import *


def produce_figures(save_file, run_bestfit_model=False):

    plots = {}

    reader = emcee.backends.HDFBackend(save_file)
    tau = reader.get_autocorr_time(quiet=True)
    burnin = int(2*np.max(tau))
    thin = int(0.5*np.min(tau))
    all_samples = reader.get_chain()
    thinned_samples = reader.get_chain(discard=burnin, flat=True, thin=1)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=1)
    log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=1)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(thinned_samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    print("flat log prior shape: {0}".format(log_prior_samples))

    fig, axes = plt.subplots(all_samples.ndim, figsize=(16, 7), sharex=True)
    labels = [r"$X_{\rm{in}}$", r"$X_{\rm{out}}$", r"$T_{\rm{jump}}$"]
    for i in range(all_samples.ndim):
        ax = axes[i]
        ax.plot(all_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(all_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");

    plots['walkers'] = fig
    
    # samples_plus_logprobs = np.concatenate((thinned_samples, log_prob_samples[:, None]), axis=1)
    # labels += ["log prob"]

    plots['corner'] = corner.corner(thinned_samples, labels=labels, #use_math_text=True, 
                                    quantiles=[0.16,0.50,0.84], bins=25)

    bestfit = [np.percentile(thinned_samples[:, x], [50])[0] for x in range(thinned_samples.ndim+1)]
    X_in, X_out, T_jump = bestfit

    if run_bestfit_model:
        models = run_convolve_and_prepare_model_spectra(data, vel_center=vel_center, 
                                                        abundance=(X_in, X_out), db=None, 
                                                        temperature_jump=T_jump*u.K)

        plots['model'] = plot_model(models, data_dict=data)

    return plots


if __name__ == '__main__':

    plots = produce_figures('model_chain_wampf.h5', run_bestfit_model=True)

    plots['walkers'].savefig("plots/walkers.pdf", bbox_inches='tight')
    plots['corner'].savefig("plots/corner.pdf", bbox_inches='tight')
    try:
        plots['model'].savefig("plots/model.pdf", bbox_inches='tight')
    except KeyError:
        pass

    plt.show()
