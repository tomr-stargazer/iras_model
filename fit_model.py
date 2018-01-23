import numpy as np
import astropy.units as u

from the_new_script import (prepare_data, convert_and_adapt_model_spectra, 
                            chisq_line, Jupper_list, fake_hyperfine_structure, plot_model,
                            model_spectrum_interpolated_onto_data, run_convolve_and_prepare_model_spectra)

vel_center=3.760

data = prepare_data(vel_center=vel_center, half_vel_span=20)

# theta: X_in, X_out, Tjump

def lnlike(theta, data):

    print("*\n"*10)
    print(theta)
    print("*\n"*10)

    X_in, X_out, temperature_jump = theta

    # models = prepare_mock_model(data, *theta)
    models = run_convolve_and_prepare_model_spectra(data, vel_center=vel_center, 
                                                    abundance=(X_in, X_out), db=None, 
                                                    temperature_jump=temperature_jump*u.K)


    return -0.5 * np.sum([chisq_line(x['T_mb'], y['T_mb'], x['rms'])
                                        for x, y in zip(data.values(), models.values())])

def lnprior(theta):

    X_in, X_out, temperature_jump = theta

    if X_in < X_out:
        return -np.inf

    if (X_in < 1e-12) or (X_in > 1e-8):
        return -np.inf

    if (temperature_jump > 120) or (temperature_jump < 30):
        return -np.inf

    if X_out < 1e-14:
        return -np.inf

    return 0


def lnprob(theta, data):

    return lnprior(theta) + lnlike(theta, data)


ndim, nwalkers = 3, 24
initial_guess = (6e-10, 1.8e-11, 70)
pos = [ initial_guess + np.array(initial_guess)/8*np.random.randn(ndim) for i in range(nwalkers)]

if __name__ == "__main__":

    import emcee

    filename = "model_chain_wampf.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), backend=backend)

    sampler.run_mcmc(pos, 100, progress=True)
