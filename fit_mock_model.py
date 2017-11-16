"""
The idea here is to fit the data with simple Gaussians, to easily facilitate 

"""

from collections import OrderedDict
import pdb
import numpy as np

from the_new_script import (prepare_data, convert_and_adapt_model_spectra, 
                            chisq_line, Jupper_list, fake_hyperfine_structure, plot_model,
                            model_spectrum_interpolated_onto_data)

# params:
# v_cen
# width 1-9
# height 1-9
# that's... 9+9+1 = 19 dimensions.
# actually I prefer to do a shared width, giving
# 1+1+9 dimensions.

def prepare_mock_model(data_dict, v_cen, v_fwhm, *heights):

    model_dict = OrderedDict()

    for i, height in enumerate(heights):

        J_upper = Jupper_list[i]
        data_spectrum_dict = data_dict[J_upper]
        # index, vel_array, jy_array = loaded_array.T

        vel_array = np.linspace(-50, 50, 200)

        sigma_width = v_fwhm / 2.35482

        flux_array = gaussian(vel_array, height, v_cen, sigma_width)

        if J_upper == 1:
            flux_array = fake_hyperfine_structure(vel_array, flux_array)

        # model_dict[J_upper] = {'vel': vel_array, 'T_mb': flux_array}

        new_model_K_array = model_spectrum_interpolated_onto_data(
            data_spectrum_dict['vel'], vel_array, flux_array, velocity_shift=0)

        model_spectrum_dict = {
            'J_upper': J_upper,
            'vel': data_spectrum_dict['vel'],
            'T_mb': new_model_K_array - new_model_K_array[0]
        }

        model_dict[J_upper] = model_spectrum_dict

    return model_dict


def gaussian(x_array, a, b, c):
    f_x = a*np.exp(-(x_array-b)**2/(2*c**2))
    return f_x


# ndim, nwalkers = 3, 100
# pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


# import emcee
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

# sampler.run_mcmc(pos, 500)

# import corner
# fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
#                       truths=[m_true, b_true, np.log(f_true)])
# fig.savefig("triangle.png")


vel_center=3.91
data = prepare_data(vel_center=vel_center, half_vel_span=20)

def lnlike(theta, data):

    models = prepare_mock_model(data, *theta)

    return -0.5 * np.sum([chisq_line(x['T_mb'], y['T_mb'], x['rms'])
                                        for x, y in zip(data.values(), models.values())])


def lnprior(theta):

    v_cen, v_fwhm, *heights = theta

    if np.abs(v_cen) > 20:
        return -np.inf
    if (v_fwhm < 0) or (v_fwhm > 20):
        return -np.inf
    for h in heights:
        if (h < 0) or (h > 20):
            return -np.inf

    return 0


def lnprob(theta, data):

    return lnprior(theta) + lnlike(theta, data)


ndim, nwalkers = 11, 100
height_guesses = (x['T_mb'].max() for x in data.values())
initial_guess = (3.91, 6, *height_guesses)
pos = [ initial_guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

import emcee


filename = "mock_chain.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(data,), backend=backend)

sampler.run_mcmc(pos, 400, progress=True)

# for result in sampler.sample(pos, iterations=400, progress=True):
#     position = result[0]
#     f = open("mock_chain.dat", "a")
#     for k in range(position.shape[0]):
#         f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))
#     f.close()