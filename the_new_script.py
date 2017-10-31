"""
Script which:

- sets up the physical model

- models the h13cn emission on the sky

- Creates synthetic spectra emulating the Herschel/etc spectra

- chi2 fits against the observed lines to derive abundance distribution situation

"""

import os
import glob
import pdb
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.units as u
import astropy.constants as c

import transphere_python.transphereRatran as tR

import reproduce_coutens_crimier_temp_profile as rcctp
from load_observed_spectra import retrieve_herschel_spectrum


Jupper_list = [1, 3, 4, 6, 7, 8, 9, 10, 11]


def prepare_real_data(vel_center=0, half_vel_span=12.5):
    """
    This produces already smoothed and baselined spectra 
    which were reduced by a totally separate script.

    It then selects only the inner velocity channels of that data.

    """

    # This bit is just for the Herschel data which start at Ju=6 . We'll have to figure out the IRAM/JCMT sitch separately.
    fit_results_path = os.path.expanduser("~/Documents/Data/Herschel_Science_Archive/IRAS16293/Fit_results")
    list_of_files = glob.glob(fit_results_path+"/H13CN*.fits")
    list_of_spectra = [x for x in list_of_files if 'spectrum.fits' in x]

    rms_noise_list = u.Quantity([14, 23, 63, 9, 9, 18, 22, 20, 31], u.mK)

    data_dict = OrderedDict()

    for i, Jupper in enumerate(Jupper_list):

        rms = rms_noise_list[i]

        try:
            spectral_fname = [x for x in list_of_spectra 
                if 'Ju={:02d}'.format(Jupper) in x][0] # assumes only one match
        except IndexError:
            vels = np.linspace(-half_vel_span+vel_center, vel_center+half_vel_span, 50)
            data_dict[Jupper] = {
                'vel' : vels,
                'jy' : np.zeros_like(vels),
                'rms': 1*u.mK
            }
            continue

        # pdb.set_trace()

        spectrum, freqs, vels = retrieve_herschel_spectrum(spectral_fname)

        # Now we want to restrict things to just the spectral region worth considering
        restricted_vels = vels[np.abs(vels-vel_center) <= half_vel_span]
        restricted_spectrum = spectrum[np.abs(vels-vel_center) <= half_vel_span]

        data_dict[Jupper] = {
            'vel': restricted_vels,
            'jy': restricted_spectrum,
            'rms': rms
        }

    return data_dict


def prepare_fake_data(vel_array):

    fake_data_dict = {}

    for Jupper in Jupper_list:
        fake_data_dict[Jupper] = {
            'vel': vel_array, 'jy': np.zeros_like(vel_array)}

    return fake_data_dict


def chisq_line(observed_data, model_data, rms_per_channel, calibration_uncertainty=0.15):

    top = (observed_data - model_data)**2
    bottom = rms_per_channel**2 + (calibration_uncertainty * observed_data)**2

    return np.sum(top/bottom)


def prepare_model(abundance, run_ratran=True):
    distance = 120 * u.pc

    # rho_nought = 9.0e-16 * u.g * u.cm**-3
    # radius_array = (np.arange(0, 1000, 50)*u.AU).to(u.cm)
    # density_array = (radius_array.to(u.AU)/(1*u.AU))**-1.5 * rho_nought

    # temp_array = np.ones_like(radius_array)*50*u.K
    radius_array = rcctp.r * u.cm
    dust_density_array = rcctp.rho_dust * u.g * u.cm**-3
    temp_array = rcctp.a['temp'][-1] * u.K
    abundance_array = np.ones_like(radius_array.value) * abundance

    # pdb.set_trace()

    # Phase 2: Run the AMC/Ratran situation.

    os.makedirs("./h13cn_emission_models/", exist_ok=True)

    transition_list = [1, 3, 4, 6, 7, 8, 9, 10, 11]
    transitions_string = ",".join(str(x) for x in transition_list)
    frequency_list = u.Quantity([86.3399, 172.6778, 259.0117, 345.3397, 431.6597, 517.9698, 604.2679,
                                 690.5520, 776.8203, 863.0706, 949.3010], u.GHz)

    if run_ratran:
        tR.ratranRun(r=radius_array.value, rho=dust_density_array.value, temp=temp_array.value, db=3.63,
                     abund=abundance_array, dpc=distance.value, trans=transitions_string,
                     molfile='h13cn.dat', unit='jypx', writeonly=0, skyonly=0,
                     channels=100, channel_width=0.23,
                     outputfile='h13cn_emission_models/ratranResult_h13cn')

    list_of_filenames = glob.glob("./h13cn_emission_models/*")

    os.makedirs("./h13cn_miriad_manipulation/", exist_ok=True)

    for i, filename in enumerate(list_of_filenames):

        J_upper = transition_list[i]
        freq = frequency_list[i]

        # pdb.set_trace()

        # make this os.system later but we're proofing it first.
        call_fn = os.system

        miriad_basename = 'h13cn_miriad_manipulation/h13cn_{0:03d}'.format(J_upper)

        remove_file_if_exists = 'rm -rf {0}.sky {0}.convol'.format(miriad_basename)
        call_fn(remove_file_if_exists)

        convert_fits_to_miriad = 'fits in={0} out={1}.sky op=xyin'.format(filename, 
                                                                          miriad_basename)
        call_fn(convert_fits_to_miriad)

        put_frequency_in_header = 'puthd in={0}.sky/restfreq value={1:.3f}'.format(miriad_basename, 
                                                                                   freq.value)
        call_fn(put_frequency_in_header)

        # Convolve map
        convolve_map = 'convol map={0}.sky fwhm={1:.2f} out={0}.convol'.format(miriad_basename, 29.1)
        call_fn(convolve_map)

        # Make a spectrum and output it
        make_spectrum = "imspect in={0}.convol region='arcsec,box(0,5,0,5)' log={0}.spectrum".format(miriad_basename)
        call_fn(make_spectrum)

        # pdb.set_trace()

    model_dict = OrderedDict()

    list_of_spectra = glob.glob("h13cn_miriad_manipulation/*.spectrum")
    for i, spectrum in enumerate(list_of_spectra):

        print(i, spectrum)

        J_upper = transition_list[i]
        freq = frequency_list[i]

        loaded_array = np.loadtxt(spectrum, skiprows=3)
        index, vel_array, jy_array = loaded_array.T

        model_dict[J_upper] = {'vel': vel_array, 'jy': jy_array-jy_array[0]}

    return model_dict

def plot_model(model_dict, data_dict=None):

    fig = plt.figure()

    ax0 = None

    for i, J_upper in enumerate(model_dict.keys()):

        ax = fig.add_subplot(3, 3, i+1, sharey=ax0)
        ax.plot(model_dict[J_upper]['vel'], model_dict[
                J_upper]['jy'], linestyle='steps-mid')
        if data_dict is not None:
            ax.plot(data_dict[J_upper]['vel'], data_dict[
                    J_upper]['jy'], 'r:', linestyle='steps-mid')

        plt.show()

        ax0 = ax

    return fig


# def compare_model_to_observations(model_dict, obs_dict)

abundance_grid = np.logspace(-10, -8, 3)
# pdb.set_trace()

# simple model: only one abundance
if True:
    for abundance in abundance_grid:

        vel_center=3.91

        # data = prepare_fake_data(models[1]['vel'])
        data = prepare_real_data(vel_center=vel_center, half_vel_span=12.5)

        models = prepare_model(abundance, run_ratran=True)
        adapted_models = adapt_models_to_data(models, data)

        # adapted_models = {
        #     J: {
        #         'vel': data[J]['vel'],  
        #         'jy': model_spectrum_interpolated_onto_data(
        #             data[J]['vel'], models[J]['vel'], models[J]['jy'], 
        #             velocity_shift=vel_center, channel_width=None)
        #              } for J in models.keys()}

        pdb.set_trace()

        placeholder_rms = 15

        for x, y, key in zip(data.values(), models.values(), data.keys()):

            print("Jupper={0}".format(key))
            line_chi2 = chisq_line(x['jy'], y['jy'], placeholder_rms)
            # pdb.set_trace()

        chi2_of_model = np.sum([chisq_line(x['jy'], y['jy'], placeholder_rms)
                                for x, y in zip(data.values(), models.values())])

        print("\n\n****************************")
        print("****************************")
        print("For X(h13cn)={0:.1e}, chi2 = {1:.2e}".format(abundance, chi2_of_model))
        print("****************************")
        print("****************************\n\n")

        fig = plot_model(models, data_dict=data)
        fig.savefig("test_plots/X={0:.1e}.png".format(abundance))

        pdb.set_trace()
