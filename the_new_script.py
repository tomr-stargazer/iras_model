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
import datetime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.units as u
import astropy.constants as c

import transphere_python.transphereRatran as tR

import reproduce_coutens_crimier_temp_profile as rcctp
from load_observed_spectra import retrieve_herschel_spectrum, retrieve_timasss_spectrum

# Parameters of the observed data.

Jupper_list = [1, 3, 4, 6, 7, 8, 9, 10, 11]
frequency_list = u.Quantity([86.3399, 259.0117, 345.3397, 517.9698, 604.2679,
                             690.5520, 776.8203, 863.0706, 949.3010], u.GHz)
efficiency_correction_list = [
    0.95/0.78,
    0.91/0.54,
    1/0.5,
    0.96/0.76,
    0.96/0.76,
    0.96/0.75,
    0.96/0.75,
    0.96/0.75,
    0.96/0.74
]

timasss_filename_list = ['iram13.fits', 'iram289.fits', 'spect466.fits']
fit_results_path = os.path.expanduser("~/Documents/Data/Herschel_Science_Archive/IRAS16293/Fit_results")
hifi_filename_list = [ os.path.join(fit_results_path, 'H13CN_Ju={:02d}_spectrum.fits'.format(Ju))
    for Ju in Jupper_list if Ju >= 6]
filename_list = timasss_filename_list + hifi_filename_list

diameter_dict = {'IRAM': 30*u.m, 'JCMT': 15*u.m, 'HIFI': 3.5*u.m}


def which_telescope(frequency):
    if (frequency >= 80 * u.GHz) and (frequency <= 285 * u.GHz):
        return 'IRAM'

    elif frequency <= 369 * u.GHz:
        return 'JCMT'

    elif frequency >= 480 * u.GHz:
        return 'HIFI'


def prepare_data(vel_center=0, half_vel_span=12.5):
    """
    This produces already smoothed and baselined spectra 
    which were reduced by a totally separate script.

    It then selects only the inner velocity channels of that data.

    """

    data_dict = OrderedDict()

    for i, Jupper in enumerate(Jupper_list):

        data_dict[Jupper] = prepare_individual_data(
            Jupper, filename_list[i], frequency_list[i], efficiency_correction_list[i],
            vel_center=vel_center, half_vel_span=half_vel_span)

    return data_dict


def prepare_individual_data(J_upper, filename, frequency, efficiency,
                            vel_center=0, half_vel_span=12.5):
    """
    For a spectral line, this does the following:
    - Loads the spectrum (flux array and velocity array) from the given file
    - Applies an efficiency correction to take antenna temperature to main beam temperature
    - Calculates the rms noise in the file
    - Extracts the relevant channels around the spectral feature (+/- `half_vel_span`)

    Parameters
    ==========
    J_upper : int
        Quantum number J of the upper rotational state of the transition.
    filename : str
        Filename of the FITS file where the spectrum data is stored.
        Format differs between TIMASSS data and Herschel data at the moment.
    frequency : astropy.Quantity (u.GHz)
        Frequency in GHz (or equivalent Unit) of the line of interest.
    efficiency : float
        Conversion factor between observed flux T*_A ("Antenna Temperature")
        and desired sky brightness T_mb ("Main Beam Temperature"). 
        Should be greater than one but unlikely to be more than 2-3.
    vel_center : float, optional (default 0)
        Central velocity of the source. Channels will be extracted symmetrically around this velocity.
    half_vel_span : float, optional (default 12.5)
        How many km/s on each side of `vel_center` to include in the extracted spectrum.
        Total width will be twice this, i.e., 25 km/s by default.

    Returns
    =======
    spectrum_dict : dict
        Contains the following:
    'J_upper' : int
        As in the input parameter.
    'vel' : np.ndarray
        Values in km/s of the velocity/spectral dimension of the spectrum. 
        Truncated to +/- `half_vel_span` km/s around `vel_center`.
    'T_mb' : np.ndarray
        The flux values in K of the main beam temperature spectrum.
        Truncated to +/- `half_vel_span` km/s around `vel_center`.
    'full_vel' : np.ndarray
        Un-truncated velocity array.
    'full_spectrum' : np.ndarray
        Un-truncated main beam temperature array.
    'rms' : float
        RMS noise (K) per channel, empirically calculated from the full input spectrum.

    """

    if frequency <= 369 * u.GHz:
        # do the TIMASSS thing
        vel_array, sp = retrieve_timasss_spectrum(filename, frequency)
        vels = vel_array.value
        spectrum = sp.flux

    elif frequency >= 480 * u.GHz:
        # do the Herschel thing
        spectrum, freqs, vels = retrieve_herschel_spectrum(filename)

    # apply efficiencies
    corrected_spectrum = spectrum * efficiency

    # find the noise. Median absolute deviation means we don't have to mask out the lines.
    rms_noise = astropy.stats.mad_std(corrected_spectrum)

    # trim some channels
    # Now we want to restrict things to just the spectral region worth considering
    restricted_vels = vels[np.abs(vels-vel_center) <= half_vel_span]
    restricted_spectrum = corrected_spectrum[np.abs(vels-vel_center) <= half_vel_span]

    spectrum_dict = {
        'J_upper': J_upper,
        'vel': restricted_vels,
        'T_mb': restricted_spectrum,
        'full_vel': vels,
        'full_spectrum': corrected_spectrum,
        'rms': rms_noise
    }

    return spectrum_dict


def prepare_fake_data(vel_array):
    """
    Possibly obsolete at the moment (since writing `prepare_data`)

    """

    fake_data_dict = {}

    for Jupper in Jupper_list:
        fake_data_dict[Jupper] = {
            'vel': vel_array, 'flux': np.zeros_like(vel_array), 'rms': 1}

    return fake_data_dict


def chisq_line(observed_data, model_data, rms_per_channel, calibration_uncertainty=0.15):

    top = (observed_data - model_data)**2
    bottom = rms_per_channel**2 + (calibration_uncertainty * observed_data)**2

    return np.sum(top/bottom)


def model_spectrum_interpolated_onto_data(data_velocity_array, model_velocity_array,
                                          old_model_spectrum, velocity_shift=0, channel_width=None):

    # To do this better, we'd actually want to convolve the old_model_spectrum 
    # with the data channel width...
    if channel_width is not None:
        kern1d = astropy.convolution.Gaussian1DKernel(channel_width/2.3) # sigma to FWHM conversion factor
        convolved_model_spectrum = astropy.convolution.convolve(old_model_spectrum, kern1d)
    else:
        convolved_model_spectrum = old_model_spectrum

    new_model_spectrum = np.interp(data_velocity_array, model_velocity_array+velocity_shift, 
                                   convolved_model_spectrum)

    return new_model_spectrum


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
    frequency_list = u.Quantity([86.3399, 259.0117, 345.3397, 517.9698, 604.2679,
                                 690.5520, 776.8203, 863.0706, 949.3010], u.GHz)

    if run_ratran:
        tR.ratranRun(r=radius_array.value, rho=dust_density_array.value, temp=temp_array.value, db=3.63,
                     abund=abundance_array, dpc=distance.value, trans=transitions_string,
                     molfile='h13cn.dat', unit='jypx', writeonly=0, skyonly=0,
                     channels=100, channel_width=0.23,
                     outputfile='h13cn_emission_models/ratranResult_h13cn')

    list_of_filenames = glob.glob("./h13cn_emission_models/*")

    os.makedirs("./h13cn_miriad_manipulation/", exist_ok=True)

    telescope_list = ['IRAM', 'IRAM', "JCMT", "HIFI", "HIFI", "HIFI", "HIFI", "HIFI", "HIFI"]
    diameter_dict = {'IRAM': 30*u.m, 'JCMT': 15*u.m, 'HIFI': 3.5*u.m}

    for i, filename in enumerate(list_of_filenames):

        J_upper = transition_list[i]
        freq = frequency_list[i]
        diameter = diameter_dict[telescope_list[i]]

        beam_fwhm = (1.22*u.radian * (c.c/freq)/(diameter) ).to(u.arcsec)

        # pdb.set_trace()

        # make this os.system later but we're proofing it first.
        call_fn = os.system

        miriad_basename = 'h13cn_miriad_manipulation/h13cn_{0:03d}'.format(J_upper)

        remove_file_if_exists = 'rm -rf {0}.sky {0}.convol'.format(miriad_basename)
        call_fn(remove_file_if_exists)

        convert_fits_to_miriad = 'fits in={0} out={1}.sky op=xyin'.format(filename, miriad_basename)
        call_fn(convert_fits_to_miriad)

        put_frequency_in_header = 'puthd in={0}.sky/restfreq value={1:.3f}'.format(miriad_basename, 
                                                                                   freq.value)
        call_fn(put_frequency_in_header)

        # Convolve map
        convolve_map = 'convol map={0}.sky fwhm={1:.2f} out={0}.convol'.format(miriad_basename, beam_fwhm.value)
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
        diameter = diameter_dict[telescope_list[i]]
        beam_fwhm = (1.22*u.radian * (c.c/freq)/(diameter) ).to(u.arcsec)

        loaded_array = np.loadtxt(spectrum, skiprows=3)
        index, vel_array, jy_array = loaded_array.T

        model_dict[J_upper] = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

        # Insert a Jy -> K conversion here.

        fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
        beam_sigma = beam_fwhm * fwhm_to_sigma
        omega_B = 2*np.pi * beam_sigma**2

        model_dict[J_upper]['T_mb'] = (model_dict[J_upper]['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))

    return model_dict


def plot_model(model_dict, data_dict=None):

    fig = plt.figure()

    for i, J_upper in enumerate(model_dict.keys()):

        ax = fig.add_subplot(3, 3, i+1)

        model_vels = model_dict[J_upper]['vel']
        model_fluxes = model_dict[J_upper]['T_mb']
        ax.plot(model_vels, model_fluxes, linestyle='steps-mid', zorder=10)

        # ax.text(0.1, 0.75, r"{0}-{1}$".format(J_upper, J_upper-1),
        #         transform=ax.transAxes, fontsize=14)

        if data_dict is not None:

            data_vels = data_dict[J_upper]['vel']
            data_fluxes = data_dict[J_upper]['T_mb']

            ax.plot(data_vels, data_fluxes, 'r:', linestyle='steps-mid', zorder=1)

            if 'rms' in data_dict[J_upper]:
                rms = data_dict[J_upper]['rms']
                ax.fill_between(data_vels, data_fluxes+rms, data_fluxes-rms, 
                                color='r', step='mid', alpha=0.1, zorder=-1)


    # plt.suptitle(r"$\rm{{H}}^{{13}}\rm{{CN}}$")
    plt.show()
    return fig


def adapt_models_to_data(models, data):

    adapted_models = {
        J: {
            'vel': data[J]['vel'],  
            'flux': model_spectrum_interpolated_onto_data(
                data[J]['vel'], models[J]['vel'], models[J]['flux'], 
                velocity_shift=vel_center, channel_width=None),
            'T_mb': model_spectrum_interpolated_onto_data(
                data[J]['vel'], models[J]['vel'], models[J]['T_mb'], 
                velocity_shift=vel_center, channel_width=None)
                 } for J in models.keys()}

    return adapted_models


# def compare_model_to_observations(model_dict, obs_dict)

abundance_grid = np.logspace(-12, -10, 6)
# pdb.set_trace()

# simple model: only one abundance
if False:
    for abundance in abundance_grid:

        vel_center=3.91

        # data = prepare_fake_data(models[1]['vel'])
        # data = prepare_real_data(vel_center=vel_center, half_vel_span=20)
        data = prepare_data(vel_center=vel_center, half_vel_span=20)

        models = prepare_model(abundance, run_ratran=True)
        adapted_models = adapt_models_to_data(models, data)

        # adapted_models = {
        #     J: {
        #         'vel': data[J]['vel'],  
        #         'flux': model_spectrum_interpolated_onto_data(
        #             data[J]['vel'], models[J]['vel'], models[J]['flux'], 
        #             velocity_shift=vel_center, channel_width=None)
        #              } for J in models.keys()}

        # pdb.set_trace()

        # placeholder_rms = 15

        for x, y, key in zip(data.values(), adapted_models.values(), data.keys()):

            print("Jupper={0}".format(key))
            line_chi2 = chisq_line(x['T_mb'], y['T_mb'], x['rms'])
            # pdb.set_trace()

        chi2_of_model = np.sum([chisq_line(x['T_mb'], y['T_mb'], x['rms'])
                                for x, y in zip(data.values(), adapted_models.values())])

        print("\n\n****************************")
        print("****************************")
        print("For X(h13cn)={0:.1e}, chi2 = {1:.2e}".format(abundance, chi2_of_model))
        print("****************************")
        print("****************************\n\n")

        fig = plot_model(adapted_models, data_dict=data)
        plt.suptitle("X(h13cn) = {0:.1e}".format(abundance))
        fig.savefig("test_plots/X={0:.1e}.png".format(abundance))

        pdb.set_trace()



def make_abundance_array(outer_radii, temp_array, target_temp, X_in, X_out, model_center=0*u.cm):
    """ Calculates the abundance in a cell that straddles T_jump. """

    r_inner = outer_radii.insert(0, model_center)[:-1]
    rau_out = outer_radii.to(u.AU)
    rau_in = r_inner.to(u.AU)

    target_radius = np.interp(target_temp, temp_array[::-1], rau_out[::-1]) * u.AU

    shell_boundaries_selection = (rau_in < target_radius) & (rau_out > target_radius)

    volume_of_whole_shell = 4/3*np.pi*((rau_out - rau_in)[shell_boundaries_selection])**3
    volume_above_temp = 4/3*np.pi*((target_radius - rau_in[shell_boundaries_selection]))**3

    volume_fraction = volume_above_temp / volume_of_whole_shell

    X_intermediate = (volume_fraction * (X_in - X_out))+X_out

    abundance_array = np.zeros_like(outer_radii.value)
    # Here's the jump abundance magic
    abundance_array[temp_array >= target_temp] = X_in
    abundance_array[temp_array < target_temp] = X_out
    abundance_array[shell_boundaries_selection] = X_intermediate

    pdb.set_trace()

    return abundance_array


ratran_output_directory = "./h13cn_emission_models"
ratran_output_prefix = 'ratranResult_h13cn'


def prepare_and_run_ratran_model(X_in, X_out, db=None, temperature_jump=100*u.K):

    distance = 120 * u.pc
    radius_array = rcctp.r * u.cm
    dust_density_array = rcctp.rho_dust * u.g * u.cm**-3
    temp_array = rcctp.a['temp'][-1] * u.K
    # abundance_array = np.zeros_like(radius_array.value)

    # # Here's the jump abundance magic
    # abundance_array[temp_array >= temperature_jump] = X_in
    # abundance_array[temp_array < temperature_jump] = X_out

    abundance_array = make_abundance_array(radius_array, temp_array, temperature_jump.value, X_in, X_out)

    if db is None:
        fwhm_linewidth =  2.91
        db = 0.6 * fwhm_linewidth

    stellar_mass = 1*u.Msun

    velocity_array = (( 2*c.G * stellar_mass / radius_array )**(0.5)).to(u.km/u.s)
    velocity_array[radius_array > 1280*u.AU] = 0

    os.makedirs(ratran_output_directory, exist_ok=True)

    transitions_string = ",".join(str(x) for x in Jupper_list)

    # write some files the way that RATRAN usually does

    tR.ratranRun(r=radius_array.value, rho=dust_density_array.value, temp=temp_array.value, 
                 vr=velocity_array.value,
                 db=db, abund=abundance_array, dpc=distance.value, trans=transitions_string,
                 molfile='h13cn.dat', unit='jypx', writeonly=0, skyonly=0,
                 channels=100, channel_width=0.23,
                 outputfile=os.path.join(ratran_output_directory, ratran_output_prefix))

    return None


miriad_directory = './h13cn_miriad_manipulation'
miriad_prefix = 'h13cn_'


def create_sky_spectra_with_miriad():

    # do it for all of them
    # create some files

    for i, J_upper in enumerate(Jupper_list):

        freq = frequency_list[i]

        if which_telescope(freq) == 'HIFI':
            offset_arcsec = 2.5 * u.arcsec
        else:
            offset_arcsec = 5 * u.arcsec

        create_individual_spectrum_with_miriad(J_upper, freq, offset_arcsec=offset_arcsec)

    return None    


def create_individual_spectrum_with_miriad(J_upper, frequency, offset_arcsec=0*u.arcsec):

    # First, figure out which telescope we're on
    diameter = diameter_dict[which_telescope(frequency)]
    beam_fwhm = (1.22*u.radian * (c.c/frequency)/(diameter) ).to(u.arcsec)

    # needs a filename
    input_filename = os.path.join(
        ratran_output_directory, ratran_output_prefix+"_{:03d}".format(J_upper)+".fits" )

    miriad_basename = os.path.join(miriad_directory, miriad_prefix)+'{0:03d}'.format(J_upper)

    def call_fn(x):
        print("*** Executing in shell: ", x)
        os.system(x)
    # call_fn = lambda x : os.system

    remove_file_if_exists = 'rm -rf {0}.sky {0}.convol'.format(miriad_basename)
    call_fn(remove_file_if_exists)

    convert_fits_to_miriad = 'fits in={0} out={1}.sky op=xyin'.format(input_filename, miriad_basename)
    call_fn(convert_fits_to_miriad)

    put_frequency_in_header = 'puthd in={0}.sky/restfreq value={1:.3f}'.format(miriad_basename, 
                                                                               frequency.value)
    call_fn(put_frequency_in_header)

    # Convolve map
    convolve_map = 'convol map={0}.sky fwhm={1:.2f} out={0}.convol'.format(miriad_basename, beam_fwhm.value)
    call_fn(convolve_map)

    # Make a spectrum and output it
    make_spectrum = "imspect in={0}.convol region='arcsec,box(0,{1:.2f},0,{1:.2f})' log={0}.spectrum".format(miriad_basename, offset_arcsec.to(u.arcsec).value)
    call_fn(make_spectrum)

    return None


#### 
# I probably chunked the python manipulation into too many named functions. 
# PROBABLY I should have just made `convert_and_adapt_individual_model` 
# an individual function. 
# We'll see. If I end up splitting off any of this stuff into a separate module
# that may help.
###

def load_miriad_spectrum(J_upper):

    miriad_basename = os.path.join(miriad_directory, miriad_prefix)+'{0:03d}'.format(J_upper)
    spectrum_filename = miriad_basename+".spectrum"

    loaded_array = np.loadtxt(spectrum_filename, skiprows=3)
    index, vel_array, jy_array = loaded_array.T

    return vel_array, jy_array


def convert_Jybm_spectrum_to_K(jy_array, frequency):

    diameter = diameter_dict[which_telescope(frequency)]
    beam_fwhm = (1.22*u.radian * (c.c/frequency)/(diameter) ).to(u.arcsec)

    fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
    beam_sigma = beam_fwhm * fwhm_to_sigma
    omega_B = 2*np.pi * beam_sigma**2

    K_array = (jy_array*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, frequency))

    return K_array


# def deserves to be its own fn here...
def fake_hyperfine_structure(vel_array, flux_array, reverse=False):

    if not np.all(np.diff(vel_array) > 0) or reverse:
        reverse=True
        # reverse both arrays
        vel_array = vel_array[::-1]
        flux_array = flux_array[::-1]

    center_velocity = -1.1805
    right_velocity = 3.681
    left_velocity = -8.472

    center_factor = 5
    right_factor = 3
    left_factor = 1

    # effectively this is 1/8 * flux_array
    base_flux = 1/(center_factor+left_factor+right_factor) * flux_array

    center_flux = np.interp(vel_array, vel_array+center_velocity, center_factor * base_flux)
    right_flux = np.interp(vel_array, vel_array+right_velocity, right_factor * base_flux)
    left_flux = np.interp(vel_array, vel_array+left_velocity, left_factor * base_flux)

    hyperfine_flux_array = center_flux + right_flux + left_flux

    if reverse:
        hyperfine_flux_array = hyperfine_flux_array[::-1]

    return hyperfine_flux_array


def convert_and_adapt_model_spectra(data_dict, vel_center=0):
    
    converted_adapted_model_dict = OrderedDict()

    for i, (Jupper, data_spectrum_dict) in enumerate(zip(Jupper_list, data_dict.values())):

        converted_adapted_model_dict[Jupper] = convert_and_adapt_individual_model(
            Jupper, frequency_list[i], data_spectrum_dict,
            vel_center=vel_center)

    return converted_adapted_model_dict


def convert_and_adapt_individual_model(J_upper, frequency, data_spectrum_dict, vel_center=0):

    model_vel_array, model_jy_array = load_miriad_spectrum(J_upper)

    model_K_array = convert_Jybm_spectrum_to_K(model_jy_array, frequency)

    if J_upper == 1:
        model_K_array = fake_hyperfine_structure(model_vel_array, model_K_array.value)*model_K_array.unit

    new_model_K_array = model_spectrum_interpolated_onto_data(
        data_spectrum_dict['vel'], model_vel_array, model_K_array, velocity_shift=vel_center)

    model_spectrum_dict = {
        'J_upper': J_upper,
        'vel': data_spectrum_dict['vel'],
        'T_mb': new_model_K_array - new_model_K_array[0]
    }

    return model_spectrum_dict


def run_convolve_and_prepare_model_spectra(data_dict, abundance=None, vel_center=0, db=None,
                                           temperature_jump=100*u.K):

    if abundance is not None:
        prepare_and_run_ratran_model(*abundance, db=db, temperature_jump=temperature_jump)

    create_sky_spectra_with_miriad()

    model_dict = convert_and_adapt_model_spectra(data_dict, vel_center=vel_center)

    return model_dict

fwhm_linewidth =  6.06
db = 0.6 * fwhm_linewidth

inner_abundances = np.logspace(-9.5, -7.5, 1)
outer_abundances = np.logspace(-11, -10, 1)
# inner_abundances = [3.16e-9]
# outer_abundances = [1e-11, 5e-11]
Tj_values = np.linspace(40, 100, 2)*u.K

chisq_grid = np.zeros((len(inner_abundances), len(outer_abundances), len(Tj_values)))

typical_elapsed_time = datetime.timedelta(minutes=1, seconds=15)

if True and __name__ == "__main__":

    beginning = datetime.datetime.now()
    beginning_string = datetime.datetime.strftime(beginning,"%Y-%m-%d %H:%M:%S")
    print("\n ** Beginning at: {0}".format(beginning_string))

    n = chisq_grid.flatten().shape

    expected_end_time = beginning + typical_elapsed_time*n[0]
    expected_end_time_string = datetime.datetime.strftime(expected_end_time, "%Y-%m-%d %H:%M:%S")
    print("\n ** Expected elapsed time: {0}".format(typical_elapsed_time*n[0]))
    print("\n ** Expected end time: {0}".format(expected_end_time_string))

    pdb.set_trace()

    for i, X_in in enumerate(inner_abundances):
        for j, X_out in enumerate(outer_abundances):
            for k, temperature_jump in enumerate(Tj_values):

                # X_in = 0.5e-9
                # X_out = 1e-11
                # db_val = 1.746
                # temperature_jump = 50*u.K

                vel_center=3.91
                data = prepare_data(vel_center=vel_center, half_vel_span=20)
                models = run_convolve_and_prepare_model_spectra(data, vel_center=vel_center, 
                                                                abundance=(X_in, X_out), db=None, 
                                                                temperature_jump=temperature_jump)

                chi2_of_model = np.sum([chisq_line(x['T_mb'], y['T_mb'], x['rms'])
                                        for x, y in zip(data.values(), models.values())])

                print("\n\n****************************")
                print("****************************")
                print("For X(h13cn)_in={0:.1e} | X(h13cn)_out={1:.1e}| Tj={2:.2f}, chi2 = {3:.2e}".format(X_in, X_out, temperature_jump, chi2_of_model))
                print("****************************")
                print("****************************\n\n")

                # chisq_grid[i,j,k] = chi2_of_model

                fig = plot_model(models, data_dict=data)
                fig.suptitle(
                    "X(h13cn)in = {0:.1e} | X(h13cn)out = {1:.1e} | Tj={2:.1f}".format(
                        X_in, X_out, temperature_jump))
                fig.savefig(
                    "test_plots/Xin={0:.1e}_Xout={1:.1e}_Tj={2:.2f}.png".format(
                        X_in, X_out, temperature_jump))


                end = datetime.datetime.now()
                end_string = datetime.datetime.strftime(end,"%Y-%m-%d %H:%M:%S")

                print("\n ** Ending at: {0}".format(end_string))

                time_elapsed = (end - beginning)
                print( "\n ** Time elapsed: {0}".format(time_elapsed))

                break
            break
        break

    print("Number of radial points: {0}".format(len(rcctp.r)))
    plt.savefig("nr={0}_spectrum_t={1}.png".format(len(rcctp.r), time_elapsed.seconds), bbox_inches='tight')

            # plt.show()

            # pdb.set_trace()


plt.show()
if not np.all(chisq_grid==0):
    np.save("chisq", chisq_grid)

if False:
    # don't save the outputs when they're just default zeroes.
    if not np.all(chisq_grid==0):
        np.save("chisq", chisq_grid)
        np.save("db_vals", db_vals)
        np.save("Xin", inner_abundances)
        np.save("Xout", outer_abundances)


    plt.figure()
    plt.imshow(np.squeeze(chisq_grid), extent=(db_values.min(), db_values.max(), outer_abundances.min(), outer_abundances.max() ))
    plt.xlabel("DB")
    plt.ylabel("Xout")
    cb = plt.colorbar()
    cb.set_label("chi squared")

    # don't save the outputs when they're just default zeroes.
    if not np.all(chisq_grid==0):
        np.save("chisq", chisq_grid)
        np.save("db_vals", db_vals)
        np.save("Xin", inner_abundances)
        np.save("Xout", outer_abundances)

    print(chisq_grid)