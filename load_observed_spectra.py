import os
import glob
import pdb
import matplotlib.pyplot as plt

import astropy.units as u

from reduce_herschel_spectra import read_in_fits_spectrum_from_class
import pyspeckit

load_a_spectrum = read_in_fits_spectrum_from_class.load_a_spectrum


def retrieve_herschel_spectrum(filename):
    """
    Simply a renaming of `reduce_herschel_spectra.read_in_fits_spectrum_from_class.load_a_spectrum`.

    Parameters
    ==========
    filename : str
        name of FITS file which contains the spectrum.

    Returns
    =======
    spectrum : np.ndarray; Unit: K
        Intensity values of spectrum, before any efficiency corrections. 
    freq_array : np.ndarray; Unit: Hz.
        Frequency values of `spectrum`.
    vel_array : np.ndarray; Unit: km/s
        Velocity values of `spectrum`.

    """

    spectrum, freq_array, vel_array = load_a_spectrum(filename)

    return spectrum, freq_array, vel_array


def retrieve_timasss_spectrum(filename, line_frequency, correct_velocity=True):

    # also to be figured out.
    data_location = os.path.expanduser("~/Documents/Data/timasss/")
    sp = pyspeckit.Spectrum(os.path.join(data_location, filename))

    sp.xarr.refX = line_frequency

    if correct_velocity:
        velocity_correction = + (sp.header['VELO-LSR']*u.m/u.s).to(u.km/u.s)
    else:
        velocity_correction = 0*u.km/u.s

    velocity_axis = sp.xarr.as_unit(u.km/u.s, velocity_convention='radio') + velocity_correction

    return velocity_axis, sp


def retrieve_apex_spectrum(filename, line_frequency, correct_velocity=True, smooth_factor=10):

    # also to be figured out.
    data_location = os.path.expanduser("~/Documents/Data/APEX/Wampfler_APEX_IRAS16293")
    sp = pyspeckit.Spectrum(os.path.join(data_location, filename))

    sp.xarr.refX = line_frequency

    # SMOOTH it
    sp.smooth(smooth_factor)

    if correct_velocity:
        velocity_correction = + (sp.header['VELO-LSR']*u.m/u.s).to(u.km/u.s)
    else:
        velocity_correction = 0*u.km/u.s

    velocity_axis = sp.xarr.as_unit(u.km/u.s, velocity_convention='radio') + velocity_correction

    return velocity_axis, sp


def demonstration_retrieve_herschel_spectra():

    fig = plt.figure()

    # A little function to retrieve the spectra  - this will outline our approach and tell us which functions we need etc

    # Lots of this is copied from fit_the_lines_script.py in reduce....
    fit_results_path = os.path.expanduser("~/Documents/Data/Herschel_Science_Archive/IRAS16293/Fit_results")
    list_of_files = glob.glob(fit_results_path+"/HCN*.fits")
    list_of_spectra = [x for x in list_of_files if 'spectrum.fits' in x]
    # Don't need the results so they're not here.

    for i, spectrum_fname in enumerate(list_of_spectra):

        ax = fig.add_subplot(3,4,i+1)

        spectrum_tuple = load_a_spectrum(spectrum_fname)
        # result_tuple = load_a_spectrum(result_fname)

        ax.plot(spectrum_tuple[2], spectrum_tuple[0], 'k', lw=1, drawstyle='steps-mid')
        # ax.plot(result_tuple[2], result_tuple[0], 'r', lw=0.75)

        ax.set_xlim(-40, 40)
        ax.set_ylim(-0.5, 1)

        pdb.set_trace()

