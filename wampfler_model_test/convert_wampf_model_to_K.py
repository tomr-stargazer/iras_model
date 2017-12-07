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


spectrum_apex_32 = "h13cn_003.apex_spectrum"

freq = 259*u.GHz
beam_fwhm = 24*u.arcsec 
loaded_array = np.loadtxt(spectrum_apex_32, skiprows=3)
index, vel_array, jy_array = loaded_array.T

model_dict_apex_32 = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

# Jy -> K conversion
fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
beam_sigma = beam_fwhm * fwhm_to_sigma
omega_B = 2*np.pi * beam_sigma**2

model_dict_apex_32['T_mb'] = (model_dict_apex_32['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))




spectrum_iram_32 = "h13cn_003.spectrum"

beam_fwhm = 9.7*u.arcsec 

loaded_array = np.loadtxt(spectrum_iram_32, skiprows=3)
index, vel_array, jy_array = loaded_array.T
model_dict_iram_32 = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

# Jy -> K conversion
beam_sigma = beam_fwhm * fwhm_to_sigma
omega_B = 2*np.pi * beam_sigma**2

model_dict_iram_32['T_mb'] = (model_dict_iram_32['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))


spectrum_iram_sourceA_32 = "h13cn_003.iram_sourceA_spectrum"

beam_fwhm = 9.7*u.arcsec 

loaded_array = np.loadtxt(spectrum_iram_sourceA_32, skiprows=3)
index, vel_array, jy_array = loaded_array.T
model_dict_iram_sourceA_32 = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

# Jy -> K conversion
beam_sigma = beam_fwhm * fwhm_to_sigma
omega_B = 2*np.pi * beam_sigma**2

model_dict_iram_sourceA_32['T_mb'] = (model_dict_iram_sourceA_32['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))


spectrum_apex_43 = "h13cn_004.apex_spectrum"

freq = 345*u.GHz
beam_fwhm = 18*u.arcsec 

loaded_array = np.loadtxt(spectrum_apex_43, skiprows=3)
index, vel_array, jy_array = loaded_array.T

model_dict_apex_43 = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

# Jy -> K conversion

beam_sigma = beam_fwhm * fwhm_to_sigma
omega_B = 2*np.pi * beam_sigma**2

model_dict_apex_43['T_mb'] = (model_dict_apex_43['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))


spectrum_jcmt_43 = "h13cn_004.spectrum"

beam_fwhm = 14.6*u.arcsec 

loaded_array = np.loadtxt(spectrum_jcmt_43, skiprows=3)
index, vel_array, jy_array = loaded_array.T

model_dict_jcmt_43 = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

# Jy -> K conversion

beam_sigma = beam_fwhm * fwhm_to_sigma
omega_B = 2*np.pi * beam_sigma**2

model_dict_jcmt_43['T_mb'] = (model_dict_jcmt_43['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))


spectrum_jcmt_sourceA_43 = "h13cn_004.jcmt_sourceA_spectrum"

beam_fwhm = 14.6*u.arcsec 

loaded_array = np.loadtxt(spectrum_jcmt_sourceA_43, skiprows=3)
index, vel_array, jy_array = loaded_array.T

model_dict_jcmt_sourceA_43 = {'vel': vel_array, 'flux': jy_array-jy_array[0]}

# Jy -> K conversion

beam_sigma = beam_fwhm * fwhm_to_sigma
omega_B = 2*np.pi * beam_sigma**2

model_dict_jcmt_sourceA_43['T_mb'] = (model_dict_jcmt_sourceA_43['flux']*u.Jy).to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))





