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

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c

import transphere_python.transphereRatran as tR

### Phase 1: set up the physical model.

# At the moment, everything here is "placeholder". I'll fill in with the real model params later.

distance = 120 * u.pc

rho_nought = 9.0e-16 * u.g * u.cm**-3
radius_array = (np.arange(0, 1000, 50)*u.AU).to(u.cm)
density_array = (radius_array.to(u.AU)/(1*u.AU))**-1.5 * rho_nought

temp_array = np.ones_like(radius_array)*50*u.K
abundance_array = np.ones_like(radius_array.value) * 1e-7

pdb.set_trace()

### Phase 2: Run the AMC/Ratran situation.

os.makedirs("./h13cn_emission_models/", exist_ok=True)

transition_list = [1,3,4,6,7,8,9,10,11]
transitions_string = ",".join(str(x) for x in transition_list)
frequency_list = u.Quantity([86.3399, 172.6778, 259.0117, 345.3397, 431.6597, 517.9698, 604.2679, 
                            690.5520, 776.8203, 863.0706, 949.3010], u.GHz)

if False:
    tR.ratranRun(r=radius_array.value, rho=density_array.value, temp=temp_array.value, db=0.3,
                 abund=abundance_array, dpc=distance.value, trans=transitions_string, 
                 molfile='h13cn.dat', unit='jypx', writeonly=0, skyonly=0,
                 outputfile='h13cn_emission_models/ratranResult_h13cn')

# Ok so this creates DATACUBES. 
# We'd like to manipulate theese datacubes to create simulated spectra.

# How... do we do that?
# Shell miriad commands? Probably.

list_of_filenames = glob.glob("./h13cn_emission_models/*")

os.makedirs("./h13cn_miriad_manipulation/", exist_ok=True)

for i, filename in enumerate(list_of_filenames):
    
    J_upper = transition_list[i]
    freq = frequency_list[i]

    pdb.set_trace()

    # make this os.system later but we're proofing it first.
    call_fn = os.system

    convert_fits_to_miriad = 'fits in={0} out=h13cn_miriad_manipulation/h13cn_{1:03d}.sky op=xyin'.format(filename, J_upper)
    call_fn(convert_fits_to_miriad)

    put_frequency_in_header = 'puthd in=h13cn_miriad_manipulation/h13cn_{0:03d}.sky/restfreq value={1:.3f}'.format(J_upper, freq.value)
    call_fn(put_frequency_in_header)

    # Make a spectrum and output it
    make_spectrum = 'imspect in=h13cn_miriad_manipulation/h13cn_{0:03d}.sky log=h13cn_{0:03d}.spectrum'.format(J_upper, freq.value)
    call_fn(make_spectrum)


# Visualize the output spectra.
fig = plt.figure()

list_of_spectra = glob.glob("*.spectrum")
for i, spectrum in enumerate(list_of_spectra):

    J_upper = transition_list[i]
    freq = frequency_list[i]

    loaded_array = np.loadtxt(spectrum, skiprows=3)
    index, vel_array, jy_array = loaded_array.T

    ax = fig.add_subplot(3, 3, i+1)
    ax.plot(vel_array, jy_array, linestyle='steps-mid')

    plt.show()

