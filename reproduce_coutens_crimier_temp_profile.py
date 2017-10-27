"""
The specific goal here is to produce (as a prototype) the Coutens-Crimier physical model of IRAS 16293, the density and temp conditions especially.

"""

import os
import sys
import math
import pdb

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapz as integrate
import scipy.interpolate

from transphere_python import astroProcs
import transphere_python.natconst as nc
import transphere_python.transphereProcs as tP
from transphere_python import plotTransphere

### Parameters of physical model

lstar    = 22                # Stellar luminosity in units of solar luminosities
tstar    = 5000.             # Stellar temperature
rstar    = lstar**0.5*(tstar/5785.0)**(-2.0)*nc.RS
rin      = 22 * nc.AU        # Inner radius of shell
rout     = 6100 * nc.AU      # Outer radius of shell
rinf     = 1280 * nc.AU      # Radius at which density distribution changes
#rho0     = 9.0e-16          # Density at reference radius (gas mass). Can be used instead of Menv (need to change code further down)
menv     = 1.9               # Envelope mass in M_sun
plrho_inner= -1.5              # Powerlaw for rho inside r_inf
plrho_outer= -2                # Powerlaw for rho outside r_inf
isrf     = 0.0               # Scaling of ISRF
dist     = 120               # Distance in pc
tbg      = -1                # Spectral shape of ISRF (Blackbody equivalent temperature). With -1 the spectrum is read from the file isrf.inp and scaled by ISRF.

### Parameters related to control of code

nr       = 200               # Nr of radial shells for grid
nref     = 100               # Nr of refinement points
rref     = 10. * nc.AU       # Refinement radius
nriter   = 30                # Maximum nr of iterations
convcrit = 0.00001           # Convergence criterion
ncst     = 10                # Nr of rays for star
ncex     = 30                # Nr of rays between star and Rin
ncnr     = 1                 # Nr of rays per radial grid point
itypemw  = 1                 # Type of mu weighting
idump    = 1                 # Dump convergence history
localdust= 0                 # Dust opacity local?

o = tP.readopac(nr='oh5')           ### Read-in dust opacity file
tP.writeopac(o, localdust, nr)      ### Write-out opacity as function of radius
kappa = tP.findKappa(localdust, o)  ### Find Kappa at 0.55 micron

r = tP.makeRadialGrid(nref, rin, rout, rref, nr)  # Make radial grid

import astropy.constants as c
import astropy.units as u

# 2.89 comes from this calculation:
# ((45*2*c.m_p + 10*4*c.m_p)/(45))/c.m_p
number_density_to_mass_density_of_H2 = 2.89*c.m_p

# this value was read off of the Crimier+2010 Figure 3 plot, using WebPlotDigitizer - so quite precise
n_at_rinf = 2.67e6 * u.cm**-3
rho_at_rinf = (n_at_rinf * number_density_to_mass_density_of_H2).to(u.g*u.cm**-3)

# print 4.0*math.pi/(3.0+plrho)*rho0*r0**(-plrho)*(rout**(3.0+plrho)-rin**(3.0+plrho))/1.989e33
# If rho0 is given above then uncomment here.
# rho0 = menv/(4.0*math.pi/(3.0+plrho)*r0**(-plrho) *
#              (rout**(3.0+plrho)-rin**(3.0+plrho))/1.989e33)

rho = np.zeros_like(r)
rho[r<=rinf] = rho_at_rinf.value * (rinf/r[r<=rinf])**(-plrho_inner)
rho[r>=rinf] = rho_at_rinf.value * (rinf/r[r>=rinf])**(-plrho_outer)

rho_dust = rho/100

number_density = rho / rho_at_rinf

# rho = 1e-2*rho0 * (r/r0)**(plrho)
#tau    = integrate(rho*kappa,r)
# print 'Tau = ',tau

model = {"rstar": rstar, "mstar": 1.0, "tstar": tstar, "r": r, "rho": rho_dust, "isrf": isrf, "tbg": tbg, 
    "freq": o['freq'], "nriter": nriter, "convcrit": convcrit, "ncst": ncst, "ncex": ncex, 
    "ncnr": ncnr, "itypemw": itypemw, "idump": idump}
tP.writeTransphereInput(model)

os.system('transphere')  # Change this to a popen call or something

# import transphere_python.readSed

# sed = transphere_python.readSed.readSed('sed.dat')


s = tP.readSpectrum()
a = tP.readConvhist()

plt.figure(1)
plotTransphere.plotSpectrum(s, dpc=dist, jy=1, pstyle='b-')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1, 3.0e3))
plt.ylim((1.0e-9, 1.0e3))
z = {"freq": o['freq'], "spectrum": math.pi *
     astroProcs.bplanck(o['freq'], tstar)*rstar**2/nc.pc**2}
plotTransphere.plotSpectrum(z, dpc=dist, jy=1, pstyle='g--')
