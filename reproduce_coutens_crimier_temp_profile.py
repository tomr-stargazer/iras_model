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

import astroProcs
import natconst as nc
import transphereProcs as tP
import plotTransphere

### Parameters of physical model

lstar    = 3                 # Stellar luminosity in units of solar luminosities
tstar    = 1000.             # Stellar temperature
rstar    = lstar**0.5*(tstar/5785.0)**(-2.0)*nc.RS
rin      = 1.0 * nc.AU       # Inner radius of shell
rout     = 1.0e4 * nc.AU     # Outer radius of shell
r0       = 10.0 * nc.AU      # Reference radius
#rho0     = 9.0e-16          # Density at reference radius (gas mass). Can be used instead of Menv (need to change code further down)
menv     = 0.5               # Envelope mass in M_sun
plrho    = -1.5              # Powerlaw for rho
isrf     = 0.0               # Scaling of ISRF
dist     = 235.0             # Distance in pc
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
