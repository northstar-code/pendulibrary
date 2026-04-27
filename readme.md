# Pendulibrary documentation

## Requirements:
### Base:
* numba
* numpy
### Specific Utils
* scipy (for linear X0 expm)
* tqdm (for loading bars)
### Plotters
* scipy
* matplotlib
* os
* plotly
* dash (3.3, NOT LATEST)
* dash_bootstrap_components


## Layout of package
* plots: generic output folder, should be empty on the repo
* scripts: For Jupyter notebooks and other misc scripts. These are user tools, *not* core functions
* src: this is the library itself
    * common: basic functionality; EOMs, Hamiltonian, etc
    * DOP853_coefs: coefficient list for integrator. Do not touch
    * integrate: integrators for state and STM
    * interpolate: contains functions for Cubic Hermite Spline interpolation. Once included RK8 integrator methods, but these have since been depricated as interpolation is only done for visual smoothness
    * targeter: contains first-order differential correction functions
    * utils: Utility functions for scripts to gain linear ICs around the equilibrium. Currently home to only two functions, should either be added to or migrated elsewhere ideally
    * continuation: contains implementations of custom continuation method with adaptive stepping and with fixed stepping, as well as forward-backward continuer for bifurcation search
    * common_targetters: targetter classes. Right now there's only one, but in theory a homo/heteroclinic connection targetter could be added, or perhaps one to abuse symmetris
    * plotters: contains generic plotting functions. I tend to dump functions here once they're robust enough to be used repeatedly


## Copyright
None, just email me or something if you're interested in using this