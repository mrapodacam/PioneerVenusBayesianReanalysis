"""
Forward Model Matrix Generation for Pioneer Venus Nephelometer
===============================================================

This module generates pre-computed Mie scattering intensity matrices for
efficient Bayesian retrieval of atmospheric particle properties from
Pioneer Venus Nephelometer measurements.

The forward model computes the expected scattering signal as a function of:
- Particle size (diameter in microns)
- Complex refractive index (real part)

The matrices are saved as NumPy arrays for fast loading during retrieval.

References
----------
- Mie scattering theory implementation via miepython
- Pioneer Venus Nephelometer instrument specifications from mission documentation
"""

import numpy as np
import scipy.stats
import csv
import matplotlib.pyplot as plt
import miepython
import scipy
import os.path
from matplotlib import image
import pandas as pd

# =============================================================================
# Pioneer Venus Nephelometer Constants
# =============================================================================

lamb = 0.9  # Wavelength in microns
inst_unit_Neph = 5e-6  # Instrument calibration constant
min_size = 0.3  # Minimum particle diameter (microns)
max_size = 20.0  # Maximum particle diameter (microns)


# =============================================================================
# Forward Model Functions
# =============================================================================

def deg_to_rad(ang):
    """
    Convert angle from degrees to radians.

    Parameters
    ----------
    ang : float or ndarray
        Angle(s) in degrees.

    Returns
    -------
    float or ndarray
        Angle(s) in radians.
    """
    return ang * np.pi / 180


def matrix_Pioneer(lambda0, a_Theta, a_f, min_a=0.3, max_a=20.0, step_a=0.001,
                   min_ri=1.25, max_ri=2.00, step_ri=0.001, inst_const=inst_unit_Neph):
    """
    Generate Mie scattering intensity matrices for the Pioneer Nephelometer.

    Computes the scattering coefficient (beta) as a function of particle size
    and refractive index, integrated over the instrument's angular response
    function. The resulting matrices can be used for fast forward model
    evaluation during Bayesian inference.

    Parameters
    ----------
    lambda0 : float
        Wavelength of incident light in microns.
    a_Theta : array-like
        Scattering angles in degrees from the instrument response function.
    a_f : array-like
        Angular response function F(theta) values corresponding to a_Theta.
    min_a : float, optional
        Minimum particle diameter in microns. Default is 0.3.
    max_a : float, optional
        Maximum particle diameter in microns. Default is 20.0.
    step_a : float, optional
        Step size for particle diameter grid in microns. Default is 0.001.
    min_ri : float, optional
        Minimum refractive index. Default is 1.25.
    max_ri : float, optional
        Maximum refractive index. Default is 2.00.
    step_ri : float, optional
        Step size for refractive index grid. Default is 0.001.
    inst_const : float, optional
        Instrument calibration constant. Default is 5e-6.

    Returns
    -------
    X : ndarray
        2D meshgrid of particle sizes (microns), shape (n_ri, n_a).
    Y : ndarray
        2D meshgrid of refractive indices, shape (n_ri, n_a).
    I_scat : ndarray
        Scattering intensity matrix, shape (n_ri, n_a).

    Notes
    -----
    The scattering coefficient beta is computed as:

        beta = I_unpol * sigma * inst_const

    where:
    - I_unpol is the unpolarized Mie scattering intensity
    - sigma = Q_sca * pi * (a/2)^2 is the scattering cross-section
    - inst_const is the instrument calibration factor

    The total intensity integrates over the angular response function.

    Examples
    --------
    >>> X, Y, Isca = matrix_Pioneer(0.9, angles, response_fn)
    >>> np.save('Isca_matrix.npy', Isca)
    """
    theta = a_Theta
    rad_theta = deg_to_rad(theta)
    mu = np.cos(rad_theta)
    lam = lambda0
    k = 2 * np.pi / lam  # wave number
    a_s = np.arange(min_a, max_a, step_a)  # min particle size is 0.3 per Ragent and Balmont
    ri_s = np.arange(min_ri, max_ri, step_ri)
    I_scat = np.zeros((len(ri_s),len(a_s)))
    # n_interp = scipy.interpolate.interp1d(layer_dis[0],layer_dis[1], fill_value='extrapolate')
    for j, a in enumerate(a_s):
        for i, m in enumerate(ri_s):
            x = (a / 2) * k
            geometric_cross_section = np.pi * (a / 2) ** 2  # it was normally in microns
            # normalised to one
            qext, qsca, qback, g = miepython.mie(m, x)
            intensity = miepython.i_unpolarized(m, x, mu, 'one')
            sigma = qsca * geometric_cross_section
            beta = intensity * sigma * inst_const
            I = beta * step_a * step_ri
            for l, f in enumerate(a_Theta):
                if l == 0:
                    weight = 0
                else:
                    # weight = np.abs(f-a_Theta[i-1]) * np.pi / 180 # This is for when we integrate in radiant but we are integrating in degrees instead!
                    weight = np.abs(f - a_Theta[l- 1])
                # Total Measured Intensity
                # I_scat[i] = beta * weight * a_f[i] * n *step_a
                I[l] = I[l] * weight * a_f[l]
            I_tot=np.sum(I)
            I_scat[i][j] = I_tot
    [X, Y] = np.meshgrid(a_s, ri_s)
    return X, Y, I_scat

data_dir = os.path.dirname(__file__)
file_path = os.path.join(data_dir, "WorkData")
file_dis = os.path.join(file_path,'LPSC_distribution.csv')
angular = os.path.join(file_path, "preflight.csv")
dfA = pd.read_csv(angular)
array_f = dfA['F(Theta)'].values
array_Theta = dfA['Angle'].values

X, Y, I_scat= matrix_Pioneer(lamb,array_Theta,array_f)

# Save the data to a CSV file
file_dir = os.path.join(data_dir,'ForwardModelMatrix_delta_001')
os.makedirs(file_dir, exist_ok=True)
np.save(os.path.join(file_dir, 'Isca_matrix.npy'), I_scat)
np.save(os.path.join(file_dir, 'X_matrix.npy'), X)
np.save(os.path.join(file_dir, 'Y_matrix.npy'), Y)