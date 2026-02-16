import numpy as np
import scipy.stats
import csv
import matplotlib.pyplot as plt
import miepython
import scipy
import os.path
from matplotlib import image
import pandas as pd

# Pioneer Variables
lamb = 0.9
inst_unit_Neph = 5 * 10 ** -6
min_size = 0.3
max_size = 20.0
# Forward Model Functions
def deg_to_rad(ang):
    return ang * np.pi / 180
def matrix_Pioneer(lambda0, a_Theta, a_f, min_a=0.3, max_a=20.0, step_a=0.001, min_ri=1.25, max_ri=2.00, step_ri=0.001, inst_const=inst_unit_Neph):
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