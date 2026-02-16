import numpy as np
import scipy.stats
import csv
import matplotlib.pyplot as plt
import matplotlib
import miepython
import random
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import corner
import scipy
import os
import os.path
import glob
from matplotlib import image
import pandas as pd
from scipy.optimize import fsolve
from collections import Counter
from collections import defaultdict
from KDEpy import FFTKDE
from scipy.signal import find_peaks
from matplotlib import image
from itertools import zip_longest

#Nested Sampling Variables
rstate = np.random.default_rng(56101)
n_Live = 1800# Num of points we are runing the NS for (MultiRI=900, NoLCPS=1800)
type_s = 'Dynamic' # Options: 'Static', 'Dynamic' or 'Both'
analysis_strat = 'Layer' # Options: 'Layer', 'independent'
type_a = 'HyperMatrix' # Options: 'HyperMatrix'
# n_dim= 19
# num_p= 1000 # Number of particle sizes
probe= 'north' # Options are: 'sounder', 'day', 'night', 'north'
type_p = 'NoLCPS'# Options are: 'MultipleRI', 'NoLCPS'
re_run= False# Default: False

# Pioneer Nephelometer Variables
lamb = 0.9
inst_unit_Neph = 5 * 10 ** -6
# inst_unit_Neph = 1
min_size = 0.3
max_size = 20.0


# Layer Plots
show_hist = False
save_hist = False
show_histM = False
save_histM = False
save_mode = True
save_mean = False
show_cornerBP = False
save_cornerBPlot = False
show_cornerMP = False
save_cornerMPlot = False
save_cornerDisMean = False
save_cornerDisMode = False
show_cornerN = False
save_cornerN = False
show_CDFN = False
save_CDFN = False
show_2D = False
save_2D = False
# Ridge Plots
show_ridgePlotRI = False
save_ridgePlotRI = False
show_ridgePlotS = False
save_ridgePlotS = False
# Region Plots
show_ridgePlotRegion = False
save_ridgePlotRegion = False
show_hist_region = False
save_hist_region = False
show_corner_region = False
save_corner_region = True
show_CDFN_region = False
save_CDFN_region = False
if save_corner_region or show_corner_region:
    show_cornerBP_region = False
    save_cornerBPlot_region = True
    show_cornerMP_region = False
    save_cornerMPlot_region = True
    show_cornerN_region = False
    save_cornerN_region = True
    save_cornerDisMean_region = False
    save_cornerDisMode_region = False
else:
    show_cornerBP_region = False
    save_cornerBPlot_region = False
    show_cornerMP_region = False
    save_cornerMPlot_region = False
    show_cornerN_region = False
    save_cornerN_region = False
    save_cornerDisMean_region = False
    save_cornerDisMode_region = False

# What to data to plot
use_mode = True# Default True (for corner plot)
use_mean = False
use_cornerMode = False
use_cornerMean = False
probs= False #Default: False


# Plotting Variables
set_range= False # Default: False
type_step = 'bins'# Options: 'bins', 'deltas', 'changing_d'
hist_deltas = 0.001
hist_bins = 250 # Default: 250
corner_bins = 75 # Default: 30
epsilon = 9*1e-1
sig_hist = 3 #Default: 1
sig_alts= 1 #Default 2
show_peaks = True # Default: False
mirror_data = True # DEfault: False
n_peaks = 4
corner_deltas = 1
figSize_x = 10 #Default: 60
figSize_y = 10 # Default:
fo = 18# Font size Default: 18
my_colorMap='Reds' #Default: 'coolwarm', Options: 'Greys', 'Reds'*
num_cont= 5
bws=0.01
low_a = 0.3
high_a = 5 #20
low_ri = 1.3 #1.25
high_ri = 1.7 #2.00
scale_type= 'Norm' # Scale of PDF Options: 'Normal' and 'Log'
log_base= 10# Default 10 ln is e
use_log_bins= True
low_pdf=1e-4 #5
high_pdf =1
smooth_corner= True #Default: False
smooth_val=1.5#0.5 #Default: 0.02
show_quantiles= True # Default:True
q_min=0.025 #Default: 0.01 for 98% confidence interval
show_fit= True #Default: True
log_min= 5e-1
# Processing
filter_values= False# Default: False
w_prior = False
filter_what= ['sigma'] #Options: 'nums', 'index','sigma'
prior_what= ['index'] #Options: 'index', 'sigma', 'rho','rho_diagonal', 'num_total'
type_Prior= ['Norm'] # Options: 'Uni', 'Norm', 'LogN', None
# prior_what= ['num_total'] #Options: 'index', 'sigma', 'rho','rho_diagonal'
# type_Prior= ['Uni'] # Options: 'Uni', 'Norm', 'LogN',None
filt_n_min= 0.005
filt_n_max= 3000
filt_s =0.27
filt_s_min =0.0125
filt_min = 1.30 # R.I
filt_max = 1.70# R.I
#Refractive index Proir
prior_min = 1.25 #1.25
prior_max = 2.00 #2.00
#Norm
prior_m = 1.44
prior_unc = 0.13/2#0.14#/2
#Uniform
# prior_m = 1.50
# prior_unc = 0.20
prior_ratio = 100

#Sigma Prior
#Norm
# prior_m_sig= 0.01
# prior_max_sig = 0.2 # MultiRI:0.1, NoLCPS:0.2
# prior_unc_sig= 0.04#0.125 # Ideal: 0.25
# prior_sig = 4 # Number of Decimal Places
# prior_m_sig= 0.033
# prior_unc_sig= 0.033#0.125 # Ideal: 0.25\
prior_m_sig= 0.020
prior_unc_sig= 0.040#0.20 # Ideal: 0.25\
prior_max_sig = 0.2 # MultiRI:0.1, NoLCPS:0.2

# prior_sig = 4 # Number of Decimal Places
#Uni
# prior_m_sig= 0.135 # Full range
# prior_unc_sig= 0.135 #0.125
# prior_m_sig= 0.141
# prior_unc_sig= 0.129 #0.125
prior_max_sig = 0.27 # MultiRI:0.1, NoLCPS:0.2
prior_sig = 4 # Number of Decimal Places
prior_ratio_sig= 100
# prior_min_sig = 1.25 #1.25
# prior_max_sig = 2.00 #2.00
# prior_m = 0.001

#Rho Prior
prior_m_rho = 0#
prior_unc_rho = 0.3#Norm: 0.2, Uni: 0.3

#Num Prior
prior_min_n = 3 #1.25
prior_max_n = 3000 #2.00
prior_m_n = 750
prior_unc_n = 550
prior_sig_n = 1


#Read Data Functions
def read_PostDist_CSV(file_j, n_modes,type_p=type_p,type_a=type_a, a_strat=analysis_strat):#,type_s=type_s,probe=probe,n_live=n_Live):
    # if probe== 'sounder':
    #     paths = os.path.join(data_dir, f"NS_{probe}_{type_p}_{type_a}_{a_strat}_{type_s}_numL{n_live}")
    # else:
    #     paths = os.path.join(data_dir, f"NS_{probe}_{type_a}_{a_strat}_{type_s}_numL{n_live}")
    # dis_path = os.path.join(paths, "PosteriorDistributions")
    # dis_path=data_dir
    # if a_strat== 'Layer':
    #     pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
    # elif a_strat== 'independent':
    #     pos_dis = os.path.join(dis_path, f"PosDist_{j}.csv")
    df = pd.read_csv(file_j)
    col_data = df.columns[:-1].dropna().values
    sample_data = df[col_data].dropna().values
    weights = np.exp(df['Weight'].dropna().values)
    if type_a == 'HyperMatrix':
        cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['rho', 'sigr', 'mur']]
        if type_p != 'MultipleRI':
            extra_cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['n', 'sig', 'mu']]
            cols_names= np.concatenate((extra_cols_names,cols_names))
        # if type_p == 'MultipleRI':
        #     cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['rho', 'sigr', 'mur']]
        # else:
        #     cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['n','sig','mu','rho', 'sigr', 'mur']]
    header = np.concatenate((cols_names, ['Weight']))
    return sample_data, weights, header
def read_myPostDist_CSV(file_j, n_modes,type_p=type_p,type_a=type_a, a_strat=analysis_strat):#,type_s=type_s,probe=probe,n_live=n_Live):
    # if probe== 'sounder':
    #     paths = os.path.join(data_dir, f"NS_{probe}_{type_p}_{type_a}_{a_strat}_{type_s}_numL{n_live}")
    # else:
    #     paths = os.path.join(data_dir, f"NS_{probe}_{type_a}_{a_strat}_{type_s}_numL{n_live}")
    # dis_path = os.path.join(paths, "PosteriorDistributions")
    # dis_path=data_dir
    # if a_strat== 'Layer':
    #     pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
    # elif a_strat== 'independent':
    #     pos_dis = os.path.join(dis_path, f"PosDist_{j}.csv")
    df = pd.read_csv(file_j)
    col_data = df.columns[:-1].dropna().values
    sample_data = df[col_data].dropna().values
    weights = df['Weight'].dropna().values
    if type_a == 'HyperMatrix':
        cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['rho', 'sigr', 'mur']]
        if type_p != 'MultipleRI':
            extra_cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['n', 'sig', 'mu']]
            cols_names= np.concatenate((extra_cols_names,cols_names))
        # if type_p == 'MultipleRI':
        #     cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['rho', 'sigr', 'mur']]
        # else:
        #     cols_names = [f'{param} {i + 1}' for i in range(n_modes) for param in ['n','sig','mu','rho', 'sigr', 'mur']]
    header = np.concatenate((cols_names, ['Weight']))
    return sample_data, weights, header
def read_LCPS_Vals(data_dir):
    file_path = os.path.join(data_dir, "WorkData")
    file_dis = os.path.join(file_path, r'LPSC_distribution.csv')
    # Read the distribution file
    with open(file_dis, 'r') as csvfile:
        reader = pd.read_csv(csvfile)
        array_Type = reader['Cloud Layer'].values
        array_MaxAlt = reader['Max Alt'].values
        array_MinAlt = reader['Min Alt'].values
        NT1, S1, M1 = reader['Nt1'].values, reader['sig_g1'].values, reader['mean_g1'].values
        NT2, S2, M2 = reader['Nt2'].values, reader['sig2'].values, reader['mean2'].values
        NT3, S3, M3 = reader['Nt3'].values, reader['sig_g3'].values, reader['mean_g3'].values

    size_dis = list(zip(NT1, S1, M1, NT2, S2, M2, NT3, S3, M3))
    return size_dis, array_MinAlt, array_MaxAlt

# Supporting Functions
def find_weighted_mode(data,weights):
    # Assume the last column is the weight
    observations = data
    weights = weights

    # Create a dictionary to store weighted counts
    weighted_counts = defaultdict(float)

    # Count occurrences, considering weights
    for obs, weight in zip(observations, weights):
        weighted_counts[tuple(obs)] += weight

    # Find the observation with the highest weighted count
    # mode = max(weighted_counts, key=weighted_counts.get)
    # counts =max(weighted_counts.values())
    # print(counts)
    sorted_counts = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)
    # return list(mode),counts
    return dict(sorted_counts)
def num_bins(arr, size):
    step = int(np.ceil(np.abs(np.max(arr) - np.min(arr)) / size))
    return max(step, 10)
def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
def load_matrices_numpy(directory):
    Isca = np.load(os.path.join(directory, 'Isca_matrix.npy'))
    X = np.load(os.path.join(directory, 'X_matrix.npy'))
    Y = np.load(os.path.join(directory, 'Y_matrix.npy'))
    return X,Y, Isca
def bivariate_distribution(arr,X,Y):
    # print('Entering Samp')
    # xs = np.linspace(0.1, 20.0, 199)
    # ys = np.linspace(1.25, 2.00, 750)
    # x,y =np.meshgrid(xs,ys)
    # x,y= np.mgrid[0.1:20.0:.1,1.25:2.00:.001]
    # x, y = np.mgrid[0.1:20.0:1, 1.25:2.00:.1]
    # pos = np.dstack((x, y))
    x_coords = X.ravel()  # ravel() is faster than flatten() for contiguous arrays
    y_coords = Y.ravel()

    # Pre-compute log coordinates (avoid repeated log calls)
    log_x = np.log(x_coords)
    log_y = np.log(y_coords)

    # Vectorized coordinate handling
    pos = np.column_stack([x_coords, y_coords])
    log_pos = np.column_stack([log_x, log_y])
    # n1, sig1, mu1, n2, sig2, mu2, n3, sig3, mu3, rho1, sigr1, mur1, rho2, sigr2, mur2, rho3, sigr3, mur3 = arr
    log_sig = np.log(arr[1])
    log_sigr = np.log(arr[10])
    var_x = log_sig ** 2 # log(sig1) **2
    var_y = log_sigr ** 2# log(sigr1) **2
    cov_xy = arr[9]* log_sig*log_sigr # rho1* sig1 * sigr1
    cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
    mean = np.array([np.log(arr[2]), np.log(arr[11])])
    rv = scipy.stats.multivariate_normal(mean, cov_matrix)# n1 * scipy.stats.multivariate_normal([mu1, mur1],[[var_x1,cov_xy1],[cov_xy1,var_y1]])
    prob_m= arr[0]*rv.pdf(log_pos)
    prob_tot = prob_m / (x_coords * y_coords)
    if not np.isnan(arr[3]):
        var_x = arr[4]**2 # sig2 ** 2
        var_y = arr[13]**2 # sigr2 ** 2
        cov_xy = arr[12]*arr[4]*arr[13] # rho2 * sig2 * sigr2
        cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        mean =np.array([arr[5], arr[14]])
        rv = scipy.stats.multivariate_normal(mean, cov_matrix) # n2 * scipy.stats.multivariate_normal([mu2, mur2], [[var_x2, cov_xy2], [cov_xy2, var_y2]])
        prob_tot += arr[3]*rv.pdf(pos)
    if not np.isnan(arr[6]):
        log_sig = np.log(arr[7])
        log_sigr = np.log(arr[16])
        var_x = log_sig ** 2  # log(sig1) **2
        var_y = log_sigr ** 2  # log(sigr1) **2
        cov_xy = arr[15] * log_sig * log_sigr  # rho1* sig1 * sigr1
        cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
        mean = np.array([np.log(arr[8]), np.log(arr[17])])
        rv = scipy.stats.multivariate_normal(mean, cov_matrix)
        prob_m = arr[6] * rv.pdf(log_pos)
        rv = prob_m / (x_coords * y_coords)
        prob_tot += rv
    prob_tot_grid = prob_tot.reshape(Y.shape)
    return prob_tot_grid
def marginal_distribution(arr,X):
    #n1, sig1, mu1, n2, sig2, mu2, n3, sig3, mu3 = arr
    pos = X
    log_sig = np.log(arr[1])
    mean = arr[2]
    rv = scipy.stats.lognorm(s=log_sig, loc=0, scale=mean)
    prob_tot= arr[0]*rv.pdf(pos)
    if len(arr)>3:
        sig = arr[4] # sig2 ** 2
        mean = arr[5]
        rv = scipy.stats.norm(scale=sig, loc=mean) # n2 * scipy.stats.multivariate_normal([mu2, mur2], [[var_x2, cov_xy2], [cov_xy2, var_y2]])
        prob_tot += arr[3]*rv.pdf(pos)
    if len(arr)>6:
        log_sig = np.log(arr[7])
        mean = arr[8]
        rv = scipy.stats.lognorm(s=log_sig, loc=0, scale=mean)
        prob_m = arr[6] * rv.pdf(pos)
        prob_tot += prob_m
    prob_tot/=np.sum(prob_tot)
    return prob_tot
def marginal_distributionNormal(arr,X):
    #n1, sig1, mu1, n2, sig2, mu2, n3, sig3, mu3 = arr
    pos = X
    log_sig = np.log(arr[1])
    mean = arr[2]
    rv = scipy.stats.lognorm(s=log_sig, loc=0, scale=mean)
    prob_tot= arr[0]*rv.pdf(pos)*pos
    if len(arr)>3:
        sig = arr[4] # sig2 ** 2
        mean = arr[5]
        rv = scipy.stats.norm(scale=sig, loc=mean) # n2 * scipy.stats.multivariate_normal([mu2, mur2], [[var_x2, cov_xy2], [cov_xy2, var_y2]])
        prob_tot += arr[3]*rv.pdf(pos)
    if len(arr)>6:
        log_sig = np.log(arr[7])
        mean = arr[8]
        rv = scipy.stats.lognorm(s=log_sig, loc=0, scale=mean)
        prob_m = arr[6] * rv.pdf(pos)*pos
        prob_tot += prob_m
    prob_tot/=np.sum(prob_tot)
    return prob_tot

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    # diff_arr = max(arr) - min(arr)
    for i in arr:
        # temp = (((i - min(arr))*diff)/diff_arr) + t_min
        temp = (i - t_min) / diff
        norm_arr.append(temp)
    return norm_arr
# Adding New Prior Functions
def trucatedNormal_Prior_distribution(value_array):
    mu_val, err_val, min_val, max_val, step_dec=value_array
    # print(mu_val, err_val, min_val, max_val, step_dec)
    num_par = int(np.ceil(np.abs(max_val - min_val) / (10 ** (-step_dec))))
    x = np.linspace(min_val, max_val, num_par)
    # Truncated Normal
    m3 = mu_val
    s3 = err_val
    low3, high3 = min_val, max_val  # lower and upper bounds
    low_n3, high_n3 = (low3 - m3) / s3, (high3 - m3) / s3  # standardize
    prob_x = scipy.stats.truncnorm.pdf(x, low_n3, high_n3, loc=m3, scale=s3)
    return x, prob_x
def stepUniform_Prior_distribution(value_array,n=prior_ratio):
    mu_val, err_val, min_val, max_val, step_dec = value_array
    # print(mu_val, err_val, min_val, max_val, step_dec)
    num_par = int(np.ceil(np.abs(max_val - min_val) / (10 ** (-step_dec))))
    x = np.linspace(min_val, max_val, num_par)
    # Truncated Normal
    m3 = mu_val
    s3 = err_val
    # Create the stepped probability distribution
    prob = np.ones_like(x)
    # Identify points in the central region (mu_val ± s3)
    central_mask = (x >= m3 - s3) & (x <= m3 + s3)
    # Assign higher probability to central region
    prob[central_mask] = n
    # Normalize so the total probability sums to 1
    prob = prob / np.sum(prob)
    return x,prob
def truncatedLogNormal_Prior_distribution(value_array):
    mu_val, err_val, min_val, max_val, step_dec = value_array
    # print(mu_val, err_val, min_val, max_val, step_dec)
    num_par = int(np.ceil(np.abs(max_val - min_val) / (10 ** (-step_dec))))
    x = np.linspace(min_val, max_val, num_par)

    # Truncated Lognormal
    # mu_val and err_val are the mean and std of the underlying normal distribution
    # (not the mean and std of the lognormal distribution itself)
    # low_bound, high_bound = min_val, max_val

    # For lognormal, we need x > 0, so ensure min_val > 0
    if min_val <= 0:
        raise ValueError("For lognormal distribution, min_val must be > 0")
    median = err_val
    mode = mu_val
    std=np.sqrt(np.log(median/mode))
    mu=np.log(median)
    means=np.exp(mu+0.5*(std)**2)
    print(mode, median, std,mu, means)
    # mode=mu_val
    # mean=err_val
    # std=np.sqrt((2/3)*np.log(mean/mode))
    # median=mode/np.exp(std**2)
    # print(mode, mean, std, median)
    # Standardize bounds for the underlying normal distribution
    # low_n = (np.log(low_bound) - mean_log) / std_log
    # high_n = (np.log(high_bound) - mean_log) / std_log

    # Calculate PDF using truncated normal on log-scale
    # prob_x = scipy.stats.truncnorm.pdf(np.log(x), low_n, high_n, loc=mean_log, scale=std_log) / x
    prob_x = scipy.stats.lognorm.pdf(x, s=std , loc=0, scale=median)
    return x, prob_x
def diagonalGaussian_Prior_distribution(value_array):
    """
    Creates a 2D Gaussian prior that favors similar values between two parameters.
    This is designed to work with rho1 and rho2 to favor them being close to each other.

    Parameters:
    -----------
    value_array : list or array
        [sigma_diagonal, min_val, max_val, step_dec]
        - sigma_diagonal: width of Gaussian around diagonal (smaller = stronger correlation)
        - min_val: minimum value for both parameters
        - max_val: maximum value for both parameters
        - step_dec: decimal places for grid resolution

    Returns:
    --------
    x : array
        1D array of parameter values
    prob_2d : 2D array
        Probability density on the 2D grid (shape: [len(x), len(x)])
    """
    sigma_diagonal, min_val, max_val, step_dec = value_array

    # Create 1D grid
    num_par = int(np.ceil(np.abs(max_val - min_val) / (10 ** (-step_dec))))
    x = np.linspace(min_val, max_val, num_par)

    # Create 2D meshgrid
    X1, X2 = np.meshgrid(x, x)

    # Distance from diagonal (where param1 = param2)
    distance_from_diagonal = X2 - X1

    # 2D Gaussian prior centered on the diagonal
    prob_2d = np.exp(-0.5 * (distance_from_diagonal / sigma_diagonal) ** 2)

    # Normalize so that the integral over the 2D space equals 1
    prob_2d = prob_2d / (np.sum(prob_2d) * (x[1] - x[0]) ** 2)

    return x, prob_2d


def diagonalPair_Prior(sel_val, val_arr, val_weight, val_labels, n_modes=1,type_p='MultipleRI',sigma_diagonal=0.1,prior_min=-1.0,prior_max=1.0,prior_sig=3):
    """
    Apply diagonal Gaussian prior to pairs of rho parameters.
    Automatically detects rho1, rho2, (rho3) based on n_modes and applies
    pairwise diagonal priors (rho1-rho2, rho2-rho3, rho1-rho3).

    Parameters:
    -----------
    sel_val : list
        Not used for diagonal prior (kept for consistency with other prior functions)
    val_arr : array
        Array of parameter values [n_samples, n_params]
    val_weight : array
        Current weights for each sample
    val_labels : list
        Labels for parameters
    n_modes : int
        Number of modes (1, 2, or 3)
    type_p : str
        'MultipleRI' or other types to determine column indices
    sigma_diagonal : float
        Width of the Gaussian around the diagonal (default: 0.1)
    prior_min : float
        Minimum value for rho parameters
    prior_max : float
        Maximum value for rho parameters
    prior_sig : int
        Decimal places for grid resolution

    Returns:
    --------
    new_weights : array
        Updated weights after applying the diagonal prior
    name_pdf : str
        Name/description of the prior
    """
    # Determine rho indices based on n_modes and type_p
    rho_indices = []
    if type_p == 'MultipleRI':
        for i in range(n_modes):
            rho_indices.append((3 * (i + 1)) - 3)  # 0, 3, 6 for modes 1, 2, 3
    else:
        for i in range(n_modes):
            rho_indices.append(9 + (3 * i))  # 9, 12, 15 for modes 1, 2, 3

    # If only one mode, no diagonal prior needed
    if len(rho_indices) < 2:
        return val_weight, 'NoDiagPrior(single_mode)'

    prior_values = [sigma_diagonal, prior_min, prior_max, prior_sig]

    # Get the 2D prior distribution
    val_dis, val_prob_2d = diagonalGaussian_Prior_distribution(prior_values)

    # Create 2D interpolator
    val_prob_interp = scipy.interpolate.RegularGridInterpolator(
        (val_dis, val_dis),
        val_prob_2d,
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    new_weights = val_weight.copy()

    # Apply diagonal prior to all pairs of rho values
    pairs_applied = []
    for i in range(len(rho_indices)):
        for j in range(i + 1, len(rho_indices)):
            idx1, idx2 = rho_indices[i], rho_indices[j]

            # Extract the two parameter values
            param1_vals = val_arr[:, idx1]
            param2_vals = val_arr[:, idx2]

            # Evaluate prior at each sample point
            points = np.column_stack([param1_vals, param2_vals])
            diagonal_prior_vals = val_prob_interp(points)

            # Apply the diagonal prior to weights
            new_weights = new_weights * diagonal_prior_vals
            pairs_applied.append(f"{val_labels[idx1]}-{val_labels[idx2]}")

    # Create name
    pair_names = ','.join(pairs_applied)
    name_pdf = f'DiagGauss({pair_names},σ={sigma_diagonal:.3f})'

    return new_weights, name_pdf
def singleValue_Prior(sel_val, val_arr, val_weight, val_labels, t_prior=type_Prior, m_val=prior_m, unc_val=prior_unc, low_val=prior_min, high_val=prior_max, sig_val=prior_sig):
    prior_values = [m_val, unc_val, low_val, high_val, sig_val]
    if t_prior == 'Norm':
        name_pdf =f'N({m_val}, {unc_val:.2f})'
        def Prior_distribution(values):
            return trucatedNormal_Prior_distribution(values)
    elif t_prior == 'Uni':
        name_pdf = f'U({low_val:.2f}, {high_val:.2f}, {prior_ratio})'
        def Prior_distribution(values):
            return stepUniform_Prior_distribution(values)
    elif t_prior == 'LogN':
        name_pdf = f'LN({m_val}, {unc_val:.2f})'
        def Prior_distribution(values):
            return truncatedLogNormal_Prior_distribution(values)
    val_dis, val_prob_dis = Prior_distribution(prior_values)
    val_prob_interp = scipy.interpolate.interp1d(val_dis, val_prob_dis, kind='linear', fill_value=0, bounds_error=False)
    new_weights = val_weight #[]
    for i in range(len(val_labels)-1):
        # print(i, val_weight)
        if i in sel_val:
            val_pdf = [y_w * val_prob_interp(y_val) for y_val, y_w in zip(val_arr[:,i], val_weight)]
            # print(i,val_labels[i])
            # new_weights.append(np.array(val_pdf))
            new_weights=new_weights*val_pdf
        else:
            # new_weights.append(val_weight)
            continue

    # new_weights=np.array(new_weights)
    return new_weights, name_pdf
def sigma_Prior(sel_val, val_arr, val_weight,prior_unc_sig=prior_unc_sig, prior_m_sig=prior_m_sig,t_prior=type_Prior, sig_val=prior_sig, n_ratio=prior_ratio_sig,prior_max_sig=prior_max_sig):
    if t_prior == 'Norm':
        name_pdf =f'N(0, .5)'
        def Prior_distribution(values):
            return trucatedNormal_Prior_distribution(values)
    elif t_prior == 'Uni':
        name_pdf = f'U({prior_m_sig- prior_unc_sig},{prior_m_sig+ prior_unc_sig}, {n_ratio})'
        def Prior_distribution(values):
            return stepUniform_Prior_distribution(values)
    elif t_prior == 'LogN':
        name_pdf = f'LN(0, {sig_val})'
        def Prior_distribution(values):
            return truncatedLogNormal_Prior_distribution(values)
    new_weights = val_weight #[]
    for n_m, i in enumerate(sel_val):
        # print(n_m,i)
        # print(min(val_arr[:,i]),max(val_arr[:,i]))
        # if t_prior =='LogN':
        #     if n_m ==1:
        #         # prior_values = [0.0005, prior_unc_sig*10, 0.0005, 0.1, sig_val]
        #         prior_values = [0.0005, prior_unc_sig, 0.0005, 0.1, sig_val]
        #     else:
        #         # print(n_m)
        #         # prior_values = [1.0005, prior_unc_sig/10, 1.0005, 1.10,sig_val]
        #         prior_values = [1.0005, 1+prior_unc_sig/10, 1.0005, 1.10, sig_val]

        if t_prior =='Norm':
            if n_m ==1:
                # prior_values = [0.0005, prior_unc_sig*10, 0.0005, 0.1, sig_val]
                prior_values = [0.0005+prior_m_sig, prior_unc_sig, 0.0005, prior_max_sig, sig_val]
            else:
                # print(n_m)
                # prior_values = [1.0005, prior_unc_sig/10, 1.0005, 1.10,sig_val]
                prior_values = [1.0005+prior_m_sig, prior_unc_sig, 1.0005, 1+prior_max_sig, sig_val]
        else:
            if n_m == 1:
                prior_values = [prior_m_sig, prior_unc_sig, 0.0000, prior_max_sig, sig_val]
            else:
                # print(n_m)
                prior_values = [1+prior_m_sig, prior_unc_sig, 1.0000, 1+prior_max_sig, sig_val]
        val_dis, val_prob_dis = Prior_distribution(prior_values)
        val_prob_interp = scipy.interpolate.interp1d(val_dis, val_prob_dis, kind='linear', fill_value=0,bounds_error=False)
        val_pdf = [y_w * val_prob_interp(y_val) for y_val, y_w in zip(val_arr[:, i], val_weight)]

        new_weights = new_weights * val_pdf
    # new_weights=np.array(new_weights)
    return new_weights, name_pdf
#
# Filter Function
def filter_funtion(sel_val, val_arr, val_weight,val_labels, low_val, high_val):
    raw_values = val_arr[sel_val]
    # print(size_values.shape)
    mask = (raw_values > high_val) | (raw_values < low_val)
    new_vales =val_arr
    for f in range(len(val_labels)-1):
        temp_arr=val_arr[f]
        filtered_values = temp_arr[mask]
        val_arr[f]= filtered_values
    new_weight=val_weight[mask]
    return new_vales, new_weight

def num_Prior(sel_val, val_arr, val_weight, t_prior=type_Prior, m_val=prior_m_n, unc_val=prior_unc_n, low_val=prior_min_n, high_val=prior_max_n,sig_val=prior_sig_n):
    prior_values = [m_val, unc_val, low_val, high_val, sig_val]

    if t_prior == 'Norm':
        name_pdf = f'N({m_val}, {unc_val:.2f})'

        def Prior_distribution(values):
            return trucatedNormal_Prior_distribution(values)
    elif t_prior == 'Uni':
        name_pdf = f'U({low_val:.2f}, {high_val:.2f}, {prior_ratio})'

        def Prior_distribution(values):
            return stepUniform_Prior_distribution(values)
    elif t_prior == 'LogN':
        name_pdf = f'LN({m_val}, {unc_val:.2f})'

        def Prior_distribution(values):
            return truncatedLogNormal_Prior_distribution(values)

    new_weights = val_weight
    n_data = val_arr[:, sel_val]
    n_total= n_data.sum(axis=1)
    print(n_total)
    val_dis, val_prob_dis = Prior_distribution(prior_values)
    val_prob_interp = scipy.interpolate.interp1d(val_dis, val_prob_dis, kind='linear', fill_value=0,bounds_error=False)
    val_pdf = [y_w * val_prob_interp(y_val) for y_val, y_w in zip(n_total, val_weight)]
    new_weights = new_weights * val_pdf
    # new_weights=np.array(new_weights)
    return new_weights, val_pdf

# Plotting Functions (in use)
def my_histogram(j, file_j, n_modes, y_min, y_max, res_path, deltas=hist_deltas, my_bins=hist_bins, type_step=type_step, type_p=type_p, a_strat=analysis_strat, show_hist=show_hist, save_hist=save_hist, save_mode=save_mode, show_peaks=show_peaks, num_peaks=n_peaks,mirror_data=mirror_data, use_mode=use_mode, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, epsilon=epsilon):
    # print(file_j)
    data, weights, header = read_PostDist_CSV(file_j, n_modes)
    # print(data[:,0])
    if filter_values:
        for f_what in filter_what:
            if f_what=='nums':
                if type_p == 'MultipleRI':
                    continue
                else:
                    n_T=data[:,0]+data[:,3]+data[:,6]
                    mask= (n_T>=filt_n_min) & (n_T<=filt_n_max)
            if f_what=='index':
                if type_p == 'MultipleRI':
                    if n_modes==1:
                        mask= (filt_max >= data[:,2]) & (data[:,2] >= filt_min)
                    elif n_modes==2:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                    else:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                else:
                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            if f_what == 'sigma':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (1+filt_s >= data[:, 1])
                    elif n_modes == 2:
                        mask = (1+filt_s >= data[:, 1]) &(filt_s >= data[:, 4])
                    else:
                        mask = (1+filt_s >= data[:, 1]) &(filt_s >= data[:, 4]) &(1+filt_s >= data[:, 7])
                else:
                    mask = (1+filt_s >= data[:, 10]) &(filt_s >= data[:, 13]) &(1+filt_s >= data[:, 16])

            data = data[mask]
            weights = weights[mask]
            # print(len(data))
    if w_prior:
        for p, p_what in enumerate(prior_what):
            sel_val = []
            if p_what == 'index':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i+1))-1)
                else:
                    for i in range(n_modes):
                        sel_val.append(11 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, header,t_prior=type_Prior[p])
            if p_what == 'sigma':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i+1))-2)
                else:
                    for i in range(n_modes):
                        sel_val.append(10 + (3 * i))
                # print(sel_val)
                weights, name_prior = sigma_Prior(sel_val, data, weights,t_prior=type_Prior[p])
            if p_what == 'rho':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i+1))-3)
                else:
                    for i in range(n_modes):
                        sel_val.append(9+ (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, header,t_prior=type_Prior[p], m_val=prior_m_rho, unc_val= prior_unc_rho, low_val=-1, high_val=1)
            # print(f'Applying Prior to: {sel_val}')
            # weights, name_prior = singleValue_Prior(sel_val, data, weights, header)
            if p_what=='num_total':
                for i in range(n_modes):
                    sel_val.append((3 * i))
                weights, name_prior = num_Prior(sel_val, data, weights,t_prior=type_Prior[p])
            if p_what == 'rho_diagonal':
                # Apply diagonal prior to favor similar rho values
                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights, val_labels=header,n_modes=n_modes, type_p=type_p, sigma_diagonal=0.1, prior_min=-1.0,prior_max=1.0, prior_sig=3)
    if use_mode==True:
        all_sorted = find_weighted_mode(data, weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        weighted_mode = np.array(weighted_mode)
        frequency = np.array(frequency)
        if save_mode== True:
            if w_prior:
                mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
                if filter_values:
                    mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
            elif filter_values:
                mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
            else:
                mode_paths = os.path.join(res_path, "My_Distributions")
            os.makedirs(mode_paths, exist_ok=True)
            if a_strat == 'Layer':
                filename = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
            elif a_strat == 'independent':
                filename = os.path.join(mode_paths, f"PosDist_{j}_jointW.csv")
            mode_data = np.column_stack((weighted_mode, frequency))
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(mode_data)
            print(f"Data saved to {filename}")
        myMode = weighted_mode[0]
    else:
        weighted_mode= np.array(data)
        frequency = np.array(weights)
    if type_p == 'MultipleRI':
        num_rows = n_modes
        num_cols = 3
    else:
        num_rows = 2
        num_cols = 9
    # mask = frequency>=2
    # weighted_mode= weighted_mode[mask]
    # frequency = frequency[mask]
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figSize_x*num_cols, figSize_y*num_rows))
    axs = axs.flatten()
    new_dis = []
    # print('Highest Weights: ', frequency[:3])
    # print('Mode: ', weighted_mode[0])
    # print(frequency.shape, weighted_mode.shape)
    for i in range(len(weighted_mode[0])):
        array_val = weighted_mode[:, i]
        low_val = np.min(array_val)
        # print(header[i],low_val)
        high_val = np.max(array_val)
        if mirror_data ==True:
            # print(f'For {n_labels[i]}: [{low_val}, {high_val}]')
            array_val = np.concatenate((array_val, 2 * low_val - array_val))
            array_w = np.concatenate((frequency, frequency))
        else:
            array_w=frequency

        if type_step=='bins':
            if mirror_data==True:
                val_bins=my_bins*2
            else:
                val_bins= my_bins
            # print(val_bins)
            delta_val =(np.max(array_val) - np.min(array_val))/val_bins
            s_hist= int(-1*np.log10(delta_val))

        elif type_step =='deltas':
            delta_val = deltas
            val_bins = num_bins(array_val, delta_val)
        elif type_step == 'changing_d':
            if type_p == 'MultipleRI':
                if i in [0,3,6]: # Correlation
                    delta_val= 0.01
                    s_hist = 2
                else: # R.I.
                    delta_val = deltas
                    s_hist = sig_hist
            else:
                if i in [0,3,6]: # number density
                    # print(low_val,high_val)
                    delta_val= 5
                    s_hist = 0
                elif i <9: # particle size
                    delta_val= 0.01
                    s_hist = 2
                elif i in [9, 12, 15]: # correlation
                    delta_val= 0.01
                    s_hist = 2
                else: # R.I.
                    delta_val = deltas
                    s_hist = sig_hist
            val_bins = num_bins(array_val, delta_val)
        if use_log_bins:
            if type_p == 'MultipleRI':
                pass
            elif i in [0,3,6]:
                # Add small epsilon to avoid log(0)

                # ri_bins = np.logspace(np.log10(max(array_ri.min(), low_ri) + epsilon),
                #                       np.log10(array_ri.max() + epsilon),
                #                       num_bins(array_ri, delta_ri))
                val_bins = np.logspace(np.log10(low_val + epsilon),np.log10(high_val + epsilon),
                                       #500)
                                     num_bins(array_val, delta_val))
        array_val = np.array(array_val)
        x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(array_val, weights=array_w)()  # Best at smoothing!
        # x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=array_w)()  # With Weights
        # Get ISJ bandwidth as baseline
        # kde_base = FFTKDE(kernel='gaussian', bw='ISJ')
        # kde_base.fit(array_val, weights=array_w)
        # bw_optimal = kde_base.bw
        #
        # # Reduce bandwidth by 30% to preserve multimodality
        # bw_multimodal = bw_optimal * 0.5
        # x_rs, y_rs = FFTKDE(kernel='gaussian', bw=bw_multimodal).fit(array_val, weights=array_w)()  # With Weights

        if mirror_data:
            mask = (x_rs < low_val) | (high_val < x_rs)
            y_rs[mask] = 0  # Set the KDE to zero outside of the domain
        if show_peaks==True:
            mask_P = (x_rs >= low_val) & (high_val >= x_rs)
            y_rsP = y_rs[mask_P]
            x_rsP = x_rs[mask_P]
            peaks_idr, peak_r = find_peaks(y_rsP, height=0)
            peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
            if len(peaks_idr) > num_peaks:
                peaks_idr = peaks_idr[:num_peaks]
            # colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
            colors = plt.cm.spring(np.linspace(0, 1, len(peaks_idr)))
            for d, p in enumerate(peaks_idr):
                peak_r = x_rsP[p]
                axs[i].axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')
        max_xi = np.argmax(y_rs)
        max_x = x_rs[max_xi]
        new_dis.append(max_x)
        axs[i].hist(array_val, weights=array_w, bins=val_bins, density=True)
        # axs[i].plot(x_rs, y_rs, color='black', label=f'Fitted Curve (bw={delta_val:.3f})')
        axs[i].plot(x_rs, y_rs, color='black', label=f'Fitted Curve (ISJ)')
        # x_rs1, y_rs1 = FFTKDE(kernel='gaussian', bw='ISJ').fit(array_val, weights=array_w)()  # With Weights
        # axs[i].plot(x_rs1, y_rs1, color='green', label=f'Fitted Curve (ISJ)')
        # x_rs2, y_rs2 = FFTKDE(kernel='gaussian', bw='silverman').fit(array_val, weights=array_w)()  # With Weights
        # axs[i].plot(x_rs2, y_rs2, color='cyan', label='Fitted Curve silverman')
        # x_rs3, y_rs3 = FFTKDE(kernel='gaussian', bw='scott').fit(array_val, weights=array_w)()  # With Weights
        # axs[i].plot(x_rs3, y_rs3, color='red', label='Fitted Scott')
        axs[i].axvline(x=max_x, color='black', linestyle='--', linewidth=2, label='FFTKDE: {0:.{1}f}'.format(max_x, sig_h))
        if use_mode==True:
            max_his = myMode[i]
            axs[i].axvline(x=max_his, color='green', linestyle='dotted', linewidth=2, label='Mode: {0:.{1}f}'.format(max_his, sig_hist))
        axs[i].set_title(header[i],fontsize=fo)
        axs[i].set_xlabel(header[i],fontsize=fo)
        axs[i].set_xlim((low_val,high_val))
        if scale_type=='Log':
            if type_p == 'MultipleRI':
                pass
            elif i in [0,3,6]:
                l_base=10
                axs[i].set_xlim((0.9,high_val))
                # print(low_val,high_val)
                # print(np.log10(low_val),np.log10(high_val))
                axs[i].set_xscale('log', base=l_base)
                axs[i].set_yscale('log', base=l_base)
                axs[i].set_ylabel(f'Log{l_base} (PDF)')
        else:
            axs[i].set_ylabel('PDF',fontsize=fo)
        # axs[i].set_ylabel('PDF', fontsize=fo)
        axs[i].minorticks_on()
        # axs[i].xaxis.set_major_locator(MultipleLocator(major_grid_x))
        # axs[i].yaxis.set_major_locator(MultipleLocator(major_grid_y))
        axs[i].grid(True, which='major', linestyle='-', alpha=0.5, linewidth=1.0, color='gray')

        # Minor grid
        # axs[i].xaxis.set_minor_locator(MultipleLocator(minor_grid_x))
        # axs[i].yaxis.set_minor_locator(MultipleLocator(minor_grid_y))
        axs[i].grid(True, which='minor', linestyle=':', alpha=0.3, linewidth=0.5, color='gray')

        axs[i].set_axisbelow(True)
        axs[i].tick_params(axis='both', labelsize=fo)
        axs[i].legend(fontsize=fo, loc='best')
    if save_mean:
        if w_prior:
            mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            mode_paths = os.path.join(res_path, "My_Distributions")
        os.makedirs(mode_paths, exist_ok=True)
        if a_strat == 'Layer':
            filename = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
        elif a_strat == 'independent':
            filename = os.path.join(mode_paths, f"PosDist_{j}_Mean.csv")
        new_dis.append(0)
        mean_data = np.array(new_dis)
        # print(mean_data)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(mean_data)
        print(f"Data saved to {filename}")
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    if a_strat== 'Layer':
        if w_prior:
            fig.suptitle('Histograms for Layer {} - {}km with Prior'.format(max_Alt, min_Alt), fontsize=fo)
        else:
            fig.suptitle('Histograms for Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_hist ==True:
        if w_prior:
            hist_paths = os.path.join(res_path, "Histograms_wPrior")
            if filter_values:
                hist_paths = os.path.join(res_path, "Histograms_wBoth")
        elif filter_values:
            hist_paths = os.path.join(res_path, f"Histograms_wFilter")
        else:
            hist_paths = os.path.join(res_path, "Histograms")
        os.makedirs(hist_paths, exist_ok=True)
        filename_parts = [r'Hist']
        if a_strat == 'Layer':
            filename_parts.append(f'_Layer{j}')
        elif a_strat == 'independent':
            filename_parts.append(f'_{j}')
        if scale_type == 'Log':
            filename_parts.append('Log')
        if use_log_bins:
            filename_parts.append('LBins')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(hist_paths, filename))
        # if a_strat == 'Layer':
        #    fig.savefig(hist_paths + '/Hist_Layer{}.png'.format(j))
        # elif a_strat == 'independent':
        #     fig.savefig(hist_paths + '/Hist_{}.png'.format(j))
    if show_hist==True:
        plt.show()
    plt.close('all')
def my_histogramMode(j, file_j, n_modes, y_min, y_max, res_path, hist_bins=hist_bins, type_p=type_p, a_strat=analysis_strat, show_hist=show_histM, save_hist=save_histM, save_mode=save_mode, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo):
    data, weights, header = read_PostDist_CSV(file_j, n_modes)

    for n in range(n_modes):
        if type_p == 'MultipleRI':
            if n == 0:
                columns = [0, 1, 2]
            elif n == 1:
                columns = [3, 4, 5]
            elif n == 2:
                columns = [6, 7, 8]
        else:
            if n==0:
                columns = [0, 1, 2, 9, 10, 11]
            elif n==1:
                columns = [3, 4, 5, 12, 13, 14]
            elif n==2:
                columns =[6, 7, 8, 15, 16, 17]
        n_data = data[:, columns]
        n_weights = weights
        n_labels = header[columns]
        all_sorted = find_weighted_mode(n_data, n_weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        weighted_mode = np.array(weighted_mode)
        frequency = np.array(frequency)
        if save_mode== True:
            n_header = np.hstack((n_labels, header[-1]))
            if w_prior:
                mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
                if filter_values:
                    mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
            elif filter_values:
                mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
            else:
                mode_paths = os.path.join(res_path, "My_Distributions")
            os.makedirs(mode_paths, exist_ok=True)
            if a_strat == 'Layer':
                filename = os.path.join(mode_paths, f"PosDistLayer{j}_Mode{n}_jointW.csv")
            elif a_strat == 'independent':
                filename = os.path.join(mode_paths, f"PosDist_{j}_Mode{n}_jointW.csv")
            mode_data = np.column_stack((weighted_mode, frequency))
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(n_header)
                writer.writerows(mode_data)
            print(f"Data saved to {filename}")
        if type_p == 'MultipleRI':
            num_rows = 1
            num_cols = 3
        else:
            num_rows = 2
            num_cols = 3
        myMode = weighted_mode[0]
        # mask = frequency>=2
        # weighted_mode= weighted_mode[mask]
        # frequency = frequency[mask]
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(figSize_x*num_cols, figSize_y*num_rows))
        axs = axs.flatten()
        new_dis = []
        for i in range(len(weighted_mode[0])):
            # if type_p == 'MultipleRI':
            #     if i in [0,3,6]:
            #         delta_val= 0.1
            #     else:
            #         delta_val = deltas
            # else:
            #     if i in [0,3,6]:
            #         delta_val= 1
            #     elif i <10 or i in [12,15]:
            #         delta_val= 0.1
            #     else:
            #         delta_val =deltas

            array_val = weighted_mode[:, i]
            low_val = np.min(array_val)
            high_val = np.max(array_val)
            # print(f'For {n_labels[i]}: [{low_val}, {high_val}]')
            array_val = np.concatenate((array_val, 2 * low_val - array_val))
            array_w = np.concatenate((frequency, frequency))
            val_bins= hist_bins
            delta_val =(high_val - low_val)/val_bins
            # val_bins = num_bins(array_val, delta_val)
            array_val = np.array(array_val)
            # x_rs, y_rs = FFTKDE(kernel='gaussian', bw='silverman').fit(array_val, weights=frequency)()  # Best at smoothing!
            x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=array_w)()  # With Weights
            mask= (high_val < x_rs) | (x_rs < low_val)
            y_rs[mask] = 0
            # x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=frequency)()  # With Weights
            max_xi = np.argmax(y_rs)
            max_x = x_rs[max_xi]
            max_his = myMode[i]
            new_dis.append(max_x)
            axs[i].hist(array_val, weights=array_w, bins=val_bins, density=True)
            axs[i].plot(x_rs, y_rs, label='Fitted Curve')
            axs[i].axvline(x=max_x, color='orange', linestyle='--', linewidth=2, label='FFTKDE: {0:.{1}f}'.format(max_x, sig_hist))
            axs[i].axvline(x=max_his, color='black', linestyle='--', linewidth=2, label='Mode: {0:.{1}f}'.format(max_his, sig_hist))
            axs[i].set_title(n_labels[i],fontsize=fo)
            axs[i].set_xlabel(n_labels[i],fontsize=fo)
            axs[i].set_xlim((low_val, high_val))
            if scale_type == 'Log':
                if type_p == 'MultipleRI':
                    pass
                elif i in [0]:
                    l_base = 10
                    axs[i].set_xscale('log', base=l_base)
                    axs[i].set_yscale('log', base=l_base)
                    axs[i].set_ylabel(f'Log{l_base} (PDF)')
            else:
                axs[i].set_ylabel('PDF', fontsize=fo)
            # axs[i].set_ylabel('PDF',fontsize=fo)
            axs[i].tick_params(axis='both', labelsize=fo)
            axs[i].legend(fontsize=fo)
        if save_mean:
            n_header = np.hstack((n_labels, header[-1]))
            if w_prior:
                mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
                if filter_values:
                    mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
            elif filter_values:
                mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
            else:
                mode_paths = os.path.join(res_path, "My_Distributions")
            os.makedirs(mode_paths, exist_ok=True)
            if a_strat == 'Layer':
                filename = os.path.join(mode_paths, f"PosDistLayer{j}_Mode{n}_Mean.csv")
            elif a_strat == 'independent':
                filename = os.path.join(mode_paths, f"PosDist_{j}_Mode{n}_Mean.csv")
            new_dis.append(0)
            mean_data = np.array(new_dis)
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(n_header)
                writer.writerow(mean_data)
            print(f"Data saved to {filename}")
        max_Alt = format(y_max, ".2f")
        min_Alt = format(y_min, ".2f")
        if a_strat== 'Layer':
            fig.suptitle('Histograms for Layer {} - {}km Mode {}'.format(max_Alt, min_Alt,n), fontsize=fo)
        plt.tight_layout()
        if save_hist ==True:
            if w_prior:
                hist_paths = os.path.join(res_path, "Histograms_wPrior")
                if filter_values:
                    hist_paths = os.path.join(res_path, "Histograms_wBoth")
            elif filter_values:
                hist_paths = os.path.join(res_path, f"Histograms_wFilter")
            else:
                hist_paths = os.path.join(res_path, "Histograms")
            os.makedirs(hist_paths, exist_ok=True)
            filename_parts = [r'Hist']
            if a_strat == 'Layer':
                filename_parts.append(f'_Layer{j}_Mode{n}')
            elif a_strat == 'independent':
                filename_parts.append(f'_{j}_Mode{n}')
            if scale_type == 'Log':
                filename_parts.append('Log')
            filename = '_'.join(filename_parts) + '.png'
            fig.savefig(os.path.join(hist_paths, filename))
            # if a_strat == 'Layer':
            #     fig.savefig(hist_paths + '/Hist_Layer{}_Mode{}.png'.format(j,n))
            # elif a_strat == 'independent':
            #     fig.savefig(hist_paths + '/Hist_{}_Mode{}.png'.format(j,n))
        if show_hist==True:
            plt.show()
        plt.close('all')
def my_cornerMaxVal(n_modes,data, weights, labels, type_p=type_p):
    max_vals = np.full(len(labels) , np.nan, dtype=float)
    for n in range(n_modes):
        if n_modes==4:
            if n == 0:
                columns = [0, 3, 6]
                # columns = [9, 12, 15]
            else:
                break
        elif n==0:
            columns = [0, 1, 2, 9, 10, 11]
        elif n==1:
            columns = [3, 4, 5, 12, 13, 14]
        elif n==2:
            columns =[6, 7, 8, 15, 16, 17]
        if type_p == 'MultipleRI':
            columns = columns[:3]
        # n_data = data[:, columns]
        # n_weights = weights
        # n_labels = labels[columns]
        a_bins=ri_bins=1000
        bws=0.01
        for k, c in enumerate(columns):
            # array_a = data[:,c]
            # array_w = weights
            low_a = np.min(data[:, c])
            high_a = np.max(data[:, c])
            array_a = np.concatenate((data[:,c], 2 * low_a - data[:,c]))
            array_w = np.concatenate((weights,weights))
            for e in columns[k+1:]:
                # array_ri= data[:,e]
                low_ri = np.min(data[:, e])
                high_ri = np.max(data[:, e])
                array_ri = np.concatenate((data[:, e], 2 * low_ri - data[:, e]))
                [values] = np.dstack([array_a, array_ri])
                grid_points_x, grid_points_y = a_bins, ri_bins
                kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))
                X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                f = points.reshape(grid_points_x, grid_points_y).T
                xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                for i in range(grid_points_y):
                    for j in range(grid_points_x):
                        if high_a < xs[i][j] or xs[i][j] < low_a or high_ri  < ys[i][j] or ys[i][j] < low_ri :
                            f[i][j] = 0
                max_value = np.max(f)
                max_indices = np.unravel_index(np.argmax(f), f.shape)
                max_x = X[max_indices[1]] #+ (bws / 2)
                max_y = Y[max_indices[0]] #+ (bws / 2)
                if use_mode==False:
                    print(f'For [{labels[c]}, {labels[e]}]={max_x:.2f},{max_y:.2f}')
                if np.isnan(max_vals[c]):
                    max_vals[c] = max_x
                if np.isnan(max_vals[e]):
                    max_vals[e] = max_y
            if use_mode:
                print(f'{labels[c]}: {max_vals[c]:.2f}')

    return max_vals
def my_cornerplot_new(j, y_min, y_max, paths, data, weights, labels, show_mycorner=show_cornerN, save_mycorner=save_cornerN,a_bins=1000, ri_bins=1000, bws=0.01,figsize=None, use_mode=False):
    n_cols = len(labels)
    max_vals = np.full(len(labels), np.nan, dtype=float)

    # Create figure with subplots
    if figsize is None:
        figsize = (3 * n_cols, 3 * n_cols)

    fig, axes = plt.subplots(n_cols, n_cols, figsize=figsize)

    # Handle case where there's only one column (single subplot)
    if n_cols == 1:
        axes = np.array([[axes]])
    elif n_cols == 2:
        # Ensure axes is 2D
        if axes.ndim == 1:
            axes = axes.reshape(n_cols, n_cols)

    # Iterate over all pairs
    for i in range(n_cols):
        for j in range(n_cols):
            ax = axes[i, j]

            # Lower triangle: 2D KDE plots (column j vs column i)
            if i > j:
                # Extract and mirror data for column j (x-axis)
                low_x = np.min(data[:, j])
                high_x = np.max(data[:, j])
                array_x = np.concatenate((data[:, j], 2 * low_x - data[:, j]))

                # Extract and mirror data for column i (y-axis)
                low_y = np.min(data[:, i])
                high_y = np.max(data[:, i])
                array_y = np.concatenate((data[:, i], 2 * low_y - data[:, i]))

                # Mirror weights
                array_w = np.concatenate((weights, weights))

                # Prepare data for KDE
                values = np.dstack([array_x, array_y])[0]

                # Compute KDE
                kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                grid, points = kde.fit(values, weights=array_w).evaluate((a_bins, ri_bins))

                X = np.unique(grid[:, 0])
                Y = np.unique(grid[:, 1])
                f = points.reshape(a_bins, ri_bins).T
                xs = grid[:, 0].reshape(a_bins, ri_bins).T
                ys = grid[:, 1].reshape(a_bins, ri_bins).T

                # Mask values outside original data range
                for ii in range(ri_bins):
                    for jj in range(a_bins):
                        if xs[ii, jj] < low_x or xs[ii, jj] > high_x or \
                                ys[ii, jj] < low_y or ys[ii, jj] > high_y:
                            f[ii, jj] = 0

                # Find maximum
                max_value = np.max(f)
                max_indices = np.unravel_index(np.argmax(f), f.shape)
                max_x = X[max_indices[1]]
                max_y = Y[max_indices[0]]

                # Store max values
                if np.isnan(max_vals[j]):
                    max_vals[j] = max_x
                if np.isnan(max_vals[i]):
                    max_vals[i] = max_y

                # Plot 2D KDE
                contour = ax.contourf(X, Y, f, levels=20, cmap='reds')
                ax.plot(max_x, max_y, 'r*', markersize=10, label='Mode')

                # Labels
                if j == 0:
                    ax.set_ylabel(labels[i])
                if i == n_cols - 1:
                    ax.set_xlabel(labels[j])

                if not use_mode:
                    print(f'For [{labels[j]}, {labels[i]}] = {max_x:.2f}, {max_y:.2f}')

            # Diagonal: 1D histograms or labels
            elif i == j:
                ax.hist(data[:, i], bins=50, weights=weights,
                        alpha=0.7, edgecolor='black')
                ax.set_ylabel('Density')
                if i == n_cols - 1:
                    ax.set_xlabel(labels[i])
                ax.text(0.5, 0.5, labels[i], transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, fontweight='bold')

            # Upper triangle: turn off
            else:
                ax.axis('off')

        # Format altitude values
        max_Alt = format(y_max, ".2f")
        min_Alt = format(y_min, ".2f")

        # Add title
        if w_prior:
            fig.suptitle('My Corner Plot for Layer {} - {}km with Prior'.format(max_Alt, min_Alt),
                         fontsize=fo + 2)
        else:
            fig.suptitle('My Corner Plot for Layer {} - {}km'.format(max_Alt, min_Alt),
                         fontsize=fo + 2)

        plt.tight_layout()

        # Save figure if requested
        if save_mycorner:
            res_path = os.path.join(paths, "Results")

            # Determine corner path based on prior and filter settings
            if w_prior:
                corner_path = os.path.join(res_path, "MyCornerPlots_wPrior")
                if filter_values:
                    corner_path = os.path.join(res_path, "MyCornerPlots_wBoth")
            elif filter_values:
                corner_path = os.path.join(res_path, "MyCornerPlots_wFilter")
            else:
                corner_path = os.path.join(res_path, "MyCornerPlots")

            os.makedirs(corner_path, exist_ok=True)

            # Build filename
            filename_parts = [f'MyCornerPlot_Layer{j}']
            if use_mode:
                filename_parts.append('jointW')
            filename = '_'.join(filename_parts) + '.png'

            fig.savefig(os.path.join(corner_path, filename), dpi=300, bbox_inches='tight')
            print(f"Plot saved to {os.path.join(corner_path, filename)}")

        # Show plot if requested
        if show_mycorner:
            plt.show()
        else:
            plt.close(fig)
def my_corner(k, n_modes,y_min, y_max, paths, use_mode=use_mode, type_p=type_p, show_cornerBP=show_cornerBP, save_cornerBP=save_cornerBPlot, save_cornerDMean=save_cornerDisMean, show_cornerMP=show_cornerMP, save_cornerMP=save_cornerMPlot, save_cornerDMode=save_cornerDisMode, deltas=corner_deltas, type_step=type_step, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, n_con=num_cont, bws=bws, scale_t=scale_type, l_base=log_base, my_colorMap=my_colorMap, show_prob=probs, corner_bins=corner_bins,log_bin_indices = [0, 3, 6], epsilon=epsilon,smooth_corner=smooth_corner,smooth_val=smooth_val):
    res_path = os.path.join(paths, "Results")
    dis_path = os.path.join(paths, "PosteriorDistributions")
    pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(k))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    if filter_values:
        for f_what in filter_what:
            if f_what == 'nums':
                if type_p == 'MultipleRI':
                    continue
                else:
                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                    mask = (n_T >= filt_n_min) & (n_T<=filt_n_max)
            if f_what == 'index':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                    elif n_modes == 2:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                    else:
                        mask = (filt_max >= data[:, 2]) & (data[:, ] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                else:
                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            if f_what == 'sigma':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (1 + filt_s >= data[:, 1])
                    elif n_modes == 2:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                    else:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & (1 + filt_s >= data[:, 7])
                else:
                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & (1 + filt_s >= data[:, 16])
            data = data[mask]
            weights = weights[mask]
    if w_prior:
        for p, p_what in enumerate(prior_what):
            sel_val = []
            if p_what == 'index':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 1)
                else:
                    for i in range(n_modes):
                        sel_val.append(11 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p])
            if p_what == 'sigma':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 2)
                else:
                    for i in range(n_modes):
                        sel_val.append(10 + (3 * i))
                # print(sel_val)
                weights, name_prior = sigma_Prior(sel_val, data, weights,t_prior=type_Prior[p])
            if p_what == 'rho':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 3)
                else:
                    for i in range(n_modes):
                        sel_val.append(9 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p], m_val=0,unc_val=prior_unc_rho, low_val=-1, high_val=1)
            if p_what == 'rho_diagonal':
                # Apply diagonal prior to favor similar rho values
                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights,val_labels=labels,n_modes=n_modes,type_p=type_p,sigma_diagonal=0.1,prior_min=-1.0, prior_max=1.0,prior_sig=3)
            if p_what=='num_total':
                for i in range(n_modes):
                    sel_val.append((3 * i))
                weights, name_prior = num_Prior(sel_val, data, weights,t_prior=type_Prior[p])
    if use_mode==True:
        all_sorted = find_weighted_mode(data, weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        data = np.array(weighted_mode)
        weights = np.array(frequency)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    if smooth_corner == False:
        smooth_val = None
    if show_cornerBP or save_cornerBP:
        # Determine number of variables
        n_vars = data.shape[1]

        # Create corner plot
        fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

        # Plot for each pair of variables
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axs[i, j]

                # Lower triangle: 2D joint distributions
                if i > j:
                    # print(i, j)
                    array_x = data[:, j]
                    array_y = data[:, i]
                    array_w = weights.copy()

                    # Remove NaN values
                    # valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                    # array_x = array_x[valid_mask]
                    # array_y = array_y[valid_mask]
                    # array_w = array_w[valid_mask]

                    if len(array_x) == 0:
                        ax.axis('off')
                        continue

                    low_x, high_x = np.min(array_x), np.max(array_x)
                    low_y, high_y = np.min(array_y), np.max(array_y)
                    if use_log_bins:
                        if type_p == 'MultipleRI':
                            use_log_x = False
                            use_log_y = False
                        else:
                            use_log_x = use_log_bins and (j in log_bin_indices)
                            use_log_y = use_log_bins and (i in log_bin_indices)
                    else:
                        use_log_x = False
                        use_log_y = False
                    if use_log_x or use_log_y:

                        # Handle x-axis log binning
                        if use_log_x:
                            # Ensure all values are positive before log transform
                            # if low_x < 1:
                            #     low_x = epsilon
                            # array_x = np.maximum(array_x, epsilon)  # Clip to minimum epsilon
                            #
                            # # Transform x data to log space BEFORE mirroring
                            # array_x = np.log10(array_x)
                            # low_x = np.log10(low_x)
                            # high_x = np.log10(np.max(array_x))
                            # clipped_x = np.where(array_x < 1, epsilon, array_x)
                            # array_x = np.log10(clipped_x)
                            low_x = np.min(array_x)
                            high_x = np.max(array_x)
                            # print('X log range:', low_x, high_x)
                            # log_bins=(1,3, corner_bins)
                            # x_bins=np.concatenate([0,1],log_bins)
                            x_bins = np.logspace(np.log10(low_x), np.log10(high_x),corner_bins+1)
                            # print(x_bins)
                        else:
                            # x_bins = corner_bins
                            x_bins=np.linspace(np.min(array_x), np.max(array_x), corner_bins+1)

                            # Handle y-axis log binning
                        if use_log_y:
                            # # Ensure all values are positive before log transform
                            # if low_y < 1:
                            #     low_y = epsilon
                            # array_y = np.maximum(array_y, epsilon)  # Clip to minimum epsilon
                            #
                            # # Transform y data to log space BEFORE mirroring
                            # array_y = np.log10(array_y)
                            # low_y = np.log10(low_y)
                            # high_y = np.log10(np.max(array_y))
                            # clipped_y = np.where(array_y < 1, epsilon, array_y)
                            # array_y = np.log10(clipped_y)
                            low_y = np.min(array_y)
                            high_y = np.max(array_y)
                            # log_bins = (1, 3, corner_bins)
                            # y_bins = np.concatenate([0, 1], log_bins)
                            # print('Y log range:', low_y, high_y)
                            y_bins = np.logspace(np.log10(low_y), np.log10(high_y),corner_bins+1)
                        else:
                            # y_bins = corner_bins
                            y_bins = np.linspace(np.min(array_y), np.max(array_y), corner_bins + 1)
                        #Trying to use a KDE
                        # X_mesh, Y_mesh = np.meshgrid(x_bins, y_bins)
                        # data_stack=np.vstack((array_x, array_y))
                        # new_kde=gaussian_kde(data_stack, weights=array_w,bw_method='silverman')
                        # if use_log_x:
                        #     bw_x=10**(np.std(np.log10(array_x)))
                        # else:
                        #     bw_x= np.std(array_x)
                        # if use_log_y:
                        #     bw_y=10**(np.std(np.log10(array_y)))
                        # else:
                        #     bw_y=np.std(array_y)
                        # if use_log_x or use_log_y:
                        #     n=len(array_x)
                        #     d=2
                        #     new_kde.set_bandwidth(bw_method=new_kde.silverman_factor()*np.array([bw_x, bw_y]).mean())
                        # positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
                        # f= new_kde(positions).reshape(X_mesh.shape)
                        # X = x_bins
                        # Y = y_bins
                        #
                        # max_indices = np.unravel_index(np.argmax(f), f.shape)
                        # max_x = X[max_indices[1]]
                        # max_y = Y[max_indices[0]]
                        # Create 2D histogram
                        # H, xedges, yedges = np.histogram2d(array_x, array_y,bins=[x_bins, y_bins],weights=array_w,density=True)
                        H, xedges, yedges = np.histogram2d(array_x, array_y, bins=[x_bins, y_bins], weights=array_w,density=True)

                        # Transpose H for correct orientation
                        H = H.T

                        # Convert to density (comparable to KDE output)
                        # dx = np.diff(xedges)
                        # dy = np.diff(yedges)
                        # bin_areas = dx[:, None] * dy[None, :]
                        # tot_h=np.sum(H)
                        # H = H / (tot_h * bin_areas)
                        #smooth
                        if smooth_corner:
                            H = gaussian_filter(H, sigma=smooth_val)

                        # Get bin centers for plotting
                        X = (xedges[:-1] + xedges[1:]) / 2
                        Y = (yedges[:-1] + yedges[1:]) / 2

                        # Find maximum for mode
                        max_indices = np.unravel_index(np.argmax(H), H.shape)
                        max_x = X[max_indices[1]]
                        max_y = Y[max_indices[0]]

                        # Apply log scale if needed
                        f = H.copy()
                        if scale_t == 'Log':
                        #     Avoid log(0) by adding small epsilon where H is zero
                            f = np.where(f > 0, np.log(f) / np.log(l_base), np.nan)

                        # Plot contour
                        X_mesh, Y_mesh = np.meshgrid(X, Y)
                        cfset = ax.contourf(X_mesh, Y_mesh, f, 100, cmap=my_colorMap)
                        # cset = ax.contour(X_mesh, Y_mesh, f, n_con, colors='k')
                        # Use logarithmically spaced levels:
                        if scale_t == 'Log':
                            # f is already in log space, so use linear spacing
                            levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                        else:
                            # f is in linear space, so use log spacing
                            f_max =np.log10(np.max(f))
                            f_min= np.min(f)
                            f_proposed = f_max - 3
                            if f_min==0:
                                f_selected=f_proposed
                            else:
                                f_selected=np.max([np.log10(f_min),f_proposed])

                            # print(f_min, f_proposed, f_selected)
                            # levels = np.logspace(f_max-4, f_max, 10)
                            levels=np.linspace(f_selected, f_max, n_con)
                            levels=10**levels
                            # print(np.log10(f_min), np.log10(f_max))
                            # levels = np.logspace(np.log10(f_min), np.log10(f_max), n_con)
                            # percentiles = np.linspace(1, 99, n_con)  # Skip extreme tails
                            # percents=100-percentiles
                            # print(percentiles)
                            # print(percents)
                            # levels = np.percentile(f, percentiles)
                        # print(levels)
                        cset = ax.contour(X, Y, f, levels=levels, colors='k')
                        ax.clabel(cset, inline=1, fontsize=8)

                        # Set axis limits and scales
                        if use_log_x:
                            ax.set_xscale('log')
                            ax.set_xlim(0.01, np.max(array_x))
                        else:
                            ax.set_xlim(low_x, high_x)

                        if use_log_y:
                            ax.set_yscale('log')
                            ax.set_ylim(0.01, np.max(array_y))
                        else:
                            ax.set_ylim(low_y, high_y)

                    # Mirror data if needed
                    else:
                        if mirror_data:
                            array_x = np.concatenate((array_x, 2 * low_x - array_x))
                            array_y = np.concatenate((array_y, 2 * low_y - array_y))
                            array_w = np.concatenate((array_w, array_w))
                            # print('Entering Mirror')
                        # Recalculate ranges after mirroring
                        delta_x = deltas if type_step != 'bins' else (high_x - low_x) / corner_bins
                        delta_y = deltas if type_step != 'bins' else (high_y - low_y) / corner_bins
                        # print('X range:', low_x, high_x)
                        # print('Y range:', low_y, high_y)
                        # print('Deltas [x,y]: ', delta_x,delta_y)
                        if delta_x > delta_y:
                            bws = delta_x
                            scal_fac = delta_x / delta_y
                            array_y = array_y * scal_fac
                        elif delta_x < delta_y:
                            scal_fac = delta_y / delta_x
                            bws = delta_y
                            array_x = array_x * scal_fac
                        else:
                            bws = delta_x

                        x_bins = num_bins(array_x, delta_x)
                        y_bins = num_bins(array_y, delta_y)
                        # print(i,j)
                        # print('# of bins:', corner_bins, type_step)
                        # print('bins [x,y]: ',x_bins,y_bins)
                        # Create 2D KDE
                        [values] = np.dstack([array_x, array_y])
                        grid_points_x, grid_points_y = x_bins, y_bins
                        kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                        grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))

                        X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                        f = points.reshape(grid_points_x, grid_points_y).T

                        # Zero out values outside domain
                        xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                        ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                        if delta_x > delta_y:
                            ys = ys / scal_fac
                        elif delta_x < delta_y:
                            xs = xs / scal_fac
                        for ii in range(grid_points_y):
                            for jj in range(grid_points_x):
                                if high_x < xs[ii][jj] or xs[ii][jj] < low_x or \
                                        high_y < ys[ii][jj] or ys[ii][jj] < low_y:
                                    f[ii][jj] = 0
                                elif mirror_data:
                                    f[ii][jj] = f[ii][jj] * 2

                        # Find maximum
                        max_value = np.max(f)
                        max_indices = np.unravel_index(np.argmax(f), f.shape)
                        if delta_x > delta_y:
                            Y = Y / scal_fac
                        elif delta_x < delta_y:
                            X = X / scal_fac
                        if use_log_x or use_log_y:
                            if use_log_x:
                                X=10**X
                            if use_log_y:
                                Y=10**Y
                        max_x = X[max_indices[1]]
                        max_y = Y[max_indices[0]]


                        # Apply log scale if needed
                        if scale_t == 'Log':# or use_log_x or use_log_y:
                            f = np.log(f) / np.log(l_base)

                        # Plot contour
                        cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                        # cset = ax.contour(X, Y, f, n_con, colors='k')
                        if scale_t == 'Log':
                            # f is already in log space, so use linear spacing
                            levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                        else:
                            # f is in linear space, so use log spacing
                            f_max =np.log10(np.max(f))
                            # levels = np.logspace(f_max-4, f_max, 10)
                            f_min = np.min(f)
                            f_proposed = f_max - 3
                            if f_min <= 0:
                                f_selected = f_proposed
                            else:
                                print(f_min)
                                f_selected = np.max([np.log10(f_min), f_proposed])

                            # print(f_min, f_proposed, f_selected)
                            # levels = np.logspace(f_max-4, f_max, 10)
                            levels = np.linspace(f_selected, f_max, n_con)
                            levels=10**levels
                        cset = ax.contour(X, Y, f, levels=levels, colors='k')
                        ax.clabel(cset, inline=1, fontsize=8)
                        ax.set_xlim(low_x, high_x)
                        ax.set_ylim(low_y, high_y)

                    # ADD GRIDLINES HERE
                    ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                    # Mark mode
                    sig_c = int(abs(np.log10(bws)))
                    ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5,label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, labels[j]))

                    ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5,label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, labels[i]))


                    if i == n_vars - 1:
                        ax.set_xlabel(labels[j], fontsize=fo)
                    if j == 0:
                        ax.set_ylabel(labels[i], fontsize=fo)
                    ax.legend(fontsize=fo)

                # Diagonal: 1D marginal distributions
                elif i == j:
                    array_val = data[:, i]
                    array_w = weights.copy()

                    # Remove NaN values
                    # valid_mask = ~np.isnan(array_val)
                    # array_val = array_val[valid_mask]
                    # array_w = array_w[valid_mask]

                    if len(array_val) == 0:
                        ax.axis('off')
                        continue

                    # NEW: Apply log binning if enabled for this variable
                    if use_log_bins and (i in log_bin_indices) and type_p == 'NoLCPS':
                        # clipped_val = np.where(array_val < 1, epsilon, array_val)
                        # array_val = clipped_val
                        low_val = np.min(array_val)
                        high_val = np.max(array_val)
                        # log_bins=np.logspace(0,3, corner_bins)
                        # val_bins=np.concatenate([[0.9,1],log_bins])
                        delta_val=1
                        val_bins = np.logspace(np.log10(low_val), np.log10(high_val), corner_bins)
                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(np.log10(array_val), weights=array_w)()
                        x_rs=10**x_rs
                        y_rs=y_rs/(x_rs* np.log(10))
                        # x_rs = (val_bins[:-1] + val_bins[1:]) / 2
                        # x_rs = val_bins
                        # y_rs = gaussian_kde(array_val, weights=array_w, bw_method='silverman').evaluate(x_rs)



                        # # Handle zero/negative values
                        # if low_val <= 0:
                        #     low_val = epsilon
                        #
                        # # Create log-spaced bins
                        # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                    else:
                        low_val = np.min(array_val)
                        high_val = np.max(array_val)
                        delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                        val_bins = num_bins(array_val, delta_val)

                        if mirror_data:
                            array_valM = np.concatenate((array_val, 2 * low_val - array_val))
                            array_wM = np.concatenate((array_w, array_w))

                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_valM, weights=array_wM)()
                        if mirror_data:
                            mask = (x_rs < low_val) | (high_val < x_rs)
                            y_rs[mask] = 0
                            y_rs = y_rs * 2
                    sig_val = int(abs(np.log10(delta_val)))


                    # Find peaks
                    if show_peaks:
                        mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                        y_rsP = y_rs[mask_P]
                        x_rsP = x_rs[mask_P]
                        peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                        peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                        if len(peaks_idr) > num_peaks:
                            peaks_idr = peaks_idr[:num_peaks]
                        colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                        for d, p in enumerate(peaks_idr):
                            peak_r = x_rsP[p]
                            ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                    # Find mode
                    max_xi = np.argmax(y_rs)
                    max_x = x_rs[max_xi]

                    # Plot
                    ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                    ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                    ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0,
                               label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                    # ADD VERTICAL GRIDLINES HERE
                    ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                    # ax.set_xlabel(region_labels[i], fontsize=fo)
                    ax.set_title(labels[i], fontsize=fo)
                    if use_log_bins and (i in log_bin_indices) and type_p == 'NoLCPS':
                        ax.set_xscale('log')
                        ax.set_xlim(0.01, high_val)
                        # low_yrs= np.min(y_rs)
                        # yrs_proposed = np.log10(np.max(y_rs)) - 5
                        # if low_yrs <= low_pdf:
                        #     sel_pdf = yrs_proposed
                        # else:
                        #     sel_pdf = np.max([np.log10(low_yrs), yrs_proposed])
                        # sel_pdf=10**sel_pdf
                        # print(sel_pdf)
                        # ax.set_ylim(bottom=sel_pdf)
                        ax.set_ylim(bottom=low_pdf)
                        ax.set_yscale('log', base=l_base)
                        ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                    else:
                        ax.set_xlim(low_val, high_val)

                        if scale_t == 'Log':
                            ax.set_yscale('log', base=l_base)
                            ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        else:
                            ax.set_ylabel('PDF', fontsize=fo)

                    ax.legend(fontsize=fo)

                # Upper triangle: hide
                else:
                    ax.axis('off')
                ax.minorticks_on()
                ax.tick_params(axis='both', labelsize=fo)

        fig.suptitle(f'Layer {k} ({max_Alt} - {min_Alt} km) - Corner Plot',fontsize=fo + 4)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save_cornerBP== True:
            if w_prior:
                corner_path = os.path.join(res_path, "MyCornerPlots_wPrior")
                if filter_values:
                    corner_path = os.path.join(res_path, "MyCornerPlots_wBoth")
            elif filter_values:
                corner_path = os.path.join(res_path, f"MyCornerPlots_wFilter")
            else:
                corner_path = os.path.join(res_path, "MyCornerPlots")
            os.makedirs(corner_path, exist_ok=True)
            filename_parts = [f'MyCornerPlot_Layer{k}']
            if use_mode:
                filename_parts.append('jointW')
            if scale_type == 'Log':
                filename_parts.append('Log')
            if use_log_bins:
                filename_parts.append('LBins')
            filename = '_'.join(filename_parts) + '.png'
            fig.savefig(os.path.join(corner_path, filename))

        if show_cornerBP==True:
            plt.show()
    if show_cornerMP or save_cornerMP:
        for n in range(n_modes):
            if n == 0:
                columns = [0, 1, 2, 9, 10, 11]
            elif n == 1:
                columns = [3, 4, 5, 12, 13, 14]
            elif n == 2:
                columns = [6, 7, 8, 15, 16, 17]

            if type_p == 'MultipleRI':
                columns= columns[:3]
            n_data = data[:, columns]
            n_weights = weights
            n_labels = labels[columns]
            # Determine number of variables
            n_vars = n_data.shape[1]

            # Create corner plot
            fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

            # Plot for each pair of variables
            for i in range(n_vars):
                for j in range(n_vars):
                    ax = axs[i, j]

                    # Lower triangle: 2D joint distributions
                    if i > j:
                        # print(i, j)
                        array_x = n_data[:, j]
                        array_y = n_data[:, i]
                        array_w = n_weights.copy()

                        # Remove NaN values
                        # valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                        # array_x = array_x[valid_mask]
                        # array_y = array_y[valid_mask]
                        # array_w = array_w[valid_mask]

                        if len(array_x) == 0:
                            ax.axis('off')
                            continue

                        low_x, high_x = np.min(array_x), np.max(array_x)
                        low_y, high_y = np.min(array_y), np.max(array_y)
                        if use_log_bins:
                            if type_p == 'MultipleRI':
                                use_log_x = False
                                use_log_y = False
                            else:
                                use_log_x = use_log_bins and (j ==0)
                                use_log_y = use_log_bins and (i ==0)
                        else:
                            use_log_x = False
                            use_log_y = False
                        if use_log_x or use_log_y:

                            # Handle x-axis log binning
                            if use_log_x:
                                # Ensure all values are positive before log transform
                                # if low_x < 1:
                                #     low_x = epsilon
                                # array_x = np.maximum(array_x, epsilon)  # Clip to minimum epsilon
                                #
                                # # Transform x data to log space BEFORE mirroring
                                # array_x = np.log10(array_x)
                                # low_x = np.log10(low_x)
                                # high_x = np.log10(np.max(array_x))
                                # clipped_x = np.where(array_x < 1, epsilon, array_x)
                                # array_x = np.log10(clipped_x)
                                low_x = np.min(array_x)
                                high_x = np.max(array_x)
                                # print('X log range:', low_x, high_x)
                                # log_bins=(1,3, corner_bins)
                                # x_bins=np.concatenate([0,1],log_bins)
                                x_bins = np.logspace(np.log10(low_x), np.log10(high_x), corner_bins + 1)
                                # print(x_bins)
                            else:
                                # x_bins = corner_bins
                                x_bins = np.linspace(np.min(array_x), np.max(array_x), corner_bins + 1)

                                # Handle y-axis log binning
                            if use_log_y:
                                # # Ensure all values are positive before log transform
                                # if low_y < 1:
                                #     low_y = epsilon
                                # array_y = np.maximum(array_y, epsilon)  # Clip to minimum epsilon
                                #
                                # # Transform y data to log space BEFORE mirroring
                                # array_y = np.log10(array_y)
                                # low_y = np.log10(low_y)
                                # high_y = np.log10(np.max(array_y))
                                # clipped_y = np.where(array_y < 1, epsilon, array_y)
                                # array_y = np.log10(clipped_y)
                                low_y = np.min(array_y)
                                high_y = np.max(array_y)
                                # log_bins = (1, 3, corner_bins)
                                # y_bins = np.concatenate([0, 1], log_bins)
                                # print('Y log range:', low_y, high_y)
                                y_bins = np.logspace(np.log10(low_y), np.log10(high_y), corner_bins + 1)
                            else:
                                # y_bins = corner_bins
                                y_bins = np.linspace(np.min(array_y), np.max(array_y), corner_bins + 1)
                            # Trying to use a KDE
                            # X_mesh, Y_mesh = np.meshgrid(x_bins, y_bins)
                            # data_stack=np.vstack((array_x, array_y))
                            # new_kde=gaussian_kde(data_stack, weights=array_w,bw_method='silverman')
                            # if use_log_x:
                            #     bw_x=10**(np.std(np.log10(array_x)))
                            # else:
                            #     bw_x= np.std(array_x)
                            # if use_log_y:
                            #     bw_y=10**(np.std(np.log10(array_y)))
                            # else:
                            #     bw_y=np.std(array_y)
                            # if use_log_x or use_log_y:
                            #     n=len(array_x)
                            #     d=2
                            #     new_kde.set_bandwidth(bw_method=new_kde.silverman_factor()*np.array([bw_x, bw_y]).mean())
                            # positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
                            # f= new_kde(positions).reshape(X_mesh.shape)
                            # X = x_bins
                            # Y = y_bins
                            #
                            # max_indices = np.unravel_index(np.argmax(f), f.shape)
                            # max_x = X[max_indices[1]]
                            # max_y = Y[max_indices[0]]
                            # Create 2D histogram
                            # H, xedges, yedges = np.histogram2d(array_x, array_y,bins=[x_bins, y_bins],weights=array_w,density=True)
                            H, xedges, yedges = np.histogram2d(array_x, array_y, bins=[x_bins, y_bins], weights=array_w, density=True)

                            # Transpose H for correct orientation
                            H = H.T

                            # Convert to density (comparable to KDE output)
                            # dx = np.diff(xedges)
                            # dy = np.diff(yedges)
                            # bin_areas = dx[:, None] * dy[None, :]
                            # tot_h=np.sum(H)
                            # H = H / (tot_h * bin_areas)
                            # smooth
                            if smooth_corner:
                                H = gaussian_filter(H, sigma=smooth_val)

                            # Get bin centers for plotting
                            X = (xedges[:-1] + xedges[1:]) / 2
                            Y = (yedges[:-1] + yedges[1:]) / 2

                            # Find maximum for mode
                            max_indices = np.unravel_index(np.argmax(H), H.shape)
                            max_x = X[max_indices[1]]
                            max_y = Y[max_indices[0]]

                            # Apply log scale if needed
                            f = H.copy()
                            if scale_t == 'Log':
                                #     Avoid log(0) by adding small epsilon where H is zero
                                f = np.where(f > 0, np.log(f) / np.log(l_base), np.nan)

                            # Plot contour
                            X_mesh, Y_mesh = np.meshgrid(X, Y)
                            cfset = ax.contourf(X_mesh, Y_mesh, f, 100, cmap=my_colorMap)
                            # cset = ax.contour(X_mesh, Y_mesh, f, n_con, colors='k')
                            # Use logarithmically spaced levels:
                            if scale_t == 'Log':
                                # f is already in log space, so use linear spacing
                                levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                            else:
                                # f is in linear space, so use log spacing
                                f_max = np.log10(np.max(f))
                                f_min = np.min(f)
                                f_proposed = f_max - 3
                                if f_min == 0:
                                    f_selected = f_proposed
                                else:
                                    f_selected = np.max([np.log10(f_min), f_proposed])

                                # print(f_min, f_proposed, f_selected)
                                # levels = np.logspace(f_max-4, f_max, 10)
                                levels = np.linspace(f_selected, f_max, n_con)
                                levels = 10 ** levels
                                # print(np.log10(f_min), np.log10(f_max))
                                # levels = np.logspace(np.log10(f_min), np.log10(f_max), n_con)
                                # percentiles = np.linspace(1, 99, n_con)  # Skip extreme tails
                                # percents=100-percentiles
                                # print(percentiles)
                                # print(percents)
                                # levels = np.percentile(f, percentiles)
                            # print(levels)
                            cset = ax.contour(X, Y, f, levels=levels, colors='k')
                            ax.clabel(cset, inline=1, fontsize=8)

                            # Set axis limits and scales
                            if use_log_x:
                                ax.set_xscale('log')
                                ax.set_xlim(0.01, np.max(array_x))
                            else:
                                ax.set_xlim(low_x, high_x)

                            if use_log_y:
                                ax.set_yscale('log')
                                ax.set_ylim(0.01, np.max(array_y))
                            else:
                                ax.set_ylim(low_y, high_y)

                        # Mirror data if needed
                        else:
                            if mirror_data:
                                array_x = np.concatenate((array_x, 2 * low_x - array_x))
                                array_y = np.concatenate((array_y, 2 * low_y - array_y))
                                array_w = np.concatenate((array_w, array_w))
                                # print('Entering Mirror')
                            # Recalculate ranges after mirroring
                            delta_x = deltas if type_step != 'bins' else (high_x - low_x) / corner_bins
                            delta_y = deltas if type_step != 'bins' else (high_y - low_y) / corner_bins
                            # print('X range:', low_x, high_x)
                            # print('Y range:', low_y, high_y)
                            # print('Deltas [x,y]: ', delta_x,delta_y)
                            if delta_x > delta_y:
                                bws = delta_x
                                scal_fac = delta_x / delta_y
                                array_y = array_y * scal_fac
                            elif delta_x < delta_y:
                                scal_fac = delta_y / delta_x
                                bws = delta_y
                                array_x = array_x * scal_fac
                            else:
                                bws = delta_x

                            x_bins = num_bins(array_x, delta_x)
                            y_bins = num_bins(array_y, delta_y)
                            # print(i,j)
                            # print('# of bins:', corner_bins, type_step)
                            # print('bins [x,y]: ',x_bins,y_bins)
                            # Create 2D KDE
                            [values] = np.dstack([array_x, array_y])
                            grid_points_x, grid_points_y = x_bins, y_bins
                            kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                            grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))

                            X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                            f = points.reshape(grid_points_x, grid_points_y).T

                            # Zero out values outside domain
                            xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                            ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                            if delta_x > delta_y:
                                ys = ys / scal_fac
                            elif delta_x < delta_y:
                                xs = xs / scal_fac
                            for ii in range(grid_points_y):
                                for jj in range(grid_points_x):
                                    if high_x < xs[ii][jj] or xs[ii][jj] < low_x or \
                                            high_y < ys[ii][jj] or ys[ii][jj] < low_y:
                                        f[ii][jj] = 0
                                    elif mirror_data:
                                        f[ii][jj] = f[ii][jj] * 2

                            # Find maximum
                            max_value = np.max(f)
                            max_indices = np.unravel_index(np.argmax(f), f.shape)
                            if delta_x > delta_y:
                                Y = Y / scal_fac
                            elif delta_x < delta_y:
                                X = X / scal_fac
                            if use_log_x or use_log_y:
                                if use_log_x:
                                    X = 10 ** X
                                if use_log_y:
                                    Y = 10 ** Y
                            max_x = X[max_indices[1]]
                            max_y = Y[max_indices[0]]

                            # Apply log scale if needed
                            if scale_t == 'Log':  # or use_log_x or use_log_y:
                                f = np.log(f) / np.log(l_base)

                            # Plot contour
                            cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                            # cset = ax.contour(X, Y, f, n_con, colors='k')
                            if scale_t == 'Log':
                                # f is already in log space, so use linear spacing
                                levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                            else:
                                # f is in linear space, so use log spacing
                                f_max = np.log10(np.max(f))
                                # levels = np.logspace(f_max-4, f_max, 10)
                                f_min = np.min(f)
                                f_proposed = f_max - 3
                                if f_min <= 0:
                                    f_selected = f_proposed
                                else:
                                    print(f_min)
                                    f_selected = np.max([np.log10(f_min), f_proposed])

                                # print(f_min, f_proposed, f_selected)
                                # levels = np.logspace(f_max-4, f_max, 10)
                                levels = np.linspace(f_selected, f_max, n_con)
                                levels = 10 ** levels
                            cset = ax.contour(X, Y, f, levels=levels, colors='k')
                            ax.clabel(cset, inline=1, fontsize=8)
                            ax.set_xlim(low_x, high_x)
                            ax.set_ylim(low_y, high_y)

                        # ADD GRIDLINES HERE
                        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                        # Mark mode
                        sig_c = int(abs(np.log10(bws)))
                        ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5,
                                   label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, n_labels[j]))

                        ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5,
                                   label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, n_labels[i]))

                        if i == n_vars - 1:
                            ax.set_xlabel(n_labels[j], fontsize=fo)
                        if j == 0:
                            ax.set_ylabel(n_labels[i], fontsize=fo)
                        ax.legend(fontsize=fo)

                    # Diagonal: 1D marginal distributions
                    elif i == j:
                        array_val = n_data[:, i]
                        array_w = n_weights.copy()

                        # Remove NaN values
                        # valid_mask = ~np.isnan(array_val)
                        # array_val = array_val[valid_mask]
                        # array_w = array_w[valid_mask]

                        if len(array_val) == 0:
                            ax.axis('off')
                            continue

                        # NEW: Apply log binning if enabled for this variable
                        if use_log_bins and i ==0 and type_p == 'NoLCPS':
                            # clipped_val = np.where(array_val < 1, epsilon, array_val)
                            # array_val = clipped_val
                            low_val = np.min(array_val)
                            high_val = np.max(array_val)
                            # log_bins=np.logspace(0,3, corner_bins)
                            # val_bins=np.concatenate([[0.9,1],log_bins])
                            delta_val = 1
                            val_bins = np.logspace(np.log10(low_val), np.log10(high_val), corner_bins)
                            x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(np.log10(array_val), weights=array_w)()
                            x_rs = 10 ** x_rs
                            y_rs = y_rs / (x_rs * np.log(10))
                            # x_rs = (val_bins[:-1] + val_bins[1:]) / 2
                            # x_rs = val_bins
                            # y_rs = gaussian_kde(array_val, weights=array_w, bw_method='silverman').evaluate(x_rs)

                            # # Handle zero/negative values
                            # if low_val <= 0:
                            #     low_val = epsilon
                            #
                            # # Create log-spaced bins
                            # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                        else:
                            low_val = np.min(array_val)
                            high_val = np.max(array_val)
                            delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                            val_bins = num_bins(array_val, delta_val)

                            if mirror_data:
                                array_valM = np.concatenate((array_val, 2 * low_val - array_val))
                                array_wM = np.concatenate((array_w, array_w))

                            x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_valM, weights=array_wM)()
                            if mirror_data:
                                mask = (x_rs < low_val) | (high_val < x_rs)
                                y_rs[mask] = 0
                                y_rs = y_rs * 2
                        sig_val = int(abs(np.log10(delta_val)))

                        # Find peaks
                        if show_peaks:
                            mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                            y_rsP = y_rs[mask_P]
                            x_rsP = x_rs[mask_P]
                            peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                            peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                            if len(peaks_idr) > num_peaks:
                                peaks_idr = peaks_idr[:num_peaks]
                            colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                            for d, p in enumerate(peaks_idr):
                                peak_r = x_rsP[p]
                                ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                        # Find mode
                        max_xi = np.argmax(y_rs)
                        max_x = x_rs[max_xi]

                        # Plot
                        ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                        ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                        ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0,
                                   label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                        # ADD VERTICAL GRIDLINES HERE
                        ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                        # ax.set_xlabel(region_labels[i], fontsize=fo)
                        ax.set_title(labels[i], fontsize=fo)
                        if use_log_bins and i==0 and type_p == 'NoLCPS':
                            ax.set_xscale('log')
                            ax.set_xlim(0.01, high_val)
                            # low_yrs= np.min(y_rs)
                            # yrs_proposed = np.log10(np.max(y_rs)) - 5
                            # if low_yrs <= low_pdf:
                            #     sel_pdf = yrs_proposed
                            # else:
                            #     sel_pdf = np.max([np.log10(low_yrs), yrs_proposed])
                            # sel_pdf=10**sel_pdf
                            # print(sel_pdf)
                            # ax.set_ylim(bottom=sel_pdf)
                            ax.set_ylim(bottom=low_pdf)
                            ax.set_yscale('log', base=l_base)
                            ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        else:
                            ax.set_xlim(low_val, high_val)

                            if scale_t == 'Log':
                                ax.set_yscale('log', base=l_base)
                                ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                            else:
                                ax.set_ylabel('PDF', fontsize=fo)

                        ax.legend(fontsize=fo)

                    # Upper triangle: hide
                    else:
                        ax.axis('off')
                    ax.minorticks_on()
                    ax.tick_params(axis='both', labelsize=fo)
            if w_prior:
                fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km with Prior'.format(n + 1, max_Alt, min_Alt), fontsize=fo)
            else:
                fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km'.format(n+1, max_Alt, min_Alt), fontsize=fo)
            plt.tight_layout()
            if save_cornerMP == True:
                if w_prior:
                    corner_path = os.path.join(res_path, "MyCornerPlots_wPrior")
                    if filter_values:
                        corner_path = os.path.join(res_path, "MyCornerPlots_wBoth")
                elif filter_values:
                    corner_path = os.path.join(res_path, f"MyCornerPlots_wFilter")
                else:
                    corner_path = os.path.join(res_path, "MyCornerPlots")
                os.makedirs(corner_path, exist_ok=True)
                filename_parts = [f'MyCornerPlot_Layer{k}_Mode{n}']
                if use_mode:
                    filename_parts.append('jointW')
                if scale_type == 'Log':
                    filename_parts.append('Log')
                if use_log_bins:
                    filename_parts.append('LBins')
                filename = '_'.join(filename_parts) + '.png'
                fig.savefig(os.path.join(corner_path, filename))
                # if use_mode:
                #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}_jointW.png'.format(j, n))
                # else:
                #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}.png'.format(j, n))
            if show_cornerMP == True:
                plt.show()
    if save_cornerDMean== True:
        quants = []
        for i in range(len(labels)-1):
            array_val = n_data[:,i]
            title_quantiles = [0.16, 0.5, 0.84]
            q_lo, q_mid, q_hi = quantile(array_val, title_quantiles, weights=n_weights)
            quants.append(q_mid)
        quants.append(0) #Adding Weight
        quants = np.array(quants)
        if w_prior:
            corner_path = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        filename = os.path.join(corner_path, f"PosDistLayer{k}_CornerMeans.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(quants)
        print(f"Data saved to {filename}")
    if save_cornerDMode:
        max_vals =my_cornerMaxVal(n_modes, data, weights, labels)
        max_vals[-1]=0
        max_vals = np.array(max_vals)
        # if use_mode:
        #     print(max_vals)
        if w_prior:
            corner_path = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        filename = os.path.join(corner_path, f"PosDistLayer{k}_CornerModes.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(max_vals)
        print(f"Data saved to {filename}")
    plt.close('all')
def my_cornerplotN(k, n_modes,y_min, y_max, paths, use_mode=use_mode, show_corner=show_cornerN, save_cornerP=save_cornerN, save_cornerD=save_cornerDisMean, deltas=corner_deltas, type_step=type_step, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, n_con=num_cont, bws=bws, scale_t=scale_type, l_base=log_base, my_colorMap=my_colorMap, show_prob=probs, corner_bins=corner_bins,log_bin_indices = [0, 3, 6], epsilon=epsilon,smooth_corner=smooth_corner,smooth_val=smooth_val):
    res_path = os.path.join(paths, "Results")
    dis_path = os.path.join(paths, "PosteriorDistributions")
    pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(k))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    if filter_values:
        for f_what in filter_what:
            if f_what == 'nums':
                if type_p == 'MultipleRI':
                    continue
                else:
                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                    mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
            if f_what == 'index':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                    elif n_modes == 2:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (
                                    data[:, 5] >= filt_min)
                    else:
                        mask = (filt_max >= data[:, 2]) & (data[:, ] >= filt_min) & (filt_max >= data[:, 5]) & (
                                    data[:, 5] >= filt_min) & (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                else:
                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (filt_max >= data[:, 14]) & (
                                data[:, 14] >= filt_min) & (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            if f_what == 'sigma':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (1 + filt_s >= data[:, 1])
                    elif n_modes == 2:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                    else:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & (1 + filt_s >= data[:, 7])
                else:
                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & (1 + filt_s >= data[:, 16])
            data = data[mask]
            weights = weights[mask]
    if w_prior:
        for p, p_what in enumerate(prior_what):
            sel_val = []
            if p_what == 'index':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 1)
                else:
                    for i in range(n_modes):
                        sel_val.append(11 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p])
            if p_what == 'sigma':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 2)
                else:
                    for i in range(n_modes):
                        sel_val.append(10 + (3 * i))
                # print(sel_val)
                weights, name_prior = sigma_Prior(sel_val, data, weights, t_prior=type_Prior[p])
            if p_what == 'rho':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 3)
                else:
                    for i in range(n_modes):
                        sel_val.append(9 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p], m_val=0, unc_val=prior_unc_rho, low_val=-1, high_val=1)
            if p_what == 'rho_diagonal':
                # Apply diagonal prior to favor similar rho values
                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights, val_labels=labels, n_modes=n_modes, type_p=type_p, sigma_diagonal=0.1, prior_min=-1.0, prior_max=1.0, prior_sig=3)
            if p_what == 'num_total':
                for i in range(n_modes):
                    sel_val.append((3 * i))
                weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])
    if use_mode == True:
        all_sorted = find_weighted_mode(data, weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        data = np.array(weighted_mode)
        weights = np.array(frequency)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    columns = [0, 3, 6]
    # columns = [9, 12, 15]
    n_data = data[:, columns]
    n_weights = weights
    n_labels = labels[columns]
    n_weights = n_weights / np.sum(n_weights)

    # Determine number of n variables (should be 3)
    n_vars = n_data.shape[1]

    fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

    # Plot for each pair of variables
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axs[i, j]

            # Lower triangle: 2D joint distributions
            if i > j:
                array_x = n_data[:, j]
                array_y = n_data[:, i]
                array_w = n_weights.copy()

                # Remove NaN values (should already be clean but double-check)
                valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                array_x = array_x[valid_mask]
                array_y = array_y[valid_mask]
                array_w = array_w[valid_mask]

                if len(array_x) == 0:
                    ax.axis('off')
                    continue
                low_x, high_x = np.min(array_x), np.max(array_x)
                low_y, high_y = np.min(array_y), np.max(array_y)
                # Use log binning for n values
                if use_log_bins:
                    x_bins = np.logspace(np.log10(low_x), np.log10(high_x), corner_bins + 1)
                    y_bins = np.logspace(np.log10(low_y), np.log10(high_y), corner_bins + 1)
                else:
                    x_bins = np.linspace(np.min(array_x), np.max(array_x), corner_bins + 1)
                    y_bins = np.linspace(np.min(array_y), np.max(array_y), corner_bins + 1)

                # Create 2D histogram
                H, xedges, yedges = np.histogram2d(array_x, array_y, bins=[x_bins, y_bins], weights=array_w,
                                                   density=True)
                # Transpose H for correct orientation
                H = H.T

                # Smooth if requested
                if smooth_corner:
                    H = gaussian_filter(H, sigma=smooth_val)

                # Get bin centers for plotting
                X = (xedges[:-1] + xedges[1:]) / 2
                Y = (yedges[:-1] + yedges[1:]) / 2

                # Find maximum for mode
                max_indices = np.unravel_index(np.argmax(H), H.shape)
                max_x = X[max_indices[1]]
                max_y = Y[max_indices[0]]

                # Apply log scale if needed
                f = H.copy()
                if scale_t == 'Log':
                    # f is already in log space, so use linear spacing
                    levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                else:
                    # f is in linear space, so use log spacing
                    f_max = np.log10(np.max(f))
                    f_min = np.min(f)
                    f_proposed = f_max - 3
                    if f_min == 0:
                        f_selected = f_proposed
                    else:
                        f_selected = np.max([np.log10(f_min), f_proposed])

                    # print(f_min, f_proposed, f_selected)
                    # levels = np.logspace(f_max-4, f_max, 10)
                    levels = np.linspace(f_selected, f_max, n_con)
                    levels = 10 ** levels
                # Plot contour
                cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                cset = ax.contour(X, Y, f, levels=levels, colors='k')
                ax.clabel(cset, inline=1, fontsize=8)

                # Add gridlines
                ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                # Mark mode
                sig_c = int(abs(np.log10(bws)))
                ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5,
                           label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, n_labels[j]))
                ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5,
                           label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, n_labels[i]))

                # Set log scale for axes
                if use_log_bins:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    # ax.set_xlim(0.01, high_x)
                    # ax.set_ylim(0.01, high_y)

                # else:
                ax.set_xlim(low_x, high_x)
                ax.set_ylim(low_y, high_y)

                if i == n_vars - 1:
                    ax.set_xlabel(n_labels[j], fontsize=fo)
                if j == 0:
                    ax.set_ylabel(n_labels[i], fontsize=fo)
                ax.legend(fontsize=fo)

            # Diagonal: 1D marginal distributions
            elif i == j:
                array_val = n_data[:, i]
                array_w = n_weights.copy()

                # Remove NaN values (should already be clean but double-check)
                valid_mask = ~np.isnan(array_val)
                array_val = array_val[valid_mask]
                array_w = array_w[valid_mask]

                if len(array_val) == 0:
                    ax.axis('off')
                    continue

                # Apply log binning for n values
                if use_log_bins:
                    # clipped_val = np.where(array_val < 1, epsilon, array_val)
                    # array_val = clipped_val
                    low_val = np.min(array_val)
                    high_val = np.max(array_val)
                    # log_bins=np.logspace(0,3, corner_bins)
                    # val_bins=np.concatenate([[0.9,1],log_bins])
                    delta_val = 1
                    val_bins = np.logspace(np.log10(low_val), np.log10(high_val), corner_bins)
                    x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(np.log10(array_val), weights=array_w)()
                    x_rs = 10 ** x_rs
                    y_rs = y_rs / (x_rs * np.log(10))

                    # # Handle zero/negative values
                    # if low_val <= 0:
                    #     low_val = epsilon
                    #
                    # # Create log-spaced bins
                    # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                else:
                    low_val = np.min(array_val)
                    high_val = np.max(array_val)
                    delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                    val_bins = num_bins(array_val, delta_val)

                    if mirror_data:
                        array_valM = np.concatenate((array_val, 2 * low_val - array_val))
                        array_wM = np.concatenate((array_w, array_w))

                    x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_valM, weights=array_wM)()
                    if mirror_data:
                        mask = (x_rs < low_val) | (high_val < x_rs)
                        y_rs[mask] = 0
                        y_rs = y_rs * 2
                sig_val = int(abs(np.log10(delta_val)))

                # Find peaks if requested
                if show_peaks:
                    mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                    y_rsP = y_rs[mask_P]
                    x_rsP = x_rs[mask_P]
                    peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                    peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                    if len(peaks_idr) > num_peaks:
                        peaks_idr = peaks_idr[:num_peaks]
                    colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                    for d, p in enumerate(peaks_idr):
                        peak_r = x_rsP[p]
                        ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                # Find mode
                max_xi = np.argmax(y_rs)
                max_x = x_rs[max_xi]

                # Plot
                ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                # ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0,
                           label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                # Add vertical gridlines
                ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                ax.set_title(n_labels[i], fontsize=fo)

                if use_log_bins:
                    ax.set_xscale('log')
                    ax.set_ylim(bottom=low_pdf)
                    ax.set_yscale('log', base=l_base)
                    ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                    # ax.set_xlim(0.01, high_val)
                # else:
                ax.set_xlim(low_val, high_val)

                if scale_t == 'Log':
                    ax.set_ylim(bottom=low_pdf)
                    ax.set_yscale('log', base=l_base)
                    ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                else:
                    ax.set_ylabel('PDF', fontsize=fo)

                ax.legend(fontsize=fo)

            # Upper triangle: hide
            else:
                ax.axis('off')

            ax.minorticks_on()
            ax.tick_params(axis='both', labelsize=fo)

    # Add title
    if w_prior:
        fig.suptitle('Corner Plot for N values,  Layer {} - {}km with Prior'.format(max_Alt, min_Alt), fontsize=fo)
    else:
        fig.suptitle('Corner Plot for N values,  Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)
    plt.tight_layout()
    if save_cornerP== True:
        if w_prior:
            corner_path = os.path.join(res_path, "MyCornerPlots_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, "MyCornerPlots_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"MyCornerPlots_wFilter")
        else:
            corner_path = os.path.join(res_path, "MyCornerPlots")
        os.makedirs(corner_path, exist_ok=True)
        filename_parts = [f'MyCornerPlot_Layer{k}_N']
        if use_mode:
            filename_parts.append('jointW')
        if scale_type == 'Log':
            filename_parts.append('Log')
        if use_log_bins:
            filename_parts.append('LBins')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(corner_path, filename))
        # fig.savefig(corner_path + r'\CornerPlot_Layer{}_N.png'.format(j))
    if show_corner==True:
        plt.show()
    # my_cornerMaxVal(4,data,weights,labels)
def my_CDFN(j, n_modes, y_min, y_max, paths, show_cdf=show_CDFN, save_cdf=save_CDFN, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, type_p=type_p,show_q=show_quantiles):
    """
    Creates cumulative distribution function (CDF) plots for n_i values
    for a single atmospheric layer.

    This is the per-layer version, analogous to how my_corner relates to my_corner_Region.
    Shows the CDF for each of the three n values (n1, n2, n3) for one layer.
    """
    import warnings
    warnings.filterwarnings('ignore', 'invalid value encountered in divide', category=RuntimeWarning)

    # Return early if MultipleRI mode
    if type_p == 'MultipleRI':
        print('Does not make sense to plot Ns for MultipleRI')
        return

    res_path = os.path.join(paths, "Results")

    # Load data for this specific layer
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")

    # Read the appropriate data file
    if use_mean:
        pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
        data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    elif use_cornerMode:
        pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
        data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    elif use_cornerMean:
        pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
        data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    elif use_mode:
        pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
        data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
    else:
        dis_path = os.path.join(paths, "PosteriorDistributions")
        pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
        data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)

    # Apply filtering if needed
    if filter_values:
        for f_what in filter_what:
            if f_what == 'nums':
                n_T = data[:, 0] + data[:, 3] + data[:, 6]
                mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
            elif f_what == 'index':
                mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & \
                       (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & \
                       (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            elif f_what == 'sigma':
                mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & \
                       (1 + filt_s >= data[:, 16])
            data = data[mask]
            weights = weights[mask]

    # Apply priors if needed
    if w_prior:
        for p, p_what in enumerate(prior_what):
            sel_val = []
            if p_what == 'num_total':
                for i in range(n_modes):
                    sel_val.append((3 * i))
                weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])

    # Normalize weights
    weights = weights / np.sum(weights)

    # Extract only n_i columns (columns 0, 3, 6)
    n_columns = [0, 3, 6]
    n_data = data[:, n_columns]
    n_weights = weights
    n_labels = labels[n_columns]

    # Remove rows with NaN values in n columns
    valid_mask = ~np.isnan(n_data).any(axis=1)
    n_data = n_data[valid_mask]
    n_weights = n_weights[valid_mask]
    if show_q:
        # Both 68% and 95% CI
        quantiles = [0.025, 0.16, 0.5, 0.84, 0.975]
        quantile_colors = {
            0.5: 'red',  # Median
            0.16: 'blue',  # 68% lower
            0.84: 'blue',  # 68% upper
            0.025: 'purple',  # 95% lower
            0.975: 'purple',  # 95% upper
            0.05: 'orange',  # 90% lower
            0.95: 'orange',  # 90% upper
            0.25: 'gray',  # Quartile
            0.75: 'gray',  # Quartile
        }
        quantile_styles = {
            0.5: '--',  # Median: dashed
            0.16: ':',  # 68%: dotted
            0.84: ':',
            0.025: '-.',  # 95%: dash-dot
            0.975: '-.',
            0.05: '--',  # 90%: dashed
            0.95: '--',
            0.25: ':',  # Quartiles: dotted
            0.75: ':',
        }

    if len(n_data) == 0:
        print(f"No valid n data for layer {j}")
        return

    # Renormalize weights after filtering
    n_weights = n_weights / np.sum(n_weights)

    # Create figure with 3 subplots (one for each n value)
    fig, axs = plt.subplots(1, 3, figsize=(figSize_x * 3, figSize_y))

    # Color scheme for the three modes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    for idx in range(3):
        ax = axs[idx]

        # Get data for this n value
        n_vals = n_data[:, idx]
        weights_i = n_weights

        # Sort values and weights together
        sort_idx = np.argsort(n_vals)
        n_sorted = n_vals[sort_idx]
        weights_sorted = weights_i[sort_idx]

        # Calculate cumulative sum of weights
        cdf = np.cumsum(weights_sorted)

        # Plot CDF
        ax.plot(n_sorted, cdf, linewidth=2.5, color=colors[idx], label='Weighted CDF')

        if show_q:
            quantile_values = {}
            # Plot each quantile
            for q in quantiles:
                q_idx = np.searchsorted(cdf, q)
                if q_idx < len(n_sorted):
                    q_val = n_sorted[q_idx]
                    quantile_values[q] = q_val

                    # Get color and style for this quantile
                    q_color = quantile_colors.get(q, 'magenta')
                    q_style = quantile_styles.get(q, '--')

                    # Determine line width based on importance
                    if q == 0.5:
                        lw = 2.0
                        alpha = 0.8
                        label = f'Median ({q * 100:.0f}%): {q_val:.1f}'
                        median = q_val
                    elif q in [0.16, 0.84, 0.025, 0.975]:
                        lw = 1.5
                        alpha = 0.6
                        label = f'{q * 100:.1f}%: {q_val:.1f}'
                    else:
                        lw = 1.0
                        alpha = 0.4
                        label = None

                    # Plot lines
                    ax.axhline(y=q, color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1)
                    ax.axvline(x=q_val, color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1,
                               label=label)
                    # Check for 68% CI (1-sigma)
                    if q == 0.16:
                        lower = q_val
                    if q == 0.84:
                        upper = q_val
                    # Check for 95% CI (2-sigma)
                    # if q==0.025:
                    #     lower = q_val
                    # if q==0.975:
                    #     upper = q_val

            title_q = r"{3} = ${{{0:.1f}}}_{{min: {1:.1f}}}^{{max: {2:.1f}}}$"
            title_q = title_q.format(median, lower, upper, n_labels[idx])
            ax.set_title(title_q, fontsize=fo)
        else:
            # Add horizontal line at 0.5 for median
            median_idx = np.searchsorted(cdf, 0.5)
            if median_idx < len(n_sorted):
                median_val = n_sorted[median_idx]
                ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                           label=f'Median: {median_val:.1f}')
                ax.axvline(x=median_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            # Add quartile lines
            # q25_idx = np.searchsorted(cdf, 0.25)
            # q75_idx = np.searchsorted(cdf, 0.75)
            # if q25_idx < len(n_sorted):
            #     q25_val = n_sorted[q25_idx]
            #     ax.axhline(y=0.25, color='magenta', linestyle=':', linewidth=1, alpha=0.5)
            #     ax.axvline(x=q25_val, color='magenta', linestyle=':', linewidth=1, alpha=0.5)
            # if q75_idx < len(n_sorted):
            #     q75_val = n_sorted[q75_idx]
            #     ax.axhline(y=0.75, color='magenta', linestyle=':', linewidth=1, alpha=0.5)
            #     ax.axvline(x=q75_val, color='magenta', linestyle=':', linewidth=1, alpha=0.5)

            ax.set_title(f'{n_labels[idx]} CDF', fontsize=fo)

        # Formatting
        ax.set_xlabel(n_labels[idx], fontsize=fo)
        ax.set_ylabel('Cumulative Probability', fontsize=fo)
        ax.set_title(f'{n_labels[idx]} CDF', fontsize=fo)
        ax.set_xlim(np.min(n_sorted), np.max(n_sorted) * 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=fo - 2)
        ax.tick_params(axis='both', labelsize=fo)

        # Optional: Use log scale for x-axis if values span multiple orders of magnitude
        if np.max(n_sorted) / np.min(n_sorted[n_sorted > 0]) > 100:
            ax.set_xscale('log')

    # Format altitude values
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")

    # Add overall title
    if w_prior:
        fig.suptitle(f'Cumulative Distribution Functions - Layer {max_Alt} - {min_Alt} km with Prior',
                     fontsize=fo + 2)
    else:
        fig.suptitle(f'Cumulative Distribution Functions - Layer {max_Alt} - {min_Alt} km',
                     fontsize=fo + 2)
    plt.tight_layout()

    # Save if requested
    if save_cdf:
        if w_prior:
            cdf_path = os.path.join(res_path, "CDF_Plots_wPrior")
            if filter_values:
                cdf_path = os.path.join(res_path, "CDF_Plots_wBoth")
        elif filter_values:
            cdf_path = os.path.join(res_path, f"CDF_Plots_wFilter")
        else:
            cdf_path = os.path.join(res_path, "CDF_Plots")
        os.makedirs(cdf_path, exist_ok=True)

        filename_parts = [f'CDF_Layer{j}_N']
        if use_mode:
            filename_parts.append('jointW')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(cdf_path, filename), dpi=300, bbox_inches='tight')

    # Show if requested
    if show_cdf:
        plt.show()

    plt.close('all')
def my_2D(j, n_modes, y_min, y_max, s_dis, paths, data_dir, n_con=num_cont, show_2D=show_2D, save_2D=save_2D, s_range=set_range,scale_t=scale_type, l_base=log_base):
    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")
    if use_mean==True:
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_Mean.csv".format(j))
    elif use_cornerMode==True:
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerModes.csv".format(j))
    elif use_cornerMean==True:
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerMeans.csv".format(j))
    elif use_mode== True:
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    model_dir = os.path.join(data_dir, 'ForwardModelMatrix_delta_001')
    X_matrix, Y_matrix, Isca_matrix = load_matrices_numpy(model_dir)
    if type_p == 'MultipleRI':
        layer_arr = np.hstack((s_dis, data[0]))
        f = bivariate_distribution(layer_arr, X_matrix, Y_matrix)
    else:
        f = bivariate_distribution(data[0], X_matrix,Y_matrix)
    if scale_t == 'Log':
        f = np.log(f) / np.log(l_base)
    fig, axs = plt.subplots(1,1,  figsize=(figSize_x, figSize_y))
    cfset =axs.contourf(X_matrix, Y_matrix, f, 100, cmap=my_colorMap)
    cbarR = plt.colorbar(cfset, format='%.2f', boundaries=[-1, 1])
    if scale_t == 'Log':
        cbarR.set_label(f'Log{l_base} ($N_T$)')
    else:
        cbarR.set_label(r'$N_T$', fontsize=fo)
    cset = axs.contour(X_matrix, Y_matrix, f, n_con, colors='k')
    if s_range == True:
        axs.set_xlim(low_a, high_a)
        axs.set_ylim(low_ri, high_ri)
    axs.clabel(cset, inline=1, fontsize=10)
    axs.set_xlabel(r'Particle Size ($\mu$m)')
    axs.set_ylabel('Refractive Index')
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    if w_prior:
        fig.suptitle('2D Distribution for Layer {} - {}km with Prior'.format(max_Alt, min_Alt), fontsize=fo)
    else:
        fig.suptitle('2D Distribution for Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)
    plt.tight_layout()
    if show_2D==True:
        plt.show()
    if save_2D== True:
        if w_prior:
            plot_path = os.path.join(res_path, "2D_Plots_wPrior")
            if filter_values:
                plot_path = os.path.join(res_path, "2D_Plots_wBoth")
        elif filter_values:
            plot_path = os.path.join(res_path, "2D_Plots_wFilter")
        else:
            plot_path = os.path.join(res_path, "2D_Plots")
        os.makedirs(plot_path, exist_ok=True)
        if use_mean == True:
            plot_subpath = os.path.join(plot_path, "Mean")
        elif use_cornerMode == True:
            plot_subpath = os.path.join(plot_path, "CornerMode")
        elif use_cornerMean == True:
            plot_subpath = os.path.join(plot_path, "CornerMean")
        elif use_mode == True:
            plot_subpath = os.path.join(plot_path, "Modes")
        os.makedirs(plot_subpath, exist_ok=True)
        filename_parts = [f'2DPlot_Layer{j}']
        if scale_type == 'Log':
            filename_parts.append('Log')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(plot_subpath, filename))
        # fig.savefig(plot_subpath + r'\2DPlot_Layer{}.png'.format(j))
    plt.close()
def RidgePlots_Size(Layers,n_Ls, paths, data_dir,probe=probe, show_ridgePlotS=show_ridgePlotS, save_ridgePlotS=save_ridgePlotS,scale_type='Log'):
    model_dir = os.path.join(data_dir, 'ForwardModelMatrix_delta_001')
    X_matrix, Y_matrix, Isca_matrix = load_matrices_numpy(model_dir)
    x= np.unique(X_matrix.ravel())
    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")
    len_layers= len(Layers)
    ri_distributions = []
    ys_max = np.zeros(len_layers)
    ys_min = np.zeros(len_layers)
    for i, j in enumerate(Layers):
        ys_min[i] = array_MinAlt[j + 1]
        ys_max[i] = array_MaxAlt[j + 1]
        if use_mean == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_Mean.csv".format(j))
        elif use_cornerMode == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerModes.csv".format(j))
        elif use_cornerMean == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerMeans.csv".format(j))
        elif use_mode == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
        data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
        my_dis=data[0]
        if type_p == 'MultipleRI':
            s_dis = size_dis[j + 1]
            for n in range(n_Ls[i]):
                my_dis=s_dis[:3*(n+1)]
        else:
            s_dis=my_dis[:9]
            my_dis=s_dis
        my_dis=np.array(my_dis)
        # print(my_dis)
        x_dis= marginal_distribution(my_dis,x)
        ri_distributions.append(x_dis)
    ri_distributions = np.array(ri_distributions)
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(figSize_x, figSize_y))
    # Plot each array as a filled line plot with some overlap

    colors = plt.cm.rainbow(np.linspace(0, 1, len_layers))  # Generate different colors
    overlap_factor=1 #Default 1 or 0.5
    # offsets = np.linspace(0, 10*(len(data)-1), len(data))  # Offsets for each array
    offsets = np.linspace(0, overlap_factor*len_layers, len_layers+1)  # Offsets for 0.001

    # offsets = np.linspace(0, 50*(len(data)-1), len(data))  # Offsets for each array
    for i, arr in enumerate(ri_distributions):
        # if scale_type == 'Log':
        #     log_arr = np.log10(arr)
        #     arr = log_arr
        log_arr = np.log10(arr)
        arr = log_arr
        vmin, vmax = arr.min(), arr.max()  # Get min and max values for the array
        # print(vmin,vmax)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps['coolwarm']  # Choose a colormap
        scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        arr_norm = (arr - vmin) / (vmax - vmin)
        arr = arr_norm
        # Vectorize to make things faster:
        offset_val = offsets[len_layers - i]
        x_vals = x[:-1]  # Exclude last element to match arr_norm length
        x_vals_next = x[1:]
        y_bottom = np.full_like(x_vals, offset_val)
        y_top = arr[:-1] + offset_val
        y_top_next = arr[1:] + offset_val
        # for j, val in enumerate(arr):
        #     if j == len(arr) - 1:
        #         break
        #         # ax.fill_between([j, j + 1],[val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1]],[offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1]],color=scalarmap.to_rgba(val))
        #     ax.fill_between([x[j], x[j + 1]],
        #                     [val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1]],
        #                     [offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1]],
        #                     color=scalarmap.to_rgba((vmin + val) * (vmax - vmin)))
        # ax.plot(x, arr + offsets[len(offsets) - i - 1], color='black')
        for j in range(len(x_vals)):
            color_val = vmin + arr[j] * (vmax - vmin) if vmax > vmin else vmin
            ax.fill_between([x_vals[j], x_vals_next[j]],[y_top[j], y_top_next[j]],[y_bottom[j], y_bottom[j]],color=scalarmap.to_rgba(color_val))

            # Plot line once per layer
        ax.plot(x, arr + offset_val, color='black')
        max_Alt = format(ys_max[i], ".1f")
        min_Alt = format(ys_min[i], ".1f")
        if scale_type == 'Log':
            ax.text(0.08, offset_val, f' {max_Alt} - {min_Alt}km', fontweight="bold",color='black', ha="left", va="center", fontsize=fo)
        else:
            ax.text(-5.5, offset_val, f' {max_Alt} - {min_Alt}km', fontweight="bold", color='black', ha="left", va="center",fontsize=fo)

    # Adjust plot settings
    ax.grid(axis='x')
    if scale_type=='Log':
        ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=fo)
    ax.set_xlabel(r'Particle Size ($\mu$m)', fontsize=fo)
    # ax.set_ylabel('Value')
    ax.set(yticks=[], ylabel="")
    # ax.set_title('Sim Data: Known Particle Size')
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 10 RI and No LCPS')
    # ax.set_title('Sounder Probe R.I. Distribution per Layer with 1000 RI (all RI w Prior)')
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 100 RI (Mean)', fontsize=fo)
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 50 RI (Mean)')
    ax.set_title(f'{probe} Particle Size Distribution per Layer ', fontsize=fo)
    ax.legend(loc='upper right', fontsize=fo)
    fig.tight_layout()
    if show_ridgePlotS:
        plt.show()
    if save_ridgePlotS:
        if w_prior:
            plot_path = os.path.join(res_path, "RidgePlots_wPrior")
            if filter_values:
                plot_path = os.path.join(res_path, "RidgePlots_wBoth")
        elif filter_values:
            plot_path = os.path.join(res_path, "RidgePlots_wFilter")
        else:
            plot_path = os.path.join(res_path, "RidgePlots")
        os.makedirs(plot_path, exist_ok=True)
        filename_parts = [r'RidgePlot_size_Layers']
        if use_mean == True:
            filename_parts.append('wMean')
        elif use_cornerMode == True:
            filename_parts.append('wCornerMode')
        elif use_cornerMean == True:
            filename_parts.append('wCornerMean')
        elif use_mode == True:
            filename_parts.append('wMode')
        if scale_type == 'Log':
            filename_parts.append('Log')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(plot_path, filename))
    plt.close()
def RidgePlots_RI(Layers,n_Ls, paths, data_dir,probe=probe, show_ridgePlotRI=show_ridgePlotRI, save_ridgePlotRI=save_ridgePlotRI,scale_type=scale_type):
    model_dir = os.path.join(data_dir, 'ForwardModelMatrix_delta_001')
    X_matrix, Y_matrix, Isca_matrix = load_matrices_numpy(model_dir)
    x= np.unique(Y_matrix.ravel())
    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")
    len_layers = len(Layers)
    ri_distributions = []
    ys_max = np.zeros(len_layers)
    ys_min = np.zeros(len_layers)
    for i, j in enumerate(Layers):
        ys_min[i] = array_MinAlt[j + 1]
        ys_max[i] = array_MaxAlt[j + 1]
        if use_mean == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_Mean.csv".format(j))
        elif use_cornerMode == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerModes.csv".format(j))
        elif use_cornerMean == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerMeans.csv".format(j))
        elif use_mode == True:
            pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
        data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
        my_dis=data[0]
        if type_p == 'MultipleRI':
            s_dis = size_dis[j + 1]
            for n in range(n_Ls[i]):
                my_dis[n*3]=s_dis[n*3]
        else:
            s_dis=my_dis[9:]
            for n in range(3):
                s_dis[n*3]=my_dis[n*3]
            my_dis=s_dis
        x_dis= marginal_distribution(my_dis,x)
        ri_distributions.append(x_dis)
        # ri_distributions= np.vstack((ri_distributions,x_dis))
        # ys_max.append(y_max)
        # ys_min.append(y_min)
    # print(ri_distributions)
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(figSize_x, figSize_y))
    # Plot each array as a filled line plot with some overlap
    colors = plt.cm.rainbow(np.linspace(0, 1, len(Layers)))  # Generate different colors
    # offsets = np.linspace(0, 10*(len(data)-1), len(data))  # Offsets for each array
    offsets = np.linspace(0, len(Layers), len(Layers) + 1)  # Offsets for 0.001
    # offsets = np.linspace(0, 50*(len(data)-1), len(data))  # Offsets for each array
    for i, arr in enumerate(ri_distributions):
        if scale_type == 'Log':
            log_arr = np.log10(arr)
            arr = log_arr
        vmin, vmax = arr.min(), arr.max()  # Get min and max values for the array
        # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.colormaps['coolwarm']  # Choose a colormap
        scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        # normalized_arr = normalize(arr,vmin, vmax)
        # print(vmin, vmax)
        arr_norm = (arr - vmin) / (vmax - vmin)
        arr = arr_norm
        for j, val in enumerate(arr):
            if j == len(arr) - 1:
                break
                # ax.fill_between([j, j + 1],[val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1]],[offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1]],color=scalarmap.to_rgba(val))
            ax.fill_between([x[j], x[j + 1]],
                            [val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1]],
                            [offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1]],
                            color=scalarmap.to_rgba(val)) #color=scalarmap.to_rgba((vmin + val) * (vmax - vmin)))

            # ax.fill_between([x[j], x[j + 1]], [val + offsets[len(offsets) - i - 1], arr[j+1] + offsets[len(offsets) - i - 1]], [offsets[len(offsets) - i - 1],offsets[len(offsets) - i - 1]],color=scalarmap.to_rgba(val))
            # ax.fill_between([min(x),max(x)], val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1],color=colors[i],alpha=normalized_arr[j])
            # ax.fill_between([j], val + offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1],color=scalarmap.to_rgba(val))
            # ax.fill_between([j-0.5,j+0.5], val + offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1],color=scalarmap.to_rgba(val))
        # ax.fill_between(np.arange(len(arr)), arr + offsets[len(offsets)-i-1], offsets[len(offsets)-i-1], color=colors[i])
        ax.plot(x, arr + offsets[len(offsets) - i - 1], color='black')
        # ax.plot(np.arange(len(arr)), arr + offsets[len(offsets) - i - 1], color='black')
        # ax.text(-250,offsets[len(offsets)-i-1] , '{} km'.format(alts_test[i]), fontweight="bold", color=colors[i], ha="left", va="center")
        max_Alt = format(ys_max[i], ".1f")
        min_Alt = format(ys_min[i], ".1f")
        ax.text(1.00, offsets[len(offsets) - i - 1], ' {} - {}km'.format(max_Alt, min_Alt), fontweight="bold",
                color='black', ha="left", va="center", fontsize=fo)

    # Adjust plot settings
    # ax.grid(axis='x')
    ax.set_xticks(np.arange(1.30, 2.0, 0.1))
    ax.set_xticks(np.arange(1.30, 2.00, 0.01), minor=True)
    ax.grid(which='major', axis='x', alpha=0.8, linestyle='-', linewidth=1)
    ax.grid(which='minor', axis='x', alpha=0.4, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', labelsize=fo)
    ax.set_xlabel('Refractive Index', fontsize=fo)
    ax.axvline(x=1.44, color='purple', linestyle='--', linewidth=2, label='1.44', alpha=0.7)
    ax.axvline(x=1.33, color='green', linestyle='--', linewidth=2, label='1.33', alpha=0.7)
    # ax.set_ylabel('Value')
    ax.set(yticks=[], ylabel="")
    # ax.set_title('Sim Data: Known Particle Size')
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 10 RI and No LCPS')
    # ax.set_title('Sounder Probe R.I. Distribution per Layer with 1000 RI (all RI w Prior)')
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 100 RI (Mean)', fontsize=fo)
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 50 RI (Mean)')
    ax.set_title(f'{probe} R.I. Distribution per Layer ', fontsize=fo)
    ax.legend(loc='upper right', fontsize=fo)
    fig.tight_layout()
    if show_ridgePlotRI:
        plt.show()
    if save_ridgePlotRI:
        if w_prior:
            plot_path = os.path.join(res_path, "RidgePlots_wPrior")
            if filter_values:
                plot_path = os.path.join(res_path, "RidgePlots_wBoth")
        elif filter_values:
            plot_path = os.path.join(res_path, "RidgePlots_wFilter")
        else:
            plot_path = os.path.join(res_path, "RidgePlots")
        os.makedirs(plot_path, exist_ok=True)
        filename_parts = [r'RidgePlot_ri_Layers']
        if use_mean == True:
            filename_parts.append('wMean')
        elif use_cornerMode == True:
            filename_parts.append('wCornerMode')
        elif use_cornerMean == True:
            filename_parts.append('wCornerMean')
        elif use_mode == True:
            filename_parts.append('wMode')
        filename = '_'.join(filename_parts) + '.png'
        if scale_type == 'Log':
            filename_parts.append('Log')
        fig.savefig(os.path.join(plot_path, filename))
    plt.close()
def RidgePlots_Region(Layers,n_Ls, paths, data_dir,probe=probe, show_ridgePlotRegion=show_ridgePlotRegion, save_ridgePlotRegion=save_ridgePlotRegion,scale_type=scale_type):
    model_dir = os.path.join(data_dir, 'ForwardModelMatrix_delta_001')
    X_matrix, Y_matrix, Isca_matrix = load_matrices_numpy(model_dir)
    x= np.unique(Y_matrix.ravel())
    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    region_val = [[1.44], [1.38, 1.40], [1.33, 1.44], [1.44], [np.nan]]
    region_name = ['D', 'C', 'B', 'A', 'L']
    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")
    len_layers = len(Layers)
    ri_distributions = []
    region_used=[]
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]
        # print('Region:', region_name[r])
        region_ri =[]
        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            if y_max <= r_max and r_min <= y_min:
                if use_mean == True:
                    pos_dis = os.path.join(mode_paths, "PosDistLayer{}_Mean.csv".format(j))
                elif use_cornerMode == True:
                    pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerModes.csv".format(j))
                elif use_cornerMean == True:
                    pos_dis = os.path.join(mode_paths, "PosDistLayer{}_CornerMeans.csv".format(j))
                elif use_mode == True:
                    pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
                data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                my_dis=data[0]
                if type_p == 'MultipleRI':
                    s_dis = size_dis[j + 1]
                    for n in range(n_Ls[i]):
                        my_dis[n*3]=s_dis[n*3]
                else:
                    s_dis=my_dis[9:]
                    for n in range(3):
                        s_dis[n*3]=my_dis[n*3]
                    my_dis=s_dis
                region_ri.append(my_dis)
        X_padded = list(zip_longest(*region_ri, fillvalue=np.nan))
        region_ri = np.array(X_padded).T
        if len(region_ri)!=0:
            # region_ri = np.array(region_ri)
            # print(region_ri.shape)
            ri_dis=np.nanmean(region_ri, axis=0)
            # print(ri_dis)
            x_dis= marginal_distribution(ri_dis,x)
            ri_distributions.append(x_dis)
            region_used.append(r)
            # ri_distributions= np.vstack((ri_distributions,x_dis))
            # ys_max.append(y_max)
            # ys_min.append(y_min)

    region_used = np.array(region_used)
    region_val = [region_val[i] for i in region_used]
    region_name = [region_name[i] for i in region_used]

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(figSize_x, figSize_y))
    # Plot each array as a filled line plot with some overlap
    colors = plt.cm.rainbow(np.linspace(0, 1, len(region_name)))  # Generate different colors
    # offsets = np.linspace(0, 10*(len(data)-1), len(data))  # Offsets for each array
    offsets = np.linspace(0, len(region_name), len(region_name)+1)  # Offsets for 0.001
    # Debugging Code
    # for i, arr in enumerate(ri_distributions):
    #     if scale_type == 'Log':
    #         log_arr = np.log10(arr)
    #         arr = log_arr
    #     vmin, vmax = arr.min(), arr.max()
    #     arr = (arr - vmin) / (vmax - vmin)  # This makes arr range 0-1
    #     current_offset = offsets[len(offsets) - i - 1]
    #     y_ma = (offsets[len(offsets) - i - 1]) / (np.max(offsets))
    #     y_mi = (offsets[len(offsets) - i - 2]) / (np.max(offsets))
    #     print(f"Region {i}: offset={current_offset}, arr min={arr.min()}, arr max={arr.max()}")
    #     print(f"Plot will span from {current_offset} to {current_offset + arr.max()}")
    #     print(f"Offsets array: {offsets}")
    #     print(f"Lines for Region {i}: y_mi={y_mi} and y_ma={y_ma}")
    for i, arr in enumerate(ri_distributions):
        if scale_type == 'Log':
            log_arr = np.log10(arr)
            arr = log_arr
        vmin, vmax = arr.min(), arr.max()  # Get min and max values for the array
        # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cmap = plt.colormaps['coolwarm']  # Choose a colormap
        scalarmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        normalized_arr = normalize(arr,vmin, vmax)
        # print(vmin, vmax)
        arr_norm = (arr - vmin) / (vmax - vmin)
        arr=arr_norm
        for j, val in enumerate(arr):
            if j == len(arr) - 1:
                break
                # ax.fill_between([j, j + 1],[val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1]],[offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1]],color=scalarmap.to_rgba(val))
            ax.fill_between([x[j], x[j + 1]],
                            [val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1]],
                            [offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1]],
                            color=scalarmap.to_rgba(val)) #color=scalarmap.to_rgba((vmin + val) * (vmax - vmin)))

            # ax.fill_between([x[j], x[j + 1]], [val + offsets[len(offsets) - i - 1], arr[j+1] + offsets[len(offsets) - i - 1]], [offsets[len(offsets) - i - 1],offsets[len(offsets) - i - 1]],color=scalarmap.to_rgba(val))
            # ax.fill_between([min(x),max(x)], val + offsets[len(offsets) - i - 1], arr[j + 1] + offsets[len(offsets) - i - 1],color=colors[i],alpha=normalized_arr[j])
            # ax.fill_between([j], val + offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1],color=scalarmap.to_rgba(val))
            # ax.fill_between([j-0.5,j+0.5], val + offsets[len(offsets) - i - 1], offsets[len(offsets) - i - 1],color=scalarmap.to_rgba(val))
        # ax.fill_between(np.arange(len(arr)), arr + offsets[len(offsets)-i-1], offsets[len(offsets)-i-1], color=colors[i])
        ax.plot(x, arr + offsets[len(offsets) - i - 1], color='black')
        y_ma = (offsets[len(offsets) - i - 1]) / (np.max(offsets))
        y_mi = (offsets[len(offsets) - i - 2]) / (np.max(offsets))
        # Fixed vertical line positioning
        # current_offset = offsets[len(offsets) - i - 1]
        # total_height = np.max(offsets) + 1
        # y_mi = current_offset / total_height
        # y_ma = (current_offset - 1) / total_height
        for v in region_val[i]:
            if np.isnan(v):
                continue
            ax.axvline(x=v, ymin=y_mi, ymax=y_ma, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(1.05, offsets[len(offsets) - i - 1], 'Region: {}'.format(region_name[i]), fontweight="bold", color='black',
                ha="left", va="center", fontsize=fo)

    # Adjust plot settings
    ax.set_xticks(np.arange(1.30, 2.0, 0.1))
    ax.set_xticks(np.arange(1.30, 2.00, 0.01), minor=True)
    ax.grid(which='major', axis='x', alpha=0.8, linestyle='-', linewidth=1)
    ax.grid(which='minor', axis='x', alpha=0.4, linestyle='--', linewidth=0.5)
    # ax.grid(axis='x')
    ax.tick_params(axis='x', labelsize=fo)
    ax.set_xlabel('Refractive Index', fontsize=fo)
    # ax.set_ylabel('Value')
    ax.set(yticks=[], ylabel="")
    # ax.set_title('Sim Data: Known Particle Size')
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 10 RI and No LCPS')
    # ax.set_title('Sounder Probe R.I. Distribution per Layer with 1000 RI (all RI w Prior)')
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 100 RI (Mean)', fontsize=fo)
    # ax.set_title('Sounder Probe Refractive Index Distribution per Layer with 50 RI (Mean)')
    ax.set_title(f'{probe} R.I. Distribution per Region ', fontsize=fo)
    ax.legend(loc='upper right', fontsize=fo)
    fig.tight_layout()
    if show_ridgePlotRegion:
        plt.show()
    if save_ridgePlotRegion:
        if w_prior:
            plot_path = os.path.join(res_path, "RidgePlots_wPrior")
            if filter_values:
                plot_path = os.path.join(res_path, "RidgePlots_wBoth")
        elif filter_values:
            plot_path = os.path.join(res_path, "RidgePlots_wFilter")
        else:
            plot_path = os.path.join(res_path, "RidgePlots")
        os.makedirs(plot_path, exist_ok=True)
        filename_parts = [r'RidgePlot_ri_Region']
        if use_mean == True:
            filename_parts.append('wMean')
        elif use_cornerMode == True:
            filename_parts.append('wCornerMode')
        elif use_cornerMean == True:
            filename_parts.append('wCornerMean')
        elif use_mode == True:
            filename_parts.append('wMode')
        if scale_type == 'Log':
            filename_parts.append('Log')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(plot_path, filename))
    plt.close()
def my_histogram_Region(Layers, n_Ls, paths, data_dir, probe=probe, show_hist=show_hist, save_hist=save_hist,deltas=hist_deltas, my_bins=hist_bins, type_step=type_step, type_p=type_p, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, epsilon=epsilon):
    """
    Creates histograms for atmospheric regions by aggregating layers within each region.
    """
    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    # region_val = [[1.44], [1.38, 1.40], [1.33, 1.44], [1.44], [np.nan]]
    region_name = ['D', 'C', 'B', 'A', 'L']

    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")
    # Process each region
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]

        # Collect data from all layers within this region
        region_data = []
        region_weights = []
        region_labels = None
        # print(n_Ls)
        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            n_modes=n_Ls[i]
            # Check if layer falls within current region
            if y_max <= r_max and r_min <= y_min:
                if use_mean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_mode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
                    data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
                else:
                    dis_path = os.path.join(paths, "PosteriorDistributions")
                    pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                    if filter_values:
                        for f_what in filter_what:
                            if f_what == 'nums':
                                if type_p == 'MultipleRI':
                                    continue
                                else:
                                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                                    mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
                            if f_what == 'index':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                                    elif n_modes == 2:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (
                                                    filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                                    else:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (
                                                    filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & (
                                                           filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                                else:
                                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (
                                                filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & (
                                                       filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
                            if f_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (1 + filt_s >= data[:, 1])
                                    elif n_modes == 2:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                                    else:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & (
                                                    1 + filt_s >= data[:, 7])
                                else:
                                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & (
                                                1 + filt_s >= data[:, 16])

                            data = data[mask]
                            weights = weights[mask]
                            # print(len(data))
                    if w_prior:
                        for p, p_what in enumerate(prior_what):
                            sel_val = []
                            if p_what == 'index':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 1)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(11 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels,t_prior=type_Prior[p])
                            if p_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 2)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(10 + (3 * i))
                                # print(sel_val)
                                weights, name_prior = sigma_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                            if p_what == 'rho':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 3)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(9 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels,t_prior=type_Prior[p], m_val=prior_m_rho,unc_val=prior_unc_rho, low_val=-1, high_val=1)
                        # print(f'Applying Prior to: {sel_val}')
                        # weights, name_prior = singleValue_Prior(sel_val, data, weights, header)
                            if p_what == 'rho_diagonal':
                                # Apply diagonal prior to favor similar rho values
                                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights, val_labels=labels, n_modes=n_modes, type_p=type_p, sigma_diagonal=0.1, prior_min=-1.0, prior_max=1.0, prior_sig=3)
                            if p_what == 'num_total':
                                for i in range(n_modes):
                                    sel_val.append((3 * i))
                                weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                # data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                region_data.append(data)
                region_weights.append(weights)
                if region_labels is None:
                    region_labels = labels
                elif len(labels)>len(region_labels):
                    region_labels=labels

        # Skip if no layers in this region
        if len(region_data) == 0:
            continue

        # Combine data from all layers in region
        max_cols = max(data.shape[1] for data in region_data)

        # Pad all arrays to match max_cols
        padded_data = []
        for data in region_data:
            if data.shape[1] < max_cols:
                # Pad with NaN to preserve original values
                padding = np.full((data.shape[0], max_cols - data.shape[1]), np.nan)
                data = np.hstack([data, padding])
            padded_data.append(data)

        combined_data = np.vstack(padded_data)
        # combined_data = np.vstack(region_data)
        combined_weights = np.concatenate(region_weights)

        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # Determine subplot layout
        if type_p == 'MultipleRI':
            # num_rows = n_modes
            num_rows=int(max_cols/3)
            num_cols = 3
        else:
            num_rows = 2
            num_cols = 9

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(figSize_x * num_cols, figSize_y * num_rows))
        axs = axs.flatten()

        # Create histogram for each variable
        for var_idx in range(len(combined_data[0])):
            array_val = combined_data[:, var_idx]
            array_w = combined_weights

            # Remove NaN values and corresponding weights
            valid_mask = ~np.isnan(array_val)
            array_val = array_val[valid_mask]
            array_w = array_w[valid_mask]

            # Skip if no valid data for this variable
            if len(array_val) == 0:
                axs[var_idx].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axs[var_idx].transAxes)
                continue

            low_val = np.min(array_val)
            high_val = np.max(array_val)

            if mirror_data:
                array_val = np.concatenate((array_val, 2 * low_val - array_val))
                array_w = np.concatenate((array_w, array_w))

            # Determine bin size based on variable type
            if type_step == 'changing_d':
                if type_p == 'MultipleRI':
                    if var_idx in [0, 3, 6]:  # Correlation
                        delta_val = 0.01
                        s_hist = 2
                    else:  # R.I.
                        delta_val = deltas
                        s_hist = sig_hist
                else:
                    if var_idx in [0, 3, 6]:  # number density
                        delta_val = 5
                        s_hist = 0
                    elif var_idx < 9:  # particle size
                        delta_val = 0.01
                        s_hist = 2
                    elif var_idx in [9, 12, 15]:  # correlation
                        delta_val = 0.01
                        s_hist = 2
                    else:  # R.I.
                        delta_val = deltas
                        s_hist = sig_h
                val_bins = num_bins(array_val, delta_val)
            elif type_step == 'bins':
                val_bins = my_bins * (2 if mirror_data else 1)
                delta_val = (np.max(array_val) - np.min(array_val)) / val_bins
                s_hist = int(-1 * np.log10(delta_val))
            else:
                delta_val = deltas
                val_bins = num_bins(array_val, delta_val)
            if use_log_bins:
                if type_p == 'MultipleRI':
                    pass
                elif var_idx in [0, 3, 6]:
                    # Add small epsilon to avoid log(0)

                    # ri_bins = np.logspace(np.log10(max(array_ri.min(), low_ri) + epsilon),
                    #                       np.log10(array_ri.max() + epsilon),
                    #                       num_bins(array_ri, delta_ri))
                    val_bins = np.logspace(np.log10(max(array_val.min(), 0.9) + epsilon),
                                           np.log10(high_val + epsilon),
                                           500)
            # Create KDE
            # x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=array_w)()
            x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(array_val, weights=array_w)()
            if mirror_data:
                mask = (x_rs < low_val) | (high_val < x_rs)
                y_rs[mask] = 0
            if show_peaks == True:
                mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                y_rsP = y_rs[mask_P]
                x_rsP = x_rs[mask_P]
                peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                if len(peaks_idr) > num_peaks:
                    peaks_idr = peaks_idr[:num_peaks]
                # colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                colors = plt.cm.spring(np.linspace(0, 1, len(peaks_idr)))
                for d, p in enumerate(peaks_idr):
                    peak_r = x_rsP[p]
                    axs[var_idx].axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')
            max_xi = np.argmax(y_rs)
            max_x = x_rs[max_xi]

            # Plot
            axs[var_idx].hist(array_val, weights=array_w, bins=val_bins, density=True, alpha=0.7)
            # axs[var_idx].plot(x_rs, y_rs, color='black', linewidth=2, label=f'Fitted Curve  (bw={delta_val:.3f})')
            axs[var_idx].plot(x_rs, y_rs, color='black', linewidth=2, label=f'Fitted Curve  (ISJ)')
            axs[var_idx].axvline(x=max_x, color='red', linestyle='--', linewidth=2, label=f'FFTKDE: {max_x:.{s_hist}f}')

            axs[var_idx].set_title(region_labels[var_idx], fontsize=fo)
            axs[var_idx].set_xlabel(region_labels[var_idx], fontsize=fo)
            axs[var_idx].set_xlim((low_val, high_val))
            if scale_type == 'Log':
                if type_p == 'MultipleRI':
                    pass
                elif var_idx in [0, 3, 6]:
                    l_base = 10
                    axs[var_idx].set_xlim((0.9, high_val))
                    axs[var_idx].set_xscale('log', base=l_base)
                    axs[var_idx].set_yscale('log', base=l_base)
                    axs[var_idx].set_ylabel(f'Log{l_base} (PDF)')
            else:
                axs[var_idx].set_ylabel('PDF', fontsize=fo)
            # axs[var_idx].set_ylabel('PDF', fontsize=fo)
            axs[var_idx].tick_params(axis='both', labelsize=fo)
            axs[var_idx].legend(fontsize=fo)

        fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) - Combined Histogram',fontsize=fo + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) #[Left, Bottom, Right, Top]

        if save_hist_region:
            if w_prior:
                hist_paths = os.path.join(res_path, "Histograms_Region_wPrior")
                if filter_values:
                    hist_paths = os.path.join(res_path, "Histograms_Region_wBoth")
            elif filter_values:
                hist_paths = os.path.join(res_path, "Histograms_Region_wFilter")
            else:
                hist_paths = os.path.join(res_path, "Histograms_Region")
            os.makedirs(hist_paths, exist_ok=True)
            if use_mean == True:
                plot_subpath = os.path.join(hist_paths, "Mean")
            elif use_cornerMode == True:
                plot_subpath = os.path.join(hist_paths, "CornerMode")
            elif use_cornerMean == True:
                plot_subpath = os.path.join(hist_paths, "CornerMean")
            elif use_mode == True:
                plot_subpath = os.path.join(hist_paths, "Modes")
            else:
                plot_subpath = os.path.join(hist_paths, "Raw")
            os.makedirs(plot_subpath, exist_ok=True)
            if scale_type == 'Log':
                fig.savefig(os.path.join(plot_subpath, f'Hist_Region_{region_name[r]}_Log.png'))
            else:
                fig.savefig(os.path.join(plot_subpath, f'Hist_Region_{region_name[r]}.png'))

        if show_hist_region:
            plt.show()

        plt.close('all')
def my_corner_Region(Layers, n_Ls, paths, data_dir, probe=probe, show_corner=show_cornerBP_region, save_corner=save_cornerBPlot_region, deltas=corner_deltas, type_step=type_step, type_p=type_p, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, n_con=num_cont, bws=bws, scale_t=scale_type, l_base=log_base, my_colorMap=my_colorMap, show_prob=probs, corner_bins=corner_bins,log_bin_indices = [0, 3, 6], epsilon=epsilon,smooth_corner=smooth_corner,smooth_val=smooth_val,show_q=show_quantiles,q_min=q_min):
    """
    Creates corner plots for atmospheric regions by aggregating layers within each region.
    Shows 2D joint distributions and 1D marginal distributions for all variable pairs.
    """
    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    region_name = ['D', 'C', 'B', 'A', 'L']

    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")

    # Process each region
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]

        # Collect data from all layers within this region
        region_data = []
        region_weights = []
        region_labels = None

        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            n_modes = n_Ls[i]

            # Check if layer falls within current region
            if y_max <= r_max and r_min <= y_min:
                if use_mean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_mode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
                    data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
                else:
                    dis_path = os.path.join(paths, "PosteriorDistributions")
                    pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)

                    # Apply filtering if needed
                    if filter_values:
                        for f_what in filter_what:
                            if f_what == 'nums':
                                if type_p != 'MultipleRI':
                                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                                    mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
                            elif f_what == 'index':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                                    elif n_modes == 2:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & \
                                               (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                                    else:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & \
                                               (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & \
                                               (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                                else:
                                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & \
                                           (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & \
                                           (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
                            elif f_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (1 + filt_s >= data[:, 1])
                                    elif n_modes == 2:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                                    else:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & \
                                               (1 + filt_s >= data[:, 7])
                                else:
                                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & \
                                           (1 + filt_s >= data[:, 16])
                            data = data[mask]
                            weights = weights[mask]

                    # Apply priors if needed
                    if w_prior:
                        for p, p_what in enumerate(prior_what):
                            sel_val = []
                            if p_what == 'index':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 1)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(11 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p])
                            elif p_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 2)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(10 + (3 * i))
                                weights, name_prior = sigma_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                            elif p_what == 'rho':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 3)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(9 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p], m_val=prior_m_rho, unc_val=prior_unc_rho, low_val=-1, high_val=1)
                            elif p_what == 'rho_diagonal':
                                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights, val_labels=labels, n_modes=n_modes, type_p=type_p, sigma_diagonal=0.1, prior_min=-1.0, prior_max=1.0, prior_sig=3)
                            elif p_what == 'num_total':
                                for i in range(n_modes):
                                    sel_val.append((3 * i))
                                weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])

                region_data.append(data)
                region_weights.append(weights)
                if region_labels is None:
                    region_labels = labels
                elif len(labels) > len(region_labels):
                    region_labels = labels

        # Skip if no layers in this region
        if len(region_data) == 0:
            continue

        # Combine data from all layers in region
        max_cols = max(data.shape[1] for data in region_data)

        # Pad all arrays to match max_cols
        padded_data = []
        for data in region_data:
            if data.shape[1] < max_cols:
                padding = np.full((data.shape[0], max_cols - data.shape[1]), np.nan)
                data = np.hstack([data, padding])
            padded_data.append(data)

        combined_data = np.vstack(padded_data)
        combined_weights = np.concatenate(region_weights)

        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # Determine number of variables
        n_vars = combined_data.shape[1]

        # Create corner plot
        fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

        # Plot for each pair of variables
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axs[i, j]

                # Lower triangle: 2D joint distributions
                if i > j:
                    # print(i, j)
                    array_x = combined_data[:, j]
                    array_y = combined_data[:, i]
                    array_w = combined_weights.copy()

                    # Remove NaN values
                    valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                    array_x = array_x[valid_mask]
                    array_y = array_y[valid_mask]
                    array_w = array_w[valid_mask]

                    if len(array_x) == 0:
                        ax.axis('off')
                        continue

                    low_x, high_x = np.min(array_x), np.max(array_x)
                    low_y, high_y = np.min(array_y), np.max(array_y)
                    if use_log_bins:
                        if type_p == 'MultipleRI':
                            use_log_x = False
                            use_log_y = False

                        else:
                            use_log_x = use_log_bins and (j in log_bin_indices)
                            use_log_y = use_log_bins and (i in log_bin_indices)
                    else:
                        use_log_x = False
                        use_log_y = False
                    if use_log_x or use_log_y:

                        # Handle x-axis log binning
                        if use_log_x:
                            # Ensure all values are positive before log transform
                            # if low_x < 1:
                            #     low_x = epsilon
                            # array_x = np.maximum(array_x, epsilon)  # Clip to minimum epsilon
                            #
                            # # Transform x data to log space BEFORE mirroring
                            # array_x = np.log10(array_x)
                            # low_x = np.log10(low_x)
                            # high_x = np.log10(np.max(array_x))
                            # clipped_x = np.where(array_x < 1, epsilon, array_x)
                            # array_x = np.log10(clipped_x)
                            if show_q:
                                q_max = 1 - q_min

                                sort_idx = np.argsort(array_x)
                                x_sorted = array_x[sort_idx]
                                weights_sortedX = array_w[sort_idx]

                                # Calculate cumulative sum of weights
                                cdf_x = np.cumsum(weights_sortedX)
                                qmin_idx = np.searchsorted(cdf_x, q_min)
                                if qmin_idx < len(x_sorted):
                                    low_x = x_sorted[qmin_idx]
                                qmax_idx = np.searchsorted(cdf_x, q_max)
                                if qmax_idx < len(x_sorted):
                                    high_x = x_sorted[qmax_idx]
                            # print('X log range:', low_x, high_x)
                            # log_bins=(1,3, corner_bins)
                            # x_bins=np.concatenate([0,1],log_bins)
                            x_bins = np.logspace(np.log10(low_x), np.log10(high_x), corner_bins + 1)
                            # print(x_bins)
                        else:
                            # x_bins = corner_bins
                            x_bins = np.linspace(np.min(array_x), np.max(array_x), corner_bins + 1)

                            # Handle y-axis log binning
                        if use_log_y:
                            # # Ensure all values are positive before log transform
                            # if low_y < 1:
                            #     low_y = epsilon
                            # array_y = np.maximum(array_y, epsilon)  # Clip to minimum epsilon
                            #
                            # # Transform y data to log space BEFORE mirroring
                            # array_y = np.log10(array_y)
                            # low_y = np.log10(low_y)
                            # high_y = np.log10(np.max(array_y))
                            # clipped_y = np.where(array_y < 1, epsilon, array_y)
                            # array_y = np.log10(clipped_y)
                            if show_q:
                                q_max = 1 - q_min

                                sort_idy = np.argsort(array_y)
                                y_sorted = array_y[sort_idy]
                                weights_sortedY = array_w[sort_idy]

                                # Calculate cumulative sum of weights
                                cdf_y = np.cumsum(weights_sortedY)
                                qmin_idy = np.searchsorted(cdf_y, q_min)
                                if qmin_idy < len(y_sorted):
                                    low_y = y_sorted[qmin_idy]
                                qmax_idy = np.searchsorted(cdf_y, q_max)
                                if qmax_idy < len(y_sorted):
                                    high_y = y_sorted[qmax_idy]
                            # log_bins = (1, 3, corner_bins)
                            # y_bins = np.concatenate([0, 1], log_bins)
                            # print('Y log range:', low_y, high_y)
                            y_bins = np.logspace(np.log10(low_y), np.log10(high_y), corner_bins + 1)
                        else:
                            # y_bins = corner_bins
                            y_bins = np.linspace(np.min(array_y), np.max(array_y), corner_bins + 1)
                        # Trying to use a KDE
                        # X_mesh, Y_mesh = np.meshgrid(x_bins, y_bins)
                        # data_stack=np.vstack((array_x, array_y))
                        # new_kde=gaussian_kde(data_stack, weights=array_w,bw_method='silverman')
                        # if use_log_x:
                        #     bw_x=10**(np.std(np.log10(array_x)))
                        # else:
                        #     bw_x= np.std(array_x)
                        # if use_log_y:
                        #     bw_y=10**(np.std(np.log10(array_y)))
                        # else:
                        #     bw_y=np.std(array_y)
                        # if use_log_x or use_log_y:
                        #     n=len(array_x)
                        #     d=2
                        #     new_kde.set_bandwidth(bw_method=new_kde.silverman_factor()*np.array([bw_x, bw_y]).mean())
                        # positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
                        # f= new_kde(positions).reshape(X_mesh.shape)
                        # X = x_bins
                        # Y = y_bins
                        #
                        # max_indices = np.unravel_index(np.argmax(f), f.shape)
                        # max_x = X[max_indices[1]]
                        # max_y = Y[max_indices[0]]
                        # Create 2D histogram
                        # H, xedges, yedges = np.histogram2d(array_x, array_y,bins=[x_bins, y_bins],weights=array_w,density=True)
                        H, xedges, yedges = np.histogram2d(array_x, array_y, bins=[x_bins, y_bins], weights=array_w,density=True)

                        # Transpose H for correct orientation
                        H = H.T

                        # Convert to density (comparable to KDE output)
                        # dx = np.diff(xedges)
                        # dy = np.diff(yedges)
                        # bin_areas = dx[:, None] * dy[None, :]
                        # tot_h = np.sum(H)
                        # H = H / (tot_h * bin_areas)
                        # smooth
                        if smooth_corner:
                            H = gaussian_filter(H, sigma=smooth_val)

                        # Get bin centers for plotting
                        X = (xedges[:-1] + xedges[1:]) / 2
                        Y = (yedges[:-1] + yedges[1:]) / 2

                        # Find maximum for mode
                        max_indices = np.unravel_index(np.argmax(H), H.shape)
                        max_x = X[max_indices[1]]
                        max_y = Y[max_indices[0]]

                        # Apply log scale if needed
                        f = H.copy()
                        # if scale_t == 'Log':
                        #     #     Avoid log(0) by adding small epsilon where H is zero
                        #     f = np.where(f > 0, np.log(f) / np.log(l_base), np.nan)

                        # Plot contour
                        X_mesh, Y_mesh = np.meshgrid(X, Y)
                        cfset = ax.contourf(X_mesh, Y_mesh, f, 100, cmap=my_colorMap)
                        # cset = ax.contour(X_mesh, Y_mesh, f, n_con, colors='k')
                        # Use logarithmically spaced levels:
                        # if scale_t == 'Log':
                        #     # f is already in log space, so use linear spacing
                        #     levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                        # else:
                        # f is in linear space, so use log spacing
                        f_max = np.log10(np.max(f))
                        f_min = np.min(f)
                        f_proposed = f_max - 3
                        if f_min == 0:
                            f_selected = f_proposed
                        else:
                            f_selected = np.max([np.log10(f_min), f_proposed])

                        # print(f_min, f_proposed, f_selected)
                        # levels = np.logspace(f_max-4, f_max, 10)
                        levels = np.linspace(f_selected, f_max, n_con)
                        levels = 10 ** levels
                            # print(np.log10(f_min), np.log10(f_max))
                            # levels = np.logspace(np.log10(f_min), np.log10(f_max), n_con)
                            # percentiles = np.linspace(1, 99, n_con)  # Skip extreme tails
                            # percents=100-percentiles
                            # print(percentiles)
                            # print(percents)
                            # levels = np.percentile(f, percentiles)
                        # print(levels)
                        cset = ax.contour(X, Y, f, levels=levels, colors='k')
                        ax.clabel(cset, inline=1, fontsize=8)

                        # Set axis limits and scales
                        if use_log_x:
                            ax.set_xscale('log')
                            ax.set_xlim(low_x, high_x)
                        else:
                            ax.set_xlim(low_x, high_x)

                        if use_log_y:
                            ax.set_yscale('log')
                            ax.set_ylim(low_y, high_y)
                        else:
                            ax.set_ylim(low_y, high_y)

                    # Mirror data if needed
                    else:
                        if mirror_data:
                            array_x = np.concatenate((array_x, 2 * low_x - array_x))
                            array_y = np.concatenate((array_y, 2 * low_y - array_y))
                            array_w = np.concatenate((array_w, array_w))
                            # print('Entering Mirror')
                        # Recalculate ranges after mirroring
                        delta_x = deltas if type_step != 'bins' else (high_x - low_x) / corner_bins
                        delta_y = deltas if type_step != 'bins' else (high_y - low_y) / corner_bins
                        # print('X range:', low_x, high_x)
                        # print('Y range:', low_y, high_y)
                        # print('Deltas [x,y]: ', delta_x,delta_y)
                        # print(array_x)
                        # print(array_y)
                        if delta_x > delta_y:
                            bws = delta_x
                            scal_fac = delta_x / delta_y
                            array_y = array_y * scal_fac
                        elif delta_x < delta_y:
                            scal_fac = delta_y / delta_x
                            bws = delta_y
                            array_x = array_x * scal_fac
                        else:
                            bws = delta_x

                        x_bins = num_bins(array_x, delta_x)
                        y_bins = num_bins(array_y, delta_y)
                        # print(i,j)
                        # print('# of bins:', corner_bins, type_step)
                        # print('bins [x,y]: ',x_bins,y_bins)
                        # Create 2D KDE
                        [values] = np.dstack([array_x, array_y])
                        grid_points_x, grid_points_y = x_bins, y_bins
                        kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                        grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))

                        X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                        f = points.reshape(grid_points_x, grid_points_y).T

                        # Zero out values outside domain
                        xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                        ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                        if delta_x > delta_y:
                            ys = ys / scal_fac
                        elif delta_x < delta_y:
                            xs = xs / scal_fac
                        for ii in range(grid_points_y):
                            for jj in range(grid_points_x):
                                if high_x < xs[ii][jj] or xs[ii][jj] < low_x or \
                                        high_y < ys[ii][jj] or ys[ii][jj] < low_y:
                                    f[ii][jj] = 0
                                elif mirror_data:
                                    f[ii][jj] = f[ii][jj] * 2

                        # Find maximum
                        max_value = np.max(f)
                        max_indices = np.unravel_index(np.argmax(f), f.shape)
                        if delta_x > delta_y:
                            Y = Y / scal_fac
                        elif delta_x < delta_y:
                            X = X / scal_fac
                        if use_log_x or use_log_y:
                            if use_log_x:
                                X = 10 ** X
                            if use_log_y:
                                Y = 10 ** Y
                        max_x = X[max_indices[1]]
                        max_y = Y[max_indices[0]]

                        # Apply log scale if needed
                        if scale_t == 'Log':  # or use_log_x or use_log_y:
                            f = np.log(f) / np.log(l_base)

                        # Plot contour
                        cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                        # cset = ax.contour(X, Y, f, n_con, colors='k')
                        # if scale_t == 'Log':
                        #     # f is already in log space, so use linear spacing
                        #     levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                        # else:
                        # f is in linear space, so use log spacing
                        f_max = np.log10(np.max(f))
                        # levels = np.logspace(f_max-4, f_max, 10)
                        f_min = np.min(f)
                        f_proposed = f_max - 3
                        if f_min <= 0:
                            f_selected = f_proposed
                        else:
                            print(f_min)
                            f_selected = np.max([np.log10(f_min), f_proposed])

                        # print(f_min, f_proposed, f_selected)
                        # levels = np.logspace(f_max-4, f_max, 10)
                        levels = np.linspace(f_selected, f_max, n_con)
                        levels = 10 ** levels
                        cset = ax.contour(X, Y, f, levels=levels, colors='k')
                        ax.clabel(cset, inline=1, fontsize=8)
                        ax.set_xlim(low_x, high_x)
                        ax.set_ylim(low_y, high_y)

                    # ADD GRIDLINES HERE
                    ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                    # Mark mode
                    sig_c = int(abs(np.log10(bws)))
                    ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5, label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, region_labels[j]))

                    ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5, label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, region_labels[i]))

                    if i == n_vars - 1:
                        ax.set_xlabel(region_labels[j], fontsize=fo)
                    if j == 0:
                        ax.set_ylabel(region_labels[i], fontsize=fo)
                    ax.legend(fontsize=fo)

                # Diagonal: 1D marginal distributions
                elif i == j:
                    array_val = combined_data[:, i]
                    array_w = combined_weights.copy()

                    # Remove NaN values
                    valid_mask = ~np.isnan(array_val)
                    array_val = array_val[valid_mask]
                    array_w = array_w[valid_mask]

                    if len(array_val) == 0:
                        ax.axis('off')
                        continue

                    # NEW: Apply log binning if enabled for this variable
                    if use_log_bins and (i in log_bin_indices)and type_p == 'NoLCPS':
                        # clipped_val = np.where(array_val < 1, epsilon, array_val)
                        # array_val = clipped_val
                        if show_q:
                            q_max = 1 - q_min

                            sort_idx = np.argsort(array_val)
                            x_sorted = array_val[sort_idx]
                            weights_sortedX = array_w[sort_idx]

                            # Calculate cumulative sum of weights
                            cdf_x = np.cumsum(weights_sortedX)
                            qmin_idx = np.searchsorted(cdf_x, q_min)
                            if qmin_idx < len(x_sorted):
                                low_val = x_sorted[qmin_idx]
                            qmax_idx = np.searchsorted(cdf_x, q_max)
                            if qmax_idx < len(x_sorted):
                                high_val = x_sorted[qmax_idx]
                        # log_bins=np.logspace(0,3, corner_bins)
                        # val_bins=np.concatenate([[0.9,1],log_bins])
                        delta_val = 1
                        val_bins = np.logspace(np.log10(low_val), np.log10(high_val), corner_bins)
                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(np.log10(array_val), weights=array_w)()
                        x_rs = 10 ** x_rs
                        y_rs = y_rs / (x_rs * np.log(10))

                        # # Handle zero/negative values
                        # if low_val <= 0:
                        #     low_val = epsilon
                        #
                        # # Create log-spaced bins
                        # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                    else:
                        low_val = np.min(array_val)
                        high_val = np.max(array_val)
                        delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                        val_bins = num_bins(array_val, delta_val)

                        if mirror_data:
                            array_valM = np.concatenate((array_val, 2 * low_val - array_val))
                            array_wM = np.concatenate((array_w, array_w))

                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_valM, weights=array_wM)()
                        if mirror_data:
                            mask = (x_rs < low_val) | (high_val < x_rs)
                            y_rs[mask] = 0
                            y_rs = y_rs * 2
                    sig_val = int(abs(np.log10(delta_val)))

                    # Find peaks
                    if show_peaks:
                        mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                        y_rsP = y_rs[mask_P]
                        x_rsP = x_rs[mask_P]
                        peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                        peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                        if len(peaks_idr) > num_peaks:
                            peaks_idr = peaks_idr[:num_peaks]
                        colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                        for d, p in enumerate(peaks_idr):
                            peak_r = x_rsP[p]
                            ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                    # Find mode
                    max_xi = np.argmax(y_rs)
                    max_x = x_rs[max_xi]

                    # Plot
                    ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                    ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                    ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0, label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                    # ADD VERTICAL GRIDLINES HERE
                    ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                    # ax.set_xlabel(region_labels[i], fontsize=fo)
                    ax.set_title(region_labels[i], fontsize=fo)
                    if use_log_bins and (i in log_bin_indices) and type_p == 'NoLCPS':
                        ax.set_xscale('log')
                        ax.set_xlim(low_val, high_val)
                        # low_yrs = np.min(y_rs)
                        # yrs_proposed = np.log10(np.max(y_rs)) - 3
                        # if low_yrs == 0:
                        #     sel_pdf = yrs_proposed
                        # else:
                        #     sel_pdf = np.max([np.log10(low_yrs), yrs_proposed])
                        # sel_pdf = 10 ** sel_pdf
                        # print(sel_pdf)
                        # ax.set_ylim(bottom=sel_pdf)
                        ax.set_ylim(bottom=low_pdf)
                        ax.set_yscale('log', base=l_base)
                        ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                    else:
                        ax.set_xlim(low_val, high_val)

                        if scale_t == 'Log':
                            ax.set_yscale('log', base=l_base)
                            ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        else:
                            ax.set_ylabel('PDF', fontsize=fo)

                    ax.legend(fontsize=fo)

                # Upper triangle: hide
                else:
                    ax.axis('off')
                ax.minorticks_on()
                ax.tick_params(axis='both', labelsize=fo)

        fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) - Corner Plot',
                     fontsize=fo + 4)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        if save_corner:
            if w_prior:
                corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wPrior")
                if filter_values:
                    corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wBoth")
            elif filter_values:
                corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wFilter")
            else:
                corner_paths = os.path.join(res_path, "MyCornerPlots_Region")
            os.makedirs(corner_paths, exist_ok=True)

            if use_mean:
                plot_subpath = os.path.join(corner_paths, "Mean")
            elif use_cornerMode:
                plot_subpath = os.path.join(corner_paths, "CornerMode")
            elif use_cornerMean:
                plot_subpath = os.path.join(corner_paths, "CornerMean")
            elif use_mode:
                plot_subpath = os.path.join(corner_paths, "Modes")
            else:
                plot_subpath = os.path.join(corner_paths, "Raw")
            os.makedirs(plot_subpath, exist_ok=True)

            if scale_t == 'Log':
                fig.savefig(os.path.join(plot_subpath,f'Corner_Region_{region_name[r]}_Log.png'))
            elif use_log_bins:
                fig.savefig(os.path.join(plot_subpath,f'Corner_Region_{region_name[r]}_LBins.png'))
            else:
                fig.savefig(os.path.join(plot_subpath,f'Corner_Region_{region_name[r]}.png'))

        if show_corner:
            plt.show()

        plt.close('all')
def my_cornerM_Region(Layers, n_Ls, paths, data_dir, probe=probe, show_corner=show_cornerMP_region, save_corner=save_cornerMPlot_region, deltas=corner_deltas, type_step=type_step, type_p=type_p, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, n_con=num_cont, bws=bws, scale_t=scale_type, l_base=log_base, my_colorMap=my_colorMap, show_prob=probs, corner_bins=corner_bins,log_bin_indices = [0, 3, 6], epsilon=epsilon,smooth_corner=smooth_corner,smooth_val=smooth_val,show_q=show_quantiles,q_min=q_min):
    """
    Creates corner plots for atmospheric regions by aggregating layers within each region.
    Shows 2D joint distributions and 1D marginal distributions for all variable pairs.
    """
    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    region_name = ['D', 'C', 'B', 'A', 'L']

    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")

    # Process each region
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]

        # Collect data from all layers within this region
        region_data = []
        region_weights = []
        region_labels = None

        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            n_modes = n_Ls[i]

            # Check if layer falls within current region
            if y_max <= r_max and r_min <= y_min:
                if use_mean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_mode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
                    data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
                else:
                    dis_path = os.path.join(paths, "PosteriorDistributions")
                    pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)

                    # Apply filtering if needed
                    if filter_values:
                        for f_what in filter_what:
                            if f_what == 'nums':
                                if type_p != 'MultipleRI':
                                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                                    mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
                            elif f_what == 'index':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                                    elif n_modes == 2:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & \
                                               (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                                    else:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & \
                                               (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & \
                                               (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                                else:
                                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & \
                                           (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & \
                                           (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
                            elif f_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (1 + filt_s >= data[:, 1])
                                    elif n_modes == 2:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                                    else:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & \
                                               (1 + filt_s >= data[:, 7])
                                else:
                                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & \
                                           (1 + filt_s >= data[:, 16])
                            data = data[mask]
                            weights = weights[mask]

                    # Apply priors if needed
                    if w_prior:
                        for p, p_what in enumerate(prior_what):
                            sel_val = []
                            if p_what == 'index':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 1)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(11 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p])
                            elif p_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 2)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(10 + (3 * i))
                                weights, name_prior = sigma_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                            elif p_what == 'rho':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 3)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(9 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p], m_val=prior_m_rho, unc_val=prior_unc_rho, low_val=-1, high_val=1)
                            elif p_what == 'rho_diagonal':
                                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights, val_labels=labels, n_modes=n_modes, type_p=type_p, sigma_diagonal=0.1, prior_min=-1.0, prior_max=1.0, prior_sig=3)
                            elif p_what == 'num_total':
                                for i in range(n_modes):
                                    sel_val.append((3 * i))
                                weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])

                region_data.append(data)
                region_weights.append(weights)
                if region_labels is None:
                    region_labels = labels
                elif len(labels) > len(region_labels):
                    region_labels = labels

        # Skip if no layers in this region
        if len(region_data) == 0:
            continue

        # Combine data from all layers in region
        max_cols = max(data.shape[1] for data in region_data)
        if type_p == 'MultipleRI':
            n_modes=int(max_cols/3)
        else:
            n_modes=3
        # Pad all arrays to match max_cols
        padded_data = []
        for data in region_data:
            if data.shape[1] < max_cols:
                padding = np.full((data.shape[0], max_cols - data.shape[1]), np.nan)
                data = np.hstack([data, padding])
            padded_data.append(data)

        combined_data = np.vstack(padded_data)
        combined_weights = np.concatenate(region_weights)

        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)
        # n_dec = [0, 2, 3]
        # Determine number of variables
        for n in range(n_modes):
            if n == 0:
                columns = [0, 1, 2, 9, 10, 11]
            elif n == 1:
                columns = [3, 4, 5, 12, 13, 14]
            elif n == 2:
                columns = [6, 7, 8, 15, 16, 17]

            if type_p == 'MultipleRI':
                columns= columns[:3]
            n_data = combined_data[:, columns]
            n_weights = combined_weights
            n_labels = region_labels[columns]
            # Determine number of variables
            n_vars = n_data.shape[1]

            # Create corner plot
            fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

            # Plot for each pair of variables
            for i in range(n_vars):
                for j in range(n_vars):
                    ax = axs[i, j]

                    # Lower triangle: 2D joint distributions
                    if i > j:
                        # print(i, j)
                        array_x = n_data[:, j]
                        array_y = n_data[:, i]
                        array_w = n_weights.copy()

                        # Remove NaN values
                        valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                        array_x = array_x[valid_mask]
                        array_y = array_y[valid_mask]
                        array_w = array_w[valid_mask]

                        if len(array_x) == 0:
                            ax.axis('off')
                            continue

                        low_x, high_x = np.min(array_x), np.max(array_x)
                        low_y, high_y = np.min(array_y), np.max(array_y)
                        if use_log_bins:
                            if type_p == 'MultipleRI':
                                use_log_x = False
                                use_log_y = False

                            else:
                                use_log_x = use_log_bins and j ==0
                                use_log_y = use_log_bins and i ==0
                        else:
                            use_log_x = False
                            use_log_y = False
                        if use_log_x or use_log_y:

                            # Handle x-axis log binning
                            if use_log_x:
                                # Ensure all values are positive before log transform
                                # if low_x < 1:
                                #     low_x = epsilon
                                # array_x = np.maximum(array_x, epsilon)  # Clip to minimum epsilon
                                #
                                # # Transform x data to log space BEFORE mirroring
                                # array_x = np.log10(array_x)
                                # low_x = np.log10(low_x)
                                # high_x = np.log10(np.max(array_x))
                                # clipped_x = np.where(array_x < 1, epsilon, array_x)
                                # array_x = np.log10(clipped_x)
                                if show_q:
                                    q_max = 1 - q_min

                                    sort_idx = np.argsort(array_x)
                                    x_sorted = array_x[sort_idx]
                                    weights_sortedX = array_w[sort_idx]

                                    # Calculate cumulative sum of weights
                                    cdf_x = np.cumsum(weights_sortedX)
                                    qmin_idx = np.searchsorted(cdf_x, q_min)
                                    if qmin_idx < len(x_sorted):
                                        low_x = x_sorted[qmin_idx]
                                    qmax_idx = np.searchsorted(cdf_x, q_max)
                                    if qmax_idx < len(x_sorted):
                                        high_x = x_sorted[qmax_idx]
                                # print('X log range:', low_x, high_x)
                                # log_bins=(1,3, corner_bins)
                                # x_bins=np.concatenate([0,1],log_bins)
                                x_bins = np.logspace(np.log10(low_x), np.log10(high_x), corner_bins + 1)
                                # print(x_bins)
                            else:
                                # x_bins = corner_bins
                                x_bins = np.linspace(np.min(array_x), np.max(array_x), corner_bins + 1)

                                # Handle y-axis log binning
                            if use_log_y:
                                # # Ensure all values are positive before log transform
                                # if low_y < 1:
                                #     low_y = epsilon
                                # array_y = np.maximum(array_y, epsilon)  # Clip to minimum epsilon
                                #
                                # # Transform y data to log space BEFORE mirroring
                                # array_y = np.log10(array_y)
                                # low_y = np.log10(low_y)
                                # high_y = np.log10(np.max(array_y))
                                # clipped_y = np.where(array_y < 1, epsilon, array_y)
                                # array_y = np.log10(clipped_y)
                                if show_q:
                                    q_max = 1 - q_min

                                    sort_idy = np.argsort(array_y)
                                    y_sorted = array_y[sort_idy]
                                    weights_sortedY = array_w[sort_idy]

                                    # Calculate cumulative sum of weights
                                    cdf_y = np.cumsum(weights_sortedY)
                                    qmin_idy = np.searchsorted(cdf_y, q_min)
                                    if qmin_idy < len(y_sorted):
                                        low_y = y_sorted[qmin_idy]
                                    qmax_idy = np.searchsorted(cdf_y, q_max)
                                    if qmax_idy < len(y_sorted):
                                        high_y = y_sorted[qmax_idy]
                                # log_bins = (1, 3, corner_bins)
                                # y_bins = np.concatenate([0, 1], log_bins)
                                # print('Y log range:', low_y, high_y)
                                y_bins = np.logspace(np.log10(low_y), np.log10(high_y), corner_bins + 1)
                            else:
                                # y_bins = corner_bins
                                y_bins = np.linspace(np.min(array_y), np.max(array_y), corner_bins + 1)
                            # Trying to use a KDE
                            # X_mesh, Y_mesh = np.meshgrid(x_bins, y_bins)
                            # data_stack=np.vstack((array_x, array_y))
                            # new_kde=gaussian_kde(data_stack, weights=array_w,bw_method='silverman')
                            # if use_log_x:
                            #     bw_x=10**(np.std(np.log10(array_x)))
                            # else:
                            #     bw_x= np.std(array_x)
                            # if use_log_y:
                            #     bw_y=10**(np.std(np.log10(array_y)))
                            # else:
                            #     bw_y=np.std(array_y)
                            # if use_log_x or use_log_y:
                            #     n=len(array_x)
                            #     d=2
                            #     new_kde.set_bandwidth(bw_method=new_kde.silverman_factor()*np.array([bw_x, bw_y]).mean())
                            # positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
                            # f= new_kde(positions).reshape(X_mesh.shape)
                            # X = x_bins
                            # Y = y_bins
                            #
                            # max_indices = np.unravel_index(np.argmax(f), f.shape)
                            # max_x = X[max_indices[1]]
                            # max_y = Y[max_indices[0]]
                            # Create 2D histogram
                            # H, xedges, yedges = np.histogram2d(array_x, array_y,bins=[x_bins, y_bins],weights=array_w,density=True)
                            H, xedges, yedges = np.histogram2d(array_x, array_y, bins=[x_bins, y_bins], weights=array_w,density=True)

                            # Transpose H for correct orientation
                            H = H.T

                            # Convert to density (comparable to KDE output)
                            # dx = np.diff(xedges)
                            # dy = np.diff(yedges)
                            # bin_areas = dx[:, None] * dy[None, :]
                            # tot_h = np.sum(H)
                            # H = H / (tot_h * bin_areas)
                            # smooth
                            if smooth_corner:
                                H = gaussian_filter(H, sigma=smooth_val)

                            # Get bin centers for plotting
                            X = (xedges[:-1] + xedges[1:]) / 2
                            Y = (yedges[:-1] + yedges[1:]) / 2

                            # Find maximum for mode
                            max_indices = np.unravel_index(np.argmax(H), H.shape)
                            max_x = X[max_indices[1]]
                            max_y = Y[max_indices[0]]

                            # Apply log scale if needed
                            f = H.copy()
                            if scale_t == 'Log':
                                #     Avoid log(0) by adding small epsilon where H is zero
                                f = np.where(f > 0, np.log(f) / np.log(l_base), np.nan)

                            # Plot contour
                            X_mesh, Y_mesh = np.meshgrid(X, Y)
                            cfset = ax.contourf(X_mesh, Y_mesh, f, 100, cmap=my_colorMap)
                            # cset = ax.contour(X_mesh, Y_mesh, f, n_con, colors='k')
                            # Use logarithmically spaced levels:
                            if scale_t == 'Log':
                                # f is already in log space, so use linear spacing
                                levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                            else:
                                # f is in linear space, so use log spacing
                                f_max = np.log10(np.max(f))
                                f_min = np.min(f)
                                f_proposed = f_max - 3
                                if f_min == 0:
                                    f_selected = f_proposed
                                else:
                                    f_selected = np.max([np.log10(f_min), f_proposed])

                                # print(f_min, f_proposed, f_selected)
                                # levels = np.logspace(f_max-4, f_max, 10)
                                levels = np.linspace(f_selected, f_max, n_con)
                                levels = 10 ** levels
                                # print(np.log10(f_min), np.log10(f_max))
                                # levels = np.logspace(np.log10(f_min), np.log10(f_max), n_con)
                                # percentiles = np.linspace(1, 99, n_con)  # Skip extreme tails
                                # percents=100-percentiles
                                # print(percentiles)
                                # print(percents)
                                # levels = np.percentile(f, percentiles)
                            # print(levels)
                            cset = ax.contour(X, Y, f, levels=levels, colors='k')
                            ax.clabel(cset, inline=1, fontsize=8)

                            # Set axis limits and scales
                            if use_log_x:
                                ax.set_xscale('log')
                                ax.set_xlim(low_x, high_x)
                            else:
                                ax.set_xlim(low_x, high_x)

                            if use_log_y:
                                ax.set_yscale('log')
                                ax.set_ylim(low_y, high_y)
                            else:
                                ax.set_ylim(low_y, high_y)

                        # Mirror data if needed
                        else:
                            if mirror_data:
                                array_x = np.concatenate((array_x, 2 * low_x - array_x))
                                array_y = np.concatenate((array_y, 2 * low_y - array_y))
                                array_w = np.concatenate((array_w, array_w))
                                # print('Entering Mirror')
                            # Recalculate ranges after mirroring
                            delta_x = deltas if type_step != 'bins' else (high_x - low_x) / corner_bins
                            delta_y = deltas if type_step != 'bins' else (high_y - low_y) / corner_bins
                            # print('X range:', low_x, high_x)
                            # print('Y range:', low_y, high_y)
                            # print('Deltas [x,y]: ', delta_x,delta_y)
                            # print(array_x)
                            # print(array_y)
                            if delta_x > delta_y:
                                bws = delta_x
                                scal_fac = delta_x / delta_y
                                array_y = array_y * scal_fac
                            elif delta_x < delta_y:
                                scal_fac = delta_y / delta_x
                                bws = delta_y
                                array_x = array_x * scal_fac
                            else:
                                bws = delta_x

                            x_bins = num_bins(array_x, delta_x)
                            y_bins = num_bins(array_y, delta_y)
                            # print(i,j)
                            # print('# of bins:', corner_bins, type_step)
                            # print('bins [x,y]: ',x_bins,y_bins)
                            # Create 2D KDE
                            [values] = np.dstack([array_x, array_y])
                            grid_points_x, grid_points_y = x_bins, y_bins
                            kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                            grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))

                            X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                            f = points.reshape(grid_points_x, grid_points_y).T

                            # Zero out values outside domain
                            xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                            ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                            if delta_x > delta_y:
                                ys = ys / scal_fac
                            elif delta_x < delta_y:
                                xs = xs / scal_fac
                            for ii in range(grid_points_y):
                                for jj in range(grid_points_x):
                                    if high_x < xs[ii][jj] or xs[ii][jj] < low_x or \
                                            high_y < ys[ii][jj] or ys[ii][jj] < low_y:
                                        f[ii][jj] = 0
                                    elif mirror_data:
                                        f[ii][jj] = f[ii][jj] * 2

                            # Find maximum
                            max_value = np.max(f)
                            max_indices = np.unravel_index(np.argmax(f), f.shape)
                            if delta_x > delta_y:
                                Y = Y / scal_fac
                            elif delta_x < delta_y:
                                X = X / scal_fac
                            if use_log_x or use_log_y:
                                if use_log_x:
                                    X = 10 ** X
                                if use_log_y:
                                    Y = 10 ** Y
                            max_x = X[max_indices[1]]
                            max_y = Y[max_indices[0]]

                            # Apply log scale if needed
                            if scale_t == 'Log':  # or use_log_x or use_log_y:
                                f = np.log(f) / np.log(l_base)

                            # Plot contour
                            cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                            # cset = ax.contour(X, Y, f, n_con, colors='k')
                            if scale_t == 'Log':
                                # f is already in log space, so use linear spacing
                                levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                            else:
                                # f is in linear space, so use log spacing
                                f_max = np.log10(np.max(f))
                                # levels = np.logspace(f_max-4, f_max, 10)
                                f_min = np.min(f)
                                f_proposed = f_max - 3
                                if f_min <= 0:
                                    f_selected = f_proposed
                                else:
                                    print(f_min)
                                    f_selected = np.max([np.log10(f_min), f_proposed])

                                # print(f_min, f_proposed, f_selected)
                                # levels = np.logspace(f_max-4, f_max, 10)
                                levels = np.linspace(f_selected, f_max, n_con)
                                levels = 10 ** levels
                            cset = ax.contour(X, Y, f, levels=levels, colors='k')
                            ax.clabel(cset, inline=1, fontsize=8)
                            ax.set_xlim(low_x, high_x)
                            ax.set_ylim(low_y, high_y)

                        # ADD GRIDLINES HERE
                        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                        # Mark mode
                        sig_c = int(abs(np.log10(bws)))
                        ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5, label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, n_labels[j]))

                        ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5, label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, n_labels[i]))

                        if i == n_vars - 1:
                            ax.set_xlabel(n_labels[j], fontsize=fo)
                        if j == 0:
                            ax.set_ylabel(n_labels[i], fontsize=fo)
                        ax.legend(fontsize=fo)

                    # Diagonal: 1D marginal distributions
                    elif i == j:
                        array_val = n_data[:, i]
                        array_w = n_weights.copy()

                        # Remove NaN values
                        valid_mask = ~np.isnan(array_val)
                        array_val = array_val[valid_mask]
                        array_w = array_w[valid_mask]

                        if len(array_val) == 0:
                            ax.axis('off')
                            continue

                        # NEW: Apply log binning if enabled for this variable
                        if use_log_bins and i ==0 and type_p == 'NoLCPS':
                            # clipped_val = np.where(array_val < 1, epsilon, array_val)
                            # array_val = clipped_val
                            if show_q:
                                q_max = 1 - q_min

                                sort_idx = np.argsort(array_val)
                                x_sorted = array_val[sort_idx]
                                weights_sortedX = array_w[sort_idx]

                                # Calculate cumulative sum of weights
                                cdf_x = np.cumsum(weights_sortedX)
                                qmin_idx = np.searchsorted(cdf_x, q_min)
                                if qmin_idx < len(x_sorted):
                                    low_val = x_sorted[qmin_idx]
                                qmax_idx = np.searchsorted(cdf_x, q_max)
                                if qmax_idx < len(x_sorted):
                                    high_val = x_sorted[qmax_idx]
                            # log_bins=np.logspace(0,3, corner_bins)
                            # val_bins=np.concatenate([[0.9,1],log_bins])
                            delta_val = 1
                            val_bins = np.logspace(np.log10(low_val), np.log10(high_val), corner_bins)
                            x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(np.log10(array_val), weights=array_w)()
                            x_rs = 10 ** x_rs
                            y_rs = y_rs / (x_rs * np.log(10))

                            # # Handle zero/negative values
                            # if low_val <= 0:
                            #     low_val = epsilon
                            #
                            # # Create log-spaced bins
                            # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                        else:
                            low_val = np.min(array_val)
                            high_val = np.max(array_val)
                            delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                            val_bins = num_bins(array_val, delta_val)

                            if mirror_data:
                                array_valM = np.concatenate((array_val, 2 * low_val - array_val))
                                array_wM = np.concatenate((array_w, array_w))

                            x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_valM, weights=array_wM)()
                            if mirror_data:
                                mask = (x_rs < low_val) | (high_val < x_rs)
                                y_rs[mask] = 0
                                y_rs = y_rs * 2
                        sig_val = int(abs(np.log10(delta_val)))

                        # Find peaks
                        if show_peaks:
                            mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                            y_rsP = y_rs[mask_P]
                            x_rsP = x_rs[mask_P]
                            peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                            peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                            if len(peaks_idr) > num_peaks:
                                peaks_idr = peaks_idr[:num_peaks]
                            colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                            for d, p in enumerate(peaks_idr):
                                peak_r = x_rsP[p]
                                ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                        # Find mode
                        max_xi = np.argmax(y_rs)
                        max_x = x_rs[max_xi]

                        # Plot
                        ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                        ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                        ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0, label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                        # ADD VERTICAL GRIDLINES HERE
                        ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                        # ax.set_xlabel(region_labels[i], fontsize=fo)
                        ax.set_title(n_labels[i], fontsize=fo)
                        if use_log_bins and i==0 and type_p == 'NoLCPS':
                            ax.set_xscale('log')
                            ax.set_xlim(low_val, high_val)
                            # low_yrs = np.min(y_rs)
                            # yrs_proposed = np.log10(np.max(y_rs)) - 3
                            # if low_yrs == 0:
                            #     sel_pdf = yrs_proposed
                            # else:
                            #     sel_pdf = np.max([np.log10(low_yrs), yrs_proposed])
                            # sel_pdf = 10 ** sel_pdf
                            # print(sel_pdf)
                            # ax.set_ylim(bottom=sel_pdf)
                            ax.set_ylim(bottom=low_pdf)
                            ax.set_yscale('log', base=l_base)
                            ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        else:
                            ax.set_xlim(low_val, high_val)

                            if scale_t == 'Log':
                                ax.set_yscale('log', base=l_base)
                                ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                            else:
                                ax.set_ylabel('PDF', fontsize=fo)

                        ax.legend(fontsize=fo)

                    # Upper triangle: hide
                    else:
                        ax.axis('off')
                    ax.minorticks_on()
                    ax.tick_params(axis='both', labelsize=fo)

            fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) - Corner Plot Mode {n+1}',fontsize=fo + 4)
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            if save_corner:
                if w_prior:
                    corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wPrior")
                    if filter_values:
                        corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wBoth")
                elif filter_values:
                    corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wFilter")
                else:
                    corner_paths = os.path.join(res_path, "MyCornerPlots_Region")
                os.makedirs(corner_paths, exist_ok=True)

                if use_mean:
                    plot_subpath = os.path.join(corner_paths, "Mean")
                elif use_cornerMode:
                    plot_subpath = os.path.join(corner_paths, "CornerMode")
                elif use_cornerMean:
                    plot_subpath = os.path.join(corner_paths, "CornerMean")
                elif use_mode:
                    plot_subpath = os.path.join(corner_paths, "Modes")
                else:
                    plot_subpath = os.path.join(corner_paths, "Raw")
                os.makedirs(plot_subpath, exist_ok=True)

                filename_parts = [f'Corner_Region_{region_name[r]}_Mode_{n+1}']
                if use_log_bins:
                    filename_parts.append('LBins')
                if scale_t == 'Log':
                    filename_parts.append('Log')
                if set_range == 'False':
                    filename_parts.append('NoMin')
                if show_fit == 'False':
                    filename_parts.append('NoFit')
                filename = '_'.join(filename_parts) + '.png'
                fig.savefig(os.path.join(plot_subpath, filename))

            if show_corner:
                plt.show()

        plt.close('all')

def my_cornerN_Region(Layers, n_Ls, paths, data_dir, probe=probe, show_corner=show_cornerN_region, save_cornerP=save_cornerN_region, save_cornerD=save_cornerDisMean, deltas=corner_deltas, type_step=type_step, type_p=type_p, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, n_con=num_cont, bws=bws, scale_t=scale_type, l_base=log_base, my_colorMap=my_colorMap, corner_bins=corner_bins, log_bin_indices=[0, 3, 6], epsilon=epsilon, smooth_corner=smooth_corner, smooth_val=smooth_val,shot_fit=show_fit,show_q=show_quantiles):
    """
    Creates corner plots of only n_i values for atmospheric regions by aggregating
    layers within each region. Similar to my_corner_Region but only for n values.
    """
    # Return early if MultipleRI mode (doesn't make sense for n values)
    if type_p == 'MultipleRI':
        print('Does not make sense to plot Ns for MultipleRI')
        return

    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    region_name = ['D', 'C', 'B', 'A', 'L']


    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")

    # Process each region
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]

        # Collect data from all layers within this region
        region_data = []
        region_weights = []
        region_labels = None

        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            n_modes = n_Ls[i]

            # Check if layer falls within current region
            if y_max <= r_max and r_min <= y_min:
                if use_mean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_mode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
                    data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
                else:
                    dis_path = os.path.join(paths, "PosteriorDistributions")
                    pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)

                # Apply filtering if needed
                if filter_values:
                    for f_what in filter_what:
                        if f_what == 'nums':
                            n_T = data[:, 0] + data[:, 3] + data[:, 6]
                            mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
                        elif f_what == 'index':
                            mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & \
                                   (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & \
                                   (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
                        elif f_what == 'sigma':
                            mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & \
                                   (1 + filt_s >= data[:, 16])
                        data = data[mask]
                        weights = weights[mask]

                # Apply priors if needed
                if w_prior:
                    for p, p_what in enumerate(prior_what):
                        sel_val = []
                        if p_what == 'num_total':
                            for i in range(n_modes):
                                sel_val.append((3 * i))
                            weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                        # Other priors could be applied but may not be relevant for just n values

                region_data.append(data)
                region_weights.append(weights)
                if region_labels is None:
                    region_labels = labels
                elif len(labels) > len(region_labels):
                    region_labels = labels

        # Skip if no layers in this region
        if len(region_data) == 0:
            continue

        # Combine data from all layers in region
        max_cols = max(data.shape[1] for data in region_data)

        # Pad all arrays to match max_cols
        padded_data = []
        for data in region_data:
            if data.shape[1] < max_cols:
                padding = np.full((data.shape[0], max_cols - data.shape[1]), np.nan)
                data = np.hstack([data, padding])
            padded_data.append(data)

        combined_data = np.vstack(padded_data)
        combined_weights = np.concatenate(region_weights)

        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # *** KEY MODIFICATION: Extract only n_i columns (columns 0, 3, 6) ***
        n_columns = [0, 3, 6]
        n_data = combined_data[:, n_columns]
        n_weights = combined_weights
        n_labels = region_labels[n_columns]
        # print(region_name[r],n_labels)
        # Remove rows with NaN values in n columns
        valid_mask = ~np.isnan(n_data).any(axis=1)
        n_data = n_data[valid_mask]
        n_weights = n_weights[valid_mask]

        if len(n_data) == 0:
            print(f"No valid n data for region {region_name[r]}")
            continue
        # print(region_name[r],len(n_data))
        # Renormalize weights after filtering
        n_weights = n_weights / np.sum(n_weights)

        # Determine number of n variables (should be 3)
        n_vars = n_data.shape[1]
        # Create corner plot using the same approach as my_corner_Region
        fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))
        # n_dec = [-1, 0, 1]
        # n_dec = [0, 1, 2]
        # n_dec=[0,2,3]
        # Plot for each pair of variables
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axs[i, j]

                # Lower triangle: 2D joint distributions
                if i > j:
                    array_x = n_data[:, j]
                    array_y = n_data[:, i]
                    array_w = n_weights.copy()

                    # Remove NaN values (should already be clean but double-check)
                    valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                    array_x = array_x[valid_mask]
                    array_y = array_y[valid_mask]
                    array_w = array_w[valid_mask]

                    if len(array_x) == 0:
                        ax.axis('off')
                        continue
                    if show_q:
                        q_min=0.01
                        q_max=1-q_min

                        sort_idx = np.argsort(array_x)
                        x_sorted = array_x[sort_idx]
                        weights_sortedX = array_w[sort_idx]

                        # Calculate cumulative sum of weights
                        cdf_x = np.cumsum(weights_sortedX)
                        qmin_idx = np.searchsorted(cdf_x, q_min)
                        if qmin_idx < len(x_sorted):
                            low_x= x_sorted[qmin_idx]
                        qmax_idx = np.searchsorted(cdf_x, q_max)
                        if qmax_idx < len(x_sorted):
                            high_x = x_sorted[qmax_idx]

                        sort_idy = np.argsort(array_y)
                        y_sorted = array_y[sort_idy]
                        weights_sortedY = array_w[sort_idy]

                        # Calculate cumulative sum of weights
                        cdf_y = np.cumsum(weights_sortedY)
                        qmin_idy = np.searchsorted(cdf_y, q_min)
                        if qmin_idy < len(y_sorted):
                            low_y = y_sorted[qmin_idy]
                        qmax_idy = np.searchsorted(cdf_y, q_max)
                        if qmax_idy < len(y_sorted):
                            high_y = y_sorted[qmax_idy]

                    else:
                        low_x, high_x = np.min(array_x), np.max(array_x)
                        low_y, high_y = np.min(array_y), np.max(array_y)
                    # low_x, high_x = 10**-(n_dec[j]), np.max(array_x)
                    # low_y, high_y = 10**-(n_dec[i]), np.max(array_y)
                    # low_x, high_x = n_dec[j], np.max(array_x)
                    # low_y, high_y = n_dec[i], np.max(array_y)
                    # Use log binning for n values
                    if use_log_bins:
                        x_bins = np.logspace(np.log10(low_x), np.log10(high_x), corner_bins + 1)
                        y_bins = np.logspace(np.log10(low_y), np.log10(high_y), corner_bins + 1)
                    else:
                        # x_bins = np.linspace(np.min(array_x), np.max(array_x), corner_bins + 1)
                        # y_bins = np.linspace(np.min(array_y), np.max(array_y), corner_bins + 1)
                        x_bins = np.linspace(low_x, high_x, corner_bins + 1)
                        y_bins = np.linspace(low_y, high_y, corner_bins + 1)

                    # Create 2D histogram
                    H, xedges, yedges = np.histogram2d(array_x, array_y, bins=[x_bins, y_bins], weights=array_w, density=True)
                    # Transpose H for correct orientation
                    H = H.T

                    # if scale_t == 'Log':
                    #     clipped_H = np.where(H< low_pdf, low_pdf, H)
                    #     g = np.log10(clipped_H)
                    #     # g = np.log10(H)
                    #     H=g.copy()
                    # Smooth if requested
                    if smooth_corner:
                        H = gaussian_filter(H, sigma=smooth_val)

                    # Get bin centers for plotting
                    X = (xedges[:-1] + xedges[1:]) / 2
                    Y = (yedges[:-1] + yedges[1:]) / 2

                    # Find maximum for mode
                    max_indices = np.unravel_index(np.argmax(H), H.shape)
                    max_x = X[max_indices[1]]
                    max_y = Y[max_indices[0]]

                    # Apply log scale if needed
                    f = H.copy()
                    # if scale_t == 'Log':
                    #     # f is already in log space, so use linear spacing
                    #     levels = np.linspace(np.min(f[f > -np.inf]), np.max(f), n_con)
                    #     # f_max = np.log10(np.max(f))
                    #     # f_min = np.min(f)
                    #     # f_proposed = f_max - 3
                    #     # if f_min == 0:
                    #     #     f_selected = f_proposed
                    #     # else:
                    #     #     f_selected = np.max([np.log10(f_min), f_proposed])
                    #     # levels = np.linspace(f_selected, f_max, n_con)
                    #     # levels = 10 ** levels
                    # else:
                    # f is in linear space, so use log spacing
                    f_max = np.log10(np.max(f))
                    f_min = np.min(f)
                    f_proposed = f_max - 3
                    if f_min == 0:
                        f_selected = f_proposed
                    else:
                        f_selected = np.max([np.log10(f_min), f_proposed])

                    # print(f_min, f_proposed, f_selected)
                    # levels = np.logspace(f_max-4, f_max, 10)
                    levels = np.linspace(f_selected, f_max, n_con)
                    levels = 10 ** levels
                    # Plot contour
                    cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                    cset = ax.contour(X, Y, f, levels=levels, colors='k')
                    ax.clabel(cset, inline=1, fontsize=8)

                    # Add gridlines
                    ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                    # Mark mode
                    sig_c = int(abs(np.log10(bws)))
                    ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5,
                               label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, n_labels[j]))
                    ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5,
                               label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, n_labels[i]))

                    # Set log scale for axes
                    if use_log_bins:
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        # ax.set_xlim(0.01, high_x)
                        # ax.set_ylim(0.01, high_y)

                    #Log Scale
                    if scale_t == 'Log':
                        ax.set_xscale('log', base=l_base)
                        ax.set_yscale('log', base=l_base)
                    if set_range:
                        ax.set_xlim(log_min, high_x)
                        ax.set_ylim(log_min, high_y)
                    else:
                        # low_x, high_x = np.min(xedges), np.max(xedges)
                        # low_y, high_y = np.min(yedges), np.max(yedges)
                        ax.set_xlim(low_x, high_x)
                        ax.set_ylim(low_y, high_y)


                    if i == n_vars - 1:
                        ax.set_xlabel(n_labels[j], fontsize=fo)
                    if j == 0:
                        ax.set_ylabel(n_labels[i], fontsize=fo)
                    ax.legend(fontsize=fo)

                # Diagonal: 1D marginal distributions
                elif i == j:
                    array_val = n_data[:, i]
                    array_w = n_weights.copy()

                    # Remove NaN values (should already be clean but double-check)
                    valid_mask = ~np.isnan(array_val)
                    array_val = array_val[valid_mask]
                    array_w = array_w[valid_mask]

                    if len(array_val) == 0:
                        ax.axis('off')
                        continue
                    if show_q:
                        q_min = 0.025
                        q_max = 1 - q_min

                        sort_idx = np.argsort(array_val)
                        x_sorted = array_val[sort_idx]
                        weights_sortedX = array_w[sort_idx]

                        # Calculate cumulative sum of weights
                        cdf_x = np.cumsum(weights_sortedX)
                        qmin_idx = np.searchsorted(cdf_x, q_min)
                        if qmin_idx < len(x_sorted):
                            low_val = x_sorted[qmin_idx]
                        qmax_idx = np.searchsorted(cdf_x, q_max)
                        if qmax_idx < len(x_sorted):
                            high_val = x_sorted[qmax_idx]


                    else:
                        # low_val = 10**-(n_dec[i])
                        low_val= np.min(array_val)
                        high_val = np.max(array_val)
                    # Apply log binning for n values
                    if use_log_bins:
                        # clipped_val = np.where(array_val < 1, epsilon, array_val)
                        # array_val = clipped_val
                        # low_val = np.min(array_val)
                        # high_val = np.max(array_val)
                        # log_bins=np.logspace(0,3, corner_bins)
                        # val_bins=np.concatenate([[0.9,1],log_bins])
                        delta_val = 1
                        val_bins = np.logspace(np.log10(low_val), np.log10(high_val), corner_bins)
                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(np.log10(array_val), weights=array_w)()
                        x_rs = 10 ** x_rs
                        y_rs = y_rs / (x_rs * np.log(10))

                        # # Handle zero/negative values
                        # if low_val <= 0:
                        #     low_val = epsilon
                        #
                        # # Create log-spaced bins
                        # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                    else:
                        # low_val = np.min(array_val)
                        # high_val = np.max(array_val)
                        delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                        val_bins = num_bins(array_val, delta_val)

                        if mirror_data:
                            array_valM = np.concatenate((array_val, 2 * low_val - array_val))
                            array_wM = np.concatenate((array_w, array_w))

                        # x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_valM, weights=array_wM)()
                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw='ISJ').fit(array_valM, weights=array_wM)()
                        if mirror_data:
                            mask = (x_rs < low_val) | (high_val < x_rs)
                            y_rs[mask] = 0
                            y_rs = y_rs * 2
                    sig_val = int(abs(np.log10(delta_val)))


                    # Find peaks if requested
                    if show_peaks:
                        mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                        y_rsP = y_rs[mask_P]
                        x_rsP = x_rs[mask_P]
                        peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                        peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                        if len(peaks_idr) > num_peaks:
                            peaks_idr = peaks_idr[:num_peaks]
                        colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                        for d, p in enumerate(peaks_idr):
                            peak_r = x_rsP[p]
                            ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                    # Find mode
                    max_xi = np.argmax(y_rs)
                    max_x = x_rs[max_xi]

                    # Plot
                    ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                    if show_fit:
                        ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                    ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0,label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                    # Add vertical gridlines
                    ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                    ax.set_title(n_labels[i], fontsize=fo)

                    if use_log_bins:
                        ax.set_xscale('log')
                        ax.set_ylim(bottom=low_pdf)
                        ax.set_yscale('log', base=l_base)
                        # ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        # ax.set_xlim(0.01, high_val)

                    if scale_t == 'Log':
                        ax.set_ylim(bottom=low_pdf)
                        ax.set_yscale('log', base=l_base)
                        # ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        ax.set_ylabel('PDF', fontsize=fo)
                        ax.set_xscale('log', base=l_base)
                    else:
                        ax.set_ylabel('PDF', fontsize=fo)
                        # ax.set_xscale('log', base=l_base

                    if set_range:
                        ax.set_xlim(log_min, high_val)
                    else:
                        ax.set_xlim(low_val, high_val)
                    ax.legend(fontsize=fo)

                # Upper triangle: hide
                else:
                    ax.axis('off')

                ax.minorticks_on()
                ax.tick_params(axis='both', labelsize=fo)

        # Add title
        if w_prior:
            fig.suptitle(
                f'Corner Plot for N values - Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) with Prior',
                fontsize=fo)
        else:
            fig.suptitle(f'Corner Plot for N values - Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km)',
                         fontsize=fo)
        plt.tight_layout()

        # Save if requested
        if save_cornerP:
            if w_prior:
                corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wPrior")
                if filter_values:
                    corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wBoth")
            elif filter_values:
                corner_paths = os.path.join(res_path, "MyCornerPlots_Region_wFilter")
            else:
                corner_paths = os.path.join(res_path, "MyCornerPlots_Region")
            os.makedirs(corner_paths, exist_ok=True)

            if use_mean:
                plot_subpath = os.path.join(corner_paths, "Mean")
            elif use_cornerMode:
                plot_subpath = os.path.join(corner_paths, "CornerMode")
            elif use_cornerMean:
                plot_subpath = os.path.join(corner_paths, "CornerMean")
            elif use_mode:
                plot_subpath = os.path.join(corner_paths, "Modes")
            else:
                plot_subpath = os.path.join(corner_paths, "Raw")
            os.makedirs(plot_subpath, exist_ok=True)

            filename_parts = [f'CornerN_Region_{region_name[r]}']
            if use_log_bins:
                filename_parts.append('LBins')
            if scale_t=='Log':
                    filename_parts.append('Log')
            if set_range=='False':
                filename_parts.append('NoMin')
            if show_fit=='False':
                filename_parts.append('NoFit')
            filename = '_'.join(filename_parts) + '.png'
            fig.savefig(os.path.join(plot_subpath, filename))

        # Show if requested

        if show_corner:
            plt.show()

        plt.close('all')
def my_CDFN_Region(Layers, n_Ls, paths, data_dir, probe=probe, show_cdf=show_CDFN_region, save_cdf=save_CDFN_region, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, type_p=type_p,show_q=show_quantiles):
    """
    Creates cumulative distribution function (CDF) plots for n_i values
    for atmospheric regions by aggregating layers within each region.

    Shows the CDF for each of the three n values (n1, n2, n3) to understand
    the probability that n_i is less than or equal to a given value.
    """
    import warnings
    warnings.filterwarnings('ignore', 'invalid value encountered in divide', category=RuntimeWarning)

    # Return early if MultipleRI mode
    if type_p == 'MultipleRI':
        print('Does not make sense to plot Ns for MultipleRI')
        return

    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    region_name = ['D', 'C', 'B', 'A', 'L']

    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")
    if show_q:
        # Both 68% and 95% CI
        quantiles = [0.025, 0.16, 0.5, 0.84, 0.975]#, 0.01,0.99]
        # quantiles =[0.5]
        quantile_colors = {
            0.5: 'red',  # Median
            0.16: 'blue',  # 68% lower
            0.84: 'blue',  # 68% upper
            0.025: 'purple',  # 95% lower
            0.975: 'purple',  # 95% upper
            0.01: 'orange',  # 90% lower Normally 0.05
            0.99: 'orange',  # 90% upper Normally 0.95
            0.25: 'gray',  # Quartile
            0.75: 'gray',  # Quartile
        }
        quantile_styles = {
            0.5: '--',  # Median: dashed
            0.16: ':',  # 68%: dotted
            0.84: ':',
            0.025: '-.',  # 95%: dash-dot
            0.975: '-.',
            0.01: '--',  # 90%: dashed Normally 0.05
            0.99: '--', #Normally 0.95
            0.25: ':',  # Quartiles: dotted
            0.75: ':',
        }
    if show_peaks:
        peaks = [0.1,1,10,100]  # , 0.01,0.99]
        peak_colors = {
            1: 'dodgerblue',  #
            10: 'forestgreen',  #
            100: 'mediumorchid',  #
            0.1: 'darkorange',  #
        }
        peak_styles = {
            1: '--',  #
            10: ':',  #
            100: '-.',  #
            0.1: '--',  #
        }
    # Process each region
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]

        # Collect data from all layers within this region
        region_data = []
        region_weights = []
        region_labels = None

        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            n_modes = n_Ls[i]

            # Check if layer falls within current region
            if y_max <= r_max and r_min <= y_min:
                if use_mean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_mode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
                    data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
                else:
                    dis_path = os.path.join(paths, "PosteriorDistributions")
                    pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)

                # Apply filtering if needed
                if filter_values:
                    for f_what in filter_what:
                        if f_what == 'nums':
                            n_T = data[:, 0] + data[:, 3] + data[:, 6]
                            mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
                        elif f_what == 'index':
                            mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & \
                                   (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & \
                                   (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
                        elif f_what == 'sigma':
                            mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & \
                                   (1 + filt_s >= data[:, 16])
                        data = data[mask]
                        weights = weights[mask]

                # Apply priors if needed
                if w_prior:
                    for p, p_what in enumerate(prior_what):
                        sel_val = []
                        if p_what == 'num_total':
                            for i in range(n_modes):
                                sel_val.append((3 * i))
                            weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])

                region_data.append(data)
                region_weights.append(weights)
                if region_labels is None:
                    region_labels = labels
                elif len(labels) > len(region_labels):
                    region_labels = labels

        # Skip if no layers in this region
        if len(region_data) == 0:
            continue

        # Combine data from all layers in region
        max_cols = max(da.shape[1] for da in region_data)

        # Pad all arrays to match max_cols
        padded_data = []
        for dat in region_data:
            if dat.shape[1] < max_cols:
                padding = np.full((dat.shape[0], max_cols - dat.shape[1]), np.nan)
                dat = np.hstack([dat, padding])
            padded_data.append(dat)

        combined_data = np.vstack(padded_data)
        combined_weights = np.concatenate(region_weights)

        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # Extract only n_i columns (columns 0, 3, 6)
        n_columns = [0, 3, 6]
        n_data = combined_data[:, n_columns]
        n_weights = combined_weights
        n_labels = region_labels[n_columns]

        # Remove rows with NaN values in n columns
        valid_mask = ~np.isnan(n_data).any(axis=1)
        n_data = n_data[valid_mask]
        n_weights = n_weights[valid_mask]

        if len(n_data) == 0:
            print(f"No valid n data for region {region_name[r]}")
            continue

        # Renormalize weights after filtering
        n_weights = n_weights / np.sum(n_weights)

        # Create figure with 3 subplots (one for each n value)
        fig, axs = plt.subplots(1, 3, figsize=(figSize_x * 3, figSize_y))

        # Color scheme for the three modes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

        for idx in range(3):
            ax = axs[idx]

            # Get data for this n value
            n_vals = n_data[:, idx]
            weights = n_weights

            # Sort values and weights together
            sort_idx = np.argsort(n_vals)
            n_sorted = n_vals[sort_idx]
            weights_sorted = weights[sort_idx]

            # Calculate cumulative sum of weights
            cdf = np.cumsum(weights_sorted)
            # Print Values of Interest
            # if idx==2:
            n_values_to_check=[0.001,0.01,0.1,1, 10, 100]
            print(f"\nRegion {region_name[r]} {n_labels[idx]}:")
            print(f"{'-' * 80}")
            print(f"{'n value':<15} {'P(n ≤ value)':<20} {'P(n > value)':<20} {'Percentile':<15}")
            print(f"{'-' * 80}")
            for n_val in sorted(n_values_to_check):
                cdf_val = get_cdf_value_at_n(n_data[:, idx], n_weights, n_val)
                percentile = cdf_val * 100
                complement = 1 - cdf_val

                print(f"{n_val:<15.4f} {cdf_val:<20.4f} {complement:<20.4f} {percentile:<15.2f}%")
            # Plot CDF
            ax.plot(n_sorted, cdf, linewidth=2.5, color=colors[idx], label='Weighted CDF')

            if show_peaks:
                for p in peaks:
                    cdf_p = get_cdf_value_at_n(n_data[:, idx], n_weights, p)
                    label = f'{p:.1f}: {cdf_p * 100:.1f}%'
                    lw = 1.5
                    alpha = 0.6
                    q_color = peak_colors.get(p, 'magenta')
                    q_style = peak_styles.get(p, '--')
                    # ax.axhline(y=cdf_p, xmin=np.min(n_sorted), xmax=p+np.min(n_sorted), color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1)
                    # ax.axvline(x=p, ymin=np.min(cdf), ymax=cdf_p, color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1,label=label)
                    ax.plot([np.min(n_sorted), p], [cdf_p,cdf_p],color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1)
                    # Vertical line from bottom to intersection
                    ax.plot([p, p], [0, cdf_p],color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1, label=label)

            if show_q:
                quantile_values = {}
                # Plot each quantile
                for q in quantiles:
                    q_idx = np.searchsorted(cdf, q)
                    if q_idx < len(n_sorted):
                        q_val = n_sorted[q_idx]
                        quantile_values[q] = q_val

                        # Get color and style for this quantile
                        q_color = quantile_colors.get(q, 'magenta')
                        q_style = quantile_styles.get(q, '--')

                        # Determine line width based on importance
                        if q in [0.01,0.99]:
                            print(f'{q * 100:.0f}%: {q_val:.3f}')
                        if q == 0.5:
                            lw = 2.0
                            alpha = 0.8
                            label = f'Median ({q * 100:.0f}%): {q_val:.1f}'
                            median=q_val
                        elif q in [0.16, 0.84, 0.025, 0.975]:
                            lw = 1.5
                            alpha = 0.6
                            label = f'{q * 100:.1f}%: {q_val:.1f}'
                        else:
                            lw = 1.0
                            alpha = 0.4
                            label = None

                        # Plot lines
                        ax.axhline(y=q, color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1)
                        ax.axvline(x=q_val, color=q_color, linestyle=q_style, linewidth=lw, alpha=alpha, zorder=1, label=label)
                        # Check for 68% CI (1-sigma)
                        if q == 0.16:
                            lower = q_val
                        if q == 0.84:
                            upper = q_val
                        # Check for 95% CI (2-sigma)
                        # if q==0.025:
                        #     lower = q_val
                        # if q==0.975:
                        #     upper = q_val

                title_q = r"{3} = ${{{0:.1f}}}_{{min: {1:.1f}}}^{{max: {2:.1f}}}$"
                title_q = title_q.format(median, lower, upper,n_labels[idx])
                ax.set_title(title_q, fontsize=fo)

            else:
                # Add horizontal line at 0.5 for median
                median_idx = np.searchsorted(cdf, 0.5)
                if median_idx < len(n_sorted):
                    median_val = n_sorted[median_idx]
                    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,label=f'Median: {median_val:.1f}')
                    ax.axvline(x=median_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

                # Add quartile lines
                # q25_idx = np.searchsorted(cdf, 0.25)
                # q75_idx = np.searchsorted(cdf, 0.75)
                # if q25_idx < len(n_sorted):
                #     q25_val = n_sorted[q25_idx]
                #     ax.axhline(y=0.25, color='magenta', linestyle=':', linewidth=1, alpha=0.5)
                #     ax.axvline(x=q25_val, color='magenta', linestyle=':', linewidth=1, alpha=0.5)
                # if q75_idx < len(n_sorted):
                #     q75_val = n_sorted[q75_idx]
                #     ax.axhline(y=0.75, color='magenta', linestyle=':', linewidth=1, alpha=0.5)
                #     ax.axvline(x=q75_val, color='magenta', linestyle=':', linewidth=1, alpha=0.5)

                ax.set_title(f'{n_labels[idx]} CDF', fontsize=fo)
            # Formatting
            ax.set_xlabel(n_labels[idx], fontsize=fo)
            ax.set_ylabel('Cumulative Probability', fontsize=fo)
            ax.set_xlim(np.min(n_sorted), np.max(n_sorted) * 1.05)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=fo - 2)
            ax.tick_params(axis='both', labelsize=fo)

            # Optional: Use log scale for x-axis if values span multiple orders of magnitude
            if np.max(n_sorted) / np.min(n_sorted[n_sorted > 0]) > 100:
                ax.set_xscale('log')

        # Add overall title
        if w_prior:
            fig.suptitle(
                f'Cumulative Distribution Functions - Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) with Prior',
                fontsize=fo + 2)
        else:
            fig.suptitle(f'Cumulative Distribution Functions - Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km)',
                         fontsize=fo + 2)
        plt.tight_layout()

        # Save if requested
        if save_cdf:
            if w_prior:
                cdf_path = os.path.join(res_path, "CDF_Plots_wPrior_Region")
                if filter_values:
                    cdf_path = os.path.join(res_path, "CDF_Plots_wBoth_Region")
            elif filter_values:
                cdf_path = os.path.join(res_path, f"CDF_Plots_wFilter_Region")
            else:
                cdf_path = os.path.join(res_path, "CDF_Plots_Region")
            os.makedirs(cdf_path, exist_ok=True)

            filename_parts = [f'CDF_Region{region_name[r]}_N']
            if use_mode:
                filename_parts.append('jointW')
            filename = '_'.join(filename_parts) + '.png'
            fig.savefig(os.path.join(cdf_path, filename), dpi=300, bbox_inches='tight')

        # Show if requested
        if show_cdf:
            plt.show()

        plt.close('all')


def get_cdf_value_at_n(n_data, n_weights, n_value):
    """
    Calculate the cumulative probability at a specific n value.

    Parameters:
    -----------
    n_data : array
        Data values for one n parameter
    n_weights : array
        Weights for the data
    n_value : float
        The n value at which to evaluate the CDF

    Returns:
    --------
    cdf_value : float
        The cumulative probability P(n <= n_value)
    """
    # Sort values and weights together
    sort_idx = np.argsort(n_data)
    n_sorted = n_data[sort_idx]
    weights_sorted = n_weights[sort_idx]

    # Calculate cumulative sum
    cdf = np.cumsum(weights_sorted)

    # Find the CDF value at the specified n_value
    # Use searchsorted to find where n_value fits in the sorted array
    idx = np.searchsorted(n_sorted, n_value, side='right')

    if idx == 0:
        # n_value is below all data points
        cdf_value = 0.0
    elif idx >= len(cdf):
        # n_value is above all data points
        cdf_value = 1.0
    else:
        # Linear interpolation between points
        if idx < len(n_sorted):
            # Get the two surrounding points
            x0, x1 = n_sorted[idx - 1], n_sorted[idx]
            y0, y1 = cdf[idx - 1] if idx > 0 else 0, cdf[idx]

            # Linear interpolation
            if x1 != x0:
                cdf_value = y0 + (y1 - y0) * (n_value - x0) / (x1 - x0)
            else:
                cdf_value = y0
        else:
            cdf_value = cdf[idx - 1]

    return cdf_value
# Past Functions
def my_histogramPrior(j, file_j, n_modes, y_min, y_max, res_path, deltas=hist_deltas, my_bins=hist_bins, type_step=type_step, type_p=type_p, a_strat=analysis_strat, show_hist=show_hist, save_hist=save_hist, save_mode=save_mode, show_peaks=show_peaks, num_peaks=n_peaks,mirror_data=mirror_data, use_mode=use_mode, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo):
    data, weights, header = read_PostDist_CSV(file_j, n_modes)
    if use_mode==True:
        all_sorted = find_weighted_mode(data, weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        weighted_mode = np.array(weighted_mode)
        frequency = np.array(frequency)
        if save_mode== True:
            mode_paths = os.path.join(res_path, "My_Distributions")
            os.makedirs(mode_paths, exist_ok=True)
            if a_strat == 'Layer':
                filename = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
            elif a_strat == 'independent':
                filename = os.path.join(mode_paths, f"PosDist_{j}_jointW.csv")
            mode_data = np.column_stack((weighted_mode, frequency))
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(mode_data)
            print(f"Data saved to {filename}")
        myMode = weighted_mode[0]
    else:
        weighted_mode= np.array(data)
        frequency = np.array(weights)
    if type_p == 'MultipleRI':
        sel_val =[]
        for i in range(n_modes):
            sel_val.append((3 * (i+1))-1)
        num_rows = n_modes
        num_cols = 3
    else:
        sel_val = []
        for i in range(n_modes):
            sel_val.append(11+(3 * i))
        num_rows = 2
        num_cols = 9
    new_weights, name_prior = singleValue_Prior(sel_val, weighted_mode, frequency, header)
    # mask = frequency>=2
    # weighted_mode= weighted_mode[mask]
    # frequency = frequency[mask]
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figSize_x*num_cols, figSize_y*num_rows))
    axs = axs.flatten()
    new_dis = []
    for i in range(len(weighted_mode[0])):
        # print(header[i])
        array_val = weighted_mode[:, i]
        frequency = new_weights#[i]
        # print(len(array_val), print(len(frequency)))
        low_val = np.min(array_val)
        high_val = np.max(array_val)
        if mirror_data ==True:
            # print(f'For {n_labels[i]}: [{low_val}, {high_val}]')
            array_val = np.concatenate((array_val, 2 * low_val - array_val))
            array_w = np.concatenate((frequency, frequency))
        else:
            array_w=frequency

        if type_step=='bins':
            if mirror_data==True:
                val_bins=my_bins*2
            else:
                val_bins= my_bins
            # print(val_bins)
            delta_val =(np.max(array_val) - np.min(array_val))/val_bins
        elif type_step =='deltas':
            delta_val = deltas
            val_bins = num_bins(array_val, delta_val)
        elif type_step == 'changing_d':
            if type_p == 'MultipleRI':
                if i in [0,3,6]:
                    delta_val= 0.1
                else:
                    delta_val = deltas
            else:
                if i in [0,3,6]:
                    delta_val= 1
                elif i <10 or i in [12,15]:
                    delta_val= 0.1
                else:
                    delta_val =deltas
            val_bins = num_bins(array_val, delta_val)
        array_val = np.array(array_val)
        # x_rs, y_rs = FFTKDE(kernel='gaussian', bw='silverman').fit(array_val, weights=frequency)()  # Best at smoothing!
        x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=array_w)()  # With Weights
        if mirror_data:
            mask = (x_rs < low_val) | (high_val < x_rs)
            y_rs[mask] = 0  # Set the KDE to zero outside of the domain
        if show_peaks==True:
            mask_P = (x_rs >= low_val) & (high_val >= x_rs)
            y_rsP = y_rs[mask_P]
            x_rsP = x_rs[mask_P]
            peaks_idr, peak_r = find_peaks(y_rsP, height=0)
            peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
            if len(peaks_idr) > num_peaks:
                peaks_idr = peaks_idr[:num_peaks]
            # colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
            colors = plt.cm.spring(np.linspace(0, 1, len(peaks_idr)))
            for d, p in enumerate(peaks_idr):
                peak_r = x_rsP[p]
                axs[i].axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_hist}f}')
        max_xi = np.argmax(y_rs)
        max_x = x_rs[max_xi]
        new_dis.append(max_x)
        axs[i].hist(array_val, weights=array_w, bins=val_bins, density=True)
        axs[i].plot(x_rs, y_rs, color='black', label='Fitted Curve')
        axs[i].axvline(x=max_x, color='black', linestyle='--', linewidth=2, label='FFTKDE: {0:.{1}f}'.format(max_x, sig_hist))
        if use_mode==True:
            max_his = myMode[i]
            axs[i].axvline(x=max_his, color='green', linestyle='dotted', linewidth=2, label='Mode: {0:.{1}f}'.format(max_his, sig_hist))
        axs[i].set_title(header[i],fontsize=fo)
        axs[i].set_xlabel(header[i],fontsize=fo)
        axs[i].set_xlim((low_val,high_val))
        if i in [0,3,6]:
            l_base=10
            # axs[i].set_xscale('log', base=l_base)
            axs[i].set_yscale('log', base=l_base)
            axs[i].set_ylabel(f'Log{l_base} (PDF)')
        else:
            axs[i].set_ylabel('PDF',fontsize=fo)
        # axs[i].set_ylabel('PDF', fontsize=fo)
        axs[i].tick_params(axis='both', labelsize=fo)
        axs[i].legend(fontsize=fo)
    if save_mean:
        mode_paths = os.path.join(res_path, "My_Distributions")
        os.makedirs(mode_paths, exist_ok=True)
        if a_strat == 'Layer':
            filename = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
        elif a_strat == 'independent':
            filename = os.path.join(mode_paths, f"PosDist_{j}_Mean.csv")
        new_dis.append(0)
        mean_data = np.array(new_dis)
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(mean_data)
        print(f"Data saved to {filename}")
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    if a_strat== 'Layer':
        fig.suptitle('Histograms with Prior {} for Layer {} - {}km'.format(name_prior,max_Alt, min_Alt), fontsize=fo)
    plt.tight_layout()
    if save_hist ==True:
        hist_paths = os.path.join(res_path, "Histograms_wPrior")
        os.makedirs(hist_paths, exist_ok=True)
        if a_strat == 'Layer':
            fig.savefig(hist_paths + '/Hist_Layer{}_wPrior.png'.format(j))
        elif a_strat == 'independent':
            fig.savefig(hist_paths + '/Hist_{}_wPrior.png'.format(j))
    if show_hist==True:
        plt.show()
    plt.close('all')
def my_cornerplotSmall(j, n_modes,y_min, y_max, paths, use_mode=use_mode, show_corner=show_cornerMP, save_cornerP=save_cornerMPlot, save_cornerD=save_cornerDisMean):
    res_path = os.path.join(paths, "Results")
    if use_mode==True:
        mode_paths = os.path.join(res_path, "My_Distributions")
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
    else:
        dis_path = os.path.join(paths, "PosteriorDistributions")
        pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(j))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    for n in range(n_modes):
        if n==0:
            columns = [0, 1, 2, 9, 10, 11]
        elif n==1:
            columns = [3, 4, 5, 12, 13, 14]
        elif n==2:
            columns =[6, 7, 8, 15, 16, 17]
        n_data = data[:, columns]
        n_weights = weights
        n_labels = labels[columns]
        fig = corner.corner(n_data, weights=n_weights, labels=n_labels, show_titles=True, title_kwargs={"fontsize": fo},label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
        fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km'.format(n, max_Alt, min_Alt), fontsize=fo)
        plt.tight_layout()
        if save_cornerP== True:
            corner_path = os.path.join(res_path, "CornerPlots")
            os.makedirs(corner_path, exist_ok=True)
            fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}.png'.format(j,n))
        if show_corner==True:
            plt.show()
    if save_cornerD== True:
        quants = []
        for i in range(len(labels)-1):
            array_val = data[:,i]
            title_quantiles = [0.16, 0.5, 0.84]
            q_lo, q_mid, q_hi = quantile(array_val, title_quantiles, weights=weights)
            quants.append(q_mid)
        quants.append(0) #Adding Weight
        quants = np.array(quants)
        corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        if use_mode == True:
            filename = os.path.join(corner_path, f"PosDistLayer{j}_CornerM.csv")
        else:
            filename = os.path.join(corner_path, f"PosDistLayer{j}_Corner.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(quants)
        print(f"Data saved to {filename}")
def my_cornerplot(j, n_modes,y_min, y_max, paths, use_mode=use_mode, show_corner=show_cornerBP, save_cornerP=save_cornerBPlot, save_cornerD=save_cornerDisMean):
    res_path = os.path.join(paths, "Results")
    if use_mode==True:
        mode_paths = os.path.join(res_path, "My_Distributions")
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
    else:
        dis_path = os.path.join(paths, "PosteriorDistributions")
        pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(j))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    fig = corner.corner(data, weights=weights, labels=labels, show_titles=True, title_kwargs={"fontsize": fo},label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
    fig.suptitle('Corner Plot for Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)
    plt.tight_layout()
    if save_cornerP== True:
        corner_path = os.path.join(res_path, "CornerPlots")
        os.makedirs(corner_path, exist_ok=True)
        fig.savefig(corner_path + r'\CornerPlot_Layer{}.png'.format(j))
    if show_corner==True:
        plt.show()
    if save_cornerD== True:
        quants = []
        for i in range(len(labels)-1):
            array_val = data[:,i]
            title_quantiles = [0.16, 0.5, 0.84]
            q_lo, q_mid, q_hi = quantile(array_val, title_quantiles, weights=weights)
            quants.append(q_mid)
        quants.append(0) #Adding Weight
        quants = np.array(quants)
        corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        if use_mode == True:
            filename = os.path.join(corner_path, f"PosDistLayer{j}_CornerM.csv")
        else:
            filename = os.path.join(corner_path, f"PosDistLayer{j}_Corner.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(quants)
        print(f"Data saved to {filename}")
    plt.close()
def my_corner_old(k, n_modes,y_min, y_max, paths, use_mode=use_mode, type_p=type_p, show_cornerBP=show_cornerBP, save_cornerBP=save_cornerBPlot, save_cornerDMean=save_cornerDisMean, show_cornerMP=show_cornerMP, save_cornerMP=save_cornerMPlot, save_cornerDMode=save_cornerDisMode, deltas=corner_deltas, type_step=type_step, mirror_data=mirror_data, figSize_x=figSize_x, figSize_y=figSize_y, fo=fo, sig_h=sig_hist, show_peaks=show_peaks, num_peaks=n_peaks, n_con=num_cont, bws=bws, scale_t=scale_type, l_base=log_base, my_colorMap=my_colorMap, show_prob=probs, corner_bins=corner_bins,log_bin_indices = [0, 3, 6], epsilon=epsilon,smooth_corner=smooth_corner,smooth_val=smooth_val):
    res_path = os.path.join(paths, "Results")
    dis_path = os.path.join(paths, "PosteriorDistributions")
    pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(k))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    if filter_values:
        for f_what in filter_what:
            if f_what == 'nums':
                if type_p == 'MultipleRI':
                    continue
                else:
                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                    mask = (n_T >= filt_n_min) & (n_T<=filt_n_max)
            if f_what == 'index':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                    elif n_modes == 2:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                    else:
                        mask = (filt_max >= data[:, 2]) & (data[:, ] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                else:
                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            if f_what == 'sigma':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (1 + filt_s >= data[:, 1])
                    elif n_modes == 2:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                    else:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & (1 + filt_s >= data[:, 7])
                else:
                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & (1 + filt_s >= data[:, 16])
            data = data[mask]
            weights = weights[mask]
    if w_prior:
        for p, p_what in enumerate(prior_what):
            sel_val = []
            if p_what == 'index':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 1)
                else:
                    for i in range(n_modes):
                        sel_val.append(11 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p])
            if p_what == 'sigma':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 2)
                else:
                    for i in range(n_modes):
                        sel_val.append(10 + (3 * i))
                # print(sel_val)
                weights, name_prior = sigma_Prior(sel_val, data, weights,t_prior=type_Prior[p])
            if p_what == 'rho':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 3)
                else:
                    for i in range(n_modes):
                        sel_val.append(9 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p], m_val=0,unc_val=prior_unc_rho, low_val=-1, high_val=1)
            if p_what == 'rho_diagonal':
                # Apply diagonal prior to favor similar rho values
                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights,val_labels=labels,n_modes=n_modes,type_p=type_p,sigma_diagonal=0.1,prior_min=-1.0, prior_max=1.0,prior_sig=3)
            if p_what=='num_total':
                for i in range(n_modes):
                    sel_val.append((3 * i))
                weights, name_prior = num_Prior(sel_val, data, weights,t_prior=type_Prior[p])
    if use_mode==True:
        all_sorted = find_weighted_mode(data, weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        data = np.array(weighted_mode)
        weights = np.array(frequency)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    if smooth_corner == False:
        smooth_val = None
    if show_cornerBP or save_cornerBP:
        # Determine number of variables
        n_vars = data.shape[1]

        # Create corner plot
        fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

        # Plot for each pair of variables
        for i in range(n_vars):
            for j in range(n_vars):
                ax = axs[i, j]

                # Lower triangle: 2D joint distributions
                if i > j:
                    # print(i, j)
                    array_x = data[:, j]
                    array_y = data[:, i]
                    array_w = weights.copy()

                    # Remove NaN values
                    # valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                    # array_x = array_x[valid_mask]
                    # array_y = array_y[valid_mask]
                    # array_w = array_w[valid_mask]

                    if len(array_x) == 0:
                        ax.axis('off')
                        continue

                    low_x, high_x = np.min(array_x), np.max(array_x)
                    low_y, high_y = np.min(array_y), np.max(array_y)
                    if use_log_bins:
                        if type_p == 'MultipleRI':
                            use_log_x = False
                            use_log_y = False

                        else:
                            use_log_x = use_log_bins and (j in log_bin_indices)
                            use_log_y = use_log_bins and (i in log_bin_indices)
                    else:
                        use_log_x = False
                        use_log_y = False
                    if use_log_x or use_log_y:

                        # Handle x-axis log binning
                        if use_log_x:
                            # Ensure all values are positive before log transform
                            # if low_x < 1:
                            #     low_x = epsilon
                            # array_x = np.maximum(array_x, epsilon)  # Clip to minimum epsilon
                            #
                            # # Transform x data to log space BEFORE mirroring
                            # array_x = np.log10(array_x)
                            # low_x = np.log10(low_x)
                            # high_x = np.log10(np.max(array_x))
                            clipped_x = np.where(array_x < 1, epsilon, array_x)
                            array_x = np.log10(clipped_x)
                            low_x = np.min(array_x)
                            high_x = np.max(array_x)
                            # print('X log range:', low_x, high_x)
                            # log_bins=(1,3, corner_bins)
                            # x_bins=np.concatenate([0,1],log_bins)

                            # Handle y-axis log binning
                        if use_log_y:
                            # # Ensure all values are positive before log transform
                            # if low_y < 1:
                            #     low_y = epsilon
                            # array_y = np.maximum(array_y, epsilon)  # Clip to minimum epsilon
                            #
                            # # Transform y data to log space BEFORE mirroring
                            # array_y = np.log10(array_y)
                            # low_y = np.log10(low_y)
                            # high_y = np.log10(np.max(array_y))
                            clipped_y = np.where(array_y < 1, epsilon, array_y)
                            array_y = np.log10(clipped_y)
                            low_y = np.min(array_y)
                            high_y = np.max(array_y)
                            # log_bins = (1, 3, corner_bins)
                            # y_bins = np.concatenate([0, 1], log_bins)
                            # print('Y log range:', low_y, high_y)

                    # Mirror data if needed
                    elif mirror_data:
                        array_x = np.concatenate((array_x, 2 * low_x - array_x))
                        array_y = np.concatenate((array_y, 2 * low_y - array_y))
                        array_w = np.concatenate((array_w, array_w))
                        # print('Entering Mirror')
                    # Recalculate ranges after mirroring
                    delta_x = deltas if type_step != 'bins' else (high_x - low_x) / corner_bins
                    delta_y = deltas if type_step != 'bins' else (high_y - low_y) / corner_bins
                    # print('X range:', low_x, high_x)
                    # print('Y range:', low_y, high_y)
                    # print('Deltas [x,y]: ', delta_x,delta_y)
                    if delta_x > delta_y:
                        bws = delta_x
                        scal_fac = delta_x / delta_y
                        array_y = array_y * scal_fac
                    elif delta_x < delta_y:
                        scal_fac = delta_y / delta_x
                        bws = delta_y
                        array_x = array_x * scal_fac
                    else:
                        bws = delta_x

                    x_bins = num_bins(array_x, delta_x)
                    y_bins = num_bins(array_y, delta_y)
                    # print(i,j)
                    # print('# of bins:', corner_bins, type_step)
                    # print('bins [x,y]: ',x_bins,y_bins)
                    # Create 2D KDE
                    [values] = np.dstack([array_x, array_y])
                    grid_points_x, grid_points_y = x_bins, y_bins
                    kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                    grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))

                    X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                    f = points.reshape(grid_points_x, grid_points_y).T

                    # Zero out values outside domain
                    xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                    ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                    if delta_x > delta_y:
                        ys = ys / scal_fac
                    elif delta_x < delta_y:
                        xs = xs / scal_fac
                    for ii in range(grid_points_y):
                        for jj in range(grid_points_x):
                            if high_x < xs[ii][jj] or xs[ii][jj] < low_x or \
                                    high_y < ys[ii][jj] or ys[ii][jj] < low_y:
                                f[ii][jj] = 0
                            elif mirror_data:
                                f[ii][jj] = f[ii][jj] * 2

                    # Find maximum
                    max_value = np.max(f)
                    max_indices = np.unravel_index(np.argmax(f), f.shape)
                    if delta_x > delta_y:
                        Y = Y / scal_fac
                    elif delta_x < delta_y:
                        X = X / scal_fac
                    if use_log_x or use_log_y:
                        if use_log_x:
                            X=10**X
                        if use_log_y:
                            Y=10**Y
                    max_x = X[max_indices[1]]
                    max_y = Y[max_indices[0]]


                    # Apply log scale if needed
                    if scale_t == 'Log':# or use_log_x or use_log_y:
                        f = np.log(f) / np.log(l_base)

                    # Plot contour
                    cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                    cset = ax.contour(X, Y, f, n_con, colors='k')
                    ax.clabel(cset, inline=1, fontsize=8)

                    # ADD GRIDLINES HERE
                    ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                    # Mark mode
                    sig_c = int(abs(np.log10(bws)))
                    ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5,label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, labels[j]))

                    ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5,label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, labels[i]))

                    # NEW: Set log scale for axes if using log bins
                    if use_log_x:
                        ax.set_xscale('log')
                        ax.set_xlim(10 ** low_x, 10 ** high_x)
                    else:
                        ax.set_xlim(low_x, high_x)

                    if use_log_y:
                        ax.set_yscale('log')
                        ax.set_ylim(10 ** low_y, 10 ** high_y)
                    else:
                        ax.set_ylim(low_y, high_y)

                    # ax.set_xlim(low_x, high_x)
                    # ax.set_ylim(low_y, high_y)

                    if i == n_vars - 1:
                        ax.set_xlabel(labels[j], fontsize=fo)
                    if j == 0:
                        ax.set_ylabel(labels[i], fontsize=fo)
                    ax.legend(fontsize=fo)

                # Diagonal: 1D marginal distributions
                elif i == j:
                    array_val = data[:, i]
                    array_w = weights.copy()

                    # Remove NaN values
                    # valid_mask = ~np.isnan(array_val)
                    # array_val = array_val[valid_mask]
                    # array_w = array_w[valid_mask]

                    if len(array_val) == 0:
                        ax.axis('off')
                        continue

                    # NEW: Apply log binning if enabled for this variable
                    if use_log_bins and (i in log_bin_indices) and type_p == 'NoLCPS':
                        # clipped_val = np.where(array_val < 1, epsilon, array_val)
                        # array_val = clipped_val
                        low_val = np.min(array_val)
                        high_val = np.max(array_val)
                        log_bins=np.logspace(0,3, corner_bins)
                        # val_bins=np.concatenate([[0.9,1],log_bins])
                        delta_val=1
                        val_bins = np.logspace(np.log10(low_val), np.log10(high_val),
                                               # 500)
                                               corner_bins)
                        # # Handle zero/negative values
                        # if low_val <= 0:
                        #     low_val = epsilon
                        #
                        # # Create log-spaced bins
                        # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                    else:
                        low_val = np.min(array_val)
                        high_val = np.max(array_val)

                        if mirror_data:
                            array_val = np.concatenate((array_val, 2 * low_val - array_val))
                            array_w = np.concatenate((array_w, array_w))

                        delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                        val_bins = num_bins(array_val, delta_val)
                    sig_val = int(abs(np.log10(delta_val)))
                    # Create KDE
                    x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=array_w)()

                    if mirror_data:
                        mask = (x_rs < low_val) | (high_val < x_rs)
                        y_rs[mask] = 0
                        y_rs = y_rs * 2

                    # Find peaks
                    if show_peaks:
                        mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                        y_rsP = y_rs[mask_P]
                        x_rsP = x_rs[mask_P]
                        peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                        peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                        if len(peaks_idr) > num_peaks:
                            peaks_idr = peaks_idr[:num_peaks]
                        colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                        for d, p in enumerate(peaks_idr):
                            peak_r = x_rsP[p]
                            ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                    # Find mode
                    max_xi = np.argmax(y_rs)
                    max_x = x_rs[max_xi]

                    # Plot
                    ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                    ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                    ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0,
                               label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                    # ADD VERTICAL GRIDLINES HERE
                    ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                    # ax.set_xlabel(region_labels[i], fontsize=fo)
                    ax.set_title(labels[i], fontsize=fo)
                    if use_log_bins and (i in log_bin_indices) and type_p == 'NoLCPS':
                        ax.set_xscale('log')
                        ax.set_xlim(0.01, high_val)
                        ax.set_ylim(bottom=low_pdf)
                        ax.set_yscale('log', base=l_base)
                        ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                    else:
                        ax.set_xlim(low_val, high_val)

                        if scale_t == 'Log':
                            ax.set_yscale('log', base=l_base)
                            ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        else:
                            ax.set_ylabel('PDF', fontsize=fo)

                    ax.legend(fontsize=fo)

                # Upper triangle: hide
                else:
                    ax.axis('off')
                ax.minorticks_on()
                ax.tick_params(axis='both', labelsize=fo)

        fig.suptitle(f'Layer {k} ({max_Alt} - {min_Alt} km) - Corner Plot',fontsize=fo + 4)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if save_cornerBP== True:
            if w_prior:
                corner_path = os.path.join(res_path, "MyCornerPlots_wPrior")
                if filter_values:
                    corner_path = os.path.join(res_path, "MyCornerPlots_wBoth")
            elif filter_values:
                corner_path = os.path.join(res_path, f"MyCornerPlots_wFilter")
            else:
                corner_path = os.path.join(res_path, "MyCornerPlots")
            os.makedirs(corner_path, exist_ok=True)
            if use_mode:
                fig.savefig(corner_path + r'\MyCornerPlot_Layer{}_jointW.png'.format(k))
            else:
                fig.savefig(corner_path + r'\MyCornerPlot_Layer{}.png'.format(k))
        if show_cornerBP==True:
            plt.show()
    if show_cornerMP or save_cornerMP:
        for n in range(n_modes):
            if n == 0:
                columns = [0, 1, 2, 9, 10, 11]
            elif n == 1:
                columns = [3, 4, 5, 12, 13, 14]
            elif n == 2:
                columns = [6, 7, 8, 15, 16, 17]

            if type_p == 'MultipleRI':
                columns= columns[:3]
            n_data = data[:, columns]
            n_weights = weights
            n_labels = labels[columns]
            # Determine number of variables
            n_vars = n_data.shape[1]

            # Create corner plot
            fig, axs = plt.subplots(n_vars, n_vars, figsize=(figSize_x * n_vars, figSize_y * n_vars))

            # Plot for each pair of variables
            for i in range(n_vars):
                for j in range(n_vars):
                    ax = axs[i, j]
                    # Lower triangle: 2D joint distributions
                    if i > j:
                        array_x = n_data[:, j]
                        array_y = n_data[:, i]
                        array_w = n_weights.copy()

                        # Remove NaN values
                        # valid_mask = ~np.isnan(array_x) & ~np.isnan(array_y)
                        # array_x = array_x[valid_mask]
                        # array_y = array_y[valid_mask]
                        # array_w = array_w[valid_mask]

                        if len(array_x) == 0:
                            ax.axis('off')
                            continue

                        low_x, high_x = np.min(array_x), np.max(array_x)
                        low_y, high_y = np.min(array_y), np.max(array_y)
                        if use_log_bins:
                            if type_p == 'MultipleRI':
                                use_log_x = False
                                use_log_y = False
                                pass
                            else:
                                use_log_x = use_log_bins and (j ==0)
                                use_log_y = use_log_bins and (i ==0)
                        else:
                            use_log_x = False
                            use_log_y = False
                        if use_log_x or use_log_y:

                            # Handle x-axis log binning
                            if use_log_x:
                                # Ensure all values are positive before log transform
                                # if low_x < 1:
                                #     low_x = epsilon
                                # array_x = np.maximum(array_x, epsilon)  # Clip to minimum epsilon
                                #
                                # # Transform x data to log space BEFORE mirroring
                                # array_x = np.log10(array_x)
                                # low_x = np.log10(low_x)
                                # high_x = np.log10(np.max(array_x))
                                clipped_x = np.where(array_x < 1, epsilon, array_x)
                                array_x = np.log10(clipped_x)
                                low_x = np.min(array_x)
                                high_x = np.max(array_x)

                                # log_bins=(1,3, corner_bins)
                                # x_bins=np.concatenate([0,1],log_bins)

                                # Handle y-axis log binning
                            if use_log_y:
                                # # Ensure all values are positive before log transform
                                # if low_y < 1:
                                #     low_y = epsilon
                                # array_y = np.maximum(array_y, epsilon)  # Clip to minimum epsilon
                                #
                                # # Transform y data to log space BEFORE mirroring
                                # array_y = np.log10(array_y)
                                # low_y = np.log10(low_y)
                                # high_y = np.log10(np.max(array_y))
                                clipped_y = np.where(array_y < 1, epsilon, array_y)
                                array_y = np.log10(clipped_y)
                                low_y = np.min(array_y)
                                high_y = np.max(array_y)
                                # log_bins = (1, 3, corner_bins)
                                # y_bins = np.concatenate([0, 1], log_bins)

                        # Mirror data if needed
                        elif mirror_data:
                            array_x = np.concatenate((array_x, 2 * low_x - array_x))
                            array_y = np.concatenate((array_y, 2 * low_y - array_y))
                            array_w = np.concatenate((array_w, array_w))
                        # Recalculate ranges after mirroring

                        delta_x = deltas if type_step != 'bins' else (high_x - low_x) / corner_bins
                        delta_y = deltas if type_step != 'bins' else (high_y - low_y) / corner_bins
                        # print(i,j)
                        # print('X range:', low_x, high_x)
                        # print('Y range:', low_y, high_y)
                        # print('Deltas [x,y]: ', delta_x,delta_y)
                        if delta_x > delta_y:
                            bws = delta_x
                            scal_fac = delta_x / delta_y
                            array_y = array_y * scal_fac
                        elif delta_x < delta_y:
                            scal_fac = delta_y / delta_x
                            bws = delta_y
                            array_x = array_x * scal_fac
                        else:
                            bws = delta_x
                        # print(delta_y, array_y)
                        x_bins = num_bins(array_x, delta_x)
                        y_bins = num_bins(array_y, delta_y)

                        # print(x_bins,y_bins)
                        # Create 2D KDE
                        [values] = np.dstack([array_x, array_y])
                        grid_points_x, grid_points_y = x_bins, y_bins
                        kde = FFTKDE(kernel='gaussian', bw=bws, norm=2)
                        grid, points = kde.fit(values, weights=array_w).evaluate((grid_points_x, grid_points_y))

                        X, Y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
                        f = points.reshape(grid_points_x, grid_points_y).T

                        # Zero out values outside domain
                        xs = grid[:, 0].reshape(grid_points_x, grid_points_y).T
                        ys = grid[:, 1].reshape(grid_points_x, grid_points_y).T
                        if delta_x > delta_y:
                            ys = ys / scal_fac
                        elif delta_x < delta_y:
                            xs = xs / scal_fac
                        for ii in range(grid_points_y):
                            for jj in range(grid_points_x):
                                if high_x < xs[ii][jj] or xs[ii][jj] < low_x or \
                                        high_y < ys[ii][jj] or ys[ii][jj] < low_y:
                                    f[ii][jj] = 0
                                elif mirror_data:
                                    f[ii][jj] = f[ii][jj] * 2

                        # Find maximum
                        max_value = np.max(f)
                        max_indices = np.unravel_index(np.argmax(f), f.shape)
                        if delta_x > delta_y:
                            Y = Y / scal_fac
                        elif delta_x < delta_y:
                            X = X / scal_fac
                        if use_log_x or use_log_y:
                            if use_log_x:
                                X = 10 ** X
                            if use_log_y:
                                Y = 10 ** Y
                        max_x = X[max_indices[1]]
                        max_y = Y[max_indices[0]]

                        # Apply log scale if needed
                        if scale_t == 'Log':  # or use_log_x or use_log_y:
                            f = np.log(f) / np.log(l_base)

                        # Plot contour
                        cfset = ax.contourf(X, Y, f, 100, cmap=my_colorMap)
                        cset = ax.contour(X, Y, f, n_con, colors='k')
                        ax.clabel(cset, inline=1, fontsize=8)

                        # ADD GRIDLINES HERE
                        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

                        # Mark mode
                        sig_c = int(abs(np.log10(bws)))
                        ax.axvline(x=max_x, color='green', linestyle='--', linewidth=1.5,
                                   label='Mode {2}: {0:.{1}f}'.format(max_x, sig_c, n_labels[j]))

                        ax.axhline(y=max_y, color='green', linestyle='--', linewidth=1.5,
                                   label='Mode {2}: {0:.{1}f}'.format(max_y, sig_c, n_labels[i]))

                        # NEW: Set log scale for axes if using log bins
                        if use_log_x:
                            ax.set_xscale('log')
                            ax.set_xlim(10 ** low_x, 10 ** high_x)
                        else:
                            ax.set_xlim(low_x, high_x)

                        if use_log_y:
                            ax.set_yscale('log')
                            ax.set_ylim(10 ** low_y, 10 ** high_y)
                        else:
                            ax.set_ylim(low_y, high_y)

                        # ax.set_xlim(low_x, high_x)
                        # ax.set_ylim(low_y, high_y)

                        if i == n_vars - 1:
                            ax.set_xlabel(n_labels[j], fontsize=fo)
                        if j == 0:
                            ax.set_ylabel(n_labels[i], fontsize=fo)
                        ax.legend(fontsize=fo)

                    # Diagonal: 1D marginal distributions
                    elif i == j:
                        array_val = n_data[:, i]
                        array_w = n_weights.copy()

                        # Remove NaN values
                        # valid_mask = ~np.isnan(array_val)
                        # array_val = array_val[valid_mask]
                        # array_w = array_w[valid_mask]

                        if len(array_val) == 0:
                            ax.axis('off')
                            continue

                        # NEW: Apply log binning if enabled for this variable
                        if use_log_bins and (i ==0):
                            clipped_val = np.where(array_val < 1, epsilon, array_val)
                            array_val = clipped_val
                            low_val = np.min(array_val)
                            high_val = np.max(array_val)
                            log_bins = np.logspace(0, 3, corner_bins)
                            # print(log_bins)
                            val_bins = np.concatenate([[0.9, 1], log_bins])
                            delta_val = 1
                            # # Handle zero/negative values
                            # if low_val <= 0:
                            #     low_val = epsilon
                            #
                            # # Create log-spaced bins
                            # val_bins = np.logspace(np.log10(low_val + epsilon), np.log10(high_val + epsilon), num_bins(array_val, delta_val))
                        else:
                            low_val = np.min(array_val)
                            high_val = np.max(array_val)
                            use_log_x = False
                            use_log_y = False

                            if mirror_data:
                                array_val = np.concatenate((array_val, 2 * low_val - array_val))
                                array_w = np.concatenate((array_w, array_w))

                            delta_val = deltas if type_step != 'bins' else (high_val - low_val) / corner_bins
                            val_bins = num_bins(array_val, delta_val)
                        sig_val = int(abs(np.log10(delta_val)))
                        # Create KDE
                        x_rs, y_rs = FFTKDE(kernel='gaussian', bw=delta_val).fit(array_val, weights=array_w)()

                        if mirror_data:
                            mask = (x_rs < low_val) | (high_val < x_rs)
                            y_rs[mask] = 0
                            y_rs = y_rs * 2

                        # Find peaks
                        if show_peaks:
                            mask_P = (x_rs >= low_val) & (high_val >= x_rs)
                            y_rsP = y_rs[mask_P]
                            x_rsP = x_rs[mask_P]
                            peaks_idr, peak_r = find_peaks(y_rsP, height=0)
                            peaks_idr = peaks_idr[np.argsort(-y_rsP[peaks_idr])]
                            if len(peaks_idr) > num_peaks:
                                peaks_idr = peaks_idr[:num_peaks]
                            colors = plt.cm.winter(np.linspace(0, 1, len(peaks_idr)))
                            for d, p in enumerate(peaks_idr):
                                peak_r = x_rsP[p]
                                ax.axvline(x=peak_r, color=colors[d], label=f'Peak {d}= {peak_r:.{sig_h}f}')

                        # Find mode
                        max_xi = np.argmax(y_rs)
                        max_x = x_rs[max_xi]

                        # Plot
                        ax.hist(array_val, bins=val_bins, weights=array_w, density=True, color='grey', alpha=0.5)
                        ax.plot(x_rs, y_rs, color='black', label='Fitted Curve')
                        ax.axvline(x=max_x, color='red', linestyle='--', linewidth=2.0,
                                   label='Mode: {0:.{1}f}'.format(max_x, sig_val))

                        # ADD VERTICAL GRIDLINES HERE
                        ax.grid(True, axis='x', which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                        # ax.set_xlabel(region_labels[i], fontsize=fo)
                        ax.set_title(n_labels[i], fontsize=fo)
                        if use_log_bins and (i ==0):
                            ax.set_xscale('log')
                            ax.set_xlim(low_val, high_val)
                            ax.set_ylim(bottom=low_pdf)
                            ax.set_yscale('log', base=l_base)
                            ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                        else:
                            ax.set_xlim(low_val, high_val)

                            if scale_t == 'Log':
                                ax.set_yscale('log', base=l_base)
                                ax.set_ylabel(f'Log{l_base} (PDF)', fontsize=fo)
                            else:
                                ax.set_ylabel('PDF', fontsize=fo)

                        ax.legend(fontsize=fo)

                    # Upper triangle: hide
                    else:
                        ax.axis('off')
                    ax.minorticks_on()
                    ax.tick_params(axis='both', labelsize=fo)
            if w_prior:
                fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km with Prior'.format(n + 1, max_Alt, min_Alt), fontsize=fo)
            else:
                fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km'.format(n+1, max_Alt, min_Alt), fontsize=fo)
            plt.tight_layout()
            if save_cornerMP == True:
                if w_prior:
                    corner_path = os.path.join(res_path, "MyCornerPlots_wPrior")
                    if filter_values:
                        corner_path = os.path.join(res_path, "MyCornerPlots_wBoth")
                elif filter_values:
                    corner_path = os.path.join(res_path, f"MyCornerPlots_wFilter")
                else:
                    corner_path = os.path.join(res_path, "MyCornerPlots")
                os.makedirs(corner_path, exist_ok=True)
                filename_parts = [f'MyCornerPlot_Layer{k}_Mode{n}']
                if use_mode:
                    filename_parts.append('jointW')
                if scale_type == 'Log':
                    filename_parts.append('Log')
                filename = '_'.join(filename_parts) + '.png'
                fig.savefig(os.path.join(corner_path, filename))
                # if use_mode:
                #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}_jointW.png'.format(j, n))
                # else:
                #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}.png'.format(j, n))
            if show_cornerMP == True:
                plt.show()
    if save_cornerDMean== True:
        quants = []
        for i in range(len(labels)-1):
            array_val = data[:,i]
            title_quantiles = [0.16, 0.5, 0.84]
            q_lo, q_mid, q_hi = quantile(array_val, title_quantiles, weights=weights)
            quants.append(q_mid)
        quants.append(0) #Adding Weight
        quants = np.array(quants)
        if w_prior:
            corner_path = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        filename = os.path.join(corner_path, f"PosDistLayer{k}_CornerMeans.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(quants)
        print(f"Data saved to {filename}")
    if save_cornerDMode:
        max_vals =my_cornerMaxVal(n_modes, data, weights, labels)
        max_vals[-1]=0
        max_vals = np.array(max_vals)
        # if use_mode:
        #     print(max_vals)
        if w_prior:
            corner_path = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        filename = os.path.join(corner_path, f"PosDistLayer{k}_CornerModes.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(max_vals)
        print(f"Data saved to {filename}")
    plt.close('all')
def my_corner_Region_old(Layers, n_Ls, paths, data_dir, probe=probe, use_mode=use_mode, type_p=type_p, show_cornerBP=show_cornerBP_region, save_cornerBP=save_cornerBPlot_region, save_cornerDMean=save_cornerDisMean_region, show_cornerMP=show_cornerMP_region, save_cornerMP=save_cornerMPlot_region, save_cornerDMode=save_cornerDisMode_region, corner_bins=corner_bins,smooth_corner=smooth_corner,smooth_val=smooth_val, fo=fo ):
    """
    Creates histograms for atmospheric regions by aggregating layers within each region.
    """
    region_yMax = [65, 57, 50, 48, 46]
    region_yMin = [57, 50, 48, 46, 32]
    # region_val = [[1.44], [1.38, 1.40], [1.33, 1.44], [1.44], [np.nan]]
    region_name = ['D', 'C', 'B', 'A', 'L']
    res_path = os.path.join(paths, "Results")
    if w_prior:
        mode_paths = os.path.join(res_path, "My_Distributions_wPrior")
        if filter_values:
            mode_paths = os.path.join(res_path, f"My_Distributions_wBoth")
    elif filter_values:
        mode_paths = os.path.join(res_path, f"My_Distributions_wFilter")
    else:
        mode_paths = os.path.join(res_path, "My_Distributions")

    # Process each region
    for r in range(len(region_name)):
        r_max = region_yMax[r]
        r_min = region_yMin[r]

        # Collect data from all layers within this region
        region_data = []
        region_weights = []
        region_labels = None
        for i, j in enumerate(Layers):
            y_min = array_MinAlt[j]
            y_max = array_MaxAlt[j]
            n_modes=n_Ls[i]

            # Check if layer falls within current region
            if y_max <= r_max and r_min <= y_min:
                if use_mean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_Mean.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerModes.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_cornerMean:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_CornerMeans.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                elif use_mode:
                    pos_dis = os.path.join(mode_paths, f"PosDistLayer{j}_jointW.csv")
                    data, weights, labels = read_myPostDist_CSV(pos_dis, n_modes)
                else:
                    dis_path = os.path.join(paths, "PosteriorDistributions")
                    pos_dis = os.path.join(dis_path, f"PosDistLayer{j}.csv")
                    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                    if filter_values:
                        for f_what in filter_what:
                            if f_what == 'nums':
                                if type_p == 'MultipleRI':
                                    continue
                                else:
                                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                                    mask = (n_T >= filt_n_min) & (n_T <= filt_n_max)
                            if f_what == 'index':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                                    elif n_modes == 2:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (
                                                    filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                                    else:
                                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (
                                                    filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & (
                                                           filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                                else:
                                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (
                                                filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & (
                                                       filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
                            if f_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    if n_modes == 1:
                                        mask = (1 + filt_s >= data[:, 1])
                                    elif n_modes == 2:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                                    else:
                                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & (
                                                    1 + filt_s >= data[:, 7])
                                else:
                                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & (
                                                1 + filt_s >= data[:, 16])

                            data = data[mask]
                            weights = weights[mask]
                            # print(len(data))
                    if w_prior:
                        for p, p_what in enumerate(prior_what):
                            sel_val = []
                            if p_what == 'index':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 1)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(11 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels,t_prior=type_Prior[p])
                            if p_what == 'sigma':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 2)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(10 + (3 * i))
                                # print(sel_val)
                                weights, name_prior = sigma_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                            if p_what == 'rho':
                                if type_p == 'MultipleRI':
                                    for i in range(n_modes):
                                        sel_val.append((3 * (i + 1)) - 3)
                                else:
                                    for i in range(n_modes):
                                        sel_val.append(9 + (3 * i))
                                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels,t_prior=type_Prior[p], m_val=prior_m_rho,unc_val=prior_unc_rho, low_val=-1, high_val=1)
                        # print(f'Applying Prior to: {sel_val}')
                        # weights, name_prior = singleValue_Prior(sel_val, data, weights, header)
                            if p_what == 'rho_diagonal':
                                # Apply diagonal prior to favor similar rho values
                                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights, val_labels=labels, n_modes=n_modes, type_p=type_p, sigma_diagonal=0.1, prior_min=-1.0, prior_max=1.0, prior_sig=3)
                            if p_what == 'num_total':
                                for i in range(n_modes):
                                    sel_val.append((3 * i))
                                weights, name_prior = num_Prior(sel_val, data, weights, t_prior=type_Prior[p])
                # data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
                region_data.append(data)
                region_weights.append(weights)
                if region_labels is None:
                    region_labels = labels
                elif len(labels)>len(region_labels):
                    region_labels=labels

        # Skip if no layers in this region
        if len(region_data) == 0:
            continue
        # else:
        #     print(len(region_data))
        # Combine data from all layers in region
        max_cols = max(data.shape[1] for data in region_data)
        n_modes = int(max_cols/3)


        # Pad all arrays to match max_cols
        padded_data = []
        for data in region_data:
            if data.shape[1] < max_cols:
                # Pad with NaN to preserve original values
                padding = np.full((data.shape[0], max_cols - data.shape[1]), np.nan)
                data = np.hstack([data, padding])
            padded_data.append(data)

        combined_data = np.vstack(padded_data)
        # combined_data = np.vstack(region_data)
        combined_weights = np.concatenate(region_weights)

        # Normalize weights
        combined_weights = combined_weights / np.sum(combined_weights)

        # NEW: Identify columns with valid data (not all NaN)
        valid_cols = []
        valid_labels = []
        for col_idx in range(combined_data.shape[1]):
            col = combined_data[:, col_idx]
            # Check if column has at least some valid (non-NaN) values
            if np.isfinite(col).any():
                valid_cols.append(col_idx)
                valid_labels.append(region_labels[col_idx])

        # Extract only valid columns
        combined_data = combined_data[:, valid_cols]

        if smooth_corner == False:
            smooth_val = None

        if show_cornerBP or save_cornerBP:
            if corner_bins == 0:
                fig = corner.corner(combined_data, weights=combined_weights, labels=region_labels, smooth=smooth_val, show_titles=True, title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
            else:
                fig = corner.corner(combined_data, weights=combined_weights, labels=region_labels, bins=corner_bins, smooth=smooth_val, show_titles=True, title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
            if w_prior:
                fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km)  with Prior - Combined Corner Plot', fontsize=fo)
            else:
                fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) - Combined Corner Plot', fontsize=fo)
            plt.tight_layout(rect=[0, 0, 1, 0.98]) #[Left, Bottom, Right, Top]
            if save_cornerBP == True:
                if w_prior:
                    corner_path = os.path.join(res_path, "CornerPlots_wPrior_Region")
                    if filter_values:
                        corner_path = os.path.join(res_path, "CornerPlots_wBoth_Region")
                elif filter_values:
                    corner_path = os.path.join(res_path, f"CornerPlots_wFilter_Region")
                else:
                    corner_path = os.path.join(res_path, "CornerPlots_Region")
                os.makedirs(corner_path, exist_ok=True)
                if use_mode:
                    fig.savefig(os.path.join(corner_path, f'CornerPlot_Region{region_name[r]}_jointW.png'))
                else:
                    fig.savefig(os.path.join(corner_path, f'CornerPlot_Region{region_name[r]}.png'))
            if show_cornerBP == True:
                plt.show()
        if show_cornerMP or save_cornerMP:
            for n in range(n_modes):
                if n == 0:
                    columns = [0, 1, 2, 9, 10, 11]
                elif n == 1:
                    columns = [3, 4, 5, 12, 13, 14]
                elif n == 2:
                    columns = [6, 7, 8, 15, 16, 17]

                if type_p == 'MultipleRI':
                    columns = columns[:3]
                n_data = combined_data[:, columns]
                n_weights = combined_weights
                n_labels = region_labels[columns]
                if corner_bins == 0:
                    fig = corner.corner(n_data, weights=n_weights, labels=n_labels, smooth=smooth_val, show_titles=True,title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo}, axis='both',labelsize=fo)
                else:
                    fig = corner.corner(n_data, weights=n_weights, labels=n_labels, bins=corner_bins, smooth=smooth_val,show_titles=True, title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo},axis='both', labelsize=fo)
                if w_prior:
                    fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km)  with Prior - Combined Corner Plot Mode {n+1}', fontsize=fo)
                else:
                    fig.suptitle(f'Region {region_name[r]} ({r_max:.1f} - {r_min:.1f} km) - Combined Corner Plot Mode {n+1}',fontsize=fo)
                plt.tight_layout(rect=[0, 0, 1, 0.98]) #[Left, Bottom, Right, Top]
                if save_cornerMP == True:
                    if w_prior:
                        corner_path = os.path.join(res_path, "CornerPlots_wPrior_Region")
                        if filter_values:
                            corner_path = os.path.join(res_path, "CornerPlots_wBoth_Region")
                    elif filter_values:
                        corner_path = os.path.join(res_path, f"CornerPlots_wFilter_Region")
                    else:
                        corner_path = os.path.join(res_path, "CornerPlots_Region")
                    os.makedirs(corner_path, exist_ok=True)
                    filename_parts = [f'CornerPlot_Region{region_name[r]}_Mode{n}']
                    if use_mode:
                        filename_parts.append('jointW')
                    if scale_type == 'Log':
                        filename_parts.append('Log')
                    filename = '_'.join(filename_parts) + '.png'
                    fig.savefig(os.path.join(corner_path, filename))
                    # if use_mode:
                    #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}_jointW.png'.format(j, n))
                    # else:
                    #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}.png'.format(j, n))
                if show_cornerMP == True:
                    plt.show()
        if save_cornerDMean == True:
            quants = []
            for i in range(len(labels) - 1):
                array_val = data[:, i]
                title_quantiles = [0.16, 0.5, 0.84]
                q_lo, q_mid, q_hi = quantile(array_val, title_quantiles, weights=weights)
                quants.append(q_mid)
            quants.append(0)  # Adding Weight
            quants = np.array(quants)
            if w_prior:
                corner_path = os.path.join(res_path, "My_Distributions_wPrior")
                if filter_values:
                    corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
            elif filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
            else:
                corner_path = os.path.join(res_path, "My_Distributions_Region")
            os.makedirs(corner_path, exist_ok=True)
            filename = os.path.join(corner_path, f"PosDistRegion{region_name[r]}_CornerMeans.csv")
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(labels)
                writer.writerow(quants)
            print(f"Data saved to {filename}")
        if save_cornerDMode:
            max_vals = my_cornerMaxVal(n_modes, data, weights, labels)
            max_vals[-1] = 0
            max_vals = np.array(max_vals)
            # if use_mode:
            #     print(max_vals)
            if w_prior:
                corner_path = os.path.join(res_path, "My_Distributions_wPrior")
                if filter_values:
                    corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
            elif filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
            else:
                corner_path = os.path.join(res_path, "My_Distributions")
            os.makedirs(corner_path, exist_ok=True)
            filename = os.path.join(corner_path, f"PosDistRegion{region_name[r]}_CornerModes.csv")
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(labels)
                writer.writerow(max_vals)
            print(f"Data saved to {filename}")
    plt.close('all')
def my_cornerplotNew(j, n_modes,y_min, y_max, paths, use_mode=use_mode, type_p=type_p, show_cornerBP=show_cornerBP, save_cornerBP=save_cornerBPlot, save_cornerDMean=save_cornerDisMean, show_cornerMP=show_cornerMP, save_cornerMP=save_cornerMPlot, save_cornerDMode=save_cornerDisMode, corner_bins=corner_bins,smooth_corner=smooth_corner,smooth_val=smooth_val):
    res_path = os.path.join(paths, "Results")
    dis_path = os.path.join(paths, "PosteriorDistributions")
    pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(j))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    if filter_values:
        for f_what in filter_what:
            if f_what == 'nums':
                if type_p == 'MultipleRI':
                    continue
                else:
                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                    mask = (n_T >= filt_n_min) & (n_T<=filt_n_max)
            if f_what == 'index':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                    elif n_modes == 2:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min)
                    else:
                        mask = (filt_max >= data[:, 2]) & (data[:, ] >= filt_min) & (filt_max >= data[:, 5]) & (data[:, 5] >= filt_min) & (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                else:
                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (filt_max >= data[:, 14]) & (data[:, 14] >= filt_min) & (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            if f_what == 'sigma':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (1 + filt_s >= data[:, 1])
                    elif n_modes == 2:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4])
                    else:
                        mask = (1 + filt_s >= data[:, 1]) & (filt_s >= data[:, 4]) & (1 + filt_s >= data[:, 7])
                else:
                    mask = (1 + filt_s >= data[:, 10]) & (filt_s >= data[:, 13]) & (1 + filt_s >= data[:, 16])
            data = data[mask]
            weights = weights[mask]
    if w_prior:
        for p, p_what in enumerate(prior_what):
            sel_val = []
            if p_what == 'index':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 1)
                else:
                    for i in range(n_modes):
                        sel_val.append(11 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p])
            if p_what == 'sigma':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 2)
                else:
                    for i in range(n_modes):
                        sel_val.append(10 + (3 * i))
                # print(sel_val)
                weights, name_prior = sigma_Prior(sel_val, data, weights,t_prior=type_Prior[p])
            if p_what == 'rho':
                if type_p == 'MultipleRI':
                    for i in range(n_modes):
                        sel_val.append((3 * (i + 1)) - 3)
                else:
                    for i in range(n_modes):
                        sel_val.append(9 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels, t_prior=type_Prior[p], m_val=0,unc_val=prior_unc_rho, low_val=-1, high_val=1)
            if p_what == 'rho_diagonal':
                # Apply diagonal prior to favor similar rho values
                weights, name_prior = diagonalPair_Prior(sel_val=[], val_arr=data, val_weight=weights,val_labels=labels,n_modes=n_modes,type_p=type_p,sigma_diagonal=0.1,prior_min=-1.0, prior_max=1.0,prior_sig=3)
            if p_what=='num_total':
                for i in range(n_modes):
                    sel_val.append((3 * i))
                weights, name_prior = num_Prior(sel_val, data, weights,t_prior=type_Prior[p])
    if use_mode==True:
        all_sorted = find_weighted_mode(data, weights)
        weighted_mode = []
        frequency = []
        for mode, freq in all_sorted.items():
            weighted_mode.append(mode)
            frequency.append(freq)
        data = np.array(weighted_mode)
        weights = np.array(frequency)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    if smooth_corner == False:
        smooth_val = None
    if show_cornerBP or save_cornerBP:
        if corner_bins==0:
            fig = corner.corner(data, weights=weights, labels=labels, smooth=smooth_val, show_titles=True,title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
        else:
            fig = corner.corner(data, weights=weights, labels=labels,bins=corner_bins, smooth=smooth_val, show_titles=True, title_kwargs={"fontsize": fo},label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
        if w_prior:
            fig.suptitle('Corner Plot for Layer {} - {}km with Prior'.format(max_Alt, min_Alt), fontsize=fo)
        else:
            fig.suptitle('Corner Plot for Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)
        plt.tight_layout()
        if save_cornerBP== True:
            if w_prior:
                corner_path = os.path.join(res_path, "CornerPlots_wPrior")
                if filter_values:
                    corner_path = os.path.join(res_path, "CornerPlots_wBoth")
            elif filter_values:
                corner_path = os.path.join(res_path, f"CornerPlots_wFilter")
            else:
                corner_path = os.path.join(res_path, "CornerPlots")
            os.makedirs(corner_path, exist_ok=True)
            if use_mode:
                fig.savefig(corner_path + r'\CornerPlot_Layer{}_jointW.png'.format(j))
            else:
                fig.savefig(corner_path + r'\CornerPlot_Layer{}.png'.format(j))
        if show_cornerBP==True:
            plt.show()
    if show_cornerMP or save_cornerMP:
        for n in range(n_modes):
            if n == 0:
                columns = [0, 1, 2, 9, 10, 11]
            elif n == 1:
                columns = [3, 4, 5, 12, 13, 14]
            elif n == 2:
                columns = [6, 7, 8, 15, 16, 17]

            if type_p == 'MultipleRI':
                columns= columns[:3]
            n_data = data[:, columns]
            n_weights = weights
            n_labels = labels[columns]
            if corner_bins==0:
                fig = corner.corner(n_data, weights=n_weights, labels=n_labels, smooth=smooth_val, show_titles=True, title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo}, axis='both',labelsize=fo)
            else:
                fig = corner.corner(n_data, weights=n_weights, labels=n_labels, bins=corner_bins, smooth=smooth_val, show_titles=True, title_kwargs={"fontsize": fo}, label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
            if w_prior:
                fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km with Prior'.format(n + 1, max_Alt, min_Alt), fontsize=fo)
            else:
                fig.suptitle('Corner Plot for Mode {},  Layer {} - {}km'.format(n+1, max_Alt, min_Alt), fontsize=fo)
            plt.tight_layout()
            if save_cornerMP == True:
                if w_prior:
                    corner_path = os.path.join(res_path, "CornerPlots_wPrior")
                    if filter_values:
                        corner_path = os.path.join(res_path, "CornerPlots_wBoth")
                elif filter_values:
                    corner_path = os.path.join(res_path, f"CornerPlots_wFilter")
                else:
                    corner_path = os.path.join(res_path, "CornerPlots")
                os.makedirs(corner_path, exist_ok=True)
                filename_parts = [f'CornerPlot_Layer{j}_Mode{n}']
                if use_mode:
                    filename_parts.append('jointW')
                if scale_type == 'Log':
                    filename_parts.append('Log')
                filename = '_'.join(filename_parts) + '.png'
                fig.savefig(os.path.join(corner_path, filename))
                # if use_mode:
                #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}_jointW.png'.format(j, n))
                # else:
                #     fig.savefig(corner_path + r'\CornerPlot_Layer{}_Mode{}.png'.format(j, n))
            if show_cornerMP == True:
                plt.show()
    if save_cornerDMean== True:
        quants = []
        for i in range(len(labels)-1):
            array_val = data[:,i]
            title_quantiles = [0.16, 0.5, 0.84]
            q_lo, q_mid, q_hi = quantile(array_val, title_quantiles, weights=weights)
            quants.append(q_mid)
        quants.append(0) #Adding Weight
        quants = np.array(quants)
        if w_prior:
            corner_path = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        filename = os.path.join(corner_path, f"PosDistLayer{j}_CornerMeans.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(quants)
        print(f"Data saved to {filename}")
    if save_cornerDMode:
        max_vals =my_cornerMaxVal(n_modes, data, weights, labels)
        max_vals[-1]=0
        max_vals = np.array(max_vals)
        # if use_mode:
        #     print(max_vals)
        if w_prior:
            corner_path = os.path.join(res_path, "My_Distributions_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, f"My_Distributions_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"My_Distributions_wFilter")
        else:
            corner_path = os.path.join(res_path, "My_Distributions")
        os.makedirs(corner_path, exist_ok=True)
        filename = os.path.join(corner_path, f"PosDistLayer{j}_CornerModes.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(labels)
            writer.writerow(max_vals)
        print(f"Data saved to {filename}")
    plt.close('all')
def my_cornerplotN_old(j, n_modes,y_min, y_max, paths, use_mode=use_mode, show_corner=show_cornerN, save_cornerP=save_cornerN, save_cornerD=save_cornerDisMean):
    res_path = os.path.join(paths, "Results")
    if use_mode==True:
        mode_paths = os.path.join(res_path, "My_Distributions")
        pos_dis = os.path.join(mode_paths, "PosDistLayer{}_jointW.csv".format(j))
    else:
        dis_path = os.path.join(paths, "PosteriorDistributions")
        pos_dis = os.path.join(dis_path, "PosDistLayer{}.csv".format(j))
    data, weights, labels = read_PostDist_CSV(pos_dis, n_modes)
    if type_p == 'MultipleRI':
        print('Does not make sense to plot Ns')
        return
    if filter_values:
        for f_what in filter_what:
            if f_what == 'nums':
                if type_p == 'MultipleRI':
                    continue
                else:
                    n_T = data[:, 0] + data[:, 3] + data[:, 6]
                    mask = (n_T >= filt_n_min) & (n_T<=filt_n_max)
            if f_what == 'index':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min)
                    elif n_modes == 2:
                        mask = (filt_max >= data[:, 2]) & (data[:, 2] >= filt_min) & (filt_max >= data[:, 5]) & (
                                    data[:, 5] >= filt_min)
                    else:
                        mask = (filt_max >= data[:, 2]) & (data[:, ] >= filt_min) & (filt_max >= data[:, 5]) & (
                                    data[:, 5] >= filt_min) & (filt_max >= data[:, 8]) & (data[:, 8] >= filt_min)
                else:
                    mask = (filt_max >= data[:, 11]) & (data[:, 11] >= filt_min) & (filt_max >= data[:, 14]) & (
                                data[:, 14] >= filt_min) & (filt_max >= data[:, 17]) & (data[:, 17] >= filt_min)
            if f_what == 'sigma':
                if type_p == 'MultipleRI':
                    if n_modes == 1:
                        mask = (1+filt_s >= data[:, 1])
                    elif n_modes == 2:
                        mask = (1+filt_s >= data[:, 1]) &(filt_s >= data[:, 4])
                    else:
                        mask = (1+filt_s >= data[:, 1]) &(filt_s >= data[:, 4]) &(1+filt_s >= data[:, 7])
                else:
                    mask = (1+filt_s >= data[:, 10]) &(filt_s >= data[:, 13]) &(1+filt_s >= data[:, 16])
            data = data[mask]
            weights = weights[mask]

    if w_prior:
        for p_what in prior_what:
            sel_val = []
            if p_what == 'index':
                if type_p == 'MultipleRI':
                    print('Does not make sense to plot Ns')
                    return
                else:
                    for i in range(n_modes):
                        sel_val.append(11 + (3 * i))
                weights, name_prior = singleValue_Prior(sel_val, data, weights, labels)
            if p_what == 'sigma':
                if type_p == 'MultipleRI':
                    print('Does not make sense to plot Ns')
                    return
                else:
                    for i in range(n_modes):
                        sel_val.append(10 + (3 * i))
                weights, name_prior = sigma_Prior(sel_val, data, weights)
    max_Alt = format(y_max, ".2f")
    min_Alt = format(y_min, ".2f")
    columns = [0, 3, 6]
    # columns = [9, 12, 15]
    n_data = data[:, columns]
    n_weights = weights
    n_labels = labels[columns]
    fig = corner.corner(n_data, weights=n_weights, labels=n_labels, show_titles=True, title_kwargs={"fontsize": fo},label_kwargs={"fontsize": fo}, axis='both', labelsize=fo)
    fig.suptitle('Corner Plot for N values,  Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)
    axes = np.array(fig.axes).reshape((n_data.shape[1], n_data.shape[1]))
    if scale_type== 'Log':
        for i in range(n_data.shape[1]):
            # Set x-axis to log scale for diagonal plots
            axes[i, i].set_xscale('log')
            axes[i, i].set_xlim((0.9, 1000))
            # Set both axes to log scale for 2D histograms
            for j in range(i):
                axes[i, j].set_xlim((0.9, 1000))
                axes[i, j].set_xscale('log')
                axes[i, j].set_yscale('log')
    # fig.suptitle(r'Corner Plot for $\rho$ values,  Layer {} - {}km'.format(max_Alt, min_Alt), fontsize=fo)
    plt.tight_layout()
    if save_cornerP== True:
        if w_prior:
            corner_path = os.path.join(res_path, f"CornerPlots_wPrior")
            if filter_values:
                corner_path = os.path.join(res_path, "CornerPlots_wBoth")
        elif filter_values:
            corner_path = os.path.join(res_path, f"CornerPlots_wFilter")
        else:
            corner_path = os.path.join(res_path, "CornerPlots")
        os.makedirs(corner_path, exist_ok=True)
        filename_parts = [f'CornerPlot_Layer{j}_N']
        if use_mode:
            filename_parts.append('jointW')
        if scale_type == 'Log':
            filename_parts.append('Log')
        filename = '_'.join(filename_parts) + '.png'
        fig.savefig(os.path.join(corner_path, filename))
        # fig.savefig(corner_path + r'\CornerPlot_Layer{}_N.png'.format(j))
    if show_corner==True:
        plt.show()
    # my_cornerMaxVal(4,data,weights,labels)

data_dir = os.path.dirname(__file__)
size_dis, array_MinAlt, array_MaxAlt = read_LCPS_Vals(data_dir)
if type_p != 'MultipleRI':
    n_modes = 3
if probe== 'sounder':
    if re_run:
        paths = os.path.join(data_dir, f"NS_{probe}_{type_p}_{type_a}_{analysis_strat}_{type_s}_numL{n_Live}_new")
    else:
        paths = os.path.join(data_dir, f"NS_{probe}_{type_p}_{type_a}_{analysis_strat}_{type_s}_numL{n_Live}")

else:
    if re_run:
        paths = os.path.join(data_dir, f"NS_{probe}_{type_a}_{analysis_strat}_{type_s}_numL{n_Live}_new")
    else:
        paths = os.path.join(data_dir, f"NS_{probe}_{type_a}_{analysis_strat}_{type_s}_numL{n_Live}")
    # paths = os.path.join(data_dir, f"NS_{probe}_{type_a}_{analysis_strat}_{type_s}_numL{n_Live}")

#Main Loop
dis_path = os.path.join(paths, "PosteriorDistributions")

if analysis_strat == 'Layer':
    files = [f for f in os.listdir(dis_path) if f.startswith('PosDistLayer') and f.endswith('.csv')]
elif analysis_strat == 'independent':
    files = [f for f in os.listdir(dis_path) if f.startswith('PosDist_') and f.endswith('.csv')]
len_f = len(files)
print(f'Number of Layers Found {len_f}')
res_path = os.path.join(paths, "Results")
os.makedirs(res_path, exist_ok=True)
Layers=[]
n_layer=[]
for file_day in files:
    file_d = os.path.join(dis_path, file_day)
    if analysis_strat == 'Layer':
        d = int(file_d.split('PosDistLayer')[1].split('.')[0])
    elif analysis_strat == 'independent':
        d = int(file_d.split('PosDist_')[1].split('.')[0])
    # print(f'Layer: {d}')
    # if d!=14:
    #     print(f'Skipping Layer: {d}')
    #     continue
    if probe=='sounder':
        y_min = array_MinAlt[d+1]
        y_max = array_MaxAlt[d+1]
        s_dis = size_dis[d + 1]
    else:
        y_min = array_MinAlt[d]
        y_max = array_MaxAlt[d]
        s_dis=size_dis[d]
    if type_p== 'MultipleRI':
        n_modes = 1
        if not np.isnan(s_dis[3]):
            n_modes += 1
        if not np.isnan(s_dis[6]):
            n_modes += 1
        # print('Number of Modes:', n_modes)
        print(f'Layer: {d}, # Modes: {n_modes},  Altitudes: {y_min:.2f} - {y_max:.2f} km ')
    else:
        print(f'Layer: {d}, Altitudes: {y_min:.2f} - {y_max:.2f} km ')
    if show_hist or save_hist or save_mode or save_mean:
        # if w_prior:
        #     my_histogramPrior(d, file_d, n_modes, y_min, y_max, res_path)
        # else:
        my_histogram(d, file_d, n_modes, y_min, y_max, res_path)
    if show_histM or save_histM:
        my_histogramMode(d, file_d, n_modes, y_min, y_max, res_path)
    if show_cornerBP or save_cornerBPlot or save_cornerDisMean or show_cornerMP or save_cornerMPlot or save_cornerDisMode:
        # my_cornerplotNew(d,n_modes, y_min, y_max, paths)
        my_corner(d, n_modes, y_min, y_max, paths)
    if show_cornerN or save_cornerN:
        my_cornerplotN(d, n_modes, y_min, y_max, paths)
    if show_CDFN or save_CDFN:
        my_CDFN(d, n_modes, y_min, y_max, paths)
    if show_2D or save_2D:
        my_2D(d, n_modes, y_min, y_max, s_dis, paths, data_dir)
    if show_ridgePlotRI or save_ridgePlotRI or show_ridgePlotS or save_ridgePlotS or show_ridgePlotRegion or save_ridgePlotRegion or show_hist_region or save_hist_region or show_corner_region or save_corner_region or show_cornerN_region or save_cornerN_region or show_CDFN_region or save_CDFN_region:
        Layers.append(d)
        n_layer.append(n_modes)
if len(Layers)>0:
    sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    n_layer, Layers = zip(*sorted_Layers)
if show_hist_region or save_hist_region:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    my_histogram_Region(Layers, n_layer, paths, data_dir)
if show_corner_region or save_corner_region:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    if save_cornerBPlot_region or show_cornerBP_region:
        my_corner_Region(Layers, n_layer, paths, data_dir)
    if save_cornerMPlot_region or show_cornerMP_region:
        my_cornerM_Region(Layers, n_layer, paths, data_dir)
if show_cornerN_region or save_cornerN_region:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    my_cornerN_Region(Layers, n_layer, paths, data_dir)
if show_CDFN_region or save_CDFN_region:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    my_CDFN_Region(Layers, n_layer, paths, data_dir)
if show_ridgePlotRI or save_ridgePlotRI:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    RidgePlots_RI(Layers,n_layer, paths,data_dir)
if show_ridgePlotRegion or save_ridgePlotRegion:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    RidgePlots_Region(Layers,n_layer, paths,data_dir)
if show_ridgePlotS or save_ridgePlotS:
    # sorted_Layers = sorted(zip(n_layer, Layers), key=lambda x: x[1])
    # n_layer, Layers = zip(*sorted_Layers)
    RidgePlots_Size(Layers,n_layer, paths,data_dir)

