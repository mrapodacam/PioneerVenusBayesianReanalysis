import numpy as np
import scipy.stats
import csv
import matplotlib.pyplot as plt
import miepython
import random
from scipy.stats import norm
from scipy.stats import lognorm
import corner
import dynesty
from dynesty import NestedSampler
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
import scipy
import os
import os.path
import glob
from matplotlib import image
import pandas as pd
from scipy.optimize import fsolve
import logging
import argparse

#Nested Sampling Variables
rstate = np.random.default_rng(56101)
n_Live = 1800# Num of points we are runing the NS for
type_s = 'Dynamic' # Options: 'Static', 'Dynamic' or 'Both'
analysis_strat = 'Layer' # Options: 'Layer', 'independent'
n_dim= 18


# Pioneer Variables
lamb = 0.9
inst_unit_Neph = 5 * 10 ** -6
# inst_unit_Neph = 1
min_size = 0.3
max_size = 20.0
# num_p= 1000 # Number of particle sizes
probe= 'sounder' # Options are: 'sounder', 'day', 'night', 'north'
n_ref= 14# Options are [14, 5, 4, 2] respectively

# Functions for Data Preping
def baseline(arr_A,arr_S,min_a=2.5,max_a=30):
    array_av = []
    for i, a in enumerate(arr_A):
        if min_a < a < max_a:
            array_av.append(arr_S[i])
    mean_av = np.mean(np.array(array_av))
    # print(mean_av)
    arr_S = arr_S - mean_av
    return arr_S
def prep_int(n_probe,ref):
    ProbeSA = os.path.join(file_path, f"{n_probe}_Probe_scatVA_new.csv")
    dfSA = pd.read_csv(ProbeSA)
    array_Sd = dfSA['Scattering'].values
    array_Ad = dfSA['Altitude'].values
    array_Ad, array_Sd = array_Ad[ref:], array_Sd[ref:]  # removing the calibration
    array_Sd = baseline(array_Ad, array_Sd)
    return array_Ad, array_Sd
def read_LCPS(filename):
    data=[]
    with open(filename, 'r') as csvfile:
        # reader = pd.read_csv(csvfile, usecols=['Particle Diameter', 'AFN Scattering Cross-section, parallel', 'AFN Scattering Cross-section, perpendicular'])
        # reader = pd.read_csv(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'array_Type' : str(row['Cloud Layer']),
                'array_MaxAlt' : float(row['Max Alt']),
                'array_MinAlt' : float(row['Min Alt']),
                'NT1' : float(row['Nt1']),
                'S1' : float(row['sig_g1']),
                'M1' : float(row['mean_g1']),
                'NT2' : row['Nt2'],
                'S2' : row['sig2'],
                'M2' : row['mean2'],
                'NT3' : row['Nt3'],
                'S3' : row['sig_g3'],
                'M3' : row['mean_g3'],
            })
    return data
def alt_dist_modes(arr_A, arr_S,LCPS_dis):
    # Clustering per Altitude ranges.
    alt_d = []
    cluster_A = []
    cluster_S = []
    # Reading the LCPS data
    arr_data=read_LCPS(LCPS_dis)
    array_MinAlt= [item['array_MinAlt'] for item in arr_data]
    array_MaxAlt = [item['array_MaxAlt'] for item in arr_data]
    size_dis = [[item['NT1'], item['S1'], item['M1'], item['NT2'], item['S2'], item['M2'], item['NT3'], item['S3'], item['M3']] for item in arr_data]
    for j in range(len(array_MinAlt)):
        y_min = array_MinAlt[j]
        y_max = array_MaxAlt[j]
        y_ds = []
        s_ds=[]
        for k in range(len(arr_A)):
            y = arr_A[k]
            s = arr_S[k]
            if y_min < y <= y_max:
                y_ds.append(y)
                s_ds.append(s)
        if len(y_ds) == 0:
            continue
        else:
            for s in range(6):
                ss=s+3
                s_dis = size_dis[j][ss]
                if s_dis == '':
                    size_dis[j][ss] = float("nan")
                else:
                    size_dis[j][ss] = float(size_dis[j][ss])
            alt_d.append(size_dis[j])
            cluster_A.append(y_ds)
            cluster_S.append(s_ds)
    arr_A=cluster_A
    arr_S=cluster_S
    return arr_A, arr_S, alt_d

# Forward Model Functions
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
        # prob_tot_grid = prob_tot.reshape(Y.shape)
    return prob_tot
def bivariate_distribution_SLow(arr,X,Y):
    # print('Entering Samp')
    # xs = np.linspace(0.1, 20.0, 199)
    # ys = np.linspace(1.25, 2.00, 750)
    # x,y =np.meshgrid(xs,ys)
    # x,y= np.mgrid[0.1:20.0:.1,1.25:2.00:.001]
    # x, y = np.mgrid[0.1:20.0:1, 1.25:2.00:.1]
    # pos = np.dstack((x, y))
    x_coords = X.flatten()
    y_coords = Y.flatten()
    pos = np.column_stack([x_coords, y_coords])
    log_pos= np.log(pos)
    # n1, sig1, mu1, n2, sig2, mu2, n3, sig3, mu3, rho1, sigr1, mur1, rho2, sigr2, mur2, rho3, sigr3, mur3 = arr
    var_x1 = np.log(arr[1]) ** 2 # log(sig1) **2
    var_y1 = np.log(arr[10]) ** 2# log(sigr1) **2
    cov_xy1 = arr[9]* np.log(arr[1])*np.log(arr[10]) # rho1* sig1 * sigr1
    prob_m1 = scipy.stats.multivariate_normal([np.log(arr[2]), np.log(arr[11])],[[var_x1,cov_xy1],[cov_xy1,var_y1]])# n1 * scipy.stats.multivariate_normal([mu1, mur1],[[var_x1,cov_xy1],[cov_xy1,var_y1]])
    prob_m1= arr[0]*prob_m1.pdf(log_pos)
    prob_tot=[p/(x*y) for x,y,p in zip(x_coords,y_coords,prob_m1)]
    if not np.isnan(arr[3]):
        var_x2 = arr[4]**2 # sig2 ** 2
        var_y2 = arr[13]**2 # sigr2 ** 2
        cov_xy2 = arr[12]*arr[4]*arr[13] # rho2 * sig2 * sigr2
        prob_m2 = scipy.stats.multivariate_normal([arr[5], arr[14]], [[var_x2, cov_xy2], [cov_xy2, var_y2]]) # n2 * scipy.stats.multivariate_normal([mu2, mur2], [[var_x2, cov_xy2], [cov_xy2, var_y2]])
        prob_tot = prob_tot + arr[3]*prob_m2.pdf(pos)
    if not np.isnan(arr[6]):
        var_x3 = np.log(arr[7]) ** 2# sig3 ** 2
        var_y3 = np.log(arr[16]) **2# sigr3 ** 2
        cov_xy3 = arr[15]*np.log(arr[7])*np.log(arr[16]) # rho3 * sig3 * sigr3
        prob_m3 = scipy.stats.multivariate_normal([np.log(arr[8]), np.log(arr[17])], [[var_x3, cov_xy3], [cov_xy3, var_y3]])# n3 * scipy.stats.multivariate_normal([mu3, mur3], [[var_x3, cov_xy3], [cov_xy3, var_y3]])
        prob_m3 = arr[6] * prob_m3.pdf(log_pos)
        prob_m3 = [p / (x * y) for x, y, p in zip(x_coords, y_coords, prob_m3)]
        prob_tot = prob_tot + prob_m3
        # prob_tot_grid = prob_tot.reshape(Y.shape)
    return prob_tot

def hyper_Pioneer(theta,X,Y,Isca):
    n = bivariate_distribution(theta,X,Y)
    # I = Isca.flatten()
    I = Isca.ravel()
    I_tot= np.sum(I*n)
    return I_tot

# Nested Sampling Functions

def NS(likelihood, prior, numL, n_var, maxiter, checkpoint_dir, num_lay, type_samp='Static'):
    # initialize our nested sampler
    # print(f"\n=== INITIALIZING NESTED SAMPLER ===")
    # print(f"Type: {type_samp}")
    # print(f"Dimensions: {n_var}")
    # print(f"Live points: {numL}")
    # print(f"Max iterations: {maxiter}")
    # "Static" nested sampling.
    if type_samp=='Static':
        check_file = os.path.join(checkpoint_dir, f"PosDistLayer{num_lay}.save")
        sampler = dynesty.NestedSampler(likelihood, prior, ndim=n_var, nlive=numL, bound='multi', sample='rwalk',rstate=rstate)
        sampler.run_nested(resume=True, maxiter=maxiter, checkpoint_file=check_file)
        results = sampler.results

    # "Dynamic" nested sampling.
    elif type_samp =='Dynamic':
        check_file = os.path.join(checkpoint_dir, f"PosDistLayer{num_lay}.save")
        # dsampler = dynesty.DynamicNestedSampler(likelihood, prior, nlive=numL, ndim=n_var, bound='multi',sample='rwalk', rstate=rstate)
        if os.path.exists(check_file):
            # Resume if file exists
            dsampler = NestedSampler.restore(check_file)
            dsampler.run_nested(resume=True, checkpoint_file=check_file, checkpoint_every=maxiter)
        else:
            # Start fresh if file doesn't exist
            dsampler = dynesty.DynamicNestedSampler(likelihood, prior, nlive=numL, ndim=n_var, bound='multi',sample='rwalk', rstate=rstate)
            dsampler.run_nested(checkpoint_file=check_file,checkpoint_every=maxiter)
        # dsampler = dynesty.DynamicNestedSampler(likelihood, prior, nlive=numL, ndim=n_var, bound='multi', sample='rwalk',rstate=rstate)
        # dsampler.run_nested(resume=True, maxiter=maxiter, checkpoint_file=check_file)
        results = dsampler.results

    # Combine results from "Static" and "Dynamic" runs.
    elif type_samp=='Both':
        # print(f'Entering:{type_samp} Nested Sampling')
        check_file_S = os.path.join(checkpoint_dir, f"PosDistLayer{num_lay}_Static.save")
        check_file_D = os.path.join(checkpoint_dir, f"PosDistLayer{num_lay}_Dynamic.save")
        # print(f'Starting:Static Nested Sampling')
        sampler = dynesty.NestedSampler(likelihood, prior, ndim=n_var, nlive=numL, bound='multi', sample='rwalk',rstate=rstate)
        sampler.run_nested(resume=True, maxiter=maxiter, checkpoint_file=check_file_S)#,print_progress=True, print_func=print)
        sresults = sampler.results
        # print(f'Starting:Dynamic Nested Sampling')
        # dsampler.run_nested(resume=True, maxiter=maxiter, checkpoint_file=check_file_D)
        if os.path.exists(check_file_D):
            # Resume if file exists
            dsampler = NestedSampler.restore(check_file_D)
            dsampler.run_nested(resume=True, checkpoint_file=check_file_D, checkpoint_every=maxiter)
        else:
            # Start fresh if file doesn't exist
            dsampler = dynesty.DynamicNestedSampler(likelihood, prior, nlive=numL, ndim=n_var, bound='multi', sample='rwalk',rstate=rstate)
            dsampler.run_nested(checkpoint_file=check_file_D, checkpoint_every=maxiter)
        dresults = dsampler.results
        results = dyfunc.merge_runs([sresults, dresults])
    # # Plot a summary of the run.
    # rfig, raxes = dyplot.runplot(results)
    #
    # # Plot traces and 1-D marginalized posteriors.
    # tfig, taxes = dyplot.traceplot(results, truths=truths, labels=labels, fig=plt.subplots(2, 2, figsize=(16, 12)))
    # tfig.tight_layout()

    # Plot the 2-D marginalized posteriors.
    # labels = [r'$a$', r'$R.I.$']
    samples = results.samples  # Samples
    weights = np.exp(results.logwt - results.logz[-1]) # Double check this since it's different
    # cfig, caxes = dyplot.cornerplot(sresults, show_titles=True, title_kwargs={'y': 1.04},labels=labels, fig=plt.subplots(2, 2, figsize=(15, 15))) # took out the truth values
    # cfig.savefig(run_path+r'/{}_figure.png'.format(file_n))
    return samples, weights

def log_likelihood_bivar(thet, y_data, max_diff, X, Y, Isca):
    # if not hasattr(log_likelihood_bivar, 'call_count'):
    #     log_likelihood_bivar.call_count = 0
    # log_likelihood_bivar.call_count += 1
    #
    # if log_likelihood_bivar.call_count % 10 == 0:
    #     print(f"Likelihood called {log_likelihood_bivar.call_count} times")
    n_T=thet[0]+thet[3]+thet[6]
    if n_T<=1:
        return -np.inf
    min_sig = 0.000005  # detector sensitivity limit
    model = hyper_Pioneer(thet, X, Y, Isca)
    model = np.array(model)
    sigs = max_diff / 6
    sigma2 = sigs ** 2
    # print('Entering Likelihood', model)
    if model < min_sig:
        print('Model out of bounds:', model)
        return -np.inf
    else:
        ll = -0.5 * np.sum((y_data - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))
    return ll


def prior_NoLCPS(utheta):
    # if not hasattr(prior_NoLCPS, 'call_count'):
    #     prior_NoLCPS.call_count = 0
    # prior_NoLCPS.call_count += 1
    #
    # if prior_NoLCPS.call_count % 6 == 0:  # Print every 100 calls
    #     print(f"Prior function called {prior_NoLCPS.call_count} times")
    values_x = utheta
    # Particle Size
    n_1 = values_x[0] * 1000
    sig_1 = 1.1 + values_x[1] * 3.9
    # Truncated Normal
    m1 = 0.5
    s1 = m1 / 2
    low1, high1 = 0.02, 1  # lower and upper bounds
    # score_1=(values_x[2]-m1)/s1
    low_n1, high_n1 = (low1 - m1) / s1, (high1 - m1) / s1  # standardize
    m1_ppf = scipy.stats.truncnorm.ppf(values_x[2], low_n1, high_n1, loc=m1, scale=s1)
    n_2 = values_x[3] * 1000
    sig_2 = 0.1 + values_x[4] * 4.9
    # Truncated Normal
    m2 = 1.5
    s2 = m2 / 2
    low2, high2 = 1.0, 5  # lower and upper bounds
    # score_2 = (values_x[5] - m2) / s2
    low_n2, high_n2 = (low2 - m2) / s2, (high2 - m2) / s2  # standardize
    m2_ppf = scipy.stats.truncnorm.ppf(values_x[5], low_n2, high_n2, loc=m2, scale=s2)
    n_3 = values_x[6] * 1000
    sig_3 = 1.1 + values_x[7] * 3.9
    # Truncated Normal
    m3 = 10
    s3 = m3 / 2
    low3, high3 = 5, 20  # lower and upper bounds
    # score_3 = (values_x[8] - m3) / s3
    low_n3, high_n3 = (low3 - m3) / s3, (high3 - m3) / s3  # standardize
    m3_ppf = scipy.stats.truncnorm.ppf(values_x[8], low_n3, high_n3, loc=m3, scale=s3)
    # print(ppf_r)
    ppf_r = [n_1, sig_1, m1_ppf, n_2, sig_2, m2_ppf, n_3, sig_3, m3_ppf]
    ri_x = values_x[9:]
    # All uniform distributions
    for m in range(3):
        idx=m*3
        ppf_rho = np.clip(2 * ri_x[idx] - 1, -0.999, 0.999)
        # ppf_rho = 2 * utheta[idx] - 1
        if m==1:
            ppf_sigr = 0.0005 + ri_x[idx + 1] * 0.2#0.375 #1.5 #Gaussian
        else:
            ppf_sigr = 1.0005 + ri_x[idx + 1] * 0.25#1.455 #2.5# Log normal
        # Uniform
        # ppf_mur = 1.25 + ri_x[idx+2]* 0.75
        # Truncated Normal
        mr = 1.43
        sr = 0.13 / 2
        lowr, highr = 1.25, 2.00  # lower and upper bounds
        low_nr, high_nr = (lowr - mr) / sr, (highr - mr) / sr  # standardize
        ppf_mur = scipy.stats.truncnorm.ppf(ri_x[idx+2], low_nr, high_nr, loc=mr, scale=sr)
        ppf_r.append(ppf_rho)
        ppf_r.append(ppf_sigr)
        ppf_r.append(ppf_mur)
    return ppf_r

# def prior_hyper_MultiRI(utheta):
#     n1, sig1, mu1, n2, sig2, mu2, n3, sig3, mu3, rho1, sigr1, mur1, rho2, sigr2, mur2, rho3, sigr3, mur3= utheta


def create_matrices_directory_NoLCPS(ri_matrix, weight_matrix):
    # pos_dis = os.path.join(base_path, "PosDistLayer{}.csv".format(layer_num))
    # Get the dimensions of both matrices
    data_dis = {'n 1': ri_matrix[0], 'sig 1': ri_matrix[1], 'mu 1': ri_matrix[2],
                'n 2': ri_matrix[3], 'sig 2': ri_matrix[4], 'mu 2': ri_matrix[5],
                'n 3': ri_matrix[6], 'sig 3': ri_matrix[7], 'mu 3': ri_matrix[8],
                'rho 1': ri_matrix[9], 'sigr 1': ri_matrix[10], 'mur 1': ri_matrix[11],
                'rho 2': ri_matrix[12], 'sigr 2': ri_matrix[13], 'mur 2': ri_matrix[14],
                'rho 3': ri_matrix[15], 'sigr 3': ri_matrix[16], 'mur 3': ri_matrix[17]}
    weight_cols_names = f"Weight"
    data_dis.update({weight_cols_names: weight_matrix})
    return data_dis
# Extracting the information off of files
data_dir = os.path.dirname(__file__)
file_path = os.path.join(data_dir, "WorkData")
file_dis = os.path.join(file_path,'LPSC_distribution.csv')
model_dir = os.path.join(file_path, 'ForwardModelMatrix_delta_001')
X_matrix, Y_matrix, Isca_matrix= load_matrices_numpy(model_dir)

array_A, array_S = prep_int(probe,n_ref)

# Creating Necessary Folders
paths = os.path.join(data_dir, f"NS_{probe}_NoLCPS_HyperMatrix_{analysis_strat}_{type_s}_numL{n_Live}_new")
os.makedirs(paths,exist_ok=True)
dis_path = os.path.join(paths, "PosteriorDistributions")
os.makedirs(dis_path,exist_ok=True)

# Separating the data into Layers
y_aD,y_dD,alt_dD= alt_dist_modes(array_A,array_S,file_dis)
alt_dD=0
n_layers = len(y_aD)
layer_indices = range(len(y_aD))
y_aD=0
# Main Loop
def get_args():
    parser = argparse.ArgumentParser(description='Run a single layer')
    parser.add_argument('--task_id', type=int, default=1, help='SLURM task id, when script is called in job array.')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='Total number of SLURM tasks when script is called in job array')
    return parser.parse_args()
if __name__ == '__main__':
    # Get command line arguments
    args = get_args()

    # Initialize logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    checkpoint_dir = os.path.join(paths,'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    layer_index = args.task_id # Task ID is >=1, so layer_index==0 is avoided, as intended.
    i = layer_index
    logging.info(f' Running task ID: {args.task_id:04d}, num tasks: {args.num_tasks:04d}')
    logging.info(f' Running layer ID: {layer_index:04d}, out of a max of {n_layers:04d} layers.')

    logging.info(f' Layer: {i}')
    # logging.info(f'Altitudes:{y_aD[i]}')
    logging.info(f'Measurement Range:{min(y_dD[i])}-{max(y_dD[i])}')
    logging.info(f'Num of measurements:{len(y_dD[i])}')
    logging.info(f'Number of Dim.:{n_dim}')
    y_d = y_dD[i]
    y_data = y_d
    # sorted_data = sorted(y_data)
    # diffs= np.diff(sorted_data)
    # max_diff=np.max(diffs)
    max_diff=np.max(y_d)-np.min(y_d)
    # print('Max difference:', max_diff)

    # n_dim= n_modes*3+1 #NoLCPS
    # Calculating the layer distribution only once per layer
    # layer_dis = new_lcps_distribution(alt)
    # alt = scipy.interpolate.interp1d(layer_dis[0],layer_dis[1], fill_value='extrapolate')
    # log_l = lambda x: log_likelihood(x, alt, y_data,num_p, array_Theta, array_f)
    # log_l = lambda x: log_likelihood_diff(x, alt, y_data, max_diff, num_p, array_Theta, array_f, inst_const=inst_unit_Neph)
    def log_l(x):
        return log_likelihood_bivar(x, y_data, max_diff, X_matrix, Y_matrix, Isca_matrix)

    def prior_M(x):
        return prior_NoLCPS(x)
    # Define breakpoints of maxiterations for which results are saved as .csv
    # maxiters = np.array([500,1000,2000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,200000,1000000])
    # maxiters=[10, 20,50,100]
    if type_s =='Dynamic':
        maxiter=60
        sam, wei = NS(log_l, prior_M, n_Live, n_dim, maxiter, checkpoint_dir, i, type_s)

        # Plot a summary of the results until now.
        sam = np.transpose(sam)
        wei = np.transpose(wei)

        data_d = create_matrices_directory_NoLCPS(sam, wei)
        dis = pd.DataFrame(data_d)
        # Save latest iteration
        pos_dis = os.path.join(dis_path, f"PosDistLayer{i}.csv")
        # pos_dis = os.path.join(dis_path, f"PosDistLayer{i}_iter{iter_count:07d}.csv")
        dis.to_csv(pos_dis, index=False)
        plt.close()
    else:
        maxiters=[3000,4000]
        # start_maxIter = 0
        # for iter_count in reversed(maxiters):
        #     if os.path.exists(dis_path + f'/PosDistLayer{i}_iter{iter_count:07d}.csv'):
        #         start_maxIter = iter_count
        #         print(f'Previous max iteration found: {start_maxIter}')
        #         break
        for maxiter in maxiters:
            # if start_maxIter != 0:
            #     if maxiter<start_maxIter:
            #         print(f'Skipping:{maxiter}')
            #         continue
            logging.info(f'********************************************')
            logging.info(f'Running sampler until maxiters: {maxiter}/{maxiters[-1]}')
            logging.info(f'********************************************')

            sam, wei = NS(log_l, prior_M, n_Live, n_dim, maxiter, checkpoint_dir, i, type_s)

            # Plot a summary of the results until now.
            sam = np.transpose(sam)
            wei = np.transpose(wei)

            data_d = create_matrices_directory_NoLCPS(sam, wei)
            dis = pd.DataFrame(data_d)
            # Save latest iteration
            pos_dis = os.path.join(dis_path, f"PosDistLayer{i}.csv")
            # pos_dis = os.path.join(dis_path, f"PosDistLayer{i}_iter{iter_count:07d}.csv")
            dis.to_csv(pos_dis, index=False)
            plt.close()
