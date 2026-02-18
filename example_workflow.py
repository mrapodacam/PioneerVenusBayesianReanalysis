"""
Example Workflow: Venus Atmospheric Particle Retrieval
=======================================================

This script demonstrates a typical analysis workflow for retrieving
atmospheric particle properties from Pioneer Venus Nephelometer data.

The workflow includes:
1. Loading and preprocessing probe measurements
2. Setting up the forward model
3. Running nested sampling retrieval (demonstration with reduced iterations)
4. Analyzing and visualizing posterior distributions

Note: For publication-quality results, use the full retrieval scripts
(NS_NoLCPS_Layer_Matrix_Retrieval.py) with appropriate iteration counts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats

# =============================================================================
# Configuration
# =============================================================================

# Select probe data: 'sounder', 'day', 'night', or 'north'
PROBE = 'sounder'

# Calibration offset for each probe (number of initial points to skip)
CALIBRATION_OFFSETS = {
    'sounder': 14,
    'day': 5,
    'night': 4,
    'north': 2
}

# Pioneer Nephelometer parameters
WAVELENGTH = 0.9  # microns
INSTRUMENT_CONSTANT = 5e-6

# =============================================================================
# Step 1: Load and Preprocess Data
# =============================================================================

def load_probe_data(probe_name, data_dir='WorkData'):
    """
    Load nephelometer scattering measurements from a probe.

    Parameters
    ----------
    probe_name : str
        Name of probe: 'sounder', 'day', 'night', or 'north'
    data_dir : str
        Path to WorkData directory

    Returns
    -------
    altitudes : ndarray
        Altitude measurements in km
    scattering : ndarray
        Baseline-corrected scattering intensities
    """
    filepath = os.path.join(data_dir, f"{probe_name}_Probe_scatVA_new.csv")
    df = pd.read_csv(filepath)

    # Extract data
    scattering = df['Scattering'].values
    altitudes = df['Altitude'].values

    # Remove calibration points
    offset = CALIBRATION_OFFSETS[probe_name]
    altitudes = altitudes[offset:]
    scattering = scattering[offset:]

    # Baseline correction: subtract mean from clear atmosphere region
    baseline_mask = (altitudes > 2.5) & (altitudes < 30)
    if np.any(baseline_mask):
        baseline = np.mean(scattering[baseline_mask])
        scattering = scattering - baseline

    return altitudes, scattering


def load_lcps_distributions(data_dir='WorkData'):
    """
    Load LCPS reference particle size distributions.

    Returns
    -------
    layers : list of dict
        Each dict contains cloud layer parameters
    """
    filepath = os.path.join(data_dir, 'LPSC_distribution.csv')
    df = pd.read_csv(filepath)

    layers = []
    for _, row in df.iterrows():
        layer = {
            'name': row['Cloud Layer'],
            'alt_max': row['Max Alt'],
            'alt_min': row['Min Alt'],
            'mode1': {'n': row['Nt1'], 'sigma': row['sig_g1'], 'mu': row['mean_g1']},
            'mode2': {'n': row['Nt2'], 'sigma': row['sig2'], 'mu': row['mean2']},
            'mode3': {'n': row['Nt3'], 'sigma': row['sig_g3'], 'mu': row['mean_g3']}
        }
        layers.append(layer)

    return layers


# =============================================================================
# Step 2: Forward Model
# =============================================================================

def load_forward_model(data_dir='WorkData'):
    """
    Load pre-computed Mie scattering matrices.

    Returns
    -------
    X : ndarray
        Particle size grid (microns)
    Y : ndarray
        Refractive index grid
    Isca : ndarray
        Scattering intensity matrix
    """
    model_dir = os.path.join(data_dir, 'ForwardModelMatrix_delta_001')

    X = np.load(os.path.join(model_dir, 'X_matrix.npy'))
    Y = np.load(os.path.join(model_dir, 'Y_matrix.npy'))
    Isca = np.load(os.path.join(model_dir, 'Isca_matrix.npy'))

    return X, Y, Isca


def bivariate_lognormal(params, X, Y):
    """
    Compute bivariate log-normal distribution over particle size and refractive index.

    Parameters
    ----------
    params : array-like
        18 parameters: [n1, sig1, mu1, n2, sig2, mu2, n3, sig3, mu3,
                        rho1, sigr1, mur1, rho2, sigr2, mur2, rho3, sigr3, mur3]
    X : ndarray
        Particle size grid
    Y : ndarray
        Refractive index grid

    Returns
    -------
    prob : ndarray
        Probability density at each (X, Y) point
    """
    x_coords = X.ravel()
    y_coords = Y.ravel()
    log_x = np.log(x_coords)
    log_y = np.log(y_coords)
    log_pos = np.column_stack([log_x, log_y])
    pos = np.column_stack([x_coords, y_coords])

    # Mode 1 (log-normal in both dimensions)
    log_sig = np.log(params[1])
    log_sigr = np.log(params[10])
    cov_matrix = np.array([
        [log_sig**2, params[9] * log_sig * log_sigr],
        [params[9] * log_sig * log_sigr, log_sigr**2]
    ])
    mean = np.array([np.log(params[2]), np.log(params[11])])
    rv = scipy.stats.multivariate_normal(mean, cov_matrix)
    prob_tot = params[0] * rv.pdf(log_pos) / (x_coords * y_coords)

    # Mode 2 (Gaussian in both dimensions)
    if not np.isnan(params[3]):
        cov_matrix = np.array([
            [params[4]**2, params[12] * params[4] * params[13]],
            [params[12] * params[4] * params[13], params[13]**2]
        ])
        mean = np.array([params[5], params[14]])
        rv = scipy.stats.multivariate_normal(mean, cov_matrix)
        prob_tot += params[3] * rv.pdf(pos)

    # Mode 3 (log-normal in both dimensions)
    if not np.isnan(params[6]):
        log_sig = np.log(params[7])
        log_sigr = np.log(params[16])
        cov_matrix = np.array([
            [log_sig**2, params[15] * log_sig * log_sigr],
            [params[15] * log_sig * log_sigr, log_sigr**2]
        ])
        mean = np.array([np.log(params[8]), np.log(params[17])])
        rv = scipy.stats.multivariate_normal(mean, cov_matrix)
        prob_tot += params[6] * rv.pdf(log_pos) / (x_coords * y_coords)

    return prob_tot


def compute_scattering(params, X, Y, Isca):
    """
    Compute predicted scattering intensity for given particle parameters.

    Parameters
    ----------
    params : array-like
        18 distribution parameters
    X, Y : ndarray
        Particle size and refractive index grids
    Isca : ndarray
        Pre-computed scattering matrix

    Returns
    -------
    intensity : float
        Total predicted scattering intensity
    """
    n = bivariate_lognormal(params, X, Y)
    I = Isca.ravel()
    return np.sum(I * n)


# =============================================================================
# Step 3: Visualization Functions
# =============================================================================

def plot_probe_profile(altitudes, scattering, title=None):
    """Plot altitude vs scattering intensity profile."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(scattering, altitudes, 'b-', linewidth=0.8)
    ax.set_xlabel('Scattering Intensity', fontsize=12)
    ax.set_ylabel('Altitude (km)', fontsize=12)
    ax.set_title(title or f'Pioneer Venus Nephelometer Profile')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([30, 70])
    return fig, ax


def plot_posterior_distributions(samples, weights, param_names, figsize=(12, 10)):
    """
    Plot marginal posterior distributions.

    Parameters
    ----------
    samples : ndarray
        Posterior samples (n_samples, n_params)
    weights : ndarray
        Sample weights
    param_names : list
        Parameter names for labels
    """
    n_params = samples.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        # Weighted histogram
        ax.hist(samples[:, i], bins=50, weights=weights,
                density=True, alpha=0.7, color='steelblue')

        # Compute weighted statistics
        mean = np.average(samples[:, i], weights=weights)
        std = np.sqrt(np.average((samples[:, i] - mean)**2, weights=weights))

        ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
        ax.axvline(mean - std, color='red', linestyle=':', alpha=0.5)
        ax.axvline(mean + std, color='red', linestyle=':', alpha=0.5)

        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[n_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


def plot_bivariate_distribution(params, X, Y, title='Particle Size vs Refractive Index'):
    """
    Plot 2D probability distribution of particle size and refractive index.
    """
    prob = bivariate_lognormal(params, X, Y)
    prob_grid = prob.reshape(Y.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot contours
    levels = np.linspace(prob_grid.max() * 0.01, prob_grid.max(), 10)
    contour = ax.contourf(X, Y, prob_grid, levels=levels, cmap='Reds')
    plt.colorbar(contour, ax=ax, label='Probability Density')

    ax.set_xlabel('Particle Size (microns)', fontsize=12)
    ax.set_ylabel('Refractive Index', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0.3, 5])
    ax.set_ylim([1.3, 1.7])

    return fig, ax


# =============================================================================
# Step 4: Load Existing Results (if available)
# =============================================================================

def load_posterior_results(layer_idx, probe='sounder', data_dir='.'):
    """
    Load posterior samples from a completed retrieval run.

    Parameters
    ----------
    layer_idx : int
        Layer index
    probe : str
        Probe name
    data_dir : str
        Base directory

    Returns
    -------
    samples : ndarray
        Posterior samples
    weights : ndarray
        Sample weights
    """
    # Find results directory
    pattern = f"NS_{probe}_NoLCPS_HyperMatrix_Layer_Dynamic_numL*"
    import glob
    dirs = glob.glob(os.path.join(data_dir, pattern))

    if not dirs:
        return None, None

    results_dir = dirs[0]
    filepath = os.path.join(results_dir, 'PosteriorDistributions', f'PosDistLayer{layer_idx}.csv')

    if not os.path.exists(filepath):
        return None, None

    df = pd.read_csv(filepath)

    # Extract samples (all columns except 'Weight')
    sample_cols = [c for c in df.columns if c != 'Weight']
    samples = df[sample_cols].values
    weights = np.exp(df['Weight'].values)
    weights /= weights.sum()  # Normalize

    return samples, weights


# =============================================================================
# Main Demonstration
# =============================================================================

def main():
    """Run example workflow demonstration."""

    print("=" * 60)
    print("Venus Atmospheric Particle Retrieval - Example Workflow")
    print("=" * 60)

    # Get script directory for data paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'WorkData')

    # --- Step 1: Load probe data ---
    print("\n[1] Loading probe data...")
    altitudes, scattering = load_probe_data(PROBE, data_dir)
    print(f"    Probe: {PROBE}")
    print(f"    Altitude range: {altitudes.min():.1f} - {altitudes.max():.1f} km")
    print(f"    Number of measurements: {len(altitudes)}")

    # --- Step 2: Load LCPS reference distributions ---
    print("\n[2] Loading LCPS reference distributions...")
    lcps_layers = load_lcps_distributions(data_dir)
    print(f"    Loaded {len(lcps_layers)} cloud layer definitions")

    # Show example layer
    example_layer = lcps_layers[5]  # Upper cloud layer
    print(f"    Example - {example_layer['name']}:")
    print(f"      Altitude: {example_layer['alt_min']:.1f} - {example_layer['alt_max']:.1f} km")
    print(f"      Mode 1: n={example_layer['mode1']['n']}, mu={example_layer['mode1']['mu']} um")

    # --- Step 3: Load forward model ---
    print("\n[3] Loading forward model matrices...")
    X, Y, Isca = load_forward_model(data_dir)
    print(f"    Particle size grid: {X.min():.2f} - {X.max():.2f} microns")
    print(f"    Refractive index grid: {Y.min():.2f} - {Y.max():.2f}")
    print(f"    Scattering matrix shape: {Isca.shape}")

    # --- Step 4: Demonstrate forward model ---
    print("\n[4] Forward model demonstration...")

    # Example parameters (typical Venus upper cloud values)
    example_params = [
        # Mode 1 (submicron): n, sigma, mu
        300, 2.16, 0.35,
        # Mode 2 (micron): n, sigma, mu
        50, 0.7, 2.3,
        # Mode 3 (large): n, sigma, mu
        np.nan, np.nan, np.nan,  # Not present in upper cloud
        # Mode 1 RI: rho, sigma_r, mu_r
        0.0, 1.2, 1.44,
        # Mode 2 RI: rho, sigma_r, mu_r
        0.0, 0.05, 1.44,
        # Mode 3 RI: rho, sigma_r, mu_r
        np.nan, np.nan, np.nan
    ]

    predicted_intensity = compute_scattering(example_params, X, Y, Isca)
    print(f"    Example predicted scattering: {predicted_intensity:.6e}")

    # --- Step 5: Check for existing results ---
    print("\n[5] Checking for existing retrieval results...")
    samples, weights = load_posterior_results(5, PROBE, script_dir)

    if samples is not None:
        print(f"    Found posterior samples: {samples.shape[0]} samples, {samples.shape[1]} parameters")

        # Parameter names for NoLCPS retrieval
        param_names = [
            'n1', 'sig1', 'mu1', 'n2', 'sig2', 'mu2', 'n3', 'sig3', 'mu3',
            'rho1', 'sigr1', 'mur1', 'rho2', 'sigr2', 'mur2', 'rho3', 'sigr3', 'mur3'
        ]

        # Compute summary statistics
        print("\n    Posterior summary (weighted mean +/- std):")
        for i, name in enumerate(param_names[:9]):  # Just particle size params
            mean = np.average(samples[:, i], weights=weights)
            std = np.sqrt(np.average((samples[:, i] - mean)**2, weights=weights))
            print(f"      {name}: {mean:.4f} +/- {std:.4f}")
    else:
        print("    No existing results found. Run retrieval first:")
        print("    python NS_NoLCPS_Layer_Matrix_Retrieval.py --task_id 5")

    # --- Step 6: Create visualization ---
    print("\n[6] Creating visualizations...")

    # Plot probe profile
    fig1, ax1 = plot_probe_profile(altitudes, scattering,
                                    f'Pioneer Venus {PROBE.title()} Probe')
    fig1.savefig(os.path.join(script_dir, 'example_probe_profile.png'), dpi=150)
    print("    Saved: example_probe_profile.png")

    # Plot example bivariate distribution
    fig2, ax2 = plot_bivariate_distribution(example_params, X, Y,
                                             'Example Particle Distribution (Upper Cloud)')
    fig2.savefig(os.path.join(script_dir, 'example_bivariate_dist.png'), dpi=150)
    print("    Saved: example_bivariate_dist.png")

    # Plot posterior if available
    if samples is not None:
        fig3 = plot_posterior_distributions(samples, weights, param_names)
        fig3.savefig(os.path.join(script_dir, 'example_posteriors.png'), dpi=150)
        print("    Saved: example_posteriors.png")

    print("\n" + "=" * 60)
    print("Workflow demonstration complete!")
    print("=" * 60)

    plt.show()


if __name__ == '__main__':
    main()
