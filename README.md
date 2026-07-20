# Venus Atmospheric Particle Retrieval

Bayesian inference framework for retrieving atmospheric particle size distributions and refractive indices from Pioneer Venus Nephelometer measurements using nested sampling and MCMC methods.

## Overview

This codebase implements a complete retrieval pipeline for characterizing Venus cloud layer aerosols:

1. **Forward Model**: Mie scattering calculations to predict nephelometer measurements given particle properties
2. **Bayesian Retrieval**: Nested sampling (via Dynesty) to infer posterior distributions of particle parameters
3. **Post-Processing**: Analysis and visualization of posterior distributions

The framework models particle populations as mixtures of up to three log-normal modes, each characterized by:
- Particle number density (n)
- Geometric standard deviation (sigma)
- Mean particle size (mu)
- Refractive index parameters (rho, sigma_r, mu_r)

Two retrieval approaches are supported:
- **NoLCPS**: Free retrieval of all 18 parameters (particle size + refractive index)
- **withLCPS**: Constrained retrieval using LCPS particle size distributions, inferring only 9 refractive index parameters

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- [miepython](https://github.com/scottprahl/miepython) - Mie scattering calculations
- [dynesty](https://github.com/joshspeagle/dynesty) - Nested sampling
- [corner](https://github.com/dfm/corner.py) - Posterior visualization
- Matplotlib
- KDEpy (for kernel density estimation)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/venus-atmospheric-retrieval.git
cd venus-atmospheric-retrieval
```

2. Create a virtual environment (recommended):
```bash
# Using conda
conda env create -f environment.yml
conda activate venus-retrieval

# Or using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import miepython; import dynesty; print('Installation successful')"
```

## Quick Start

### 1. Generate Forward Model Matrices (one-time setup)

```python
python ForwardModel_Matrix.py
```

This creates pre-computed Mie scattering matrices in `WorkData/ForwardModelMatrix_delta_001/`:
- `Isca_matrix.npy` - Scattering intensity grid
- `X_matrix.npy` - Particle size grid (0.3-20 microns)
- `Y_matrix.npy` - Refractive index grid (1.25-2.00)

### 2. Run Retrieval

For a single layer:
```python
python NS_NoLCPS_Layer_Matrix_Retrieval.py --task_id 1
```

For HPC batch processing (all layers):
```bash
sbatch process_all_layers_NoLCPS.sh
```

### 3. Analyze Results

```python
python dataAnalysis.py
```

Or see `example_workflow.py` for a step-by-step demonstration.

## Data Requirements

### Pioneer Venus Probe Data

The code expects nephelometer scattering measurements in CSV format with columns:
- `Scattering`: Measured scattering intensity
- `Altitude`: Altitude in km

Four probe datasets are included in `WorkData/`:
- `sounder_Probe_scatVA_new.csv` - Sounder probe
- `day_Probe_scatVA_new.csv` - Day-side probe
- `night_Probe_scatVA_new.csv` - Night-side probe
- `north_Probe_scatVA_new.csv` - North polar probe

### LCPS Reference Distributions

Cloud layer particle size distributions from the Large Probe Cloud Particle Size Spectrometer are provided in `WorkData/LPSC_distribution.csv`.

### Instrument Response

The Pioneer Nephelometer angular response function is in `WorkData/preflight.csv`.

### Accessing Original Data

Pioneer Venus probe data can be accessed through:
- [NASA Planetary Data System (PDS)](https://pds.nasa.gov/)
- [NSSDC Master Catalog](https://nssdc.gsfc.nasa.gov/nmc/)

## Project Structure

```
Code/
├── ForwardModel_Matrix.py          # Mie scattering forward model
├── NS_NoLCPS_Layer_Matrix_Retrieval.py   # Free parameter retrieval
├── NS_withLCPS_Layer_Matrix_Retrieval.py # LCPS-constrained retrieval
├── dataAnalysis.py                 # Post-processing and visualization
├── example_workflow.py             # Example usage script
├── process_all_layers_NoLCPS.sh    # SLURM batch script
├── requirements.txt                # pip dependencies
├── environment.yml                 # Conda environment
├── LICENSE                         # MIT License
└── WorkData/
    ├── preflight.csv               # Instrument angular response
    ├── LPSC_distribution.csv       # Reference particle distributions
    ├── *_Probe_scatVA_new.csv      # Probe measurements
    └── ForwardModelMatrix_delta_001/
        ├── Isca_matrix.npy         # Pre-computed scattering matrices
        ├── X_matrix.npy
        └── Y_matrix.npy
```

## Key Parameters

### Nested Sampling Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_Live` | 1800 (NoLCPS) / 900 (withLCPS) | Number of live points |
| `type_s` | 'Dynamic' | Sampling mode: 'Static', 'Dynamic', or 'Both' |
| `bound` | 'multi' | Bounding method (multi-ellipsoid) |
| `sample` | 'rwalk' | Sampling method (random walk) |

### Physical Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Wavelength | 0.9 microns | Nephelometer laser wavelength |
| Particle size | 0.3-20 microns | Particle diameter range |
| Refractive index | 1.25-2.00 | Real refractive index range |

## Output

Results are saved to `NS_{probe}_{type}_HyperMatrix_Layer_{sampling}_numL{nlive}_new/`:
- `PosteriorDistributions/PosDistLayer{i}.csv` - Posterior samples with weights
- `checkpoints/` - Dynesty checkpoint files for resuming runs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2024venus,
  title={[Paper Title]},
  author={[Authors]},
  journal={Science Advances},
  year={2024},
  note={In preparation}
}
```

**Citation placeholder**: Paper citation will be added upon publication in Science Advances.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**M. Regina A. Moreno**
Massachusetts Institute of Technology, Aeronautics and Astronautics Department and Department of Earth, Atmospheric and Planetary Sciences
Email: mapodaca[AT]mit.edu

## Acknowledgments

- Pioneer Venus mission data courtesy of NASA
- LCPS particle distributions from [Knollenberg & Hunten, 1980]
- Mie scattering calculations via [miepython](https://github.com/scottprahl/miepython)
- Nested sampling via [dynesty](https://github.com/joshspeagle/dynesty)
