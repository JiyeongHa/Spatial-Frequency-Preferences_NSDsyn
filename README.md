# Spatial Frequency Maps in Human Visual Cortex: </br>A Replication and Extension

This repository includes code to reproduce the analysis and figures from a manuscript,  "Spatial Frequency Maps in Human Visual Cortex: A Replication and Extension"

for [spatial frequency preferences in human visual cortex](https://jov.arvojournals.org/article.aspx?articleid=2792643).

Citation
```
Ha, J., Broderick, W. F., Kay, K., & Winawer, J. (2025). Spatial Frequency Maps in Human Visual Cortex: A Replication and Extension. *bioRxiv*, 2025.01.21.634150. https://doi.org/10.1101/2025.01.21.634150

```

Table of Contents

* [Dependencies](#dependencies)
     * [Conda environment](#conda-environment)
* [Data](#data)
   * [Processed data](#processed-data)
   * [NSD synthetic data](#nsd-synthetic-data)
* [Analysis pipeline](#analysis-pipeline)
    * [Reproducing the figures](#reproducing-the-figures)

# Dependencies
All code is written in Python 3.8. We use [mamba](https://github.com/mamba-org/mamba) (a faster drop-in replacement for conda) to manage dependencies.

## Environment setup

1. Install [Miniforge](https://github.com/conda-forge/miniforge#install) (comes with `conda` and `mamba` pre-installed).
   If you already have Miniconda/Anaconda, install mamba into your base env instead:
   ```bash
   conda install mamba -n base -c conda-forge
   ```

2. Clone and enter the repository:
   ```bash
   git clone git@github.com:JiyeongHa/Spatial-Frequency-Preference_NSDsyn.git
   cd Spatial-Frequency-Preference_NSDsyn
   ```

3. Create the environment:
   ```bash
   mamba env create -f environment.yml
   ```

4. Activate and install the local package:
   ```bash
   conda activate sfp
   pip install -e .
   ```

5. Verify the setup:
   ```bash
   python verify_env.py
   ```
   All checks should show `[PASS]`. See [Troubleshooting](#troubleshooting) if any fail.

6. Configure data paths by editing `config.json`:
   - `NSD_DIR`: path to NSD synthetic dataset
   - `OUTPUT_DIR`: path for analysis outputs
   - `BRODERICK_DIR`: path to Broderick dataset

   
# Data 
## NSD synthetic data
The access to the NSD synthetic data will be granted after filling out the form on the NSD website (https://naturalscenesdataset.org/).

## processed data
The data for this project is available on OSF (https://osf.io/umqkw/).  

# Analysis pipeline
## Reproducing the figures
Jupyter notebooks for reproducing each figure are available in the `figures/` directory:

| Notebook | Figures | Description |
|----------|---------|-------------|
| `fig1.ipynb` | 1a–c | NSD synthetic stimuli examples, Broderick et al. comparison stimulus, and stimulus parameter space |
| `fig4.ipynb` | 4 | 1D model results: Spatial frequency tuning curves (subject-level and group-level) |

| `fig5.ipynb` | 5a–d | 1D model results: Preferred period and bandwidth as a function of eccentricity |
| `fig6-11.ipynb` | 6–11 | 2D model parameters for replication goal (Broderick et al. vs. NSD V1) |
| `fig12-17.ipynb` | 12–17 | 2D model parameters for extension goal (V1, V2, & V3) |

Figures 2 and 3 were created manually and are not included in the notebooks.
