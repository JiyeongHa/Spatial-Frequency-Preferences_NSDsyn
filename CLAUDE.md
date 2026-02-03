# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains analysis code for the paper "Spatial Frequency Maps in Human Visual Cortex: A Replication and Extension" (Ha, Broderick, Kay, & Winawer, 2022). The project investigates spatial frequency preferences in human visual cortex (V1, V2, V3) using fMRI data from the Natural Scenes Dataset (NSD) synthetic stimuli and the Broderick et al. dataset.

# ⚠️ CRITICAL INSTRUCTIONS - READ FIRST ⚠️

## Before Writing Any Code - MANDATORY WORKFLOW
**FOLLOW THIS ORDER:**
1. **Ask clarifying questions** if ANY part is unclear
2. **Before starting, ALWAYS explain the summary of each step how the edits will be done**. For example, when I ask to write a function, explain the steps how you will write it. 
3. **State verification plan** - How will you prove this works? (test, command, visual check, etc.)
4. **Run verification** and iterate until it passes
5. **SUMMARY** ALWAYS make a summary of changes and steps. Example: When writing a function, explain: 1. What the function does 2. Key steps in the implementation 3. Why certain approaches were chosen. Keep explanations concise but informative

**CODE QUALITY:**
- ALWAYS prioritize efficiency - use vectorization (numpy, pandas operations) over loops 
- Avoid unnecessary use of pandas when vectorization is possible

## Jupyter Notebook Rules - MANDATORY
**EXECUTION (NON-NEGOTIABLE):**
- NEVER use `print()` unless I explicitly say "print X" or "show me X". Use print() ONLY when I specifically request it
- ALL package imports MUST go in the FIRST cell of the notebook. When adding new packages, edit the first cell - don't create new import cells. Don't add packages in cells other than the first cell.
- DO NOT use reload package when I have `%load_ext autoreload %autoreload 2` in the first cell of my notebook.  
- When I say "edit", "run", "implement", or "execute" → ALWAYS do it in Jupyter notebook in VS Code IDE
- NEVER run these tasks in terminal - I need to see outputs in the notebook
- DO NOT just write code without running it - execution is required

## Terminal Rules - MANDATORY

**SHELL:**
- ALWAYS use `zsh`, never bash

**CONDA ENVIRONMENT:**
- `sfp` is the conda environment that has to be used for this project
- Always use `conda run` instead of `conda activate` for commands:**
  - ✅ Correct: `conda run -n sfp snakemake -n <target>`
  - ❌ Wrong: `conda activate sfp && snakemake -n <target>`

---
# Snakemake Workflow Implementation Notes
Snakemake is the file used to manage workflow.
**Test Case:** subj01, roi V1
---

## Snakemake Patterns & Best Practices
### UNLOCK 
When unlock, you need to specify file path with -- unlock. 
- ❌ Wrong: snakemake --unlock
- ✅ Correct: snakemake -j1 /Volumes/server/Projects/natSF/derivatives/model/training/fullgratings/weights/channel-6x6sub-subj01_roi-V1_extending-122_padmode-None.npy --unlock 

### 1. Conditional Inputs
**Pattern:** Use lambda functions returning empty list for optional inputs
```python
rule example:
    input:
        required="path/to/required.txt",
        optional=lambda wc: "path/to/file.txt" if wc.param == "value" else []
    output:
        "output/{param}/result.txt"
    run:
        # Handle both cases in run block
```

**Why:** 
- Empty list `[]` cleanly signals "no input" to Snakemake
- Avoids issues with `None` or conditional string concatenation
- Allows rule logic to branch based on wildcards

### 2. Output Path Design
**Rule:** Include ALL varying parameters as wildcards in output paths
```python
# ❌ BAD: Missing parameter causes overwrites
output: "results/{subj}_{roi}.h5"

# ✅ GOOD: All parameters represented
output: "results/weights-{weight_type}/channel-{c_sf}x{c_ori}/padding-{padding}/sub-{subj}_roi-{roi}.h5"
```

**Why:**
- Prevents different parameter combinations from overwriting same file
- Makes dependency tracking explicit
- Easier to organize and find outputs

### 3. Running Multiple Targets
```bash
# ❌ May fail with ValueError
snakemake -j4 target1.h5 target2.h5

# ✅ Reliable approach
snakemake -j2 target1.h5
snakemake -j2 target2.h5
```

### 4. Background Jobs for Long Tasks
```bash
# Start background job
snakemake -j4 /path/to/target.h5 &

# Monitor progress
# 515 files × 3-4 sec each ≈ 30 min with -j4
```

---

## Common Issues & Solutions

### Output Path Collisions
**Symptom:** Files unexpectedly overwrite each other  
**Diagnosis:** Check if all wildcards are in output path  
**Fix:** Add missing parameters to directory structure

### Conditional Logic in Rules
**Pattern:** Use `if/elif/else` in `run:` block for parameter-dependent behavior
```python
run:
    if wildcards.param == "option1":
        # Load from file
    elif wildcards.param == "option2":
        # Generate programmatically
    else:
        raise ValueError(f"Unknown param: {wildcards.param}")
```

## Lessons Learned

1. **Design paths before implementing rules** - Prevents migration headaches
2. **Lambda functions are powerful for conditional inputs** - More flexible than string manipulation
3. **HDF5 needs string types** - Convert categoricals before saving
4. **Explicit error handling** - Add `ValueError` for unknown wildcard values
5. **Background jobs for batch processing** - Use `&` for long-running tasks
6. **Test incrementally** - Run existing configs before generating new data

---
## Key Commands

### Workflow Execution (Snakemake)

**Run all analysis and generate figures:**
```bash
snakemake -j1 plot_all
```

**Dry-run to preview workflow:**
```bash
snakemake -N plot_all
```

**Run specific analysis steps:**
```bash
# Prepare all NSD synthetic data
snakemake -j1 prep_all_nsdsyn

# Fit 1D tuning curves for all subjects/ROIs
snakemake -j1 fit_tuning_all

# Run 2D model fitting
snakemake -j1 results_2D

# Cross-validation results
snakemake -j1 cvresults_all

# Bootstrap analysis
snakemake -j1 all_bootstraps

# Generate all visualizations
snakemake -j1 visualize_all

# Generate fsaverage brain surface maps
snakemake -j1 fsaverage_all
```

**Run simulations:**
```bash
snakemake -j1 run_simulation_all
```

**Test snakemake setup:**
```bash
snakemake -j1 test_run
snakemake -j1 test_shell
```

### Configuration

Edit [config.json](config.json) to set paths for:
- `NSD_DIR`: Natural Scenes Dataset directory
- `OUTPUT_DIR`: Analysis outputs and derivatives
- `FIG_DIR`: Figure outputs
- `BRODERICK_DIR`: Broderick dataset directory
- `PYSURFER_DIR`: PySurfer helpers path

## Code Architecture

### Main Python Package: `sfp_nsdsyn/`

**Data Loading & Preparation:**
- [utils.py](sfp_nsdsyn/utils.py): Subject ID conversion, dataframe utilities, ROI mapping
- [make_dataframes.py](sfp_nsdsyn/make_dataframes.py): Load NSD stimuli, process beta values, calculate local stimulus properties
- [Broderick_dataset.py](sfp_nsdsyn/Broderick_dataset.py): Broderick dataset loading, subject/ROI anatomical masks
- [voxel_selection.py](sfp_nsdsyn/voxel_selection.py): Voxel filtering, counting, coordinate transformations

**Data Processing:**
- [binning.py](sfp_nsdsyn/binning.py): Eccentricity binning, ROI splitting, summary statistics
- [create_gratings.py](sfp_nsdsyn/create_gratings.py): Grating stimulus generation

**Modeling:**
- [one_dimensional_model.py](sfp_nsdsyn/one_dimensional_model.py): 1D log-Gaussian tuning curve fitting using PyTorch
  - Fits spatial frequency tuning to eccentricity-binned data
  - Estimates preferred frequency (mode), width (sigma), amplitude
- [two_dimensional_model.py](sfp_nsdsyn/two_dimensional_model.py): 2D spatial frequency model (Broderick et al.)
  - 9 parameters: sigma, slope, intercept, phase preferences (p_1-p_4), amplitudes (A_1-A_2)
  - PyTorch-based optimization with GPU support
- [cross_validation_2d_model.py](sfp_nsdsyn/cross_validation_2d_model.py): 4-class held-out cross-validation
- [bootstrapping.py](sfp_nsdsyn/bootstrapping.py): Bootstrap resampling over runs for confidence intervals
- [simulation.py](sfp_nsdsyn/simulation.py): Synthetic voxel generation, BOLD response synthesis

**Visualization:**
- [visualization/](sfp_nsdsyn/visualization/): Plotting modules for 1D/2D model results, tuning curves, parameter distributions

### Analysis Pipeline (Jupyter Notebooks)

Notebooks in [notebooks/](notebooks/) follow a numbered sequence:
- **Step 0**: Stimulus validation (`0-*.ipynb`)
- **Step 1**: Data preparation (`1-*.ipynb`)
- **Step 2**: 1D log-Gaussian model fitting (`2-*.ipynb`)
- **Step 3**: 2D spatial frequency model fitting/analysis (`3-*.ipynb`)

Key notebooks:
- [behav.ipynb](notebooks/behav.ipynb): Behavioral analysis
- [pRF_pSF_simulation.ipynb](notebooks/pRF_pSF_simulation.ipynb): Population RF/SF simulation
- [null-distribution-comparison.ipynb](notebooks/null-distribution-comparison.ipynb): Statistical null distribution testing

### Workflow Orchestration

The [Snakefile](Snakefile) defines the complete analysis pipeline with 60+ rules organized by:
1. **Data preparation rules** (`prep_*`): Load and preprocess NSD/Broderick data
2. **Binning rules** (`binning_*`): Eccentricity-based data binning
3. **Model fitting rules** (`fit_*`, `run_model*`): 1D and 2D model optimization
4. **Analysis rules** (`run_cross_validation`, `bootstrap_*`): Statistical testing
5. **Plotting rules** (`plot_*`): Figure generation for all analyses
6. **Simulation rules** (`run_simulation*`): Synthetic data validation
7. **Brain mapping rules** (`fsaverage_*`): Surface-based visualization

## Key Concepts

### Datasets
- **NSD Synthetic**: 8 subjects, gratings with varying spatial frequency/orientation
- **Broderick et al.**: 12 subjects, original dataset for model validation

### Visual ROIs
- V1, V2, V3 (with dorsal/ventral sub-regions)
- Left/right hemispheres analyzed separately
- Native subject space and fsaverage template space

### Stimulus Classes
- Annulus, pinwheel, forward-spiral, reverse-spiral
- 6 main + 4 intermediate spatial frequency levels (28 total conditions)

### Models
**1D Model**: Log-Gaussian function fit to eccentricity-binned spatial frequency tuning
**2D Model**: Broderick et al. model accounting for both spatial frequency and visual field position dependencies

## Important Snakemake Variables

Defined at the top of [Snakefile](Snakefile:1-40):
- `STIM_LIST`: Stimulus classes
- `ROIS`: Visual areas analyzed
- `SN_LIST`: NSD subject numbers (01-08)
- `broderick_subj_list`: Broderick dataset subjects
- `LR_1D`, `LR_2D`: Learning rates for 1D/2D models
- `MAX_EPOCH_1D`, `MAX_EPOCH_2D`: Training epochs
- `PARAMS_2D`: 2D model parameter names

## Data Access

- **NSD Synthetic data**: Requires access form at naturalscenesdataset.org
- **Processed data**: Available on OSF (https://osf.io/umqkw/)
- Paths configured in [config.json](config.json)
