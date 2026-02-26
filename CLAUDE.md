# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains analysis code for the paper "Spatial Frequency Maps in Human Visual Cortex: A Replication and Extension" (Ha, Broderick, Kay, & Winawer, 2022). The project investigates spatial frequency preferences in human visual cortex (V1, V2, V3) using fMRI data from the Natural Scenes Dataset (NSD) synthetic stimuli and the Broderick et al. dataset.

# ⚠️ CRITICAL INSTRUCTIONS - READ FIRST ⚠️

## Before Writing Any Code - MANDATORY WORKFLOW
**FOLLOW THIS ORDER:**
1. **Always respond in a concise and direct manner, providing only relevant information**
2. **Ask clarifying questions** if ANY part is unclear
3. **Before starting, ALWAYS explain the summary of each step how the edits will be done**. For example, when I ask to write a function, explain the steps how you will write it. 
4. **State verification plan** - How will you prove this works? (test, command, visual check, etc.)
5. **Run verification** and iterate until it passes
6. **SUMMARY** ALWAYS make a summary of changes and steps. Example: When writing a function, explain: 1. What the function does 2. Key steps in the implementation 3. Why certain approaches were chosen. Keep explanations concise but informative

**CODE QUALITY:**
- ALWAYS prioritize efficiency - use vectorization (numpy, pandas operations) over loops 
- Avoid unnecessary use of pandas when vectorization is possible
- Keep Your Code modular. If one function or rule gets too long, break down your code into smaller, reusable functions and classes. Each function or class should have a single responsibility.
```bash
# Bad
def process_data(data):
    # Load data
    # Clean data
    # Analyze data
    # Save results

# Good
def load_data(path):
    pass

def clean_data(data):
    pass

def analyze_data(data):
    pass

def save_results(results):
    pass
```
## Jupyter Notebook Rules - MANDATORY
**EXECUTION (NON-NEGOTIABLE):**
- NEVER use `print()` unless I explicitly say "print X" or "show me X". Use print() ONLY when I specifically request it

## Terminal Rules - MANDATORY
**SHELL:**
- ALWAYS use `zsh`, never bash
- macOS `grep` does not support `-P` (Perl regex). Use `sed` or `grep -E` instead:
  - ❌ Wrong: `grep -oP 'sub-\d+'` (fails on macOS)
  - ✅ Correct: `sed -n 's/.*sub-\([0-9]*\).*/\1/p'` or `grep -oE 'sub-[0-9]+'`

**CONDA ENVIRONMENT:**
- `sfp` is the conda environment that has to be used for this project
- Always use `conda run` instead of `conda activate` for commands
- **IMPORTANT:** Must source `~/.zshrc` first to initialize conda:
  - ✅ Correct: `source ~/.zshrc && conda run -n sfp snakemake -n <target>`
  - ❌ Wrong: `conda run -n sfp snakemake -n <target>` (conda not initialized)
  - ❌ Wrong: `conda activate sfp && snakemake -n <target>`

---
# Snakemake Workflow Implementation Notes
Snakemake is the file used to manage workflow.

## Important Snakemake Variables
Defined at the top of [Snakefile](Snakefile:1-40):
- `STIM_LIST`: Stimulus classes
- `ROIS`: Visual areas analyzed
- `SN_LIST`: NSD subject numbers (01-08). related to wildcard 'subj'
- `broderick_subj_list`: Broderick dataset subjects
- `LR_1D`, `LR_2D`: Learning rates for 1D/2D models. related to the wildcard 'lr'
- `MAX_EPOCH_1D`, `MAX_EPOCH_2D`: Training epochs. related to wildcard 'max_epoch'
- `PARAMS_2D`: 2D model parameter names
- wildcard 'vs' is voxel selection method, for 2D model it's always 'pRFsize'

## Snakemake Patterns & Best Practices
### Always Use -j Flag
Every snakemake command requires `-j` to specify cores:
- ❌ Wrong: `snakemake --touch <target>` (missing -j)
- ✅ Correct: `snakemake -j1 --touch <target>`

### Do not make the run section too long
Make functions in a python script in `sfp_nsdsyn' folder that is relevant  and import it instead of writing a long run section.

### UNLOCK
When unlock, you need to specify file path with --unlock:
- ❌ Wrong: snakemake --unlock
- ✅ Correct: snakemake -j1 /path/to/file.npy --unlock

### Rules with Wildcards
Rules containing wildcards cannot be called by name - must specify concrete file path:
- ❌ Wrong: `snakemake -n my_rule` (if rule has wildcards like `{subj}`, `{roi}`)
- ✅ Correct: `snakemake -n /path/to/output_subj01_roi-V1.csv` 

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

```

## Common Issues & Solutions
### Incomplete Files Exception
**Symptom:** `IncompleteFilesException` when running snakemake
**Cause:** Previous run was interrupted, leaving incomplete output files
**Fix:** Add `--rerun-incomplete` flag to rerun incomplete jobs:
```bash
source ~/.zshrc && conda run -n sfp snakemake -n --rerun-incomplete <target>
```

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

### Updated Input Files (Skip Re-runs)
**Symptom:** Jobs want to re-run because input files are newer than outputs
**Cause:** Input CSVs were modified after model outputs were created
**Fix:** Use `--touch` to mark outputs as up-to-date without re-running:
```bash
source ~/.zshrc && conda run -n sfp snakemake -j1 --touch /path/to/output1.pt /path/to/output2.pt
```

### Syntax Error: EOF in Multi-line Statement
**Symptom:** `SyntaxError: EOF in multi-line statement` when running snakemake
**Cause:** Unbalanced parentheses, often from redundant nested `os.path.join(os.path.join(...))`
**Fix:** Check for duplicate `os.path.join(` calls - usually only one is needed:
```python
# ❌ Wrong: nested os.path.join
os.path.join(os.path.join(config['DIR'], 'subdir', 'file.txt'))

# ✅ Correct: single os.path.join
os.path.join(config['DIR'], 'subdir', 'file.txt')
```

### NameError in run: Block for Wildcards
**Symptom:** `NameError: name 'n_perm' is not defined` in Snakemake `run:` block
**Cause:** Wildcards must be accessed via `wildcards.<name>`, not directly by name
**Fix:**
```python
run:
    # ❌ Wrong: direct wildcard name
    title = f'Results (n={n_perm})'

    # ✅ Correct: access via wildcards object
    title = f'Results (n={wildcards.n_perm})'
```

### Output Name Mismatch in run: Block
**Symptom:** `AttributeError: 'OutputFiles' object has no attribute 'plot'`
**Cause:** Using `output.plot` but output is named `plot1` in the rule definition
**Fix:** Output names in `run:` block must match exactly what's defined in `output:`
```python
output:
    plot1 = "path/to/file.png",
    plot2 = "path/to/file.svg"
run:
    # ❌ Wrong: output.plot doesn't exist
    save_path=output.plot

    # ✅ Correct: use exact names
    save_path=output.plot1
    fig.savefig(output.plot2)
```

### --allowed-rules Flag Blocks Dependency Checking
**Symptom:** Snakemake runs `test_run` or default rule instead of target rule when using `--allowed-rules`
**Cause:** `--allowed-rules` blocks dependency rules from being considered, even if their outputs already exist
**Fix:** When all inputs exist, run without `--allowed-rules`. Use dry-run first to confirm only target rule will execute:
```bash
# ❌ May not work even when inputs exist
snakemake -j1 --allowed-rules my_rule /path/to/output.csv

# ✅ Correct: run without flag when inputs exist (verify with dry-run first)
snakemake -j1 -n /path/to/output.csv  # dry-run to confirm only target runs
snakemake -j1 /path/to/output.csv     # actual run
```

## Lessons Learned
1. **Design paths before implementing rules** - Prevents migration headaches
2. **Lambda functions are powerful for conditional inputs** - More flexible than string manipulation
3. **HDF5 needs string types** - Convert categoricals before saving
4. **Explicit error handling** - Add `ValueError` for unknown wildcard values
6. **Test incrementally** - Run existing configs before generating new data
7. **Large directories** - Use `find` instead of `ls` when directories have thousands of files to avoid "argument list too long" errors
8. **Plotting functions only plot** - Never put data computation (pooled_std, standardized_mean, etc.) inside visualization functions. Compute upstream and pass pre-computed arrays
9. **Match array ordering to param lists** - When passing arrays indexed by position alongside a param list, ensure both use the same order (e.g., `params_ordered` vs `params_no_sigma` have same elements but different order)
10. **`conda run` may resolve to wrong env** - If `conda run -n <env>` uses the wrong Python, use the full path instead: `/Users/jh7685/opt/miniconda3/envs/<env>/bin/python`
11. **`sfp` env: Python 3.8 + torch 2.2.2 + numpy 1.23.5** - After accidental deletion and reinstall, the working sfp env config is: install torch via `conda install -c pytorch pytorch=2.2.2 torchvision=0.17.2`, then `pip install numpy==1.23.5`. numpy 1.21.5 (original) segfaults with torch 2.2.2 on macOS 16; numpy 1.24+ removes `np.float` breaking nibabel. numpy 1.23.5 is the sweet spot.
12. **Snakefile import order matters for torch** - `from sfp_nsdsyn import *` must appear BEFORE `pickle.HIGHEST_PROTOCOL = 4` in the Snakefile. Setting HIGHEST_PROTOCOL=4 before torch loads causes `AssertionError` in `pickletools.py` because torch expects protocol 5 opcodes during initialization.
13. **`sfp_torch` env for standalone torch notebooks** - Use `sfp_torch` (Python 3.11 + PyTorch 2.2.2) for notebooks that only need torch/numpy/matplotlib, since it avoids sfp's older package constraints.


---
## Key Commands
### Workflow Execution (Snakemake)

**Dry-run to preview workflow:**
```bash
snakemake -N plot_all
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

## Data Access

- **NSD Synthetic data**: Requires access form at naturalscenesdataset.org
- **Processed data**: Available on OSF (https://osf.io/umqkw/)
- Paths configured in [config.json](config.json)
