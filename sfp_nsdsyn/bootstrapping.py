import sys
from . import utils as utils
import pandas as pd
import numpy as np


def sample_run_and_average(df, class_idx=range(28), sample_size=8,
                           to_group=['class_idx','voxel','run','vroinames','sub','hemi','names'], 
                           replace=True):
    """This function is used to sample the data over runs for each class index. 
    For all the class_idx, it will sample 8 runs with replacement. 
    Each sampled run will be averaged.   
    
    Args:
        df: Input dataframe containing the data
        to_group: List of columns to group by
        replace: Whether to sample with replacement
        
    Returns:
        DataFrame containing bootstrapped samples
    """
    if class_idx is None:
        class_idx = df.class_idx.unique()
    bts_df = pd.DataFrame({})
    for c in class_idx:
        sample_df = df.query('class_idx == @c')
        runs = sample_df['run'].unique()
        runs_sample = np.random.choice(runs, size=sample_size, replace=replace)
        
        # Create empty list to store sampled dataframes
        run_dfs = []
        # For each run in our bootstrap sample
        for i, run in enumerate(runs_sample):
            # Get data for this run and append to list
            run_df = sample_df[sample_df['run'] == run]
            run_df = run_df.groupby(to_group).mean().reset_index()
            run_df['sample'] = i
            run_dfs.append(run_df)
        # Concatenate all sampled run dataframes
        sample_df = pd.concat(run_dfs, ignore_index=True)
        bts_df = pd.concat([bts_df, sample_df], ignore_index=True)
    return bts_df

def bootstrap_over_runs(df, n_bootstraps=100, class_idx=range(28), sample_size=8, replace=True, 
                        to_group=['class_idx','voxel','run','vroinames','sub','hemi','names'],
                        print_every=5):

    bts_df = pd.DataFrame({})
    for b in range(n_bootstraps):
        if print_every is not None:
            if b % print_every == 0:
                print(f'Bootstrap {b} of {n_bootstraps} started!')
        sample_df = sample_run_and_average(df, class_idx=class_idx, sample_size=sample_size, replace=replace, to_group=to_group)
        sample_df['bootstrap'] = b
        bts_df = pd.concat([bts_df, sample_df], ignore_index=True)
    return bts_df


def bootstrap_sample(data, stat=np.mean, n_select=8, n_bootstrap=100):
    """ Bootstrap sample from data"""
    bootstrap = []
    for i in range(n_bootstrap):
        samples = np.random.choice(data, size=n_select, replace=True)
        i_bootstrap = stat(samples)
        bootstrap.append(i_bootstrap)
    return bootstrap


def bootstrap_dataframe(df, n_bootstrap=100,
                        to_sample='avg_betas',
                        to_group=['voxel', 'names', 'freq_lvl'], replace=True):
    """ Bootstrap using a dataframe. Progress bar will be displayed according to the
    number of the voxels for each subject."""

    selected_cols = to_group + [to_sample]
    all_df = pd.DataFrame(columns=selected_cols)
    for i_v in df.voxel.unique():
        sample_df = df.query('voxel == @i_v')
        for i in range(n_bootstrap):
            tmp = sample_df[selected_cols].groupby(to_group).sample(n=8, replace=replace)
            tmp = tmp.groupby(to_group).mean().reset_index()
            tmp['bootstrap'] = i
            tmp['bootstrap'] = tmp['bootstrap'].astype(int)
            all_df = pd.concat([all_df, tmp], ignore_index=True)

    return all_df

def bootstrap_dataframe_all_subj(sn_list, df, n_bootstrap=100,
                        to_sample='betas',
                        to_group=['subj', 'voxel', 'names', 'freq_lvl'], replace=True):
    """ Bootstrap for each subject's dataframe. Message will be displayed for each subject."""

    selected_cols = to_group + [to_sample]
    all_df = pd.DataFrame(columns=selected_cols)
    for sn in sn_list:
        subj = utils.sub_number_to_string(sn)
        tmp = df.query('subj == @subj')
        print(f'***{subj} bootstrapping start!***')
        tmp = bootstrap_dataframe(tmp,
                                  n_bootstrap=n_bootstrap,
                                  to_sample=to_sample,
                                  to_group=to_group,
                                  replace=replace)
        all_df = pd.concat([all_df, tmp], ignore_index=True)

    return all_df


def sigma_vi(bts_df, power, to_sd='normed_betas', to_group=['sub', 'voxel', 'class_idx']):
    sigma_vi_df = bts_df.groupby(to_group)[to_sd].apply(lambda x: (abs(np.percentile(x, 84)-np.percentile(x, 16))/2)**power)
    sigma_vi_df = sigma_vi_df.reset_index().rename(columns={to_sd: 'sigma_vi'})
    return sigma_vi_df

def sigma_v(bts_df, power, to_sd='normed_betas', to_group=['voxel', 'sub']):
    selected_cols = to_group + ['class_idx']
    sigma_vi_df = sigma_vi(bts_df, power, to_sd=to_sd, to_group=selected_cols)
    sigma_v_df = sigma_vi_df.groupby(to_group)['sigma_vi'].mean().reset_index()
    sigma_v_df = sigma_v_df.rename(columns={'sigma_vi': 'sigma_v'})
    return sigma_v_df

def get_multiple_sigma_vs(df, power, columns, to_sd='normed_betas', to_group=['voxel','subj']):
    """Generate multiple sigma_v_squared using different powers. power argument must be passed as a list."""
    sigma_v_df = sigma_v(df, power=power, to_sd=to_sd, to_group=to_group)
    sigma_v_df = sigma_v_df.rename(columns={'sigma_v': 'tmp'})
    sigma_v_df[columns] = pd.DataFrame(sigma_v_df['tmp'].to_list(), columns=columns)
    sigma_v_df = sigma_v_df.drop(columns=['tmp'])
    return sigma_v_df

def normalize_betas_by_frequency_magnitude(betas_df, betas='betas', freq_lvl='freq_lvl'):
    tmp = betas_df.groupby(['voxel', freq_lvl])[betas].mean().reset_index()
    tmp = tmp.pivot('voxel', freq_lvl, betas)
    index_col = tmp.index.to_numpy().reshape(-1, 1)
    tmp = np.linalg.norm(tmp, axis=1, keepdims=True)
    length = np.concatenate((index_col, tmp), axis=1)
    length = pd.DataFrame(length, columns=['voxel','length'])
    new_df = pd.merge(betas_df, length, on='voxel')
    new_df['normed_betas'] = np.divide(new_df['betas'], new_df['length'])
    return new_df

def get_sigma_v_for_whole_brain(betas_df, betas, class_list=None, sigma_power=2):
    """This function has the same purpose as the functions above, but is designed to perform faster
    to decrease the processing time, usually for whole brain voxels.
    precision_vi contains a matrix (voxel X 8 phases) for each class i.
    Then this matrix is normalized for each voxel.
    For all the classes, we average these normalized matrices and take a mean to get a single value for each voxel."""
    sigma_squared_v = []
    if class_list is None:
        class_list = betas_df.class_idx.unique()
    for class_i in class_list:
        sigma_vi = betas_df.query('class_idx == @class_i')[betas].to_numpy().reshape((betas_df.voxel.nunique(), -1))
        sigma_squared_v.append(np.std(sigma_vi, axis=1) ** sigma_power)
    return np.mean(sigma_squared_v, axis=0)

def merge_sigma_v_to_main_df(bts_v_df, subj_df, on=['subj', 'voxel']):
    return subj_df.merge(bts_v_df, on=on)

def average_sigma_v_across_voxels(df, subset=['subj']):

    if all(df.groupby(['voxel']+subset)['sigma_v_squared'].count() == 1) == False:
        df = df.drop_duplicates(['voxel']+subset)
    avg_sigma_v_df = df[subset+['sigma_v_squared']].groupby(subset).mean().reset_index()
    avg_sigma_v_df = avg_sigma_v_df.rename(columns={'sigma_v_squared': 'sigma_squared_s'})
    return avg_sigma_v_df

def get_precision_s(df, subset):
    avg_sigma_v_df = average_sigma_v_across_voxels(df, subset)
    avg_sigma_v_df['precision'] = 1 / avg_sigma_v_df['sigma_squared_s']
    return avg_sigma_v_df[subset + ['precision']]


def pooled_std(df, group_col, params=None):
    """Calculate pooled standard deviation for two groups across multiple parameters.

    The pooled standard deviation is calculated using the formula:
        s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2))

    Args:
        df: Input dataframe containing the parameter columns and a grouping column
        group_col: Name of the column that divides the data into two groups
        params: List of parameter column names to calculate pooled SD for.
            If None, defaults to ['sigma', 'slope', 'intercept', 'p_1', 'p_2',
            'p_3', 'p_4', 'A_1', 'A_2']

    Returns:
        DataFrame with one row containing the pooled SD for each parameter

    Example:
        >>> pooled_sd_df = pooled_std(final_params, group_col='dset_type')
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, but found {len(groups)}: {groups}")

    group1 = df[df[group_col] == groups[0]]
    group2 = df[df[group_col] == groups[1]]

    n1 = len(group1)
    n2 = len(group2)

    pooled_sd_values = {}
    for param in params:
        s1 = group1[param].std(ddof=1)
        s2 = group2[param].std(ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        pooled_sd_values[param] = pooled_sd

    return pd.DataFrame([pooled_sd_values])


def standardized_mean(df, pooled_sd_df, group_col, params=None):
    """Calculate mean of each parameter per group, divided by pooled standard deviation.

    Args:
        df: Input dataframe containing the parameter columns and a grouping column
        pooled_sd_df: DataFrame with pooled standard deviations (output from pooled_std)
        group_col: Name of the column that divides the data into groups
        params: List of parameter column names. If None, defaults to
            ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    Returns:
        DataFrame with rows for each group, containing mean/pooled_sd for each parameter

    Example:
        >>> pooled_sd_df = pooled_std(final_params, group_col='dset_type')
        >>> std_mean_df = standardized_mean(final_params, pooled_sd_df, group_col='dset_type')
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    group_means = df.groupby(group_col)[params].mean()
    standardized = group_means / pooled_sd_df.iloc[0][params].values

    return standardized.reset_index()


def shuffle_class_idx(df, to_shuffle=['betas'],
                      groupby_cols=['voxel', 'sub'], same_perm=False):
    """Shuffle values across class_idx within each group (e.g., voxel).

    This function performs a permutation of the specified columns across
    different class_idx values within each group. All specified columns are
    shuffled together (same permutation applied to all columns), which is useful
    for permutation testing to create null distributions.

    Args:
        df: Input dataframe containing 'class_idx' column and the columns to shuffle
        to_shuffle: List of column names whose values will be shuffled across class_idx
            (default: ['betas'])
        groupby_cols: List of columns to group by before shuffling (default: ['voxel', 'sub'])
            Shuffling happens independently within each group.
        same_perm: If True, apply the same permutation to all groups (default: False).
            When True, all voxels will be shuffled in the same way.

    Returns:
        DataFrame with shuffled values in the specified columns

    Example:
        If a voxel originally has:
            class_idx=0, betas=0.1, 
            class_idx=1, betas=0.2, 
            class_idx=2, betas=0.3, 
        After shuffling (e.g., permutation [2,0,1]), it becomes:
            class_idx=0, betas=0.3, 
            class_idx=1, betas=0.1, 
            class_idx=2, betas=0.2, 
    """
    df_shuffled = df.copy()
    # Filter to only columns that exist in the dataframe
    cols_to_shuffle = [col for col in to_shuffle if col in df_shuffled.columns]

    if same_perm:
        # Generate one permutation based on number of class_idx
        n_classes = df['class_idx'].nunique()
        perm_idx = np.random.permutation(n_classes)

        def shuffle_within_group(group):
            group = group.sort_values('class_idx')
            for col in cols_to_shuffle:
                group[col] = group[col].values[perm_idx]
            return group
    else:
        def shuffle_within_group(group):
            n_rows = len(group)
            perm_idx = np.random.permutation(n_rows)
            for col in cols_to_shuffle:
                group[col] = group[col].values[perm_idx]
            return group

    df_shuffled = df_shuffled.groupby(groupby_cols, group_keys=False).apply(shuffle_within_group)
    return df_shuffled


def shuffle_betas_within_freq_group(df, to_shuffle=['betas'],
                                    groupby_cols=['voxel', 'sub']):
    """Shuffle betas within each frequency group for orientation null test.

    Within each freq_lvl group, betas are permuted across stimulus classes
    that share the same base frequency. Mixtures (class_idx 24-27) are
    treated as their own separate group. The same permutation is applied
    to all voxels (within each freq group).

    This breaks the orientation-response relationship while preserving
    the frequency-response relationship.

    Args:
        df: Input dataframe with 'class_idx', 'freq_lvl', and columns to shuffle.
            Must be averaged across task/phase (1 row per voxel x class_idx).
        to_shuffle: Column names to shuffle (default: ['betas']).
        groupby_cols: Columns defining voxel groups (default: ['voxel', 'sub']).

    Returns:
        DataFrame with betas shuffled within each frequency group.
    """
    df_shuffled = df.copy()
    cols_to_shuffle = [col for col in to_shuffle if col in df_shuffled.columns]

    # Build frequency groups: regular freq_lvl groups + mixtures as separate group
    # Mixtures (class_idx 24-27) share freq_lvl=3 with regular stimuli,
    # so we create an explicit group column
    is_mixture = df_shuffled['class_idx'] >= 24
    df_shuffled['_freq_group'] = df_shuffled['freq_lvl'].astype(int)
    df_shuffled.loc[is_mixture, '_freq_group'] = -1  # sentinel for mixtures

    # Generate one permutation per freq group (same perm across all voxels)
    freq_groups = sorted(df_shuffled['_freq_group'].unique())
    perm_per_group = {}
    for fg in freq_groups:
        n_in_group = df_shuffled.loc[df_shuffled['_freq_group'] == fg, 'class_idx'].nunique()
        perm_per_group[fg] = np.random.permutation(n_in_group)

    def shuffle_within_voxel(group):
        group = group.sort_values('class_idx')
        for fg, perm_idx in perm_per_group.items():
            mask = group['_freq_group'] == fg
            for col in cols_to_shuffle:
                group.loc[mask, col] = group.loc[mask, col].values[perm_idx]
        return group

    df_shuffled = df_shuffled.groupby(groupby_cols, group_keys=False).apply(shuffle_within_voxel)
    df_shuffled = df_shuffled.drop(columns='_freq_group')
    return df_shuffled


def shuffle_eccentricities(df, groupby_cols=['sub']):
    """Shuffle eccentricity values across voxels within a group.

    Each voxel has a single eccentricity shared across all stimulus conditions.
    This randomly reassigns eccentricities across voxels, breaking the
    voxel-position mapping while preserving all other properties (betas,
    local_sf, angle, class_idx, etc.).

    Mirrors the MATLAB shuffleEccentricities() approach.

    Args:
        df: DataFrame with 'voxel' and 'eccentricity' columns.
            Each voxel must have a consistent eccentricity across all rows.
        groupby_cols: Groups within which to shuffle independently (default: ['sub']).

    Returns:
        DataFrame with eccentricity values randomly permuted across voxels.
    """
    df_shuffled = df.copy()

    def shuffle_within_group(group):
        voxel_ecc = group.groupby('voxel')['eccentricity'].first()
        shuffled_ecc = voxel_ecc.values[np.random.permutation(len(voxel_ecc))]
        ecc_map = dict(zip(voxel_ecc.index, shuffled_ecc))
        group['eccentricity'] = group['voxel'].map(ecc_map)
        return group

    df_shuffled = df_shuffled.groupby(groupby_cols, group_keys=False).apply(shuffle_within_group)
    return df_shuffled


def shuffle_local_sf(df, groupby_cols=['sub']):
    """Shuffle local_sf across stimulus classes, same permutation for all voxels.

    Each voxel has 28 stimulus classes with different local_sf values.
    This randomly permutes which stimulus class gets which local_sf,
    using the SAME permutation for all voxels within a group.

    Mirrors the MATLAB shuffleSpatialFrequencies() approach.

    Args:
        df: DataFrame with 'voxel', 'class_idx', and 'local_sf' columns.
            Must have one row per (voxel, class_idx).
        groupby_cols: Groups within which to shuffle independently (default: ['sub']).

    Returns:
        DataFrame with local_sf values permuted across stimulus classes.
    """
    df_shuffled = df.copy()

    def shuffle_within_group(group):
        class_indices = np.sort(group['class_idx'].unique())
        perm = np.random.permutation(len(class_indices))
        class_perm_map = dict(zip(class_indices, class_indices[perm]))

        sf_lookup = group.set_index(['voxel', 'class_idx'])['local_sf']
        permuted_class = group['class_idx'].map(class_perm_map)
        idx = pd.MultiIndex.from_arrays([group['voxel'], permuted_class])
        group['local_sf'] = sf_lookup.reindex(idx).values
        return group

    df_shuffled = df_shuffled.groupby(groupby_cols, group_keys=False).apply(shuffle_within_group)
    return df_shuffled


def calculate_error_per_param(df, reference, params=None, metric='mse'):
    """
    Calculate error metric for each parameter between two dataframes.

    Compares the mean of each parameter in df against the mean in reference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to analyze
    reference : pd.DataFrame
        Reference dataset to compare against
    params : list of str, optional
        List of column names (parameters) to calculate error metric for.
        Defaults to 9 standard model parameters.
    metric : str, optional
        Error metric to calculate. Options:
        - 'mse': Mean Squared Error (default)
        - 'mae': Mean Absolute Error

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with error value for each parameter as columns

    Raises
    ------
    ValueError
        If metric is not 'mse' or 'mae'
        If params don't exist in df or reference

    Examples
    --------
    >>> df1 = pd.DataFrame({'sigma': [2, 4], 'slope': [1, 3]})
    >>> df2 = pd.DataFrame({'sigma': [5, 7], 'slope': [4, 6]})
    >>> calculate_error_per_param(df1, df2, params=['sigma', 'slope'])
       sigma  slope
    0    9.0    9.0
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    if metric not in ['mse', 'mae']:
        raise ValueError(f"metric must be 'mse' or 'mae', got '{metric}'")

    missing_df = [p for p in params if p not in df.columns]
    missing_ref = [p for p in params if p not in reference.columns]
    if missing_df:
        raise ValueError(f"Parameters {missing_df} not found in df")
    if missing_ref:
        raise ValueError(f"Parameters {missing_ref} not found in reference")

    df_means = df[params].mean().values
    ref_means = reference[params].mean().values

    if metric == 'mse':
        errors = (df_means - ref_means) ** 2
    else:  # mae
        errors = np.abs(df_means - ref_means)

    return pd.DataFrame([errors], columns=params)


def calculate_mse(df, groupby=None, params=None, metric='mse', melt=True, reference=None):
    """
    Calculate error metric for groups/parameters.

    Two modes:
    1. Within-group (reference=None): Calculates variance within each group
    2. Between-group (reference=df): Calculates MSE between df means and reference means

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to analyze
    groupby : str, optional
        Column name to group by (e.g., 'perm' for permutations).
        Required when reference=None.
    params : list of str
        List of column names (parameters) to calculate error metric for.
        Defaults to 9 standard model parameters.
    metric : str, optional
        Error metric to calculate. Options:
        - 'mse': Mean Squared Error (default)
        - 'mae': Mean Absolute Error
    melt : bool, optional
        If True, return long format (only for within-group mode)
    reference : pd.DataFrame, optional
        Reference dataset for between-group comparison. When provided,
        calculates MSE between df's parameter means and reference's means.

    Returns
    -------
    float or pd.DataFrame
        - If reference is provided: single MSE/MAE value (float)
        - If reference is None: DataFrame with error per group/parameter

    Raises
    ------
    ValueError
        If metric is not 'mse' or 'mae'
        If groupby column or value columns don't exist in df

    Examples
    --------
    >>> # Between-group mode
    >>> df1 = pd.DataFrame({'sigma': [2, 4], 'slope': [1, 3]})
    >>> df2 = pd.DataFrame({'sigma': [5, 7], 'slope': [4, 6]})
    >>> calculate_mse(df1, reference=df2, params=['sigma', 'slope'])
    9.0

    >>> # Within-group mode
    >>> df = pd.DataFrame({
    ...     'perm': [0, 0, 1, 1],
    ...     'sigma': [2.1, 2.2, 2.5, 2.6],
    ...     'slope': [0.11, 0.12, 0.15, 0.16]
    ... })
    >>> result = calculate_mse(df, groupby='perm', params=['sigma', 'slope'])
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    # Validate metric parameter
    if metric not in ['mse', 'mae']:
        raise ValueError(f"metric must be 'mse' or 'mae', got '{metric}'")

    # BETWEEN-GROUP MODE: Compare df means to reference means
    if reference is not None:
        errors_per_param = calculate_error_per_param(df, reference, params=params, metric=metric)
        return errors_per_param.values.mean()

    # WITHIN-GROUP MODE: Original behavior (variance within groups)
    if groupby is None:
        raise ValueError("groupby is required when reference is None")
    if groupby not in df.columns:
        raise ValueError(f"groupby column '{groupby}' not found in dataframe")

    missing_cols = [col for col in params if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Parameters {missing_cols} not found in dataframe")

    # Vectorized calculation using transform to broadcast group means
    group_means = df.groupby(groupby)[params].transform('mean')

    if metric == 'mse':
        errors = (df[params] - group_means) ** 2
    else:  # mae
        errors = (df[params] - group_means).abs()

    result = errors.groupby(df[groupby]).mean()
    result = result.reset_index()

    if melt:
        result = result.melt(id_vars=[groupby], value_vars=params,
                            var_name='parameter', value_name='value')

    return result


def calculate_null_mse_distribution(null_df, reference_df, perm_col='perm', params=None):
    """
    Calculate MSE for each permutation in null distribution.

    Parameters
    ----------
    null_df : pd.DataFrame
        DataFrame containing null data with permutation column
    reference_df : pd.DataFrame
        Reference dataset to compare against
    perm_col : str
        Column name containing permutation indices
    params : list of str, optional
        Parameter columns. Defaults to 9 standard params.

    Returns
    -------
    list of dict
        List with {'perm': int, 'mse': float} for each permutation
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    null_mse_list = []
    for p in null_df[perm_col].unique():
        perm_df = null_df[null_df[perm_col] == p]
        mse = calculate_mse(perm_df, reference=reference_df, params=params)
        null_mse_list.append({perm_col: int(p), 'mse': mse})

    return null_mse_list


def create_metric_comparison_df(actual_values, null_result_list, metric='both'):
    """
    Create DataFrame with actual and null distribution metric values.

    Parameters
    ----------
    actual_values : float or tuple
        If metric='mse' or 'corr': single float value
        If metric='both': tuple of (actual_mse, actual_corr)
    null_result_list : list of dict
        List with dicts containing 'perm' and metric value(s) for each null permutation
    metric : str
        Which metric(s): 'mse', 'corr', or 'both'

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [perm, mse/corr, type]
        - perm=-1 for actual, 0+ for null permutations
        - type='actual' or 'null'
    """
    result_df = pd.DataFrame(null_result_list)
    result_df['type'] = 'null'

    if metric == 'mse':
        actual_row = pd.DataFrame([{'perm': -1, 'mse': actual_values, 'type': 'actual'}])
    elif metric == 'corr':
        actual_row = pd.DataFrame([{'perm': -1, 'corr': actual_values, 'type': 'actual'}])
    else:  # metric == 'both'
        actual_mse, actual_corr = actual_values
        actual_row = pd.DataFrame([{'perm': -1, 'mse': actual_mse, 'corr': actual_corr, 'type': 'actual'}])

    return pd.concat([actual_row, result_df], ignore_index=True)


# Backward compatible alias
def create_mse_comparison_df(actual_mse, null_mse_list):
    """Backward compatible wrapper. See create_metric_comparison_df."""
    return create_metric_comparison_df(actual_mse, null_mse_list, metric='mse')


def calculate_null_error_per_param_distribution(null_df, reference_df, perm_col='perm', params=None, metric='mse'):
    """
    Calculate per-parameter error for each permutation in null distribution.

    Parameters
    ----------
    null_df : pd.DataFrame
        DataFrame containing null data with permutation column
    reference_df : pd.DataFrame
        Reference dataset to compare against
    perm_col : str
        Column name containing permutation indices
    params : list of str, optional
        Parameter columns. Defaults to 9 standard params.
    metric : str, optional
        Error metric: 'mse' or 'mae'. Default 'mse'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [perm, param1, param2, ...] containing
        error values for each parameter per permutation
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    results = []
    for p in null_df[perm_col].unique():
        perm_df = null_df[null_df[perm_col] == p]
        error_df = calculate_error_per_param(perm_df, reference_df, params=params, metric=metric)
        error_df[perm_col] = int(p)
        results.append(error_df)

    return pd.concat(results, ignore_index=True)


def calculate_standardized_error_per_param_comparison(nsd_df, broderick_df, null_nsd_df,
                                                       nsd_dset_type='NSD V1',
                                                       broderick_dset_type='Broderick et al. V1',
                                                       null_dset_type='Null NSD V1',
                                                       perm_col='perm', params=None,
                                                       standardize=True):
    """Calculate per-parameter squared error for actual and null distributions.

    When standardize=True, divides each parameter by pooled SD before computing errors.
    When standardize=False, computes raw squared errors between group means.

    Parameters
    ----------
    nsd_df : pd.DataFrame
        NSD dataset
    broderick_df : pd.DataFrame
        Broderick dataset
    null_nsd_df : pd.DataFrame
        Null NSD dataset with permutation column
    nsd_dset_type : str
        Label for NSD data (default: 'NSD V1')
    broderick_dset_type : str
        Label for Broderick data (default: 'Broderick et al. V1')
    null_dset_type : str
        Label for null NSD data (default: 'Null NSD V1')
    perm_col : str
        Column name for permutation indices (default: 'perm')
    params : list of str, optional
        Parameter columns. Defaults to 9 standard params.
    standardize : bool
        If True, divide by pooled SD before computing errors. Default True.

    Returns
    -------
    actual_errors : pd.DataFrame
        Single-row DataFrame with squared error per parameter
    null_errors_df : pd.DataFrame
        DataFrame with columns [param1, ..., paramN, perm] containing
        squared error per parameter per permutation
    pooled_sd_df : pd.DataFrame or None
        Pooled standard deviations (None when standardize=False)
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    nsd_df = nsd_df.copy()
    broderick_df = broderick_df.copy()
    null_nsd_df = null_nsd_df.copy()

    nsd_df['dset_type'] = nsd_dset_type
    broderick_df['dset_type'] = broderick_dset_type
    null_nsd_df['dset_type'] = null_dset_type

    actual_combined = pd.concat([nsd_df, broderick_df], axis=0)

    if standardize:
        pooled_sd_df = pooled_std(actual_combined, group_col='dset_type', params=params)
        std_actual = standardized_mean(actual_combined, pooled_sd_df, group_col='dset_type', params=params)
        nsd_means = std_actual[std_actual['dset_type'] == nsd_dset_type][params].values.squeeze()
        brod_means = std_actual[std_actual['dset_type'] == broderick_dset_type][params].values.squeeze()
        std_null = standardized_mean(null_nsd_df, pooled_sd_df,
                                      group_col=['dset_type', perm_col], params=params)
        null_means_array = std_null[params].values
        perm_values = std_null[perm_col].values
    else:
        pooled_sd_df = None
        group_means = actual_combined.groupby('dset_type')[params].mean().reset_index()
        nsd_means = group_means[group_means['dset_type'] == nsd_dset_type][params].values.squeeze()
        brod_means = group_means[group_means['dset_type'] == broderick_dset_type][params].values.squeeze()
        null_group_means = null_nsd_df.groupby(['dset_type', perm_col])[params].mean().reset_index()
        null_means_array = null_group_means[params].values
        perm_values = null_group_means[perm_col].values

    actual_errors = pd.DataFrame([(nsd_means - brod_means) ** 2], columns=params)
    null_errors_array = (null_means_array - brod_means) ** 2
    null_errors_df = pd.DataFrame(null_errors_array, columns=params)
    null_errors_df[perm_col] = perm_values

    return actual_errors, null_errors_df, pooled_sd_df


def create_error_per_param_comparison_df(actual_errors, null_errors_df, perm_col='perm'):
    """
    Create DataFrame with actual and null per-parameter errors.

    Parameters
    ----------
    actual_errors : pd.DataFrame
        Single-row DataFrame with error for each parameter
    null_errors_df : pd.DataFrame
        DataFrame with per-parameter errors for each permutation

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [perm, param1, param2, ..., type]
        - perm=-1 for actual, 0+ for null permutations
        - type='actual' or 'null'
    """
    null_errors_df = null_errors_df.copy()
    null_errors_df['type'] = 'null'

    actual_errors = actual_errors.copy()
    actual_errors[perm_col] = -1
    actual_errors['type'] = 'actual'

    return pd.concat([actual_errors, null_errors_df], ignore_index=True)


def calculate_standardized_metric_comparison(nsd_df, broderick_df, null_nsd_df,
                                              nsd_dset_type='NSD V1',
                                              broderick_dset_type='Broderick et al. V1',
                                              null_dset_type='Null NSD V1',
                                              perm_col='perm', params=None,
                                              metric='both', standardize=True):
    """
    Calculate MSE and/or correlation between NSD and Broderick data.

    When standardize=True, divides by pooled SD before computing metrics.
    When standardize=False, computes metrics from raw group means.

    Parameters
    ----------
    nsd_df : pd.DataFrame
        NSD dataset with 'dset_type' column
    broderick_df : pd.DataFrame
        Broderick dataset with 'dset_type' column
    null_nsd_df : pd.DataFrame
        Null NSD dataset with 'dset_type' and 'perm' columns
    nsd_dset_type : str
        Value in 'dset_type' column for NSD data (default: 'NSD V1')
    broderick_dset_type : str
        Value in 'dset_type' column for Broderick data (default: 'Broderick et al. V1')
    null_dset_type : str
        Value in 'dset_type' column for null NSD data (default: 'Null NSD V1')
    perm_col : str
        Column name containing permutation indices (default: 'perm')
    params : list of str, optional
        Parameter columns. Defaults to 9 standard params.
    metric : str, optional
        Which metric(s) to calculate. Options:
        - 'mse': Mean Squared Error only
        - 'corr': Pearson correlation only
        - 'both': Both MSE and correlation (default)
    standardize : bool
        If True, divide by pooled SD before computing metrics. Default True.

    Returns
    -------
    tuple
        If metric='mse': (actual_mse: float, null_mse_list: list of dict)
        If metric='corr': (actual_corr: float, null_corr_list: list of dict)
        If metric='both': ((actual_mse, actual_corr): tuple, null_result_list: list of dict)
            - null_result_list contains dicts with keys: 'perm', 'mse', 'corr'
    """
    if params is None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']

    if metric not in ['mse', 'corr', 'both']:
        raise ValueError(f"metric must be 'mse', 'corr', or 'both', got '{metric}'")

    nsd_df = nsd_df.copy()
    broderick_df = broderick_df.copy()
    null_nsd_df = null_nsd_df.copy()

    nsd_df['dset_type'] = nsd_dset_type
    broderick_df['dset_type'] = broderick_dset_type
    null_nsd_df['dset_type'] = null_dset_type

    actual_combined = pd.concat([nsd_df, broderick_df], axis=0)

    if standardize:
        pooled_sd_df = pooled_std(actual_combined, group_col='dset_type', params=params)
        standardized_actual = standardized_mean(actual_combined, pooled_sd_df, group_col='dset_type', params=params)
        nsd_means = standardized_actual[standardized_actual['dset_type'] == nsd_dset_type][params].values.squeeze()
        broderick_means = standardized_actual[standardized_actual['dset_type'] == broderick_dset_type][params].values.squeeze()
        standardized_null = standardized_mean(null_nsd_df, pooled_sd_df,
                                              group_col=['dset_type', perm_col], params=params)
        null_means_array = standardized_null[params].values
        perm_indices = standardized_null[perm_col].values
    else:
        group_means = actual_combined.groupby('dset_type')[params].mean().reset_index()
        nsd_means = group_means[group_means['dset_type'] == nsd_dset_type][params].values.squeeze()
        broderick_means = group_means[group_means['dset_type'] == broderick_dset_type][params].values.squeeze()
        null_group_means = null_nsd_df.groupby(['dset_type', perm_col])[params].mean().reset_index()
        null_means_array = null_group_means[params].values
        perm_indices = null_group_means[perm_col].values

    # Calculate actual metrics
    actual_mse = np.mean((nsd_means - broderick_means) ** 2)
    actual_corr = utils.pearson_r(nsd_means, broderick_means)

    # Null metrics (vectorized)
    diff_squared = (null_means_array - broderick_means) ** 2
    null_mse_array = np.mean(diff_squared, axis=1)

    broderick_broadcasted = np.broadcast_to(broderick_means, null_means_array.shape)
    null_corr_array = utils.pearson_r(null_means_array, broderick_broadcasted, axis=1)

    # Create output based on metric parameter
    if metric == 'mse':
        null_result_list = [{perm_col: int(p), 'mse': mse}
                            for p, mse in zip(perm_indices, null_mse_array)]
        return actual_mse, null_result_list
    elif metric == 'corr':
        null_result_list = [{perm_col: int(p), 'corr': corr}
                            for p, corr in zip(perm_indices, null_corr_array)]
        return actual_corr, null_result_list
    else:  # metric == 'both'
        null_result_list = [{perm_col: int(p), 'mse': mse, 'corr': corr}
                            for p, mse, corr in zip(perm_indices, null_mse_array, null_corr_array)]
        return (actual_mse, actual_corr), null_result_list


# Backward compatible alias
def calculate_standardized_mse_comparison(nsd_df, broderick_df, null_nsd_df,
                                          nsd_dset_type='NSD V1',
                                          broderick_dset_type='Broderick et al. V1',
                                          null_dset_type='Null NSD V1',
                                          perm_col='perm', params=None):
    """Backward compatible wrapper. See calculate_standardized_metric_comparison."""
    return calculate_standardized_metric_comparison(
        nsd_df, broderick_df, null_nsd_df,
        nsd_dset_type=nsd_dset_type,
        broderick_dset_type=broderick_dset_type,
        null_dset_type=null_dset_type,
        perm_col=perm_col, params=params,
        metric='mse')