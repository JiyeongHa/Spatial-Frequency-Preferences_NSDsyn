import sys
from . import utils as utils
import pandas as pd
import numpy as np
from tqdm import tqdm



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
    for i_v in tqdm(df.voxel.unique()):
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

