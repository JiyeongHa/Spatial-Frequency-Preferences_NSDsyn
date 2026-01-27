import sys
import os
sys.path.append('../')
import numpy as np
import pandas as pd
import random
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_dataframe(output_dir, dset, subj, roi, vs, precision=True):
    """Load subject data and precision information"""
    # Load main dataframe
    subj_df_path = os.path.join(output_dir, 'dataframes', dset, 'model',
                               f'dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv')
    subj_df = pd.read_csv(subj_df_path)
    if precision:
        # Load precision dataframe
        precision_path = os.path.join(output_dir, 'dataframes', dset, 'precision', 
                                    f'precision-v_sub-{subj}_roi-{roi}_vs-{vs}.csv')
        precision_df = pd.read_csv(precision_path)
        # Merge dataframes
        df = subj_df.merge(precision_df, on=['sub', 'voxel', 'vroinames'])

    return df

def sub_number_to_string(sub_number, dataset="nsdsyn"):
    """ Return number (1,2,3,..) to "subj0x" form """
    if dataset == "nsdsyn":
        return "subj%02d" % sub_number
    elif dataset == "broderick":
        return "wlsubj{:03d}".format(sub_number)


def remove_subj_strings(subj_list):
    """ Remove 'subj' from the list and change the list type into integers """
    if not isinstance(subj_list, list):
        subj_list = subj_list.unique().tolist()
    num_list = [int(i.replace('subj', '')) for i in subj_list]
    return num_list

def sort_a_df_column(df_vroinames):
    """ Input should be the whole column of a dataframe.
    Sort a column that contains either strings or numbers in a descending order"""

    roi_list = df_vroinames.unique().tolist()
    if df_vroinames.name == 'vroinames':
        roi_list.sort(key=lambda x: int(x[1]))
    elif df_vroinames.name != 'vroinames':
        if all(isinstance(item, str) for item in roi_list):
            roi_list.sort()
        elif all(isinstance(item, float) for item in roi_list):
            roi_list.sort(key=lambda x: int(x))

    return roi_list

def load_df(sn, df_dir='/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/derivatives/first_level_analysis',
            df_name='results_1D_model.csv', dataset="nsd"):
    subj = sub_number_to_string(sn, dataset)
    df_path = os.path.join(df_dir, subj + '_' + df_name)
    df = pd.read_csv(df_path)
    return df

def load_all_subj_df(subj_to_run,
                     df_dir='/Volumes/server/Projects/sfp_nsd/derivatives/dataframes/',
                     df_name='results_1D_model.csv', dataset="nsd"):
    all_subj_df = []
    for sn in subj_to_run:
        tmp_df = load_df(sn, df_dir=df_dir, df_name=df_name, dataset=dataset)
        if not 'subj' in tmp_df.columns:
            tmp_df['subj'] = sub_number_to_string(sn)
        all_subj_df.append(tmp_df)
    all_subj_df = pd.concat(all_subj_df, ignore_index=True)
    return all_subj_df

def create_empty_df(col_list=None):
    empty_df = pd.DataFrame(columns=col_list)
    return empty_df

def save_df_to_csv(df, output_path, indexing=False):
    """Save dataframe to .csv files under the designated path. Make a directory if it's needed."""
    parent_path = Path(output_path)
    if not os.path.exists(parent_path.parent.absolute()):
        os.makedirs(parent_path.parent.absolute())
    df.to_csv(output_path, index=indexing)

def count_voxels(df, to_group=['subj', 'vroinames']):
    n_voxel_df = df.groupby(to_group, as_index=False)['voxel'].nunique()
    n_voxel_df = n_voxel_df.rename(columns={"voxel": "n_voxel"})
    return n_voxel_df

def check_28cond(df, print_msg=True):
    rand_subj = sub_number_to_string(random.randint(1, 9))
    rand_voxel = np.random.choice(df.query('subj == @rand_subj').voxel.unique())
    new_df = df.query('subj == @rand_subj & voxel == @rand_voxel')
    if print_msg:
        print(f'Voxel no.{rand_voxel} for {rand_subj} has {new_df.shape[0]} conditions.')

    return new_df.shape[0]

def complete_path(dir):
    """returns absolute path of the directory"""
    return os.path.abspath(dir)


def old_save_fig(save_fig, save_dir, y_label, x_label, f_name):
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir, y_label + '_vs_' + x_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, f_name)
        plt.savefig(save_path, bbox_inches='tight')

def save_fig(save_path):
    if save_path is not None:
        parent_path = Path(save_path)
        if not os.path.exists(parent_path.parent.absolute()):
            os.makedirs(parent_path.parent.absolute())
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

def plot_voxels(df, n_voxels=1, to_plot="normed_beta", save_fig=False, save_path=None):
    x = np.arange(df[to_plot].shape[0])
    fig = plt.figure()
    color = plt.cm.rainbow(np.linspace(0,1,5))
    plt.plot(x, df[to_plot], color='k', label="betas", linewidth=2, linstyle='dashed', markersize=12, marker='o')
    plt.legend(title='Noise SD')
    plt.ylabel('Synthetic BOLD')
    plt.title('1 Synthetic voxel with noise')
    plt.tight_layout()
    save_fig(save_fig, save_path)


def melt_df(df, value_vars, var_name="type", value_name="value"):
    """This function uses pd.melt to melt df while maintaining all columns"""
    id_cols = df.drop(columns=value_vars).columns.tolist()
    long_df = pd.melt(df, id_vars=id_cols, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return long_df

def melt_params(df, value_name='value', params=None):
    if params == None:
        params = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
    id_cols = df.drop(columns=params).columns.tolist()
    long_df = pd.melt(df, id_vars=id_cols,
                      value_vars=params,
                      var_name='params', value_name=value_name)
    return long_df

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def convert_rgb_to_seaborn_color_palette(rgb_list, n_colors):
    return sns.color_palette(np.array(rgb_list) / 255, n_colors)

def color_husl_palette_different_shades(n_colors, hex_hue):
    """hue must be seaborn palettes._ColorPalette"""
    pal = sns.color_palette(f'light:{hex_hue}', n_colors=n_colors)
    return pal

def make_dset_palettes(dset=None):
    c_list = sns.diverging_palette(130, 300, s=100, l=30, n=2, as_cmap=False)
    hex_color = c_list.as_hex()
    if dset == 'broderick':
        pal = color_husl_palette_different_shades(12, hex_color[1])
        pal.reverse()
    elif dset == 'nsdsyn':
        pal = color_husl_palette_different_shades(8, hex_color[0])
        pal.reverse()
    else:
        palette = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
               (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
               (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
               (189, 189, 189)]
        pal = convert_rgb_to_seaborn_color_palette(palette)
    return pal

def get_colors(to_color, to_plot=None):
    if to_color == "dset":
        return get_dset_colors(to_plot)
    elif to_color == "sub":
        return get_subject_colors('nsdsyn', to_plot)
    elif to_color == "roi":
        return get_roi_colors(to_plot)
    elif to_color == "names":
        return get_stim_colors(to_plot)
    elif to_color is None:
        return sns.color_palette("magma")


def _map_colors_and_list(pal_list, pal, convert_to_sns=True):
    if convert_to_sns:
        pal = convert_rgb_to_seaborn_color_palette(pal, len(pal))
    map_dict = dict(zip(pal_list, pal))
    return map_dict

def get_dset_colors(to_plot):
    dset_list = ['broderick', 'nsdsyn']
    dset_pals = [(72, 122, 23), (81, 31, 127)]
    map_dict = _map_colors_and_list(dset_list, dset_pals)
    pal = [c for k,c in map_dict.items() if k in to_plot]
    return pal

def get_roi_colors(to_plot):
    if 'V1 Broderick' in to_plot:
        roi_list = ['V1 Broderick', 'V1', 'V2', 'V3']
        roi_pals = [(72, 122, 23), (81, 31, 127), (151, 62, 164), (206, 144, 201)]
    else:
        roi_list = ['V1', 'V2', 'V3']
        roi_pals = [(81, 31, 127), (151, 62, 164), (206, 144, 201)]
    map_dict = _map_colors_and_list(roi_list, roi_pals)
    pal = [c for k, c in map_dict.items() if k in to_plot]
    return pal

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def hex_to_rgb(hex):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i + 2], 16)
        rgb.append(decimal)

    return tuple(rgb)

def get_stim_colors(to_plot):
    if to_plot is None:
        pal = ['k']
    else:
        default_palette = dict(zip(['annulus', 'reverse spiral', 'pinwheel', 'forward spiral'],
                           sns.color_palette("deep", 4)))
        pal = [v for k,v in default_palette.items() if k in to_plot]
    return pal

def get_subject_colors(to_plot, dset='nsdsyn'):
    c_list = sns.diverging_palette(130, 300, s=100, l=30, n=2, as_cmap=False)
    hex_color = c_list.as_hex()
    subj_list = [sub_number_to_string(sn, dset) for sn in np.arange(1, 9)]
    if dset == 'broderick':
        pal = sns.color_palette(f"deep:{hex_code}", as_cmap=False)
        #pal = color_husl_palette_different_shades(12, hex_color[1])
        #pal.reverse()
    elif dset == 'nsdsyn':
        hex_code = rgb_to_hex(88, 5, 145)
        pal = sns.color_palette(f"light:{hex_code}", n_colors=len(subj_list)+4, as_cmap=False)
        pal = pal[4:]
    else:
        subj_list = [sub_number_to_string(sn, 'nsdsyn') for sn in np.arange(1, 9)]
        palette = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
                   (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
                   (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
                   (189, 189, 189)]
        # expects RGB triplets to lie between 0 and 1, not 0 and 255
        pal = convert_rgb_to_seaborn_color_palette(palette, len(palette))
    map_dict = _map_colors_and_list(subj_list, pal, False)
    sub_list_pal = [c for k,c in map_dict.items() if k in to_plot]
    return sub_list_pal

def new_subj_colors(subj_list):
    palette = [(235, 172, 35), (0, 187, 173), (184, 0, 88), (0, 140, 249),
               (0, 110, 0), (209, 99, 230), (178, 69, 2), (135, 133, 0),
               (89, 84, 214), (255, 146, 135), (0, 198, 248), (0, 167, 108),
               (189, 189, 189)]
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    pal = convert_rgb_to_seaborn_color_palette(palette, len(palette))
    map_dict = _map_colors_and_list(subj_list, pal, False)
    sub_list_pal = [c for k,c in map_dict.items() if k in subj_list]
    return sub_list_pal



def get_continuous_colors(n, hex_code):
    pal = sns.color_palette(f"light:{hex_code}", n_colors=2*n, as_cmap=False)
    pal = pal[::2]
    return pal

def weighted_mean(x, **kws):
    """store weights as imaginery number"""
    return np.sum(np.real(x) * np.imag(x)) / np.sum(np.imag(x))

def match_wildcards_with_col(arg):
    switcher = {'roi': 'vroinames',
                'eph': 'max_epoch',
                'lr': 'lr_rate',
                'class': 'names'}
    return switcher.get(arg, arg)

def load_dataframes(file_list, *args):
    """args: ['subj', 'dset', 'lr_rate', 'max_epoch', 'vroinames', 'names'].
    maybe I can make use of something like [k.split('-')[0] for k in file_name.split('_')] later"""
    history_df = pd.DataFrame({})
    for f in file_list:
        f_type = f.split('.')[-1]
        if f_type == 'h5'or f_type == 'hdf':
            tmp = pd.read_hdf(f)
        elif f_type == 'csv':
            tmp = pd.read_csv(f)
        for arg in args:
            f = f.split('/')[-1]
            ff = f.split(f'.{f_type}')[0]
            tmp[match_wildcards_with_col(arg)] = [k for k in ff.split('_') if arg in k][0][len(arg)+1:].replace('-', ' ')
        history_df = history_df.append(tmp)
    return history_df

def decimal_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def decimal_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def set_rcParams(rc):
    if rc is None:
        pass
    else:
        for k, v in rc.items():
            plt.rcParams[k] = v

def set_fontsize(small, medium, large):
    font_rc = {'font.size': medium,
          'axes.titlesize': large,
          'axes.labelsize': medium,
          'xtick.labelsize': small,
          'ytick.labelsize': small,
          'legend.fontsize': small,
          'figure.titlesize': large}
    set_rcParams(font_rc)

def pick_random_voxels(voxel_list, n):
    return np.random.choice(voxel_list, n,replace=False)

def combine_dorsal_and_ventral_rois(all_df):
    """Replace 'V1v' and 'V1d' with 'V1', and 'V2v' and 'V2d' with 'V2'"""
    all_df['ROI'] = all_df['ROI'].replace({'V1v': 'V1', 'V1d': 'V1',
                                           'V2v': 'V2', 'V2d': 'V2',
                                           'V3v': 'V3', 'V3d': 'V3'})
    return all_df

def get_height_based_on_width(width, aspect_ratio):

    return width/aspect_ratio

def scale_fonts(font_scale):
    import matplotlib as mpl
    font_keys = ["axes.labelsize", "axes.titlesize", "legend.fontsize",
             "xtick.labelsize", "ytick.labelsize", "font.size"]
    font_dict = {k: mpl.rcParams[k] * font_scale for k in font_keys}
    mpl.rcParams.update(font_dict)


def calculate_weighted_mean(df, values, weight, groupby=['vroinames'], n_decimals=3):
    if weight is None:
        result = df.groupby(groupby)[values].apply(lambda x: x.mean())
    else:
        result = df.groupby(groupby).apply(
            lambda x: {value: np.round((x[value] * x[weight]).sum() / x[weight].sum(),n_decimals) for value in values}
        )
    # Convert the resulting dictionary into a dataframe
    result_df = result.apply(pd.Series).reset_index()

    return result_df

def bootstrap_weighted_mean(df, values, weight, n_samples, groupby=['vroinames'], n_bootstrap=100):
    bootstrap_results = pd.DataFrame({})
    np.random.RandomState(None)

    for i in range(n_bootstrap):
        sampled_df = df.sample(n=n_samples, replace=True)
        weighted_mean = calculate_weighted_mean(sampled_df, values, weight, groupby)
        bootstrap_results = pd.concat([bootstrap_results, weighted_mean], axis=0, ignore_index=True)
    return bootstrap_results

def calculate_confidence_intervals(df, params_list, dset_types, ci_quantiles=[0.16, 0.84], n_boot=5000, seed=None, n_decimals=3):
    all_dset_df = pd.DataFrame({})
    for dset_type in dset_types:
        tmp = df.query('dset_type == @dset_type')
        dset_df = pd.DataFrame({})
        dset_df['dset_type'] = [dset_type]
        for param in params_list:
            x = tmp.apply(lambda row: row[param] + row.precision * 1j, axis=1)
            x = x.to_numpy()
            reps = sns.algorithms.bootstrap(x, func=weighted_mean, n_boot=n_boot, seed=seed)
            lo, hi = np.round(np.quantile(reps, ci_quantiles), n_decimals)  # 68% CI
            dset_df[param] = [[lo, hi]]
        all_dset_df = pd.concat((all_dset_df, dset_df), axis=0)
    return all_dset_df


def pearson_r(x, y, axis=0):
    """Calculate Pearson correlation coefficient between two arrays.

    Args:
        x: First numpy array
        y: Second numpy array (must have same shape as x)
        axis: Axis along which to compute the correlation (default: 0)

    Returns:
        Pearson correlation coefficient(s). If x and y are 1D, returns a scalar.
        If x and y are 2D, returns an array of correlations computed along the
        specified axis.

    Example:
        >>> x = np.array([[1, 2, 3], [4, 5, 6]])
        >>> y = np.array([[1, 2, 3], [6, 5, 4]])
        >>> pearson_r(x, y, axis=1)  # correlations for each row
        array([ 1., -1.])
    """
    x_mean = np.mean(x, axis=axis, keepdims=True)
    y_mean = np.mean(y, axis=axis, keepdims=True)

    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = np.sum(x_centered * y_centered, axis=axis)
    denominator = np.sqrt(np.sum(x_centered**2, axis=axis) * np.sum(y_centered**2, axis=axis))

    return numerator / denominator
