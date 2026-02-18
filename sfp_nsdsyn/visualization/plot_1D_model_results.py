import sys
import os
sys.path.append('../../')
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sfp_nsdsyn import utils as utils
from sfp_nsdsyn import one_dimensional_model as tuning
import pandas as pd
import matplotlib as mpl
import matplotlib.patheffects as pe
from sfp_nsdsyn.visualization import plot_2D_model_results as vis2D
mpl.rcParams.update(mpl.rcParamsDefault)
rc = {'text.color': 'black',
      'axes.labelcolor': 'black',
      'legend.labelcolor': 'black',
      'xtick.color': 'black',
      'ytick.color': 'black',
      'axes.edgecolor': 'black',
      'font.family': 'Helvetica',
      'axes.linewidth': 1,
      'axes.labelpad': 3,
      'axes.spines.right': False,
      'axes.spines.top': False,
      'xtick.major.pad': 5,
      'xtick.major.width': 1,
      'ytick.major.width': 1,
      'lines.linewidth': 1,
      'font.size': 11,
      'axes.titlesize': 11,
      'axes.labelsize': 11,
      'xtick.labelsize': 11,
      'ytick.labelsize': 11,
      'legend.title_fontsize': 11,
      'legend.fontsize': 11,
      'figure.titlesize': 11,
      'figure.dpi': 72 * 3,
      'savefig.dpi': 72 * 4
      }
mpl.rcParams.update(rc)

def _get_y_pdf(row):
    y_pdf = tuning.np_log_norm_pdf(row['local_sf'], row['slope'], row['mode'], row['sigma'])
    return y_pdf





def merge_pdf_values(bin_df, model_df, on=["sub", "vroinames", "ecc_bin"]):
    merge_df = bin_df.merge(model_df, on=on)
    merge_df['pdf'] = merge_df.apply(_get_y_pdf, axis=1)
    return merge_df


def plot_average_tuning_curves_NSD_with_errorbar(data_df, params_df,
                                                 bins_to_plot, pal,
                                                 x='local_sf', y='betas',
                                                 markersize=20, normalize=True,
                                                 width=6.5, height=2.6, save_path=None):

    rc.update({'xtick.major.pad': 3,
               'xtick.labelsize': 9,
               'axes.titlepad': 10,
               'legend.title_fontsize': 10,
               'legend.fontsize': 10,
               })
    utils.set_rcParams(rc)

    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 3, figsize=(width, height),
                             sharex=True, sharey=True)

    subj_list = data_df['sub'].unique()
    for i, nsd_roi in enumerate(['V1', 'V2', 'V3']):
        for cur_bin, ls, fc in zip(bins_to_plot, ['--', '-'], ['w', pal[i]]):
            min_val = data_df.query('ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].min()
            max_val = data_df.query('ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].max()
            tmp_subj_df = data_df.query('ecc_bin == @cur_bin & vroinames == @nsd_roi')

            if normalize:
                all_pred_y = []
                all_subj_df = pd.DataFrame({})
                mean_scaling_factor = []
                for subj in subj_list:
                    subj_df = tmp_subj_df.query('sub == @subj')
                    tmp_tuning_df = params_df.query('ecc_bin == @cur_bin & vroinames == @nsd_roi & sub == @subj')
                    pred_x, pred_y = tuning._get_x_and_y_prediction(min_val * 0.7,
                                                            max_val * 1.2,
                                                            tmp_tuning_df['slope'].item(),
                                                            tmp_tuning_df['mode'].item(),
                                                            tmp_tuning_df['sigma'].item())
                    pred_y = np.array(pred_y)
                    scaling_factor = tmp_tuning_df['slope'].item()
                    mean_scaling_factor.append(scaling_factor)
                    subj_df[y] = subj_df[y] / scaling_factor
                    pred_y = pred_y / scaling_factor
                    all_pred_y.append(pred_y)
                    all_subj_df = pd.concat([all_subj_df, subj_df], ignore_index=True)

                all_pred_y = np.array(all_pred_y)
                mean_scaling_factor = np.array(mean_scaling_factor).mean()
                all_pred_y = all_pred_y * mean_scaling_factor
                pred_y = np.mean(all_pred_y, axis=0)
                all_subj_df[y] = all_subj_df[y] * mean_scaling_factor
                tmp_subj_df = all_subj_df
            else:
                mean_tuning_df = params_df.query('ecc_bin == @cur_bin & vroinames == @nsd_roi')
                pred_x, pred_y = tuning._get_x_and_y_prediction(min_val * 0.7,
                                                        max_val * 1.2,
                                                        mean_tuning_df['slope'].mean(),
                                                        mean_tuning_df['mode'].mean(),
                                                        mean_tuning_df['sigma'].mean())
                pred_y = np.array(pred_y)

            axes[i].plot(pred_x, pred_y,
                         color=pal[i],
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0, clip_on=False)
            tmp_df = tmp_subj_df.groupby(['freq_lvl'])[x].mean().reset_index()
            # Map shared local_sf from tmp_df onto tmp_subj_df by freq_lvl
            tmp_subj_df = tmp_subj_df.merge(tmp_df[['freq_lvl', x]].rename(columns={x: 'shared_local_sf'}),
                                            on='freq_lvl', how='left')
            axes[i] = sns.lineplot(
                                data=tmp_subj_df,
                                x='shared_local_sf',
                                y=y,
                                color=pal[i],
                                marker='o',
                                markersize=4,
                                markerfacecolor=fc,
                                markeredgecolor=pal[i],
                                linestyle='none',
                                linewidth=1.5,
                                ax=axes[i],
                                estimator='mean',
                                label=None, errorbar=('ci', 68), err_style='bars')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].set_title(f'{nsd_roi}')

    for g in range(len(axes)):
        axes[g].set_xscale('log')
        axes[g].set(xlim=[0.1, 40])
        axes[g].set(ylim=[1,4], yticks=[1,2,3,4])



        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        axes[g].tick_params(axis='both')
        axes[g].legend(title=None, loc=(-0.1, 0.00), frameon=False, handletextpad=0.08)


    # axes[-1].legend(title='Eccentricity band', bbox_to_anchor=(1, 0.85), frameon=False)
    # leg = axes[-1].get_legend()
    # leg.legendHandles[0].set_edgecolor('black')
    # leg.legendHandles[1].set_color('black')
    fig.supxlabel('Spatial frequency (cpd)')
    if not normalize:
            fig.supylabel('Response\n(% BOLD signal change)', ha='center')
    else:
        fig.supylabel('BOLD response\n(Normalized amplitude)', ha='center')
    fig.subplots_adjust(wspace=0.32, left=0.12, bottom=0.2)
    utils.save_fig(save_path)
    return fig, axes

def plot_tuning_curves_NSD_with_errorbar(data_df, params_df,
                                          bins_to_plot, pal,
                                         x='local_sf', y='betas',
                                         markersize=20, normalize=True,
                                         width=6.5, height=2.6, save_path=None):

    rc.update({'xtick.major.pad': 3,
               'xtick.labelsize': 9,
               'axes.titlepad': 10,
               'legend.title_fontsize': 10,
               'legend.fontsize': 10,
               })
    utils.set_rcParams(rc)
    
    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 3, figsize=(width, height),
                             sharex=True, sharey=True)
    for i, nsd_roi in enumerate(['V1', 'V2', 'V3']):
        for cur_bin, ls, fc in zip(bins_to_plot, ['--', '-'], ['w', pal[i]]):
            min_val = data_df.query('ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].min()
            max_val = data_df.query('ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].max()
            tmp_subj_df = data_df.query('ecc_bin == @cur_bin & vroinames == @nsd_roi')
            tmp_tuning_df = params_df.query('ecc_bin == @cur_bin & vroinames == @nsd_roi')
            pred_x, pred_y = tuning._get_x_and_y_prediction(min_val * 0.7,
                                                     max_val * 1.2,
                                                     tmp_tuning_df['slope'].item(),
                                                     tmp_tuning_df['mode'].item(),
                                                     tmp_tuning_df['sigma'].item())
            if normalize:
                scaling_factor = np.max(pred_y)
                tmp_subj_df[y] = tmp_subj_df[y] / scaling_factor
                pred_y = pred_y / scaling_factor
            axes[i].plot(pred_x, pred_y,
                         color=pal[i],
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0, clip_on=False)
            summary_betas = (tmp_subj_df.groupby(['freq_lvl'])[y]
                      .agg(['mean', 'std', 'count'])
                      .assign(se=lambda x: x['std'] / np.sqrt(x['count'])))
            summary_local_sf = (tmp_subj_df.groupby(['freq_lvl'])[x]
                      .mean().reset_index())
            # axes[i].errorbar(summary_local_sf['local_sf'], 
            #                  summary_betas['mean'],
            #                  yerr=summary_betas['se'], alpha=0.95, color=fc, markerfacecolor=fc, linestyle='none',
            #                  markeredgecolor=pal[i],
            #                  ecolor=pal[i],
            #                  fmt='o-', capsize=None, markersize=3, elinewidth=1.3, zorder=10)
            tmp_df = tmp_subj_df.groupby(['freq_lvl'])[x].mean().reset_index()
            # Map shared local_sf from tmp_df onto tmp_subj_df by freq_lvl
            tmp_subj_df = tmp_subj_df.merge(tmp_df[['freq_lvl', x]].rename(columns={x: 'shared_local_sf'}),
                                            on='freq_lvl', how='left')
            axes[i] = sns.lineplot(
                                data=tmp_subj_df,
                                x='shared_local_sf',
                                y=y,
                                color=pal[i],
                                marker='o',
                                markersize=4,
                                markerfacecolor=fc,
                                markeredgecolor=pal[i],
                                linestyle='none',
                                linewidth=1.5,
                                ax=axes[i],
                                estimator='mean',
                                label=None, errorbar=('ci', 68), err_style='bars')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            axes[i].set_title(f'{nsd_roi}')

    for g in range(len(axes)):
        axes[g].set_xscale('log')
        axes[g].set(xlim=[0.1, 40])
        if normalize:
            axes[g].set(ylim=[0, 1.05], yticks=[0, 0.5, 1])

        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        axes[g].tick_params(axis='both')
        axes[g].legend(title=None, loc=(-0.1, 0.00), frameon=False, handletextpad=0.08)


    # axes[-1].legend(title='Eccentricity band', bbox_to_anchor=(1, 0.85), frameon=False)
    # leg = axes[-1].get_legend()
    # leg.legendHandles[0].set_edgecolor('black')
    # leg.legendHandles[1].set_color('black')

    fig.supxlabel('Spatial frequency (cpd)')
    fig.supylabel('Response\n(% BOLD signal change)', ha='center')
    fig.subplots_adjust(wspace=0.32, left=0.12, bottom=0.2)
    utils.save_fig(save_path)
    return fig, axes

def plot_tuning_curves_NSD(data_df, params_df,
                           subj, bins_to_plot, pal,
                           x='local_sf', y='betas',
                           markersize=20, normalize=True,
                           width=6.5, height=2.6, save_path=None):

    rc.update({'xtick.major.pad': 3,
               'xtick.labelsize': 9,
               'axes.titlepad': 10,
               'legend.title_fontsize': 10,
               'legend.fontsize': 10,
               })
    utils.set_rcParams(rc)
    
    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 3, figsize=(width, height),
                             sharex=True, sharey=True)
    for i, nsd_roi in enumerate(['V1', 'V2', 'V3']):
        for cur_bin, ls, fc in zip(bins_to_plot, ['--', '-'], ['w', pal[i]]):
            min_val = data_df.query('sub == @subj & ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].min()
            max_val = data_df.query('sub == @subj & ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].max()
            tmp_subj_df = data_df.query('sub == @subj & ecc_bin == @cur_bin & vroinames == @nsd_roi')
            tmp_tuning_df = params_df.query('sub == @subj & ecc_bin == @cur_bin & vroinames == @nsd_roi')
            pred_x, pred_y = tuning._get_x_and_y_prediction(min_val * 0.7,
                                                     max_val * 1.2,
                                                     tmp_tuning_df['slope'].item(),
                                                     tmp_tuning_df['mode'].item(),
                                                     tmp_tuning_df['sigma'].item())
            if normalize:
                scaling_factor = np.max(pred_y)
                tmp_subj_df[y] = tmp_subj_df[y] / scaling_factor
                pred_y = pred_y / scaling_factor
            axes[i].plot(pred_x, pred_y,
                         color=pal[i],
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0, clip_on=False)
            axes[i].scatter(tmp_subj_df[x], tmp_subj_df[y],
                            s=markersize,
                            facecolor=fc,
                            alpha=0.95,
                            label=cur_bin,
                            edgecolor=pal[i], linewidth=1.3,
                            zorder=10, clip_on=False)
            axes[i].set_title(f'{nsd_roi}')

    for g in range(len(axes)):
        axes[g].set_xscale('log')
        axes[g].set(xlim=[0.1, 40])
        if normalize:
            axes[g].set(ylim=[0, 1.05], yticks=[0, 0.5, 1])
        else:
            axes[g].set(ylim=[0.5, 3.5], yticks=[ 1, 2, 3])
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        axes[g].tick_params(axis='both')
        axes[g].legend(title=None, loc=(-0.1, 0.00), frameon=False, handletextpad=0.08)


    #axes[-1].legend(title='Eccentricity band', bbox_to_anchor=(1, 0.85), frameon=False)
    #leg = axes[-1].get_legend()
    #leg.legendHandles[0].set_edgecolor('black')
    #leg.legendHandles[1].set_color('black')

    fig.supxlabel('Spatial frequency (cpd)')
    fig.supylabel('BOLD response\n(Normalized amplitude)', ha='center')
    fig.subplots_adjust(wspace=0.32, left=0.12, bottom=0.2)
    utils.save_fig(save_path)
    return fig, axes

def plot_sf_curves_with_broderick(nsd_subj, nsd_subj_df, nsd_tuning_df, nsd_bins_to_plot,
                                  broderick_subj, broderick_subj_df, broderick_tuning_df, broderick_bins_to_plot,
                                  pal, markersize=20,
                                  width=7, height=2.5, save_path=None):

    rc.update({'xtick.major.pad': 3,
               'xtick.labelsize': 9,
               'axes.titlepad': 10,
               'legend.title_fontsize': 10,
               'legend.fontsize': 10,
               })
    sns.set_theme("paper", style='ticks', rc=rc)
    fig, axes = plt.subplots(1, 4, figsize=(width, height),
                             sharex=True, sharey=True)
    for cur_bin, ls, fc in zip(broderick_bins_to_plot, ['--', '-'], ['w', 'gray']):
        tmp_subj_df = broderick_subj_df.query('sub == @broderick_subj & ecc_bin == @cur_bin & vroinames == "V1"')
        tmp_tuning_df = broderick_tuning_df.query('sub == @broderick_subj & ecc_bin == @cur_bin & vroinames == "V1"')
        tmp_subj_df['betas'] = tmp_subj_df['betas'] / tmp_subj_df['betas'].max()
        pred_x, pred_y = _get_x_and_y_prediction(tmp_subj_df['local_sf'].min() * 0.8,
                                                 tmp_subj_df['local_sf'].max() * 1.4,
                                                 tmp_tuning_df['slope'].item(),
                                                 tmp_tuning_df['mode'].item(),
                                                 tmp_tuning_df['sigma'].item())
        pred_y = pred_y / np.max(pred_y)
        axes[0].plot(pred_x, pred_y,
                     color='gray',
                     linestyle=ls,
                     linewidth=2,
                     path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                   pe.Normal()],
                     zorder=0, clip_on=False)
        axes[0].scatter(tmp_subj_df['local_sf'], tmp_subj_df['betas'],
                        s=markersize,
                        facecolor=fc,  # colors[c],

                        alpha=0.95,
                        label=cur_bin,
                        edgecolor='gray', linewidth=1.3,
                        zorder=10, clip_on=False)
        axes[0].set_title('Broderick et al.\n(2022)')

    for i, nsd_roi in enumerate(['V1', 'V2', 'V3']):
        for cur_bin, ls, fc in zip(nsd_bins_to_plot, ['--', '-'], ['w', pal[i]]):
            min_val = nsd_subj_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].min()
            max_val = nsd_subj_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == "V2"')['local_sf'].max()
            tmp_subj_df = nsd_subj_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == @nsd_roi')
            tmp_tuning_df = nsd_tuning_df.query('sub == @nsd_subj & ecc_bin == @cur_bin & vroinames == @nsd_roi')
            tmp_subj_df['betas'] = tmp_subj_df['betas'] / tmp_subj_df['betas'].max()
            pred_x, pred_y = tuning._get_x_and_y_prediction(min_val * 0.7,
                                                     max_val * 1.2,
                                                     tmp_tuning_df['slope'].item(),
                                                     tmp_tuning_df['mode'].item(),
                                                     tmp_tuning_df['sigma'].item())
            pred_y = pred_y / np.max(pred_y)
            axes[i+1].plot(pred_x, pred_y,
                         color=pal[i],
                         linestyle=ls,
                         linewidth=2,
                         path_effects=[pe.Stroke(linewidth=1, foreground='black'),
                                       pe.Normal()],
                         zorder=0, clip_on=False)
            axes[i+1].scatter(tmp_subj_df['local_sf'], tmp_subj_df['betas'],
                            s=markersize,
                            facecolor=fc,
                            alpha=0.95,
                            label=cur_bin,
                            edgecolor=pal[i], linewidth=1.3,
                            zorder=10, clip_on=False)
            axes[i+1].set_title(f'NSD {nsd_roi}')

    for g in range(len(axes)):
        axes[g].set_xscale('log')
        axes[g].set(ylim=[0, 1.05], yticks=[0, 0.5, 1])
        axes[g].spines['top'].set_visible(False)
        axes[g].spines['right'].set_visible(False)
        axes[g].tick_params(axis='both')
        axes[g].legend(title=None, loc=(-0.1, 0.00), frameon=False,handletextpad=0.08)


    #axes[-1].legend(title='Eccentricity band', bbox_to_anchor=(1, 0.85), frameon=False)
    #leg = axes[-1].get_legend()
    #leg.legendHandles[0].set_edgecolor('black')
    #leg.legendHandles[1].set_color('black')

    fig.supxlabel('Local spatial frequency (cpd)')
    fig.supylabel('BOLD response\n(Normalized amplitude)', ha='center')
    fig.subplots_adjust(wspace=0.3, left=0.1, bottom=0.2)
    utils.save_fig(save_path)
    return fig, axes


def _get_middle_ecc(row):

    label = row['ecc_bin']
    bin_e1 = float(label.split('-')[0])
    bin_e2 = float(label.split('-')[1][:-4])
    return np.round((bin_e1+bin_e2)/2, 2)

def _add_jitter(df, to_jitter, subset, jitter_scale=0.01):
    rand_vals = (0.5-np.random.random(df[subset].nunique()))*jitter_scale
    jitters = dict(zip(df[subset].unique(), rand_vals))
    new_col = df.apply(lambda row: row[to_jitter] + jitters[row[subset]], axis=1)
    return new_col

def plot_preferred_period(df, preferred_period, precision, hue, hue_order, fit_df,
                          pal=sns.color_palette("tab10"),
                          lgd_title=None,
                          row=None, row_order=None,
                          suptitle=None, height=2.5, width=3.4, errorbar=("ci", 68),
                          save_path=None):
    rc.update({'axes.titlepad': 10})
    sns.set_theme("notebook", style='ticks', rc=rc)

    new_df = df.copy()
    new_df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', subset=hue, jitter_scale=0.08)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', 'ecc', jitter_scale=0.03)


    new_df['value_and_weight'] = [v + w * 1j for v, w in zip(new_df[preferred_period], new_df[precision])]
    grid = sns.FacetGrid(new_df,
                         row=row,
                         row_order=row_order,
                         height=height,
                         aspect=width/height,
                         sharex=False, sharey=False)
    g = grid.map(sns.lineplot, 'ecc', 'value_and_weight',
                 hue, hue_order=hue_order, marker='o', palette=pal,
                 linestyle='', markersize=5, mew=0.1, mec='white',
                 estimator=utils.weighted_mean, errorbar=errorbar,
                 err_style='bars', err_kws={'elinewidth': 1.5}, alpha=0.9, zorder=10)

    if lgd_title is not None:
        if row is not None:
            handles, labels = g.axes[-1,0].get_legend_handles_labels()
        else:
            handles, labels = g.ax.get_legend_handles_labels()

        # Modifying the properties of the handles, e.g., increasing the line width
        for handle in handles:
            handle.set_marker('o')
            handle.set_alpha(0.9)
            handle.set_linewidth(2)
        if row is not None:
            grid.axes[0,0].legend(handles=handles, labels=labels, loc=(1.1, 0.3), title=lgd_title, frameon=False)
        else:
            grid.ax.legend(handles=handles, labels=labels, loc=(1.1, 0.3), title=lgd_title, frameon=False)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(None)

    if fit_df is not None:
        for ax, col_name in zip(g.axes.flatten(), row_order):
            for i, cur_hue in enumerate(hue_order):
                tmp_fit_df = fit_df.copy()
                if row is not None:
                    tmp_fit_df = tmp_fit_df[tmp_fit_df[row] == col_name]
                tmp_fit_df = tmp_fit_df[tmp_fit_df[hue] == cur_hue]

                ax.plot(tmp_fit_df['ecc'], tmp_fit_df['fitted'], alpha=1,
                        color=pal[i], linestyle='-', linewidth=1.5, zorder=10-i)



    grid.axes[0,0].set(xlim=(0,11), xticks=[0,2,4,6,8,10], ylim=(0,2), yticks=[0, 1, 2])
    grid.axes[-1,0].set(xlim=(0,4), xticks=[0,1,2,3,4], ylim=(0,1), yticks=[0, 0.5, 1])
    grid.set_axis_labels('Eccentricity (deg)', 'Deg per cycle')
    grid.fig.text(0.55, 0.95, suptitle, fontweight='bold', ha='center', fontsize=rc['figure.titlesize'])
    grid.fig.subplots_adjust(hspace=0.5)
    utils.save_fig(save_path)

    return g

def plot_bandwidth_in_octaves(df, bandwidth, precision, hue, hue_order, fit_df=None,
                              pal=sns.color_palette("tab10"),
                              lgd_title=None,
                              row=None, row_order=None,
                              suptitle=None, height=2.5, width=3.4, errorbar=("ci", 68),
                              save_path=None):

    rc.update({'axes.titlepad': 10})
    sns.set_theme("notebook", style='ticks', rc=rc)

    new_df = df.copy()
    new_df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', subset=hue, jitter_scale=0.08)
    new_df['ecc'] = _add_jitter(new_df, 'ecc', 'ecc', jitter_scale=0.03)
    new_df['value_and_weight'] = [v + w * 1j for v, w in zip(new_df[bandwidth], new_df[precision])]
    grid = sns.FacetGrid(new_df,
                         row=row,
                         row_order=row_order,
                         height=height,
                         aspect=width/height,
                         sharex=False, sharey=False)
    g = grid.map(sns.lineplot, 'ecc', 'value_and_weight',
                 hue, hue_order=hue_order, marker='o', palette=pal,
                 linestyle='', markersize=5, mew=0.1, mec='white',
                 estimator=utils.weighted_mean, errorbar=errorbar,
                 err_style='bars', err_kws={'elinewidth': 1.5}, alpha=0.85, zorder=20)
    if lgd_title is not None:
        if row is not None:
            handles, labels = g.axes[-1,0].get_legend_handles_labels()
        else:
            handles, labels = g.ax.get_legend_handles_labels()

        # Modifying the properties of the handles, e.g., increasing the line width
        for handle in handles:
            handle.set_marker('o')
            handle.set_alpha(0.9)
            handle.set_linewidth(0)
        if row is not None:
            grid.axes[0,0].legend(handles=handles, labels=labels, loc=(1.1, 0.3), title=lgd_title, frameon=False)
        else:
            grid.ax.legend(handles=handles, labels=labels, loc=(1.1, 0.3), title=lgd_title, frameon=False)
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(None)

    if fit_df is not None:
        for ax, col_name in zip(g.axes.flatten(), row_order):
            for i, cur_hue in enumerate(hue_order):
                tmp_fit_df = fit_df.copy()
                if row is not None:
                    tmp_fit_df = tmp_fit_df[fit_df[row] == col_name]
                tmp_fit_df = tmp_fit_df[tmp_fit_df[hue] == cur_hue]
                ax.plot(tmp_fit_df['ecc'], tmp_fit_df['fitted'], alpha=1,
                        color=pal[i], linestyle='-', linewidth=1.5, zorder=10-i)

    grid.set_axis_labels('Eccentricity (deg)', 'FWHM (in octaves)')
    grid.axes[0,0].set(xlim=(0,11), xticks=[0,2,4,6,8,10],ylim=(0,8), yticks=[0,4,8])
    grid.axes[1,0].set(xlim=(0,4), xticks=[0,1,2,3,4], ylim=(0,12.2), yticks=[0,4,8,12])
    grid.fig.text(0.55, 0.95, suptitle, weight='bold', ha='center', fontsize=rc['figure.titlesize'])
    grid.fig.subplots_adjust(hspace=0.5)
    utils.save_fig(save_path)
    return g


def calculate_weighted_mean(df, value, precision, groupby=['vroinames']):
    df['ecc'] = df.apply(_get_middle_ecc, axis=1)
    new_df = df.groupby(groupby+['ecc']).apply(lambda x: (x[value] * x[precision]).sum() / x[precision].sum())
    new_df = new_df.reset_index().rename(columns={0: 'weighted_mean'})
    return new_df


def _add_ecc_0(df, groupby=['vroinames']):
    unique_df = df.drop_duplicates(subset=groupby, inplace=False)
    unique_df['ecc'] = 0
    return pd.concat((df, unique_df), axis=0)


def fit_line_to_weighted_mean(df, values, precision, groupby=['vroinames']):
    weighted_mean_df = calculate_weighted_mean(df, values, precision, groupby)
    if 'ecc' not in weighted_mean_df.columns:
        weighted_mean_df['ecc'] = weighted_mean_df.apply(_get_middle_ecc, axis=1)
    weighted_mean_df = weighted_mean_df.sort_values(by='ecc')
    coeff_df = weighted_mean_df.groupby(groupby).apply(lambda x: np.polyfit(x['ecc'], x['weighted_mean'], 1))
    coeff_df = coeff_df.reset_index().rename(columns={0: 'coefficient'})
    fit_df = pd.merge(weighted_mean_df, coeff_df, on=groupby)
    fit_df = _add_ecc_0(fit_df, groupby)
    fit_df['fitted'] = fit_df.apply(lambda x: x['coefficient'][0] * x['ecc'] + x['coefficient'][1], axis=1)
    return fit_df

def merge_and_plot(subj_to_run, fitting_df, subj_df, merge_on_cols=["subj", "vroinames", "eccrois"],
                   to_subplot="vroinames", n_sp_low=2, legend_out=True, to_label="eccrois",
                   dp_to_x_axis="local_sf", dp_to_y_axis='avg_betas', plot_pdf=True,
                   ln_y_axis="y_lg_pdf", x_axis_label="Spatial Frequency", y_axis_label="Beta",
                   legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                   save_fig=True, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                   save_file_name='.png'):
    merged_df = _merge_pdf_values(fitting_df=fitting_df, subj_df=subj_df, merge_on_cols=merge_on_cols,
                                  merge_output_df=True)
    grid = plot_beta_all_subj(subj_to_run=subj_to_run, merged_df=merged_df, to_subplot=to_subplot, n_sp_low=n_sp_low,
                              legend_out=legend_out, to_label=to_label, dp_to_x_axis=dp_to_x_axis,
                              dp_to_y_axis=dp_to_y_axis, plot_pdf=plot_pdf,
                              ln_y_axis=ln_y_axis, x_axis_label=x_axis_label, y_axis_label=y_axis_label,
                              legend_title=legend_title, labels=labels,
                              save_fig=save_fig, save_dir=save_dir,
                              save_file_name=save_file_name)

    return merged_df, grid


def plot_parameter_mean(output_df, subj_to_run=None, to_subplot="vroinames", n_sp_low=4,
                           legend_out=True, to_label="eccrois",
                           dp_to_x_axis='params', dp_to_y_axis='value',
                           x_axis_label="Parameters", y_axis_label="Value",
                           legend_title="Eccentricity", labels=['~0.5°', '0.5-1°', '1-2°', '2-4°', '4+°'],
                           save_fig=False, save_dir='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/',
                           save_file_name='.png'):
    #new_output_df = output_df.query('eccrois == @cur_ecc')
    new_output_df = pd.melt(output_df, id_vars=['subj', 'vroinames', 'eccrois'], value_vars=['slope', 'mode', 'sigma'], ignore_index=True).rename(columns={'variable': 'params'})
    col_order = utils.sort_a_df_column(new_output_df[to_subplot])
    grid = sns.FacetGrid(new_output_df,
                         col=to_subplot, row='eccrois',
                         col_order=col_order,
                         hue=to_label,
                         palette=sns.color_palette("husl"),
                         legend_out=legend_out,
                         sharex=True, sharey=True)
    grid.map(sns.lineplot, dp_to_x_axis, dp_to_y_axis, ci=68, marker='o', linestyle='', err_style='bars', hue_order=utils.sort_a_df_column(output_df[to_label]))
    grid.set_axis_labels(x_axis_label, y_axis_label)
    lgd = grid.fig.legend(title=legend_title, bbox_to_anchor=(1.05, 0.8), loc="upper left", labels=labels, fontsize=15)
    grid.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
    n_subj = len(output_df['subj'].unique())
    grid.fig.suptitle(f'N = {n_subj}', fontsize=18, fontweight="bold")
    if save_fig:
        if not save_dir:
            raise Exception("Output directory is not defined!")
        fig_dir = os.path.join(save_dir + y_axis_label + '_vs_' + x_axis_label)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        save_path = os.path.join(fig_dir, 'ecc_seperately_all_subj')
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.show()

    return grid


def plot_dots(df, y, col, hue, lgd_title, height=5, save_path=None):
    grid = sns.FacetGrid(df,
                         col=col,
                         col_order=df[col].unique(),
                         hue=hue,
                         hue_order=df[hue].unique(),
                         height=height,
                         palette=sns.color_palette("rocket", df[hue].nunique()),
                         sharex=True, sharey=True)
    g = grid.map(sns.lineplot, 'local_sf', y, marker='o', err_style='bars', linestyle='')
    g = grid.map(sns.lineplot, 'local_sf', 'normed_pred', marker='', err_style='bars', linestyle='dotted')
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    grid.fig.legend(title=lgd_title, labels=df[hue].unique())
    plt.xscale('log')

def plot_histogram_for_each_sub_and_roi(df, x, y, x_order, y_label=None,
                                     hue=None, hue_order=None,
                                     lgd_title=None,
                                     height=5, col=None, col_wrap=None,
                                     save_path=None, ylim=None,
                                     **kwargs):
    # Plot
    rc = {'axes.labelpad': 15}
    rc.update({'font.family': 'sans-serif',
          'font.sans-serif': ['HelveticaNeue', 'Helvetica', 'Arial'],
          'lines.linewidth': 2})
    utils.set_rcParams(rc)
    sns.set_context("notebook", font_scale=1.5, rc=rc)
    grid = sns.FacetGrid(df,
                         col=col, col_wrap=col_wrap,
                         height=height, hue=hue, hue_order=hue_order,
                         aspect=1.6,
                         sharex=True, sharey=True, **kwargs)
    g = grid.map(sns.stripplot, x, y,
                 order=x_order,
                 jitter=True, dodge=True,
                 marker='o', size=8, edgecolor='gray', linewidth=0.5)
    y_label = y_label if y_label else y
    grid.set_axis_labels('Regions of Interest (ROI)', y_label)
    if ylim is not None:
        g.set(ylim=ylim)
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    utils.save_fig(save_path)
    return g

def plot_median_for_each_sub_and_roi(df, x, y, x_order, y_label=None,
                                     hue=None, hue_order=None,
                                     lgd_title=None, aspect=1.6,
                                     height=5, col=None, col_wrap=None,
                                     save_path=None, to_logscale=False, ylim=None,
                                     **kwargs):
    # Plot
    rc = {'axes.labelpad': 15}
    rc.update({'font.family': 'sans-serif',
          'font.sans-serif': ['HelveticaNeue', 'Helvetica', 'Arial'],
          'lines.linewidth': 2})
    utils.set_rcParams(rc)
    sns.set_context("notebook", font_scale=1.5, rc=rc)
    grid = sns.FacetGrid(df,
                         col=col, col_wrap=col_wrap,
                         height=height, hue=hue, hue_order=hue_order,
                         aspect=aspect,
                         sharex=True, sharey=True, **kwargs)
    g = grid.map(sns.stripplot, x, y,
                 order=x_order,
                 jitter=True, dodge=True, alpha=0.95,
                 marker='o', size=8, edgecolor='gray', linewidth=0.5)
    y_label = y_label if y_label else y
    grid.set_axis_labels('Regions of Interest (ROI)', y_label)
    if to_logscale:
        grid.set(yscale='log')
    if ylim is not None:
        g.set(ylim=ylim)
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    utils.save_fig(save_path)
    return grid, g

def plot_datapoints(df, x, y, hue, hue_order=None, lgd_title=None,
                    col='names', col_order=None, suptitle=None,
                    height=5,
                    save_path=None, **kwargs):
    rc = {'text.color': 'black',
          'axes.labelcolor': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'axes.labelpad': 20,
          'axes.linewidth': 3,
          'axes.titlepad': 40,
          'axes.titleweight': "bold",
          "axes.spines.right": False,
          "axes.spines.top": False,
          'xtick.major.pad': 10,
          'ytick.major.pad': 10,
          'xtick.major.width': 3,
          'xtick.minor.width': 3,
          'ytick.major.width': 3,
          'xtick.major.size': 10,
          'xtick.minor.size': 6,
          'ytick.major.size': 10,
          'grid.linewidth': 3,
          'font.family': 'Helvetica',
          'lines.linewidth': 2}
    sns.set_theme(style="ticks", context='notebook', rc=rc, font_scale=2)

    grid = sns.FacetGrid(df,
                         col=col,
                         col_order=col_order,
                         hue=hue,
                         height=height,
                         hue_order=hue_order,
                         sharex=True, sharey=True, **kwargs)
    grid.fig.suptitle(suptitle, fontweight="bold")
    g = grid.map(sns.scatterplot, x, y, s=90, alpha=0.9, edgecolor='gray')
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    for subplot_title, ax in grid.axes_dict.items():
        ax.set_title(f"{subplot_title.title()}")
    if lgd_title is not None:
        g.add_legend(title=lgd_title, loc='upper right')
    grid.set(xscale='log')
    utils.save_fig(save_path)

    return g
def assign_goal(dset_type):
    if dset_type == 'Broderick et al. V1':
        return 'replication'
    elif 'NSD' in dset_type:
        return 'extension'
    else:
        return None

def create_goal_columns(tuning_df, fit_df):

    tmp_tuning_df = tuning_df.query('dset_type == "NSD V1"')
    tmp_fit_df = fit_df.query('dset_type == "NSD V1"')
    tmp_tuning_df['goal'] = 'replication'
    tmp_fit_df['goal'] = 'replication'

    tuning_df['goal'] = tuning_df['dset_type'].apply(assign_goal)
    fit_df['goal'] = fit_df['dset_type'].apply(assign_goal)

    tuning_df = pd.concat((tuning_df, tmp_tuning_df), axis=0)
    fit_df = pd.concat((fit_df, tmp_fit_df), axis=0)
    return tuning_df, fit_df

def extend_NSD_line(fit_df):
    ecc_max = fit_df.query('dset_type == "Broderick et al. V1"').ecc.max()
    tmp_fit_df = fit_df.query('dset_type == "NSD V1" & goal == "replication" & ecc == 0')
    coeff = tmp_fit_df['coefficient'].iloc[0].tolist()
    tmp_fit_df['ecc'] = ecc_max
    tmp_fit_df['fitted'] = coeff[0] * ecc_max + coeff[1]
    fit_df = pd.concat((fit_df, tmp_fit_df), axis=0)
    return fit_df

def plot_prf_size_vs_eccentricity(data, x, y, hue, ax, pal, hue_order=None, save_path=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.set_theme("paper", style='ticks')
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, markers=True, 
    dashes=False, errorbar=('ci', 68), palette=pal, ax=ax, legend=False)
    ax.set_xlim(0, 4)
    ax.set_xticks([0.5, 1,1.5, 2, 2.5, 3, 3.5, 4])
    ax.set_yticks([0, 1, 2])
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xlabel('Eccentricity (deg)')
    ax.set_ylabel('pRF Size (deg)')
    ax.set_title('pRF Size vs Eccentricity')
    ax = sns.lineplot(ax=ax, data=data.query('ecc != 0'), 
                        x='ecc', y='size', palette=pal,
                                            hue=hue, hue_order=hue_order, marker='o', linestyle='', 
                                            markersize=5, errorbar=('ci', 68), 
                                            err_style='bars', 
                                            err_kws={'elinewidth': 1}, 
                                            alpha=0.9, zorder=10)
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))#ax.get_legend().remove()
    #plt.tight_layout()
    utils.save_fig(save_path)
    return ax


def plot_prf_size_vs_fwhm(data, x, y, hue, ax, pal, hue_order=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.set_theme("paper", style='ticks', rc=rc, palette=pal)
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order, 
    palette=pal, ax=ax)
    ax.set_xlabel('pRF Size (deg)')
    ax.set_xlim(0, 2)
    ax.set_xticks([0, 1, 2])
    ax.set_ylabel('Bandwidth (FWHM)')
    ax.set_ylim((0,15))
    #ax.legend(frameon=False)

    return ax

def plot_v1_precision_vs_dprime(df, x, y, hue, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.set_theme("paper", style='ticks', rc=rc)
    sns.scatterplot(
        data=df, 
        x=x, y=y, hue=hue, ax=ax
    )

    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
    return ax
