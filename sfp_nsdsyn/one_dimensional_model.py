import sys
import os
sys.path.append('..')
import numpy as np
import pandas as pd
import torch
import sfp_nsdsyn.utils as utils
from timeit import default_timer as timer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize


def torch_log_norm_pdf(x, amp, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
    # the preferred period, the ivnerse of the mode / the peak spatial frequency.
    pdf = amp * torch.exp(-(torch.log2(x) - torch.log2(mode)) ** 2 / (2 * sigma ** 2))

    return pdf


def np_log_norm_pdf(x, amp, mode, sigma):
    """the pdf of the log normal distribution, with a scale factor
    """
    # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
    # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is
    # the preferred period, the inverse of the mode = the peak spatial frequency.
    pdf = amp * np.exp(-(np.log2(x) - np.log2(mode)) ** 2 / (2 * sigma ** 2))
    return pdf


def _track_loss_df(values, create_loss_df=True, loss_df=None,
                   keys=["subj", "vroinames", "eccrois", "alpha", "epoch", "start_loss", "final_loss"]):
    if create_loss_df:
        loss_df = utils.create_empty_df(col_list=keys)
    else:
        if loss_df is None:
            raise Exception("Dataframe for saving loss log is not defined!")

    loss_df = loss_df.append(dict(zip(keys, values)), ignore_index=True)
    return loss_df


def set_initial_params(init_list, to_torch=True):
    if init_list == "random":
        init_list = np.random.random(3)
    if to_torch:
        amp = torch.tensor([init_list[0]], dtype=torch.float32, requires_grad=True)
        mode = torch.tensor([init_list[1]], dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor([init_list[2]], dtype=torch.float32, requires_grad=True)
    else:
        amp = np.random.random(1)
        mode = np.random.random(1) + 0.5
        sigma = np.random.random(1) + 0.5
    return amp, mode, sigma


def _df_column_to_torch(df, column):
    torch_column_val = torch.from_numpy(df[column].values)
    return torch_column_val


def fit_1D_model(df, sn, stim_class, varea, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):
    subj = utils.sub_number_to_string(sn)
    eroi_list = utils.sort_a_df_column(df[ecc_bins])

    # Initialize output df
    loss_history_df = {}
    model_history_df = {}

    # start fitting process
    cur_df = df.query('(subj == @subj) & (names == @stim_class) & (vroinames == @varea)')
    print(f'##{subj}-{stim_class}-{varea} start!##')
    start = timer()
    for cur_ecc in eroi_list:
        tmp_df = cur_df[cur_df[ecc_bins] == cur_ecc]
        x = _df_column_to_torch(tmp_df, "local_sf")
        y = _df_column_to_torch(tmp_df, "betas")
        # set initial parameters
        amp, mode, sigma = _set_initial_params(initial_val)
        # select an optimizer
        optimizer = torch.optim.Adam([amp, mode, sigma], lr=lr)
        # set a loss function - mean squared error
        criterion = torch.nn.MSELoss()

        loss_history = []
        model_history = []
        for t in range(epoch):
            optimizer.zero_grad()
            loss = criterion(torch_log_norm_pdf(x, amp, mode, sigma), y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            model_history.append([amp.item(), mode.item(), sigma.item()])
            if (t + 1) % n_print == 0:
                print(f'Loss at epoch {str(t + 1)}: {str(round(loss.item(), 3))}')

        print(f'###Eccentricity bin {str(eroi_list.index(cur_ecc) + 1)} out of {len(eroi_list)} finished!###')
        print(
            f'Final parameters: amp {amp.item()}, mode {mode.item()}, sigma {sigma.item()}\n')
        model_history_df[cur_ecc] = pd.DataFrame(model_history, columns=['amp', 'mode', 'sigma'])
        loss_history_df[cur_ecc] = pd.DataFrame(loss_history, columns=['loss'])
    end = timer()
    elapsed_time = end - start
    print(f'###{subj}-{stim_class}-{varea} finished!###')
    print(f'Elapsed time: {np.round(end - start, 2)} sec')
    model_history_df = pd.concat(model_history_df).reset_index().rename(
        columns={'level_0': "ecc_bin", 'level_1': "epoch"})
    loss_history_df = pd.concat(loss_history_df).reset_index().rename(
        columns={'level_0': "ecc_bin", 'level_1': "epoch"})
    loss_history_df['lr'] = lr
    return model_history_df, loss_history_df, elapsed_time


def run_1D_model(df, sn_list, stim_class_list, varea_list, ecc_bins="bins", n_print=1000,
                 initial_val="random", epoch=5000, lr=1e-3):
    model_history_df = {}
    loss_history_df = {}
    for sn in sn_list:

        subj = utils.sub_number_to_string(sn)
        m_m_df = {}
        l_l_df = {}
        for stim_class in stim_class_list:
            m_df = {}
            l_df = {}
            for varea in varea_list:
                m_df[varea], l_df[varea], e_time = fit_1D_model(df, sn=sn, stim_class=stim_class,
                                                                varea=varea, ecc_bins=ecc_bins, n_print=n_print,
                                                                initial_val=initial_val, epoch=epoch, lr=lr)

            m_m_df[stim_class] = pd.concat(m_df)
            l_l_df[stim_class] = pd.concat(l_df)
        model_history_df[subj] = pd.concat(m_m_df)
        loss_history_df[subj] = pd.concat(l_l_df)
    col_replaced = {'level_0': 'subj', 'level_1': 'names', 'level_2': 'vroinames'}
    model_history_df = pd.concat(model_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)
    loss_history_df = pd.concat(loss_history_df).reset_index().drop(columns='level_3').rename(columns=col_replaced)

    return model_history_df, loss_history_df


def sim_fit_1D_model(cur_df, ecc_bins="bins", n_print=1000,
                     initial_val="random", epoch=5000, lr=1e-3):
    eroi_list = utils.sort_a_df_column(cur_df[ecc_bins])

    # Initialize output df
    loss_history_df = {}
    model_history_df = {}

    # start fitting process
    start = timer()
    for cur_ecc in eroi_list:
        tmp_df = cur_df[cur_df[ecc_bins] == cur_ecc]
        x = _df_column_to_torch(tmp_df, "local_sf")
        y = _df_column_to_torch(tmp_df, "betas")
        # set initial parameters
        amp, mode, sigma = _set_initial_params(initial_val)
        # select an optimizer
        optimizer = torch.optim.Adam([amp, mode, sigma], lr=lr)
        # set a loss function - mean squared error
        criterion = torch.nn.MSELoss()

        loss_history = []
        model_history = []
        for t in range(epoch):
            optimizer.zero_grad()
            loss = criterion(torch_log_norm_pdf(x, amp, mode, sigma), y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            model_history.append([amp.item(), mode.item(), sigma.item()])
            if (t + 1) % n_print == 0:
                print(f'Loss at epoch {str(t + 1)}: {str(round(loss.item(), 3))}')

        print(f'###Eccentricity bin {str(eroi_list.index(cur_ecc) + 1)} out of {len(eroi_list)} finished!###')
        print(
            f'Final parameters: amp {amp.item()}, mode {mode.item()}, sigma {sigma.item()}\n')
        model_history_df[cur_ecc] = pd.DataFrame(model_history, columns=['amp', 'mode', 'sigma'])
        loss_history_df[cur_ecc] = pd.DataFrame(loss_history, columns=['loss'])
    end = timer()
    elapsed_time = end - start
    print(f'Elapsed time: {np.round(end - start, 2)} sec')
    model_history_df = pd.concat(model_history_df).reset_index().rename(
        columns={'level_0': "bins", 'level_1': "epoch"})
    loss_history_df = pd.concat(loss_history_df).reset_index().rename(
        columns={'level_0': "bins", 'level_1': "epoch"})
    loss_history_df['lr'] = lr
    return model_history_df, loss_history_df, elapsed_time


def get_bin_labels(e1, e2, enum):
    if enum == "log3":
        enum_only = enum[3:]
        bin_list = np.logspace(np.log2(float(e1)), np.log2(float(e2)), num=int(enum_only) + 1, base=2)
    else:
        bin_list = np.linspace(float(e1), float(e2), int(enum) + 1)
    bin_list = np.round(bin_list, 2)
    bin_labels = [f'{str(a)}-{str(b)} deg' for a, b in zip(bin_list[:-1], bin_list[1:])]
    return bin_list, bin_labels


def bin_ecc(to_bin, bin_list, bin_labels=None):
    if bin_labels is None:
        bin_labels = [f'{str(a)}-{str(b)}deg' for a, b in zip(bin_list[:-1], bin_list[1:])]
    ecc_bin = pd.cut(to_bin, bins=bin_list, include_lowest=True, labels=bin_labels)
    return ecc_bin


def summary_stat_for_ecc_bin(df,
                             to_group=['subj', 'ecc_bin', 'freq_lvl', 'names', 'vroinames'],
                             to_bin=["betas", "local_sf"],
                             central_tendency="mean"):
    if central_tendency == "mode":
        c_df = df.groupby(to_group)[to_bin].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    else:
        c_df = df.groupby(to_group)[to_bin].agg(central_tendency).reset_index()
    # this should be fixed for cases where there are more than two central tendencies.
    return c_df


class LogGaussianTuningDataset:
    """Tranform dataframes to pivot style. x axis represents ecc_bin, y axis is freq_lvl."""

    def __init__(self, local_sf, betas):
        self.target = torch.tensor(betas.to_numpy()).double()
        self.sf = torch.tensor(local_sf.to_numpy()).double()


# numpy to torch function
def _cast_as_tensor(x):
    """ Change numpy vector to torch vector. The input x should be either a column of dataframe,
     a list, or numpy vector.You can also pass a torch vector but it will print out warnings."""
    if type(x) == pd.Series:
        x = x.values
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32)


def _cast_as_param(x, requires_grad=True):
    """ Change input x to parameter"""
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)


class LogGaussianTuningModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = _cast_as_param(np.random.random(1))
        self.mode = _cast_as_param(np.random.random(1) + 0.5)
        self.sigma = _cast_as_param(np.random.random(1) + 0.5)

    def forward(self, x):
        """the pdf of the log normal distribution, with a scale factor
        """
        # note that mode here is the actual mode, for us, the peak spatial frequency. this differs from
        # the 2d version we have, where we we have np.log2(x)+np.log2(p), so that p is the inverse of
        # the preferred period, the inverse of the mode / the peak spatial frequency.
        pdf = self.slope * torch.exp(-(torch.log2(x) - torch.log2(self.mode)) ** 2 / (2 * self.sigma ** 2))
        return torch.clamp(pdf, min=1e-6)


def fit_tuning_curves(my_model, my_dataset, learning_rate=1e-4, max_epoch=5000, print_every=100,
                      anomaly_detection=False, amsgrad=False, eps=1e-8, save_path=None, seed=None, verbose=True):
    """Fit log normal Gaussian tuning curves.
    This function will allow you to run a for loop for N times set as max_epoch,
    and return the output of the training; loss history, model history."""
    torch.autograd.set_detect_anomaly(anomaly_detection)
    if seed is not None:
        np.random.seed(seed)
    my_parameters = [p for p in my_model.parameters() if p.requires_grad]
    params_col = [name for name, param in my_model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(my_parameters, lr=learning_rate, amsgrad=amsgrad, eps=eps)
    loss_fn = torch.nn.MSELoss()
    loss_history = []
    model_history = []
    start = timer()
    for t in range(max_epoch):
        optimizer.zero_grad()  # clear previous gradients
        pred = my_model.forward(x=my_dataset.sf)
        loss = loss_fn(pred, my_dataset.target)
        param_values = [p.detach().numpy().item() for p in my_model.parameters() if p.requires_grad]
        loss_history.append(loss.item())
        model_history.append(param_values)  # more than one item here
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients

        if (t + 1) % print_every == 0 or t == 0:
            content = f'**epoch no.{t} loss: {np.round(loss.item(), 5)}'
            print(content)
    elapsed_time = timer() - start
    my_model.eval()
    param_values = [p.detach().numpy().item() for p in my_model.parameters() if p.requires_grad]
    if verbose:
        print(f'**epoch no.{max_epoch}: Finished! final params {dict(zip(params_col, np.round(param_values,3)))}')
        print(f'Elapsed time: {np.round(elapsed_time, 2)} sec \n')
    if save_path is not None:
        torch.save(my_model.state_dict(), save_path)
    loss_history = pd.DataFrame(loss_history, columns=['loss']).reset_index().rename(columns={'index': 'epoch'})
    model_history = pd.DataFrame(model_history, columns=params_col).reset_index().rename(columns={'index': 'epoch'})
    return loss_history, model_history


def fit_tuning_curves_for_each_bin(bin_labels, df, learning_rate=1e-4, max_epoch=5000, print_every=100,
                                   anomaly_detection=False, amsgrad=False, eps=1e-8, save_path_list=None):
    loss_history = {}
    model_history = {}
    for bin, save_path in zip(bin_labels, save_path_list):
        c_df = df.query('ecc_bin == @bin')
        my_dataset = LogGaussianTuningDataset(c_df)
        my_model = LogGaussianTuningModel()
        loss_history[bin], model_history[bin] = fit_tuning_curves(my_model, my_dataset, learning_rate, max_epoch,
                                                                  print_every,
                                                                  anomaly_detection, amsgrad, eps, save_path)
    loss_history = pd.concat(loss_history).reset_index().drop(columns='level_1').rename(columns={'level_0': 'ecc_bin'})
    model_history = pd.concat(model_history).reset_index().drop(columns='level_1').rename(
        columns={'level_0': 'ecc_bin'})
    return loss_history, model_history


def load_history_df_1D(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, df_dir):
    subj = utils.sub_number_to_string(sn, dataset=dset)
    f_name = f'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr_rate}_eph-{max_epoch}_{roi}_vs-pRFcenter_e{e1}-{e2}_nbin-{nbin}.h5'
    return pd.read_hdf(os.path.join(df_dir, f_name))


def load_history_1D_all_subj(sn_list, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, df_dir):
    df = pd.DataFrame({})
    for sn in sn_list:
        tmp = load_history_df_1D(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, df_dir)
        df = pd.concat((df, tmp), axis=0)
    return df


def load_binned_df_1D(sn, dset, stat, roi, e1, e2, nbin, df_dir):
    subj = utils.sub_number_to_string(sn, dataset=dset)
    f_name = f'binned_e{e1}-{e2}_nbin-{nbin}_{subj}_stim_voxel_info_df_vs-pRFcenter_{roi}_{stat}.csv'
    df = pd.read_csv(os.path.join(df_dir, f_name))
    if 'subj' not in df.columns.tolist():
        df['subj'] = subj
    return df


def load_binned_df_1D_all_subj(sn_list, dset, stat, roi, e1, e2, nbin, df_dir):
    df = pd.DataFrame({})
    for sn in sn_list:
        tmp = load_binned_df_1D(sn, dset, stat, roi, e1, e2, nbin, df_dir)
        df = pd.concat((df, tmp), axis=0)
    return df


def _merge_fitting_output_df_to_subj_df(model_history, binned_df, merge_on=["subj", "vroinames", "ecc_bin", 'names']):
    max_epoch = model_history.epoch.max()
    model_history = model_history.query('epoch == @max_epoch')
    merged_df = binned_df.merge(model_history, on=merge_on)
    return merged_df


def load_and_merge_1D_df(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, input_dir, output_dir):
    model_history = load_history_df_1D(sn, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, output_dir)
    binned_df = load_binned_df_1D(sn, dset, stat, roi, e1, e2, nbin, input_dir)
    return _merge_fitting_output_df_to_subj_df(model_history, binned_df, merge_on=['ecc_bin', 'names'])


def load_and_merge_1D_df_all_subj(sn_list, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin, input_dir,
                                  output_dir):
    model_history = load_history_1D_all_subj(sn_list, dset, stat, df_type, roi, lr_rate, max_epoch, e1, e2, nbin,
                                             output_dir)
    binned_df = load_binned_df_1D_all_subj(sn_list, dset, stat, roi, e1, e2, nbin, input_dir)
    return _merge_fitting_output_df_to_subj_df(model_history, binned_df,
                                               merge_on=['subj', 'vroinames', 'ecc_bin', 'names'])


def plot_datapoints(df, col='names', hue='ecc_bin', lgd_title='Eccentricity', height=5, subplot_right=0.9,
                    sup_title=None,
                    save_fig=False, save_path='/Volumes/server/Project/sfp_nsd/derivatives/figures/1D_results.png'):
    col_order = utils.sort_a_df_column(df[col])
    sns.set_context("notebook", font_scale=1.5)
    grid = sns.FacetGrid(df,
                         col='names',
                         hue=hue,
                         hue_order=df[hue].unique(),
                         palette=sns.color_palette("rocket", df[hue].nunique()),
                         col_wrap=4,
                         height=height,
                         legend_out=True,
                         sharex=True, sharey=True)
    g = grid.map(sns.scatterplot, 'local_sf', 'betas', edgecolor="gray")
    grid.set_axis_labels('Spatial Frequency', 'Beta')
    grid.fig.legend(title=lgd_title, labels=df[hue].unique())
    if sup_title is not None:
        grid.fig.suptitle(sup_title)
        grid.fig.subplots_adjust(right=subplot_right, top=0.86)
    else:
        grid.fig.subplots_adjust(right=subplot_right)
    plt.xscale('log')
    utils.save_fig(save_fig, save_path)
    return grid


def _get_x_and_y_prediction(min, max, slope, peak, sigma, n_points=500):
    x = np.linspace(min, max, n_points)
    y = [np_log_norm_pdf(k, slope, peak, sigma) for k in x]
    return x, y


def _get_x_and_y_prediction_from_2D(sf_min, sf_max, fnl_param_df, voxel_info):
    voxel_info = voxel_info.drop_duplicates(subset=['subj', 'voxel'])
    voxel_info['local_ori'] = 1
    sf_range = np.logspace(sf_min, sf_max, num=50, base=2)
    return x, y


def plot_param_history(df,
                       to_label=None, label_order=None,
                       lgd_title=None, height=5,
                       save_fig=False, save_path='/Users/jh7685/Dropbox/NYU/Projects/SF/MyResults/.png',
                       ci=68, n_boot=100, log_y=True):
    sns.set_context("notebook", font_scale=1.5)
    to_x = "epoch"
    to_y = "value"
    x_label = "Epoch"
    y_label = "Parameter value"
    n_labels = df[to_label].nunique()
    # expects RGB triplets to lie between 0 and 1, not 0 and 255
    pal = sns.color_palette("rocket", n_labels)
    grid = sns.FacetGrid(df,
                         hue=to_label,
                         hue_order=label_order,
                         col="names",
                         row='params',
                         height=height,
                         palette=pal,
                         legend_out=True,
                         sharex=True, sharey=False)
    g = grid.map(sns.lineplot, to_x, to_y, linewidth=2, ci=ci, n_boot=n_boot)
    grid.set_axis_labels(x_label, y_label)
    if lgd_title is not None:
        grid.add_legend(title=lgd_title)
    if log_y is True:
        plt.semilogy()
    utils.save_fig(save_fig, save_path)


def plot_ecc_bin_prediction_from_2D(pred_df, pred_y, hue, lgd_title, title, save_path=None):
    fig, ax = plt.subplots()
    sns.set_context("notebook", font_scale=1.5)
    sns.lineplot(x="local_sf", y=pred_y, hue=hue, data=pred_df,
                 palette=sns.color_palette("rocket", pred_df[hue].nunique()), ax=ax, marker='', ci=None, linestyle='-')
    ax.set(xlabel='Spatial Frequency', ylabel='Betas')
    plt.title(title, fontsize=20)
    ax.set_xscale('log')
    utils.save_fig(save_path != None, save_path)
    plt.show()


def plot_sf_curves_from_2D(pred_df, pred_y, y, hue, lgd_title, t, save_path=None, lines="data"):
    sns.set_context("notebook", font_scale=1.5)
    grid = sns.FacetGrid(pred_df,
                         col='names',
                         hue=hue,
                         hue_order=pred_df[hue].unique(),
                         palette=sns.color_palette("rocket", pred_df[hue].nunique()),
                         height=5,
                         legend_out=True,
                         sharex=True, sharey=True)
    if lines == "data":
        grid.map(sns.lineplot, "local_sf", y, marker='o', ci=68, linestyle='-', err_style='bars', linewidth=3)
    elif lines == "pred":
        grid.map(sns.lineplot, "local_sf", y, marker='o', ci=None, linestyle='', linewidth=2)
        grid.map(sns.lineplot, "local_sf", pred_y, marker='', ci=68, err_style='band', linestyle='-', linewidth=2)
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    grid.add_legend(title=lgd_title)
    grid.set(ylim=(0.06, 0.32), xlim=(0.08, 35))  # y=0.05, 20
    grid.fig.suptitle(t)
    plt.xscale('log')
    grid.fig.subplots_adjust(top=0.8)
    utils.save_fig(save_path != None, save_path)


def plot_sf_curves_from_2D_voxel(df, y, pred_df, pred_y, hue, lgd_title, t, save_path=None):
    sns.set_context("notebook", font_scale=1.5)
    grid = sns.FacetGrid(df,
                         col='eccentricity',
                         hue=hue,
                         hue_order=pred_df[hue].unique(),
                         palette=sns.color_palette("rocket", pred_df[hue].nunique()),
                         height=4,
                         legend_out=True,
                         sharex=True, sharey=True)
    grid.map(sns.lineplot, "local_sf", y, marker='o', ci=68, err_style='bars', linestyle='', linewidth=2)
    for subplot_title, ax in grid.axes_dict.items():
        tmp = pred_df.query('eccentricity == @subplot_title')
        ax.errorbar(tmp['local_sf'], y=tmp[pred_y], yerr=tmp['err'], color='k', linewidth=1)
    grid.set_axis_labels('Spatial Frequency', 'Betas')
    grid.add_legend(title=lgd_title)
    grid.set(ylim=(0, 0.22), xlim=(0.01, 100))
    plt.xscale('log')
    grid.fig.suptitle(t)
    utils.save_fig(save_path != None, save_path)


def match_wildcards_with_col(arg):
    switcher = {'roi': 'vroinames',
                'eph': 'max_epoch',
                'lr': 'lr_rate',
                'class': 'names'}
    return switcher.get(arg, arg)



def load_LogGaussianTuningModel(pt_file_path):
    my_model = LogGaussianTuningModel()
    my_model.load_state_dict(torch.load(pt_file_path, map_location='cpu'))
    my_model.eval()
    return my_model


def _find_bin(row):
    _, bin_labels = get_bin_labels(row.e1, row.e2, row.nbin)
    return bin_labels[int(row.curbin)]


def model_to_df(pt_file_path=None, my_model=None, *args):
    if pt_file_path is not None:
        my_model = load_LogGaussianTuningModel(pt_file_path)
    model_dict = {}
    for name, param in my_model.named_parameters():
        model_dict[name] = param.detach().numpy()
    model_df = pd.DataFrame(model_dict)
    for arg in args:
        model_df[match_wildcards_with_col(arg)] = [k for k in pt_file_path.split('_') if arg in k][0][
                                                  len(arg) + 1:].replace('-', ' ')
    return model_df


def load_all_models(pt_file_path_list, *args):
    model_df = pd.DataFrame({})
    for pt_file_path in pt_file_path_list:
        tmp = model_to_df(pt_file_path, None, *args)
        model_df = model_df.append(tmp)
    model_df['ecc_bin'] = model_df.apply(_find_bin, axis=1)
    return model_df

def fit_logGaussian_curves(df,
                           x, y,
                           initial_params,
                           goodness_of_fit=True,
                           maxfev=100000,
                           tol = 1.5e-08,
                           amp_bounds=(0,10),
                           mode_bounds=(2**(-5), 2**11),
                           sigma_bounds=(0.01, 10)):
    tmp = df.sort_values(x)
    p_opt, p_cov = optimize.curve_fit(f=np_log_norm_pdf,
                                      xdata=tmp[x].to_list(),
                                      ydata=tmp[y].to_list(),
                                      maxfev=maxfev,
                                      ftol=tol, xtol=tol,
                                      p0=initial_params,
                                      bounds=list(zip(amp_bounds, mode_bounds, sigma_bounds))
                                      )
    p_opt = pd.DataFrame(p_opt.reshape(1,-1), columns=['amp','mode','sigma'])
    if goodness_of_fit is True:
        r2, rmse = calculate_goodness_of_fit(p_opt, df, x, y)
        p_opt['r2'] = r2
        p_opt['rmse'] = rmse
    return p_opt, p_cov

def calculate_goodness_of_fit(p_opt, df, x, y):
    tmp = df.sort_values(x)
    y_pred = np_log_norm_pdf(tmp[x].to_list(),
                             p_opt['amp'].values,
                             p_opt['mode'].values,
                             p_opt['sigma'].values)
    # Compute residuals
    residuals = tmp[y].to_list() - y_pred
    # Calculate the total sum of squares (TSS), variance of the observed data
    ss_tot = np.sum((tmp[y] - np.mean(tmp[y])) ** 2)
    # Calculate the residual sum of squares (RSS)
    ss_res = np.sum(residuals ** 2)
    # Calculate R^2
    r_squared = 1 - (ss_res / ss_tot)
    # Calculate RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))
    return r_squared, rmse

def plot_logGaussian_fit(df, x, y, p_opt, ax=None,
                         x_label='base frequency', ax_title=None):
    tmp = df.sort_values(x)
    new_x_vals = np.logspace(np.log10(tmp[x].min()), np.log10(tmp[x].max()), 100)
    y_pred = np_log_norm_pdf(new_x_vals,
                             p_opt['amp'].values,
                             p_opt['mode'].values,
                             p_opt['sigma'].values)
    if ax is None:
        fig, ax = plt.subplots()
    # Plot the data and the fitted curve on the provided Axes object
    ax.plot(tmp[x], tmp[y], 'o', label='data', color='black')
    ax.plot(new_x_vals, y_pred, label='fit', linewidth=2, color='r')
    ax.set_xscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel('betas')
    ax.set_title(ax_title)
    #ax.legend()
    plt.tight_layout()
    return ax

def get_bandwidth_in_octave(data_df, local_sf, slope, peak, sigma):
    xx = data_df[local_sf]
    x_min, x_max = np.floor(np.log(peak) - 5 * sigma), np.ceil(np.log(peak) + 5 * sigma)
    pred_x, pred_y = _get_x_and_y_prediction(x_min, x_max, slope, peak, sigma, n_points=1000)
    # Find the peak freq's beta value

    # find the half of the peak based on the lowest beta value in the prediction
    results_half_y = 0.5*slope
    # find the x values that are the closest to the half of the peak
    half_y = np.abs([k-results_half_y for k in pred_y])
    results_half_x = half_y.argsort()

    results_half_x_min = results_half_x[pred_x[results_half_x] < peak]
    f_low = pred_x[results_half_x_min][0]

    results_half_x_max = results_half_x[pred_x[results_half_x] > peak]
    f_high = pred_x[results_half_x_max][0]
    # Calculate the bandwidth in octave
    fwhm_octaves = np.log2(f_high) - np.log2(f_low)
    return fwhm_octaves

# Define the log Gaussian function
def log_gaussian(x, mode, sigma, A):
    return A * np.exp(-((np.log(x) - np.log(mode)) ** 2) / (2 * sigma ** 2))

# Define the function to find the half maximum points
def half_max_function(x, mode, sigma, A):
    return log_gaussian(x, mode, sigma, A) - (A / 2)

def get_bandwidth_chatgpt(mode, sigma, slope):
    from scipy.optimize import fsolve

    # Define the function to find the half maximum points

    # Solving for x1 and x2
    x1 = fsolve(half_max_function, 1e-4, args=(mode, sigma, slope))
    x2 = fsolve(half_max_function, mode*2, args=(mode, sigma, slope))

    # Calculate FWHM in octaves
    fwhm_octaves = np.log2(x2 / x1)

    print("x1 (lower half-max point):", x1)
    print("x2 (upper half-max point):", x2)
    print("FWHM in octaves:", fwhm_octaves)
    return fwhm_octaves