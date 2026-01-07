# sys.path.append('../../')
import os
import pandas as pd
import numpy as np
import torch
import itertools
from timeit import default_timer as timer
from . import utils as utils
from pathlib import Path
import warnings

def stim_w_r_and_w_a(stim_df,
                      stim_class=['pinwheel','forward spiral','annulus','reverse spiral']):
    stim_info = pd.read_csv(stim_df)
    stim_info.drop_duplicates('names', inplace=True)
    stim_info = stim_info[['names','w_r','w_a']]
    stim_info.query('names in @stim_class', inplace=True)
    return stim_info

def merge_synthetic_values_to_df(df, var_name, val_range, n_vals,endpoint=True):
    vals = np.linspace(val_range[0], val_range[-1], num=n_vals, endpoint=endpoint)
    synthetic_df = pd.DataFrame({})
    for val in vals:
        df[var_name] = val
        synthetic_df = synthetic_df.append(df, ignore_index=True)
    return synthetic_df

def create_synthetic_cols(df, var_list, val_range_list, n_val_list, endpoint=True):
    if (len(var_list) == len(val_range_list) == len(n_val_list)) is False:
        raise Exception("var_list nad val_range_list must have the same length!")
    vals = [np.linspace(val_range_list[k][0],
                        val_range_list[k][-1],
                        n_val_list[k],
                        endpoint=endpoint) for k in range(len(n_val_list))]
    all_combinations = list(itertools.product(*vals))
    synthetic_df = pd.DataFrame({})
    for combination in all_combinations:
        pd_dict = {col:val for col,val in zip(var_list, combination)}
        tmp = df.assign(**pd_dict)
        synthetic_df = synthetic_df.append(tmp, ignore_index=True)
    return synthetic_df

def break_down_phase(df):
    dv_to_group = ['subj', 'freq_lvl', 'names', 'voxel', 'hemi']
    df = df.groupby(dv_to_group).mean().reset_index()

    return df


def get_Pv_row(row, params, modulator=True):
    if params.shape[0] != 1:
        raise Exception('params should be a df with one row!')
    #params = params.iloc[0]
    ecc_dependency = params.slope * row.eccentricity + params.intercept
    if modulator is True:
        Pv = ecc_dependency * (1 + params.p_1 * np.cos(2 * row.local_ori) +
                               params.p_2 * np.cos(4 * row.local_ori) +
                               params.p_3 * np.cos(2 * (row.local_ori - row.angle)) +
                               params.p_4 * np.cos(4 * (row.local_ori - row.angle)))
    elif modulator is False:
        Pv = ecc_dependency
    return Pv

def get_p1v_row(row, params):
    if params.shape[0] != 1:
        raise Exception('params should be a df with one row!')
    #params = params.iloc[0]
    ecc_dependency = params.slope * row.eccentricity + params.intercept
    p1v = ecc_dependency * (1 + params.p_1 * np.cos(2 * row.local_ori))
    return p1v

def get_Pv(slope, eccentricity, intercept,
           local_ori, angle,
           p_1, p_2, p_3, p_4,
           modulator=True):
    ecc_dependency = slope * eccentricity + intercept
    if modulator is True:
        Pv = ecc_dependency * (1 + p_1 * np.cos(2 * local_ori) +
                               p_2 * np.cos(4 * local_ori) +
                               p_3 * np.cos(2 * (local_ori - angle)) +
                               p_4 * np.cos(4 * (local_ori - angle)))
    elif modulator is False:
        Pv = ecc_dependency

    return Pv


def normalize(voxel_info, to_norm, to_group=["subj", "voxel"], phase_info=False):
    """calculate L2 norm for each voxel and normalized using the L2 norm"""

    if type(voxel_info) == pd.DataFrame:
        if phase_info is False:
            if all(voxel_info.groupby(to_group).size() == 28) is False:
                warnings.warn('There are more than 28 conditions for one voxel!\n')
        tmp = voxel_info.groupby(to_group)[to_norm].apply(lambda x: x / np.linalg.norm(x)).reset_index()
        tmp.rename(columns={to_norm: f'normed_{to_norm}'}, inplace=True)
        normed = pd.merge(voxel_info, tmp[to_group + [f'normed_{to_norm}']], on=to_group)

    elif type(voxel_info) == list:
        normed = torch.empty(to_norm.shape, dtype=torch.float64)
        for idx in voxel_info:
            voxel_idx = voxel_info == idx
            print(voxel_idx)
            normed[voxel_idx] = to_norm[voxel_idx] / torch.linalg.norm(to_norm[voxel_idx])
    return normed


def normalize_pivotStyle(betas):
    """calculate L2 norm for each voxel and normalize using the L2 norm"""
    return betas / torch.linalg.norm(betas, ord=2, dim=1, keepdim=True)


# numpy to torch function
def _cast_as_tensor(x):
    """ Change numpy vector to torch vector. The input x should be either a column of dataframe,
     a list, or numpy vector.You can also pass a torch vector but it will print out warnings."""
    if type(x) == pd.Series:
        x = x.values
    # needs to be float32 to work with the Hessian calculations
    return torch.tensor(x, dtype=torch.float32)


def _cast_as_param(x, requires_grad=True):
    """ Change input x to """
    return torch.nn.Parameter(_cast_as_tensor(x), requires_grad=requires_grad)



def count_nan_in_torch_vector(x):
    return torch.nonzero(torch.isnan(torch.log2(x).view(-1)))


class SpatialFrequencyDataset:
    """Tranform dataframes to pivot style. x axis represents voxel, y axis is class_idx."""
    def __init__(self, df, beta_col='betas'):
        self.target = torch.tensor(df.pivot('voxel', 'class_idx', beta_col).to_numpy())
        self.ori = torch.tensor(df.pivot('voxel', 'class_idx', 'local_ori').to_numpy())
        self.angle = torch.tensor(df.pivot('voxel', 'class_idx', 'angle').to_numpy())
        self.eccen = torch.tensor(df.pivot('voxel', 'class_idx', 'eccentricity').to_numpy())
        self.sf = torch.tensor(df.pivot('voxel', 'class_idx', 'local_sf').to_numpy())
        self.sigma_v_squared = torch.tensor(df.pivot('voxel', 'class_idx', 'sigma_v_squared').to_numpy())
        self.voxel_info = df.pivot('voxel', 'class_idx', beta_col).index.astype(int).to_list()

class SpatialFrequencyModel(torch.nn.Module):
    def __init__(self, params=None, model=7, random_state=42):
        """ The input subj_df should be across-phase averaged prior to this class."""
        super().__init__()  # Allows us to avoid using the base class name explicitly
        self.A_3 = 0
        self.A_4 = 0
        self.random_state = random_state
        np.random.seed(random_state)
        if params is not None:
            self._init_existing_params(params)
        else:
            if model == 0:  # sigma and slope
                self.sigma = _cast_as_param(np.random.random(1) + 0.5)
                self.slope = _cast_as_param(np.random.random(1))
                self.intercept = 0
                self.p_1 = self.p_2 = self.p_3 = self.p_4 = self.A_1 = self.A_2 = 0

            elif model == 1:  # sigma and intercept
                self.sigma = _cast_as_param(np.random.random(1) + 0.5)
                self.slope = 0
                self.intercept = _cast_as_param(np.random.random(1))
                self.p_1 = self.p_2 = self.p_3 = self.p_4 = self.A_1 = self.A_2 = 0

            elif model == 2:  # sigma, intercept, and slope
                self.sigma = _cast_as_param(np.random.random(1) + 0.5)
                self.slope = _cast_as_param(np.random.random(1))
                self.intercept = _cast_as_param(np.random.random(1))
                self.p_1 = self.p_2 = self.p_3 = self.p_4 = self.A_1 = self.A_2 = 0

            elif model == 4:  # no relative orientation and amplitude modulation
                self.sigma = _cast_as_param(np.random.random(1) + 0.5)
                self.slope = _cast_as_param(np.random.random(1))
                self.intercept = _cast_as_param(np.random.random(1))
                self.p_1 = _cast_as_param(np.random.random(1) / 10)
                self.p_2 = _cast_as_param(np.random.random(1) / 10)
                self.p_3 = 0
                self.p_4 = 0
                self.A_1 = 0
                self.A_2 = 0

            elif model == 5:  # no amplitude modulation
                self.sigma = _cast_as_param(np.random.random(1) + 0.5)
                self.slope = _cast_as_param(np.random.random(1))
                self.intercept = _cast_as_param(np.random.random(1))
                self.p_1 = _cast_as_param(np.random.random(1) / 10)
                self.p_2 = _cast_as_param(np.random.random(1) / 10)
                self.p_3 = _cast_as_param(np.random.random(1) / 10)
                self.p_4 = _cast_as_param(np.random.random(1) / 10)
                self.A_1 = 0
                self.A_2 = 0

            elif model == 6:  # no relative orientation
                self.sigma = _cast_as_param(np.random.random(1) + 0.5)
                self.slope = _cast_as_param(np.random.random(1))
                self.intercept = _cast_as_param(np.random.random(1))
                self.p_1 = _cast_as_param(np.random.random(1) / 10)
                self.p_2 = _cast_as_param(np.random.random(1) / 10)
                self.p_3 = 0
                self.p_4 = 0
                self.A_1 = _cast_as_param(np.random.random(1))
                self.A_2 = _cast_as_param(np.random.random(1))

            elif model == 7:  # original model
                self.sigma = _cast_as_param(np.random.random(1))
                self.slope = _cast_as_param(np.random.random(1))
                self.intercept = _cast_as_param(np.random.random(1))
                self.p_1 = _cast_as_param(np.random.random(1) / 10)
                self.p_2 = _cast_as_param(np.random.random(1) / 10)
                self.p_3 = _cast_as_param(np.random.random(1) / 10)
                self.p_4 = _cast_as_param(np.random.random(1) / 10)
                self.A_1 = _cast_as_param(np.random.random(1))
                self.A_2 = _cast_as_param(np.random.random(1))
    
    def print_current_params(self):
        """Print the current parameters of the model."""
        param_names = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        for param in param_names:
            print(f"{param}: {getattr(self, param)}")
            
    def _init_existing_params(self, params):
        param_names = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        for param in param_names:
            setattr(self, param, _cast_as_param(params[param]))
        
    def get_Av(self, theta_l, theta_v):
        """ Calculate A_v (formula no. 7 in Broderick et al. (2022)) """
        theta_l = _cast_as_tensor(theta_l)
        theta_v = _cast_as_tensor(theta_v)
        Av = 1 + self.A_1 * torch.cos(2 * theta_l) + \
                self.A_2 * torch.cos(4 * theta_l) + \
                self.A_3 * torch.cos(2 * (theta_l - theta_v)) + \
                self.A_4 * torch.cos(4 * (theta_l - theta_v))
        return torch.clamp(Av, min=1e-6)

    def get_Pv(self, theta_l, theta_v, r_v):
        """ Calculate p_v (formula no. 6 in Broderick et al. (2022)) """
        theta_l = _cast_as_tensor(theta_l)
        theta_v = _cast_as_tensor(theta_v)
        r_v = _cast_as_tensor(r_v)
        ecc_dependency = self.slope * r_v + self.intercept
        Pv = ecc_dependency * (1 + self.p_1 * torch.cos(2 * theta_l) +
                                self.p_2 * torch.cos(4 * theta_l) +
                                self.p_3 * torch.cos(2 * (theta_l - theta_v)) +
                                self.p_4 * torch.cos(4 * (theta_l - theta_v)))
        return torch.clamp(Pv, min=1e-6)

    def forward(self, theta_l, theta_v, r_v, w_l, to_numpy=False):
        """ In the forward function we accept a Variable of input data and we must
        return a Variable of output data. Return predicted BOLD response
        in eccentricity (formula no. 5 in Broderick et al. (2022)) """
        w_l = _cast_as_tensor(w_l)
        Av = self.get_Av(theta_l, theta_v)
        Pv = self.get_Pv(theta_l, theta_v, r_v)
        pred = Av * torch.exp(-(torch.log2(w_l) + torch.log2(Pv)) ** 2 / (2 * self.sigma ** 2))
        if to_numpy:
            pred = pred.detach().numpy()
        return pred

def loss_fn(sigma_v_info, prediction, target):
    """"""
    norm_pred = normalize_pivotStyle(prediction)
    norm_measured = normalize_pivotStyle(target)
    loss_all_voxels = torch.mean((1/sigma_v_info) * (norm_pred - norm_measured)**2, dim=1)
    return loss_all_voxels

def fit_model(sfp_model, dataset, learning_rate=1e-4, max_epoch=1000, print_every=100, save_path=None,
              loss_all_voxels=False, anomaly_detection=False, amsgrad=False, eps=1e-8):
    """Fit the model. This function will allow you to run a for loop for N times set as max_epoch,
    and return the output of the training; loss history, model history."""
    torch.autograd.set_detect_anomaly(anomaly_detection)
    # [sigma, slope, intercept, p_1, p_2, p_3, p_4, A_1, A_2]
    my_parameters = [p for p in sfp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(my_parameters, lr=learning_rate, amsgrad=amsgrad, eps=eps)
    losses_history = []
    loss_history = []
    model_history = []
    start = timer()
    for t in range(max_epoch):
        pred = sfp_model.forward(theta_l=dataset.ori, theta_v=dataset.angle, r_v=dataset.eccen, w_l=dataset.sf)  # predictions should be put in here
        losses = loss_fn(dataset.sigma_v_squared, pred, dataset.target) # loss should be returned here
        loss = torch.mean(losses)
        if loss_all_voxels is True:
            losses_history.append(losses.detach().numpy())
        if (t + 1) % print_every == 0 or t == 0:
            print(f'**epoch no.{t} loss: {np.round(loss.item(), 10)}')
        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute gradients of all variables wrt loss
        optimizer.step()  # perform updates using calculated gradients
        model_values = [v.detach().numpy().item() for v in sfp_model.parameters() if v.requires_grad]  # output needs to be put in there
        loss_history.append(loss.item())
        model_history.append(model_values)  # more than one item here
    
    sfp_model.eval()
    elapsed_time = timer() - start
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(sfp_model.state_dict(), save_path)
    params_col = [name for name, param in sfp_model.named_parameters() if param.requires_grad]
    params_val = [param for name, param in sfp_model.named_parameters() if param.requires_grad]
    print(f'*epoch no.{max_epoch}: Finished! final model params...\n {dict(zip(params_col, np.round(model_values,2)))}') 
    print(f'Elapsed time: {np.round(elapsed_time, 2)} sec')

    voxel_list = dataset.voxel_info
    loss_history = pd.DataFrame(loss_history, columns=['loss']).reset_index().rename(columns={'index': 'epoch'})
    model_history = pd.DataFrame(model_history, columns=params_col).reset_index().rename(columns={'index': 'epoch'})
    if loss_all_voxels is True:
        losses_history = pd.DataFrame(np.asarray(losses_history), columns=voxel_list).reset_index().rename(columns={'index': 'epoch'})
    return loss_history, model_history, losses_history

def shape_losses_history(losses_history, syn_df):
    voxel_list = losses_history.drop(columns=['epoch']).columns.tolist()
    losses_history = pd.melt(losses_history, id_vars=['epoch'], value_vars=voxel_list, var_name='voxel', value_name='loss')
    losses_history = losses_history.merge(syn_df[['voxel','noise_SD','sigma_v_squared']], on=['voxel'])
    return losses_history

def melt_history_df(history_df):
    return pd.concat(history_df).reset_index().rename(columns={'level_0': 'subj', 'level_1': 'epoch'})


def add_param_type_column(model_history_df, params):
    id_cols = model_history_df.drop(columns=params).columns.tolist()
    df = pd.melt(model_history_df, id_vars=id_cols, value_vars=params,
                 var_name='params', value_name='value')
    return df

def group_params(df, params=['sigma', 'slope', 'intercept'], group=[1, 2, 2]):
    """Create a new column in df based on params values.
    Params and group should be 1-on-1 matched."""
    df = add_param_type_column(df, params)
    conditions = []
    for i in params:
        tmp = df['params'] == i
        conditions.append(tmp)
    df['group'] = np.select(conditions, group, default='other')
    return df



def load_history_df_subj(output_dir, dataset, stat, model_type, sn_list, lr_rate, max_epoch, df_type, roi):
    all_history_df = pd.DataFrame()
    subj_list = [utils.sub_number_to_string(x, dataset) for x in sn_list]
    for cur_model_type, cur_subj, cur_lr, cur_epoch, cur_roi in itertools.product(model_type, subj_list, lr_rate, max_epoch, roi):
        model_history_path = os.path.join(output_dir,
                                          f'{df_type}_history_dset-{dataset}_bts-{stat}_model-{cur_model_type}_{cur_subj}_lr-{cur_lr}_eph-{cur_epoch}_{cur_roi}.h5')
        tmp = pd.read_hdf(model_history_path)
        if {'lr_rate', 'max_epoch', 'model_type', 'subj'}.issubset(tmp.columns) is False:
            tmp['dset'] = dataset
            tmp['lr_rate'] = cur_lr
            tmp['max_epoch'] = cur_epoch
            tmp['model_type'] = cur_model_type
            tmp['subj'] = cur_subj
            tmp['vroinames'] = cur_roi
        all_history_df = pd.concat((all_history_df, tmp), axis=0, ignore_index=True)
    return all_history_df


def load_loss_and_model_history_Broderick_subj(output_dir, dataset, stat, model_type, sn_list, lr_rate, max_epoch, roi, losses=True):
    loss_history = load_history_df_subj(output_dir, dataset, stat, model_type, sn_list, lr_rate, max_epoch, "loss", roi)
    model_history = load_history_df_subj(output_dir, dataset, stat, model_type, sn_list, lr_rate, max_epoch, "model", roi)
    if losses is True:
        losses_history = load_history_df_subj(output_dir, dataset, stat, model_type, sn_list, lr_rate, max_epoch, "losses", roi)
    else:
        losses_history = []
    return loss_history, model_history, losses_history


def get_mean_and_error_for_each_param(df, err="sem", to_group=['params']):
    if 'params' not in df.columns:
        value_vars = ['sigma','slope','intercept','p_1','p_2','p_3','p_4','A_1','A_2']
        id_vars = [x for x in df.columns.to_list() if not x in value_vars]
        df = pd.melt(df, id_vars, value_vars, var_name='params', value_name='value')
    val_name = [col for col in df.columns if 'value' in col][0]
    m_df = df.groupby(to_group)[val_name].mean().reset_index().rename(columns={val_name: 'mean_value'})
    if err == "std":
        err_df = df.groupby(to_group)[val_name].std().reset_index().rename(columns={val_name: 'std_value'})
    elif err == "sem":
        err_df = df.groupby(to_group)[val_name].sem().reset_index().rename(columns={val_name: 'std_value'})
    return m_df.merge(err_df, on=to_group)


def load_SpatialFrequencyModel(pt_file_path):
    model = SpatialFrequencyModel()
    model.load_state_dict(torch.load(pt_file_path))
    model.eval()
    return model

def model_to_df(pt_file_path, *args):
    model = load_SpatialFrequencyModel(pt_file_path)
    model_dict = {}
    for name, param in model.named_parameters():
        model_dict[name] = param.detach().numpy()
    model_df = pd.DataFrame(model_dict)
    for arg in args:
        cur_pt_file = pt_file_path.split('/')[-1]
        model_df[utils.match_wildcards_with_col(arg)] = [k for k in cur_pt_file.split('_') if arg in k][0][len(arg)+1:].replace('-', ' ')
    return model_df

def load_all_models(pt_file_path_list, *args):
    model_df = pd.DataFrame({})
    for pt_file_path in pt_file_path_list:
        tmp = model_to_df(pt_file_path, *args)
        model_df = pd.concat((model_df, tmp), axis=0)
    return model_df