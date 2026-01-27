configfile:
    "config.json"
import os
import sys
sys.path.append(config['PYSURFER_DIR'])
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import pickle
pickle.HIGHEST_PROTOCOL = 4
from sfp_nsdsyn import *
from pysurfer.freeview_helper import retinotopy_colors


STIM_LIST=['annulus', 'pinwheel', 'forward-spiral', 'reverse-spiral'] #'avg'
ARGS_1D = ['sub', 'class', 'lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
ARGS_2D = ['lr','eph','sub','roi']
LR_1D = [0.005]
MAX_EPOCH_1D = [8000]
LR_2D = [0.0005]
MAX_EPOCH_2D = [30000]
MEASURED_NOISE_SD = np.round(0.03995, 2)  # unnormalized 1.502063
LR_RATE = [0.0005] #[0.0007]#np.linspace(5,9,5)*1e-4
MULTIPLES_OF_NOISE_SD = [1]
NOISE_SD = [np.round(MEASURED_NOISE_SD*x, 2) for x in [1]]
N_VOXEL = [100]
FULL_VER = ["True"]
PW = ["True"]
SN_LIST = ["{:02d}".format(sn) for sn in np.arange(1,9)]
broderick_sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
broderick_subj_list = [utils.sub_number_to_string(sn, dataset="broderick") for sn in broderick_sn_list]
ROIS = ["V1","V2","V3"]
PARAMS_2D = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
PARAMS_GROUP_2D = [0,1,1,2,2,3,3,4,4]
ROI_PAL = [(0.5490196078431373, 0.03137254901960784, 0.0),
           (0.07058823529411765, 0.44313725490196076, 0.10980392156862745),
           (0.0, 0.10980392156862745, 0.4980392156862745)]  #[sns.color_palette('dark', 10)[:][k] for k in [3,2,0]]

# small tests to make sure snakemake is playing nicely with the job management
# system.
rule test_run:
     log: os.path.expanduser('~/test_run-%j.log')
     run:
         import numpy
         print("success!", numpy)

rule test_shell:
     log: os.path.expanduser('~/test_shell-%j.log')
     shell:
         "python -c 'import numpy; print(numpy)'; echo success!"

def get_sn_list(dset):
    if dset == "broderick":
        sn_list = [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]
    elif dset == "nsdsyn":
        sn_list = np.arange(1,9)
    return sn_list

def make_subj_list(dset):
    if dset == "broderick":
        return [utils.sub_number_to_string(sn, dataset="broderick") for sn in get_sn_list(dset)]
    elif dset == "nsdsyn":
        return [utils.sub_number_to_string(sn, dataset="nsdsyn") for sn in get_sn_list(dset)]

def get_stim_size_in_degree(dset):
    if dset == 'nsdsyn':
        fixation_radius = vs.pix_to_deg(42.878)
        stim_radius = vs.pix_to_deg(714/2)
    else:
        fixation_radius = 1
        stim_radius = 12
    return fixation_radius, stim_radius

rule prep_nsdsyn_description_file:
    output:
        os.path.join(config['NSD_DIR'], 'nsdsyn_stim_description.csv')
    run:
        from sfp_nsdsyn import prep
        prep.load_stim_info_as_df(output[0], drop_phase=False, force_download=True)

rule prep_nsdsyn_data:
    input:
        design_mat = os.path.join(config['NSD_DIR'], 'nsddata', 'experiments', 'nsdsynthetic', 'nsdsynthetic_expdesign.mat'),
        stim_info = os.path.join(config['NSD_DIR'], 'nsdsyn_stim_description.csv'),
        lh_prfs = expand(os.path.join(config['NSD_DIR'], 'nsddata', 'freesurfer','{{subj}}', 'label', 'lh.prf{prf_param}.mgz'), prf_param= ["eccentricity", "angle", "size", "exponent", "R2"]),
        lh_rois = expand(os.path.join(config['NSD_DIR'], 'nsddata', 'freesurfer','{{subj}}', 'label', 'lh.prf-{roi_file}.mgz'), roi_file= ["visualrois", "eccrois"]),
        lh_betas = os.path.join(config['NSD_DIR'], 'nsddata_betas', 'ppdata', '{subj}', 'nativesurface', 'nsdsyntheticbetas_fithrf_GLMdenoise_RR', 'lh.betas_nsdsynthetic.hdf5'),
        rh_prfs= expand(os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{{subj}}','label','rh.prf{prf_param}.mgz'), prf_param=["eccentricity", "angle", "size", "exponent", "R2"]),
        rh_rois= expand(os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{{subj}}','label','rh.prf-{roi_file}.mgz'), roi_file=["visualrois", "eccrois"]),
        rh_betas= os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{subj}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','rh.betas_nsdsynthetic.hdf5')
    output:
        spiral_betas = os.path.join(config['OUTPUT_DIR'], 'dataframes', 'nsdsyn', 'model', 'dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}_tavg-{tavg}.csv'),
    params:
        rois_vals = lambda wildcards: [prep.convert_between_roi_num_and_vareas(wildcards.roi), [1,2,3,4,5]],
        task_keys = ['fixation', 'memory'],
        stim_size= get_stim_size_in_degree('nsdsyn')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs','dataframes','nsdsyn','model', 'dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}_tavg-{tavg}.log')
    run:
        from sfp_nsdsyn import prep
        from sfp_nsdsyn import vs
        lh_df = prep.make_sf_dataframe(stim_info=input.stim_info,
                                       design_mat=input.design_mat,
                                       rois=input.lh_rois, rois_vals=params.rois_vals,
                                       prfs=input.lh_prfs,
                                       betas=input.lh_betas,
                                       drop_phase=False, force_download=False,
                                       task_keys=params.task_keys, task_average=(wildcards.tavg=="True"),
                                       angle_to_radians=True)
        rh_df = prep.make_sf_dataframe(stim_info=input.stim_info,
                                       design_mat=input.design_mat,
                                       rois=input.rh_rois, rois_vals=params.rois_vals,
                                       prfs=input.rh_prfs,
                                       betas=input.rh_betas,
                                       drop_phase=False, force_download=False,
                                       task_keys=params.task_keys, task_average=(wildcards.tavg=="True"),
                                       angle_to_radians=True)
        sf_df = prep.concat_lh_rh_df(lh_df, rh_df)
        if wildcards.vs != 'None':
            sf_df = vs.select_voxels(sf_df, drop_by=wildcards.vs,
                                     inner_border=params.stim_size[0],
                                     outer_border=params.stim_size[1],
                                     eccentricity='eccentricity', size='size',
                                     to_group=['voxel'], return_voxel_list=False)
        sf_df['sub'] = wildcards.subj
        sf_df.to_csv(output[0], index=False)


rule prep_all_nsdsyn:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], 'dataframes', 'nsdsyn', 'model', 'dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}_tavg-{tavg}.csv'),
               subj=make_subj_list('nsdsyn'),roi=['V1', 'V2', 'V3'], vs='pRFcenter', tavg='True')


rule prep_broderick_data:
    input:
        stim_info = os.path.join(config['BRODERICK_DIR'], 'stimuli', 'task-sfprescaled_stim_description_after_correction.csv'),
        lh_eccentricity = os.path.join(config['BRODERICK_DIR'], 'derivatives', 'prf_solutions', 'sub-{subj}', 'bayesian_posterior', 'lh.inferred_eccen.mgz'),
        lh_angle = os.path.join(config['BRODERICK_DIR'], 'derivatives', 'prf_solutions', 'sub-{subj}', 'bayesian_posterior', 'lh.inferred_angle.mgz'),
        lh_size = os.path.join(config['BRODERICK_DIR'], 'derivatives', 'prf_solutions', 'sub-{subj}', 'bayesian_posterior', 'lh.inferred_sigma.mgz'),
        lh_rois = os.path.join(config['BRODERICK_DIR'], 'derivatives', 'prf_solutions', 'sub-{subj}', 'bayesian_posterior', 'lh.inferred_varea.mgz'),
        rh_eccentricity=os.path.join(config['BRODERICK_DIR'],'derivatives','prf_solutions','sub-{subj}','bayesian_posterior','rh.inferred_eccen.mgz'),
        rh_angle=os.path.join(config['BRODERICK_DIR'],'derivatives','prf_solutions','sub-{subj}','bayesian_posterior','rh.inferred_angle.mgz'),
        rh_size=os.path.join(config['BRODERICK_DIR'],'derivatives','prf_solutions','sub-{subj}','bayesian_posterior','rh.inferred_sigma.mgz'),
        rh_rois=os.path.join(config['BRODERICK_DIR'],'derivatives','prf_solutions','sub-{subj}','bayesian_posterior','rh.inferred_varea.mgz'),
        betas= os.path.join(config['BRODERICK_DIR'], 'derivatives', 'GLMdenoise', 'sub-{subj}_ses-04_task-sfprescaled_results.mat')
    output:
        spiral_betas = os.path.join(config['OUTPUT_DIR'], 'dataframes', 'broderick', 'model', 'dset-broderick_sub-{subj}_roi-{roi}_vs-{vs}_tavg-{tavg}.csv'),
    params:
        stim_size= get_stim_size_in_degree('broderick')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs','dataframes','broderick','model', 'dset-broderick_sub-{subj}_roi-{roi}_vs-{vs}_tavg-{tavg}.log')
    run:
        from sfp_nsdsyn.Broderick_dataset import _transform_angle_corrected #with the corrected stim_info, you need to use this function
        if wildcards.tavg == 'True':
            results_names = ['modelmd']
        elif wildcards.tavg == 'False':
            results_names = ['models']

        lh_prf_path_list = [input.lh_eccentricity, input.lh_angle, input.lh_size]
        rh_prf_path_list = [input.rh_eccentricity, input.rh_angle, input.rh_size]
        bd_df = brod.make_broderick_sf_dataframe(input.stim_info,
                                                 input.lh_rois, input.rh_rois,
                                                 input.lh_eccentricity, input.rh_eccentricity,
                                                 lh_prf_path_list, rh_prf_path_list,
                                                 input.betas,
                                                 transform_func=_transform_angle_corrected,
                                                 eccen_range=params.stim_size, roi=wildcards.roi,
                                                 angle_to_radians=True,
                                                 results_names=results_names)
        bd_df = vs.select_voxels(bd_df, drop_by=wildcards.vs,
                                 inner_border=params.stim_size[0],
                                 outer_border=params.stim_size[1],
                                 to_group=['voxel'],return_voxel_list=False)
        bd_df['sub'] = wildcards.subj
        bd_df['vroinames'] = wildcards.roi
        bd_df.to_csv(output[0], index=False)

rule make_precision_v_df:
    input:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','model', 'dset-{dset}_sub-{subj}_roi-{roi}_vs-pRFsize_tavg-False.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}', 'precision', 'precision-v_sub-{subj}_roi-{roi}_vs-pRFsize.csv')
    params:
        p_dict = {1: 'noise_SD', 2: 'sigma_v_squared'}
    run:
        subj_vs_df = pd.read_csv(input[0])
        power_val = [1,2]
        power_var = [col for p, col in params.p_dict.items() if p in power_val]
        sigma_v_df = bts.get_multiple_sigma_vs(subj_vs_df,
                                               power=power_val,
                                               columns=power_var,
                                               to_sd='betas', to_group=['sub','voxel', 'vroinames'])
        sigma_v_df.to_csv(output[0], index=False)

def define_rois_for_precision_s_df(dset):
    if dset == "nsdsyn":
        return ROIS
    else:
        return ['V1']

rule make_precision_s_df:
    input:
        precision_v = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],'dataframes','{{dset}}','precision', 'precision-v_sub-{subj}_roi-{roi}_vs-pRFsize.csv'), subj=make_subj_list(wildcards.dset), roi=define_rois_for_precision_s_df(wildcards.dset)),
    output:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision', 'precision-s_dset-{dset}_vs-pRFsize.csv')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs','dataframes','{dset}','precision', 'precision-s_dset-{dset}_vs-pRFsize.log')
    run:
        precision_v = utils.load_dataframes(input.precision_v)
        precision_s = precision_v.groupby(['sub','vroinames']).mean().reset_index()
        precision_s['precision'] = 1 / precision_s['sigma_v_squared']
        precision_s.to_csv(output[0], index=False)

def get_ecc_bin_list(wildcards):
    bin_list, bin_labels = tuning.get_bin_labels(wildcards.e1, wildcards.e2, wildcards.enum)
    return bin_list, bin_labels

rule binning_nsdsyn:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'model', 'dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv')
    output:
        os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'binned', 'e1-{e1}_e2-{e2}_nbin-{enum}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'dataframes', '{dset}', 'binned', 'e1-{e1}_e2-{e2}_nbin-{enum}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    params:
        bin_info = lambda wildcards: get_ecc_bin_list(wildcards)
    run:
        from sfp_nsdsyn import tuning
        df = pd.read_csv(input.subj_df)
        my_phase = int(wildcards.phase)

        df['ecc_bin'] = tuning.bin_ecc(df['eccentricity'], bin_list=params.bin_info[0], bin_labels=params.bin_info[1])
        c_df = tuning.summary_stat_for_ecc_bin(df,
                                               to_group= ['sub', 'ecc_bin', 'freq_lvl', 'names', 'vroinames'],
                                               to_bin=['betas', 'local_sf'],
                                               central_tendency='mean')
        c_df.to_csv(output[0], index=False)

def get_trained_model_for_all_bins(wildcards):
    only_num = int(wildcards.enum.replace('log', ''))
    BINS = [f'model-history_class-{wildcards.stim_class}_lr-{wildcards.lr}_eph-{wildcards.max_epoch}_e1-{wildcards.e1}_e2-{wildcards.e2}_nbin-{wildcards.enum}_dset-{wildcards.dset}_sub-{wildcards.subj}_roi-{wildcards.roi}_vs-{wildcards.vs}.pt' for bin in np.arange(1,only_num+1)]
    return [os.path.join(config['OUTPUT_DIR'], "sfp_model","results_1D", f'{wildcards.dset}', bin_path) for bin_path in BINS]

def normalize(group):
    min_val = group.min()
    max_val = group.max()
    return (group - min_val) / (max_val - min_val)

rule fit_tuning_curves:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'], 'dataframes', "{dset}", 'binned', 'e1-{e1}_e2-{e2}_nbin-{enum}_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'model-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'loss-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'model-params_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model", "results_1D", "{dset}", 'loss-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    resources:
        cpus_per_task = 1,
        mem_mb = 2000
    params:
        bin_info = lambda wildcards: get_ecc_bin_list(wildcards)
    run:
        subj_df = pd.read_csv(input.input_path)
        if wildcards.stim_class == "avg":
            subj_df = subj_df.groupby(['sub','ecc_bin','vroinames','freq_lvl']).mean().reset_index()
        else:
            save_stim_type_name = wildcards.stim_class.replace('-',' ')
            subj_df = subj_df.query('names == @save_stim_type_name')
        if wildcards.subj == "avg":
            beta_col = "normed_betas"
        else:
            beta_col = "betas"

        save_ecc_bin_name = params.bin_info[1][int(wildcards.curbin)]
        subj_df = subj_df.query('ecc_bin == @save_ecc_bin_name')
        my_model = tuning.LogGaussianTuningModel()
        my_dataset = tuning.LogGaussianTuningDataset(subj_df['local_sf'], subj_df[beta_col])
        loss_history, model_history = tuning.fit_tuning_curves(my_model, my_dataset,
                                                               learning_rate=float(wildcards.lr),
                                                               max_epoch=int(wildcards.max_epoch),
                                                               print_every=1000, save_path=output.model)
        model_history['ecc_bin'] = save_ecc_bin_name
        loss_history['ecc_bin'] = save_ecc_bin_name
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')

rule fit_tuning_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'model-params_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
        subj='avg', dset='nsdsyn',  roi=ROIS, vs='pRFcenter',curbin=np.arange(0,7),enum=7, e1=0.5, e2=4, stim_class='avg', lr=LR_1D, max_epoch=MAX_EPOCH_1D)

def _get_curbin(enum):
    return np.arange(0, int(enum.replace('log', '')))

def bin_to_plot(bins_to_plot):
   return [int(k) for k in bins_to_plot.split('-')]

rule plot_tuning_curves_NSD:
    input:
        model_df = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "nsdsyn",  'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_sub-{{subj}}_roi-{roi}_vs-{{vs}}.pt'), curbin=_get_curbin(wildcards.enum), roi=ROIS),
        binned_df = expand(os.path.join(config['OUTPUT_DIR'], 'dataframes', "nsdsyn", 'binned', 'e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_sub-{{subj}}_roi-{roi}_vs-{{vs}}.csv'), roi=ROIS)
    output: 
        'tclass-{stim_class}_normalize-{normalize}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{bins_to_plot}_dset-nsdsyn_sub-{subj}_roi-all_vs-{vs}.{fig_format}'
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "figures", "sfp_model","results_1D", "nsdsyn", 'tclass-{stim_class}_normalize-{normalize}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{bins_to_plot}_dset-nsdsyn_sub-{subj}_roi-all_vs-{vs}.{fig_format}.log')
    params:
        roi_list=ROIS,
        roi_pal=ROI_PAL
    run:
        bin_list, bin_labels = tuning.get_bin_labels(float(wildcards.e1), float(wildcards.e2), enum=wildcards.enum)
        final_params = tuning.load_all_models(input.model_df, *ARGS_1D)
        bin_df = utils.load_dataframes(input.binned_df, *['sub','roi'])
        if wildcards.stim_class == "avg":
            bin_df = bin_df.groupby(['sub', 'ecc_bin', 'vroinames', 'freq_lvl']).mean().reset_index()
        else:
            save_stim_type_name = wildcards.stim_class.replace('-',' ')
            bin_df = bin_df.query('names == @save_stim_type_name')
        bins_to_plot = [bin_labels[k] for k in bin_to_plot(wildcards.bins_to_plot)]
        vis1D.plot_tuning_curves_NSD(data_df=bin_df, subj=wildcards.subj,
                                     params_df=final_params,
                                     bins_to_plot= bins_to_plot,
                                     x='local_sf', y='betas',
                                     pal=params.roi_pal,
                                     normalize=(wildcards.normalize=="True"),
                                     save_path=output[0])

rule plot_preferred_period_1D_with_broderick_et_al:
    input:
        nsd_models = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "nsdsyn",  'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{{vs}}.pt'), curbin=_get_curbin(wildcards.enum), subj=make_subj_list('nsdsyn'), roi=ROIS),
        nsd_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','precision-s_dset-nsdsyn_vs-pRFsize.csv'),
        broderick_models = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "broderick", 'tfunc-uncorrected_model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-1_e2-12_nbin-11_curbin-{broderick_curbin}_dset-broderick_sub-{broderick_subj}_roi-V1_vs-{{vs}}.pt'), broderick_curbin=np.arange(0,10), broderick_subj=make_subj_list('broderick')),
        broderick_precision_s = os.path.join(config['OUTPUT_DIR'], 'dataframes','broderick','precision','precision-s_dset-broderick_vs-pRFsize.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures', "sfp_model","results_1D", "all",'pperiod_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_vs-{vs}.{fig_format}')
    params:
        roi_list=['Broderick et al. V1', 'NSD V1', 'NSD V2', 'NSD V3'],
        roi_pal=ROI_PAL
    run:
        args = ['sub', 'class','lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
        tuning_with_precision_df = pd.DataFrame({})
        fit_df = pd.DataFrame({})
        for precision_input, model_input in zip([input.nsd_precision_s, input.broderick_precision_s],
                                                [input.nsd_models, input.broderick_models]):
            precision_s = pd.read_csv(precision_input)
            params_df = tuning.load_all_models(model_input,*args)
            tmp_tuning_df = params_df.merge(precision_s[['sub', 'vroinames', 'precision']],on=['sub', 'vroinames'])
            tmp_tuning_df['pp'] = 1 / tmp_tuning_df['mode']
            if precision_input == input.nsd_precision_s:
                tmp_tuning_df['dset_type'] = tmp_tuning_df['vroinames'].apply(lambda x: f'NSD {x}')
            else:
                tmp_tuning_df['dset_type'] = 'Broderick et al. V1'
            tmp_fit_df = vis1D.fit_line_to_weighted_mean(tmp_tuning_df,'pp','precision',groupby=['vroinames', 'dset_type'])
            tuning_with_precision_df = pd.concat((tuning_with_precision_df, tmp_tuning_df), axis=0)
            fit_df = pd.concat((fit_df, tmp_fit_df), axis=0)
        tuning_with_precision_df, fit_df = vis1D.create_goal_columns(tuning_with_precision_df, fit_df)
        fit_df = vis1D.extend_NSD_line(fit_df)
        params.roi_pal.insert(0, (0.5,0.5,0.5))
        g = vis1D.plot_preferred_period(tuning_with_precision_df,fit_df=fit_df,
                                        row='goal',row_order=['replication', 'extension'],
                                        preferred_period='pp',precision='precision',hue='dset_type',
                                        hue_order=params.roi_list,lgd_title='Dataset & ROI',
                                        pal=params.roi_pal, height=2.5, width=3.4,
                                        suptitle='Preferred period',
                                        save_path=output[0])

rule plot_fwhm_1D_with_broderick_et_al:
    input:
        nsd_models = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "nsdsyn", 'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_sub-{subj}_roi-{roi}_vs-{{vs}}.pt'), curbin=_get_curbin(wildcards.enum), subj=make_subj_list('nsdsyn'), roi=ROIS),
        nsd_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','precision-s_dset-nsdsyn_vs-pRFsize.csv'),
        broderick_models = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "broderick", 'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-1_e2-12_nbin-11_curbin-{broderick_curbin}_dset-broderick_sub-{broderick_subj}_roi-V1_vs-{{vs}}.pt'), broderick_curbin=np.arange(0,10), broderick_subj=make_subj_list('broderick')),
        broderick_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','broderick','precision','precision-s_dset-broderick_vs-pRFsize.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_1D","all", 'fwhm_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_vs-{vs}.{fig_format}')
    params:
        roi_list=['Broderick et al. V1', 'NSD V1', 'NSD V2', 'NSD V3'],
        roi_pal=ROI_PAL
    run:
        args = ['sub', 'class', 'lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
        tuning_with_precision_df = pd.DataFrame({})
        fit_bandwidth_df = pd.DataFrame({})
        for precision_input, model_input in zip([input.nsd_precision_s, input.broderick_precision_s],
                                                [input.nsd_models, input.broderick_models]):
            precision_s = pd.read_csv(precision_input)
            params_df = tuning.load_all_models(model_input,*args)
            tmp_tuning_df = params_df.merge(precision_s[['sub', 'vroinames', 'precision']],on=['sub', 'vroinames'])
            if precision_input == input.nsd_precision_s:
                tmp_tuning_df['dset_type'] = tmp_tuning_df['vroinames'].apply(lambda x: f'NSD {x}')
            else:
                tmp_tuning_df['dset_type'] = 'Broderick et al. V1'
            tmp_tuning_df['fwhm'] = tmp_tuning_df['sigma'] * 2.335  #fwhm
            tmp_fit_bandwidth_df = vis1D.fit_line_to_weighted_mean(tmp_tuning_df,'fwhm','precision',groupby=['vroinames','dset_type'])
            tuning_with_precision_df = pd.concat((tuning_with_precision_df, tmp_tuning_df), axis=0)
            fit_bandwidth_df = pd.concat((fit_bandwidth_df, tmp_fit_bandwidth_df), axis=0)

        tuning_with_precision_df, fit_bandwidth_df = vis1D.create_goal_columns(tuning_with_precision_df, fit_bandwidth_df)
        #fit_bandwidth_df = vis1D.extend_NSD_line(fit_bandwidth_df)
        params.roi_pal.insert(0, (0.5,0.5,0.5))
        vis1D.plot_bandwidth_in_octaves(tuning_with_precision_df,
                                        fit_df=None, bandwidth='fwhm',precision='precision',
                                        row='goal', row_order=['replication','extension'],
                                        hue='dset_type', hue_order=params.roi_list, height=2.5, width=3.4,
                                        pal=params.roi_pal,lgd_title='ROIs', suptitle='Bandwidth',
                                        errorbar=("ci", 68), save_path=output[0])

rule shuffle_df:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'model', 'dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv'),
        precision = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}', 'precision', 'precision-v_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        shuffled_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'perm', 'perm-{perm}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_precision_merged.csv'),
    run:
        from sfp_nsdsyn.bootstrapping import shuffle_class_idx
        np.random.seed(int(wildcards.perm))
        subj_df = pd.read_csv(input.subj_df)
        precision_df = pd.read_csv(input.precision)
        df = subj_df.merge(precision_df, on=['sub', 'vroinames', 'voxel'])
        df = df.groupby(['sub','voxel','class_idx','vroinames']).mean().reset_index()
        df = shuffle_class_idx(df, groupby_cols=['sub','voxel'], to_shuffle=['betas'], same_perm=True)
        df.to_csv(output.shuffled_df, index=False)

rule run_model_shuffled:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'perm', 'perm-{perm}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_precision_merged.csv')
    output:
        model_history = os.path.join(config['OUTPUT_DIR'],  "sfp_model", "results_2D", "{dset}", 'perm', 'perm-{perm}_model-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}",'perm', 'perm-{perm}_loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{dset}", 'perm','perm-{perm}_model-params_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model","results_2D", "{dset}",'perm','perm-{perm}_loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.log'),
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "sfp_model","results_2D", "{dset}",'perm','perm-{perm}_loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.txt'),
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        subj_df = pd.read_csv(input.subj_df)
        subj_model = model.SpatialFrequencyModel()
        subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
        loss_history, model_history, _ = model.fit_model(subj_model, subj_dataset,
                                                         learning_rate=float(wildcards.lr),
                                                         max_epoch=int(wildcards.max_epoch),
                                                         save_path=output.model,
                                                         print_every=10000,
                                                         loss_all_voxels=False,
                                                         anomaly_detection=False,
                                                         amsgrad=False,
                                                         eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')    


rule run_model_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}", 'perm', 'perm-{perm}_model-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
               dset='nsdsyn', lr=LR_2D, max_epoch=MAX_EPOCH_2D, subj=make_subj_list('nsdsyn')[:1], roi=['V1'], vs='pRFsize', perm=np.arange(0,100))
 

rule run_shuffle_model_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}", 'perm', 'perm-{perm}_model-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
               dset='nsdsyn', lr=LR_2D, max_epoch=MAX_EPOCH_2D, subj=make_subj_list('nsdsyn')[1:], roi=['V1'], vs='pRFsize', perm=np.arange(0,100))
 
rule run_model:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'model', 'dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv'),
        precision = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}', 'precision', 'precision-v_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}", 'model-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}", 'loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{dset}", 'model-params_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model","results_2D", "{dset}",'loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.log'),
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "sfp_model","results_2D", "{dset}",'loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.txt'),
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        subj_df = pd.read_csv(input.subj_df)
        precision_df = pd.read_csv(input.precision)
        df = subj_df.merge(precision_df, on=['sub', 'vroinames', 'voxel'])
        df = df.groupby(['sub','voxel','class_idx','vroinames']).mean().reset_index()
        subj_model = model.SpatialFrequencyModel()
        subj_dataset = model.SpatialFrequencyDataset(df, beta_col='betas')
        loss_history, model_history, _ = model.fit_model(subj_model, subj_dataset,
                                                         learning_rate=float(wildcards.lr),
                                                         max_epoch=int(wildcards.max_epoch),
                                                         save_path=output.model,
                                                         print_every=10000,
                                                         loss_all_voxels=False,
                                                         anomaly_detection=False,
                                                         amsgrad=False,
                                                         eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')

rule predict_Pv_based_on_model:
    input:
        stim = os.path.join(config['NSD_DIR'], 'nsdsyn_stim_description.csv'),
        model = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D", "{dset}", 'model-params_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-pRFsize.pt'),
    output:
        os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","{dset}",'sfstimuli-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-pRFsize.h5')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs', "sfp_model","prediction_2D","{dset}",'sfstimuli-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-pRFsize.log')
    run:
        stim_info = vis2D.get_w_a_and_w_r_for_each_stim_class(input.stim)
        final_params = model.model_to_df(input.model, *ARGS_2D)
        syn_df = vis2D.calculate_preferred_period_for_synthetic_df(stim_info,final_params,
                                                                   ecc_range=(float(wildcards.ecc1), float(wildcards.ecc2)),
                                                                   n_ecc=int(wildcards.n_ecc),
                                                                   angle_range=(np.deg2rad(float(wildcards.ang1)), np.deg2rad(float(wildcards.ang2))),
                                                                   n_angle=int(wildcards.n_ang),
                                                                   ecc_col='eccentricity',angle_col='angle',
                                                                   angle_in_radians=True, sfstimuli=wildcards.frame)
        syn_df.to_hdf(output[0], key='stage', mode='w')

rule pv_again:
    input:
        a = expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","{dset}", 'sfstimuli-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-pRFsize.h5'),
                    dset=['nsdsyn'], frame=['scaled','constant'], ecc1=[0], ecc2=[12], n_ecc=[121], ang1=[0], ang2=[360], n_ang=[361], lr=LR_2D, max_epoch=MAX_EPOCH_2D, subj=make_subj_list('nsdsyn'), roi=ROIS),
        # b = expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","{dset}", 'sfstimuli-{frame}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-pRFsize.h5'),
        #             dset=['broderick'], frame=['scaled','constant'], ecc1=[0], ecc2=[12], n_ecc=[121], ang1=[0], ang2=[360], n_ang=[361], lr=LR_2D, max_epoch=MAX_EPOCH_2D, subj=make_subj_list('broderick'), roi=['V1'])

rule get_summary_table:
    input:
        nsd_params = expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "nsdsyn",'model-params_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-{roi}_vs-pRFsize.pt'), subj=make_subj_list('nsdsyn'), roi=ROIS),
        nsd_precision = os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','precision-s_dset-nsdsyn_vs-pRFsize.csv'),
        broderick_params = expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "broderick",'model-params_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-V1_vs-pRFsize.pt'), subj=make_subj_list('broderick')),
        broderick_precision = os.path.join(config['OUTPUT_DIR'],'dataframes','broderick','precision','precision-s_dset-broderick_vs-pRFsize.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'sfp_model','results_2D','mean_parameter_estimates_with_ci.csv'),
    run:
        nsd_params = model.load_all_models(input.nsd_params, *ARGS_2D)
        nsd_precision = pd.read_csv(input.nsd_precision)
        nsd_df = pd.merge(nsd_params, nsd_precision[['sub','vroinames','precision']], on=['sub','vroinames'])
        nsd_df['dset_type'] = nsd_df['vroinames'].apply(lambda x: f'NSD {x}')
        broderick_params = model.load_all_models(input.broderick_params, *ARGS_2D)
        broderick_precision = pd.read_csv(input.broderick_precision)
        broderick_df = pd.merge(broderick_params, broderick_precision[['sub','vroinames','precision']], on=['sub','vroinames'])
        broderick_df['dset_type'] = 'Broderick et al. V1'
        final_params = pd.concat((nsd_df, broderick_df), axis=0)
        dset_types = ['Broderick et al. V1', 'NSD V1', 'NSD V2', 'NSD V3']
        all_dset_df_mean = utils.calculate_weighted_mean(final_params, PARAMS_2D, 'precision', groupby=['dset_type'], n_decimals=3)
        all_dset_df_68 = utils.calculate_confidence_intervals(final_params, PARAMS_2D, dset_types, ci_quantiles=[0.16, 0.84], n_boot=5000, n_decimals=2, seed=None)
        all_dset_df_95 = utils.calculate_confidence_intervals(final_params, PARAMS_2D, dset_types, ci_quantiles=[0.025, 0.975], n_boot=5000, n_decimals=2, seed=None)
        # Combine all_dset_df_mean, all_dset_df_68, and all_dset_df_95
        combined_df = all_dset_df_mean.copy()
        # Iterate over each dataset type and parameter
        for dset_type in dset_types:
            for param in PARAMS_2D:
                # Get mean, ci_68, and ci_95 for this param and dset_type
                mean_val = combined_df.loc[combined_df['dset_type'] == dset_type, param].values[0]
                ci_68_val = all_dset_df_68.loc[all_dset_df_68['dset_type'] == dset_type, f"{param}"].values[0]
                ci_95_val = all_dset_df_95.loc[all_dset_df_95['dset_type'] == dset_type, f"{param}"].values[0]
                # Format: 'mean, [ci_68], [ci_95]'
                combined_df.loc[combined_df['dset_type'] == dset_type, param] = f"{mean_val}, {ci_68_val}, {ci_95_val}"

        combined_df = combined_df.set_index('dset_type').transpose().reset_index().rename_axis(None, axis=1).rename(columns={'index': 'param'})
        combined_df.to_csv(output[0], index=False)
        
rule run_cross_validation:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','model','dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv'),
        precision = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}', 'precision', 'precision-v_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'sfp_model','cross_validation','{dset}','model-{model_type}','cvresults_bts-False_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
        os.path.join(config['OUTPUT_DIR'],'sfp_model','cross_validation','{dset}','model-{model_type}','cvlosses_bts-False_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.npy')
    run:
        df = pd.read_csv(input.subj_df)
        precision_df = pd.read_csv(input.precision)
        df = df.merge(precision_df, on=['sub', 'vroinames', 'voxel'])
        df = df.groupby(['sub','voxel','class_idx','vroinames']).mean().reset_index()
        sfp_model = model.SpatialFrequencyModel(model=int(wildcards.model_type))
        cv_results, cv_losses = cv2d.run_cross_validation(df, 
                                                          sfp_model, 
                                                          n_folds=7, 
                                                          n_test_classes=4, 
                                                          learning_rate=float(wildcards.lr), 
                                                          max_epoch=int(wildcards.max_epoch), 
                                                          print_every=10000, 
                                                          random_state=42)
        cv_results.to_csv(output[0], index=False)
        np.save(output[1], cv_losses.detach().numpy())

rule cvresults_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],'sfp_model','cross_validation','{dset}','model-{model_type}','cvlosses_bts-False_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.npy'),
               dset='nsdsyn', model_type=[1,2,3,4,5,6,7], subj=make_subj_list('nsdsyn'), roi=['V1'], vs='pRFsize', lr=LR_2D, max_epoch=MAX_EPOCH_2D)

rule bootstrap_data:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','model','dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tavg-False.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','bootstraps', 'bootstrap-{bts}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs', 'dataframes','{dset}','bootstraps','bootstrap-{bts}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    run:
        import sfp_nsdsyn.bootstrapping as bts
        subj_df = pd.read_csv(input.subj_df)
        if int(wildcards.bts) % 5 == 0:
            print(f'Bootstrap {int(wildcards.bts)} started!')
        sample_df = bts.sample_run_and_average(subj_df, 
                                            class_idx=range(28), 
                                            sample_size=8, 
                                            replace=True, 
                                            to_group=['class_idx','voxel','run','vroinames','sub','hemi','names'])
        sample_df['bootstrap'] = int(wildcards.bts)
        sample_df.to_csv(output[0], index=False)

rule all_bootstraps:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','bootstraps', 'bootstrap-{bts}_dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
               subj=make_subj_list('nsdsyn'), bts=np.arange(10,100), roi=['V1'], vs='pRFsize')
        

rule run_model_for_bootstraps:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','bootstraps', 'bootstrap-{bts}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
        precision = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision', 'precision-v_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}", 'bootstraps', 'bootstrap-{bts}_model-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{dset}",'bootstraps', 'bootstrap-{bts}_loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{dset}", 'bootstraps', 'bootstrap-{bts}_model-params_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model","results_2D", "{dset}",'bootstraps', 'bootstrap-{bts}_loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.log'),
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "sfp_model","results_2D", "{dset}",'bootstraps', 'bootstrap-{bts}_loss-history_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.txt'),
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        subj_df = pd.read_csv(input.subj_df)
        precision_df = pd.read_csv(input.precision)
        df = subj_df.merge(precision_df,on=['sub', 'vroinames', 'voxel'])
        df = df.groupby(['sub', 'voxel', 'class_idx', 'names','vroinames']).mean().reset_index()
        subj_model = model.SpatialFrequencyModel()
        subj_dataset = model.SpatialFrequencyDataset(df, beta_col='betas')
        loss_history, model_history, _ = model.fit_model(subj_model,subj_dataset,
                                                        learning_rate=float(wildcards.lr),
                                                        max_epoch=int(wildcards.max_epoch),
                                                        save_path=output.model,
                                                        print_every=10000,
                                                        loss_all_voxels=False,
                                                        anomaly_detection=False,
                                                        amsgrad=False,
                                                        eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')                                                


rule  fit_to_bootstraps:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{dset}", 'bootstraps', 'bootstrap-{bts}_model-params_lr-{lr}_eph-{max_epoch}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
               bts=np.arange(0,1000), lr=LR_2D, max_epoch=MAX_EPOCH_2D, dset='nsdsyn', subj=make_subj_list('nsdsyn'), roi=ROIS, vs='pRFsize')

rule plot_avg_model_parameters:
    input:
        nsd_model_params = expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "nsdsyn",'{{stimtest}}', 'model-params_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-{roi}_vs-pRFsize.pt'), subj=make_subj_list('nsdsyn'), roi=ROIS),
        nsd_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','{stimtest}','precision-s_dset-nsdsyn_vs-pRFsize.csv'),
        broderick_model_params= expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "broderick",'{{stimtest}}', 'model-params_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-V1_vs-pRFsize.pt'),  subj=make_subj_list('broderick')),
        broderick_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','broderick','precision','{stimtest}','precision-s_dset-broderick_vs-pRFsize.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D", '{stimtest}', "{goal}", 'fig-params_lr-{lr}_eph-{max_epoch}_sub-avg.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs',"figures", "sfp_model","results_2D", '{stimtest}', "{goal}", 'fig-params_lr-{lr}_eph-{max_epoch}_sub-avg_format-{fig_format}.log')
    run:

        final_df = pd.DataFrame({})
        for dset, model_df, precision_df in zip(['nsdsyn', 'broderick'], [input.nsd_model_params, input.broderick_model_params], [input.nsd_precision_s, input.broderick_precision_s]):
            params = model.load_all_models(model_df, *ARGS_2D)
            precision_s = pd.read_csv(precision_df)
            tmp_df = params.merge(precision_s[['sub','vroinames','precision']], on=['sub','vroinames'])
            tmp_df['dset'] = dset
            final_df = pd.concat((final_df, tmp_df), axis=0)
        final_df['dset_type'] = final_df.apply(lambda row: f"NSD {row['vroinames']}"
                                               if row['dset'] == 'nsdsyn'
                                               else 'Broderick et al. V1', axis=1)
        final_df['goal'] = final_df['vroinames'].apply(lambda x: 'replication' if x == "V1" else 'extension')
        tmp = final_df.query('"NSD V1" in dset_type')
        tmp['goal'] = 'extension'

        final_df = pd.concat((tmp, final_df),axis=0)
        params_list = [['sigma'], ['slope', 'intercept'], ['p_1', 'p_2'], ['A_1', 'A_2'], ['p_3', 'p_4']]
        tmp, hue_order, pal = vis2D.filter_for_goal(final_df, wildcards.goal)
        ylim_list, ytick_list = vis2D.get_y_values_for_summary_fig(wildcards.goal)
        vis2D.make_param_summary_fig(tmp,'dset_type', hue_order=hue_order,
                                     pal=pal, params_list=params_list,
                                     ylim_list=ylim_list, yticks_list=ytick_list,
                                     width_ratios=(0.8, 2, 1.5, 1.5, 1.5), fig_size=(8.6, 1.5),
                                     dot_scale=0.9, errwidth=2,
                                     save_path=output[0])

rule plot_replication_prediction_figures:
    input:
        nsd_model_params=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D","nsdsyn",'{{stimtest}}','model-params_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-V1_vs-pRFsize.pt'), subj=make_subj_list('nsdsyn')),
        nsd_precision_s=os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','{stimtest}','precision-s_dset-nsdsyn_vs-pRFsize.csv'),
        nsd_prediction=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","nsdsyn",'{{stimtest}}', 'sfstimuli-{{frame}}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-V1_vs-pRFsize.h5'), subj=make_subj_list('nsdsyn')),
        broderick_model_params=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D","broderick",'{{stimtest}}','model-params_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-V1_vs-pRFsize.pt'),subj=make_subj_list('broderick')),
        broderick_precision_s=os.path.join(config['OUTPUT_DIR'],'dataframes','broderick','precision','{stimtest}','precision-s_dset-broderick_vs-pRFsize.csv'),
        broderick_prediction=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","broderick",'{{stimtest}}', 'sfstimuli-{{frame}}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-{{lr}}_eph-{{max_epoch}}_sub-{subj}_roi-V1_vs-pRFsize.h5'), subj=make_subj_list('broderick'))
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","prediction_2D","all", 'replication', '{stimtest}','param-{param}_n_lr-{lr}_eph-{max_epoch}_sub-avg.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'],'logs','figures',"sfp_model","prediction_2D","all",'replication','{stimtest}','param-{param}_n_lr-{lr}_eph-{max_epoch}_sub-avg_{fig_format}.log')
    params:
        local_ori=np.deg2rad(90),
        ecc=5,
        angle=np.deg2rad(180)
    run:
        nsd_df = vis2D.merge_model_and_precision(input.nsd_model_params,input.nsd_precision_s, *ARGS_2D)
        broderick_df = vis2D.merge_model_and_precision(input.broderick_model_params,input.broderick_precision_s,*ARGS_2D)
        nsd_df['dset_type'] = 'NSD V1'
        broderick_df['dset_type'] = 'Broderick et al. V1'
        final_params = pd.concat((nsd_df, broderick_df), axis=0)
        final_params['goal'] = 'replication'

        tmp, hue_order, pal = vis2D.filter_for_goal(final_params, 'replication')
        weighted_mean_df = vis2D.get_weighted_average_of_params(tmp, ['dset_type'], params.ecc, params.angle, params.local_ori)
        my_param = vis2D.get_params(wildcards.param)
        param_ylim, param_yticks = vis2D.get_param_y_values(wildcards.param, 'replication')
        pred_ylim, pred_yticks = vis2D.get_prediction_y_values(wildcards.param, 'replication')

        vis2D.plot_param_and_prediction(params_df=tmp,
                                        params=my_param,
                                        hue='dset_type',
                                        hue_order=hue_order,
                                        prediction_df=weighted_mean_df,
                                        prediction_y=None,
                                        pal=pal,
                                        params_ylim=param_ylim,
                                        params_yticks=param_yticks,
                                        prediction_ylim=pred_ylim,
                                        prediction_yticks=pred_yticks,
                                        title='Bandwidth of V1 spatial frequency tuning curves',
                                        prediction_ylabel='Preferred period (deg)',
                                        figsize=(3.5, 1.8),
                                        width_ratios=[1.5, 4],
                                        save_path=output[0])



rule results_2D:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D", 'corrected', "{goal}", 'fig1-params_lr-{lr}_eph-{max_epoch}_sub-avg.{fig_format}'),
               goal=['replication','extension'], lr=LR_2D, max_epoch=MAX_EPOCH_2D, fig_format=['svg'])

rule binning_broderick:
    input:
        subj_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}','dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}_tfunc-{tfunc}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'], 'dataframes', '{dset}', 'binned', 'tfunc-{tfunc}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    log:
        os.path.join(config['OUTPUT_DIR'], 'logs', 'dataframes', '{dset}', 'binned', 'tfunc-{tfunc}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    params:
        bin_info = lambda wildcards: get_ecc_bin_list(wildcards)
    run:
        from sfp_nsdsyn import tuning
        df = pd.read_csv(input.subj_df)
        df = df.query('~names.str.contains("mixture").values')
        df['ecc_bin'] = tuning.bin_ecc(df['eccentricity'], bin_list=params.bin_info[0], bin_labels=params.bin_info[1])
        c_df = tuning.summary_stat_for_ecc_bin(df,
                                               to_group= ['sub', 'ecc_bin', 'freq_lvl', 'names', 'vroinames'],
                                               to_bin=['betas', 'local_sf'],
                                               central_tendency='mean')
        c_df.to_csv(output[0], index=False)



rule fit_tuning_curves_broderick:
    input:
        input_path = os.path.join(config['OUTPUT_DIR'], 'dataframes', "{dset}", 'binned', 'tfunc-{tfunc}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv'),
    output:
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'normalize-{normalize}_tfunc-{tfunc}_model-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'normalize-{normalize}_tfunc-{tfunc}_loss-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.h5'),
        model = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{dset}", 'normalize-{normalize}_tfunc-{tfunc}_model-params_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.pt'),
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model", "results_1D", "{dset}", 'normalize-{normalize}_tfunc-{tfunc}_loss-history_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{curbin}_set-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.log')
    resources:
        cpus_per_task = 1,
        mem_mb = 2000
    params:
        bin_info = lambda wildcards: get_ecc_bin_list(wildcards)
    run:


        subj_df = pd.read_csv(input.input_path)
        if wildcards.stim_class == "avg":
            subj_df = subj_df.groupby(['sub','ecc_bin','vroinames','freq_lvl']).mean().reset_index()
            if wildcards.normalize == "True":
                # Group by the specified columns and apply the normalization to 'betas'
                subj_df['betas'] = subj_df.groupby(['sub', 'ecc_bin', 'vroinames'])[
                    'betas'].transform(normalize)
        else:
            save_stim_type_name = wildcards.stim_class.replace('-',' ')
            subj_df = subj_df.query('names == @save_stim_type_name')

        save_ecc_bin_name = params.bin_info[1][int(wildcards.curbin)]
        subj_df = subj_df.query('ecc_bin == @save_ecc_bin_name')
        model_path_list = get_trained_model_for_all_bins(wildcards)
        my_model = tuning.LogGaussianTuningModel()
        my_dataset = tuning.LogGaussianTuningDataset(subj_df['local_sf'], subj_df['betas'])
        loss_history, model_history = tuning.fit_tuning_curves(my_model, my_dataset,
                                                               learning_rate=float(wildcards.lr),
                                                               max_epoch=int(wildcards.max_epoch),
                                                               print_every=2000, save_path=output.model)
        model_history['ecc_bin'] = save_ecc_bin_name
        loss_history['ecc_bin'] = save_ecc_bin_name
        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')


rule plot_tuning_curves_broderick:
    input:
        model_df = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{{dset}}", 'tfunc-{{tfunc}}_model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_dset-{{dset}}_sub-{{subj}}_roi-{{roi}}_vs-{{vs}}.pt'), curbin=_get_curbin(wildcards.enum)),
        binned_df = os.path.join(config['OUTPUT_DIR'], 'dataframes', "{dset}", 'binned', 'tfunc-{tfunc}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],"figures", "sfp_model","results_1D", "{dset}", 'tfunc-{tfunc}_tuning_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{bins_to_plot}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "figures", "sfp_model","results_1D", "{dset}", 'tfunc-{tfunc}_tuning_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_curbin-{bins_to_plot}_dset-{dset}_sub-{subj}_roi-{roi}_vs-{vs}.{fig_format}.log')
    params:
        roi_pal=ROI_PAL
    run:
        bin_list, bin_labels = tuning.get_bin_labels(float(wildcards.e1), float(wildcards.e2), enum=int(wildcards.enum))
        final_params = tuning.load_all_models(input.model_df, *ARGS_1D)
        bin_df = utils.load_dataframes([input.binned_df], *['sub','dset','roi'])
        if wildcards.stim_class == "avg":
            bin_df = bin_df.groupby(['sub', 'ecc_bin', 'vroinames', 'freq_lvl']).mean().reset_index()
        else:
            save_stim_type_name = wildcards.stim_class.replace('-',' ')
            bin_df = bin_df.query('names == @save_stim_type_name')
        bins_to_plot = [bin_labels[k] for k in bin_to_plot(wildcards.bins_to_plot)]
        vis1D.plot_sf_curves_only_for_V1(df=bin_df.query('ecc_bin in @bins_to_plot'),
                             params_df=final_params.query('ecc_bin in @bins_to_plot'),
                             x='local_sf', y='betas', hue='ecc_bin', hue_order=bins_to_plot,
                             col_title=wildcards.roi, lgd_title='Eccentricity band',
                             palette=params.roi_pal[0],
                             save_path=output[0])

rule plot_preferred_period_1D:
    input:
        models = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{{dset}}", 'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_dset-{{dset}}_sub-{subj}_roi-{roi}_vs-{{vs}}.pt'), curbin=_get_curbin(wildcards.enum), subj=make_subj_list(wildcards.dset), roi=ROIS),
        precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-pRFsize.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures', "sfp_model","results_1D", "{dset}",'pperiod_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_vs-{vs}.{fig_format}')
    params:
        roi_list=ROIS,
        roi_pal=ROI_PAL
    run:
        args = ['sub', 'class','dset', 'lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
        precision_s = pd.read_csv(input.precision_s)
        params_df= tuning.load_all_models(input.models, *args)
        tuning_with_precision_df = params_df.merge(precision_s[['sub', 'vroinames', 'precision']],on=['sub', 'vroinames'])
        tuning_with_precision_df['pp'] = 1 / tuning_with_precision_df['mode']

        fit_df = vis1D.fit_line_to_weighted_mean(tuning_with_precision_df,'pp','precision',groupby=['vroinames'])

        g = vis1D.plot_preferred_period(tuning_with_precision_df,preferred_period='pp',precision='precision',hue='vroinames',hue_order=params.roi_list,fit_df=fit_df,pal=params.roi_pal,lgd_title='ROI',height=4,width=3,save_path=
        output[0])

rule plot_full_width_half_maximum_1D:
    input:
        models=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_1D","{{dset}}",'model-params_class-{{stim_class}}_lr-{{lr}}_eph-{{max_epoch}}_e1-{{e1}}_e2-{{e2}}_nbin-{{enum}}_curbin-{curbin}_dset-{{dset}}_sub-{subj}_roi-{roi}_vs-{{vs}}.pt'),curbin=_get_curbin(wildcards.enum),subj=make_subj_list('nsdsyn'),roi=ROIS),
        precision_s=os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-pRFsize.csv'),
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_1D","{dset}", "fwhm_class-{stim_class}_lr-{lr}_eph-{max_epoch}_e1-{e1}_e2-{e2}_nbin-{enum}_dset-{dset}_vs-{vs}.{fig_format}")

    params:
        roi_list=ROIS,
        roi_pal=ROI_PAL
    run:
        args = ['sub', 'class', 'dset', 'lr', 'eph', 'roi', 'e1', 'e2', 'nbin', 'curbin']
        precision_s = pd.read_csv(input.precision_s)
        params_df = tuning.load_all_models(input.models,*args)
        tuning_with_precision_df = params_df.merge(precision_s[['sub', 'vroinames', 'precision']],on=['sub', 'vroinames'])
        tuning_with_precision_df['fwhm'] = tuning_with_precision_df['sigma']*2.335 #fwhm
        fit_bandwidth_df = vis1D.fit_line_to_weighted_mean(tuning_with_precision_df, 'fwhm', 'precision', groupby=['vroinames'])
        vis1D.plot_bandwidth_in_octaves(tuning_with_precision_df,bandwidth='fwhm',precision='precision',hue='vroinames',hue_order=params.roi_list,fit_df=fit_bandwidth_df,pal=params.roi_pal,lgd_title='ROIs',row=None,row_order=None,suptitle=None,height=4,width=3,errorbar=(
            "ci", 68),save_path=output[0])

rule plot_2D_model_loss_history_broderick_et_al:
    input:
        loss_history = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{{dset}}",'tfunc-{{tfunc}}_loss-history_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list(wildcards.dset))
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "{dset}",'tfunc-{tfunc}_loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_roi-{roi}_vs-{vs}.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "figures", "sfp_model", "results_2D", "{dset}",'tfunc-{tfunc}_loss-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_roi-{roi}_vs-{vs}.{fig_format}.log')
    run:
        loss_df = pd.DataFrame({})

        for l_file in input.loss_history:
            tmp_df = utils.load_dataframes([l_file], *ARGS_2D)
            tmp_df = tmp_df[tmp_df.epoch % 100 == 0]
            loss_df = pd.concat((loss_df, tmp_df), axis=0)
        subj_list = make_subj_list(wildcards.dset)
        kwargs = {'palette': utils.new_subj_colors(subj_list)}
        vis.plot_loss_history(loss_df,
                              hue='sub',
                              lgd_title='Subjects',
                              hue_order=subj_list,
                              log_y=True,
                              save_path=output[0],
                              **kwargs)

rule plot_2D_model_param_history_broderick_et_al:
    input:
        param_history = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "{{dset}}", 'tfunc-{{tfunc}}_param-history_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list(wildcards.dset))
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "{dset}",'tfunc-{tfunc}_param-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_roi-{roi}_vs-{vs}.{fig_format}')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "figures", "sfp_model", "results_2D", "{dset}",'tfunc-{tfunc}_param-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        params_list = PARAMS_2D,
    run:
        params_df = pd.DataFrame({})
        for m_file in input.param_history:
            tmp_df = utils.load_dataframes([m_file],*ARGS_2D)
            tmp_df = tmp_df[tmp_df.epoch % 100 == 0]
            params_df = pd.concat((params_df, tmp_df), axis=0)
        subj_list = make_subj_list(wildcards.dset)
        kwargs = {'palette': utils.new_subj_colors(subj_list)}
        vis.plot_param_history(params_df, params=PARAMS_2D,
                                hue='sub',lgd_title='Subjects',hue_order=subj_list, save_path=output[0], **kwargs)


rule plot_precision_weighted_avg_2D_model_parameters:
    input:
        broderick_model_params = expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "broderick", 'tfunc-corrected_model_lr-{{lr}}_eph-{{max_epoch}}_dset-broderick_sub-{broderick_subj}_roi-V1_vs-{{vs}}.pt'), broderick_subj=make_subj_list('broderick')),
        broderick_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','broderick', 'precision', 'precision-s_dset-broderick_vs-{vs}.csv'),
        nsd_model_params = expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "nsdsyn", 'model-params_lr-{{lr}}_eph-{{max_epoch}}_dset-nsdsyn_sub-{subj}_roi-{roi}_vs-{{vs}}.pt'), roi=ROIS, subj=make_subj_list('nsdsyn')),
        nsd_precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn', 'precision', 'precision-s_dset-nsdsyn_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "all", 'params-avg_lr-{lr}_eph-{max_epoch}_vs-{vs}.{fig_format}')
    params:
        ylim_list = [(1.6, 4.75), (0.04, 0.43), (-0.15, 0.15),(-0.7, 0.15), (-0.04, 0.065)],
        ytick_list=  [[2, 3, 4], [0.1, 0.2,0.3,0.4], [-0.15,0,0.15], [-0.6, -0.3,0], [-0.03,0,0.03,0.06]],
        param_group = [1,2,2,3,3,4,4,5,5],
        roi_pal=ROI_PAL
    run:
        broderick_params = model.load_all_models(input.broderick_model_params, *ARGS_2D)
        broderick_precision_s =  pd.read_csv(input.broderick_precision_s)
        broderick_df = pd.merge(broderick_params, broderick_precision_s[['sub','vroinames','precision']], on=['sub','vroinames'])
        broderick_df['dset_type'] = 'Broderick et al. V1'

        nsd_params = model.load_all_models(input.nsd_model_params, *ARGS_2D)
        nsd_precision_s =  pd.read_csv(input.nsd_precision_s)
        nsd_df = pd.merge(nsd_params, nsd_precision_s[['sub','vroinames','precision']], on=['sub','vroinames'])
        nsd_df['dset_type'] = nsd_df['vroinames'].apply(lambda x: f'NSD {x}')

        final_df = pd.concat((broderick_df, nsd_df),axis=0)
        params.roi_pal.insert(0, (0.5,0.5,0.5))

        grid = vis2D.plot_precision_weighted_avg_parameters(final_df,
                                                            PARAMS_2D,
                                                            params.param_group,
                                                            hue='dset_type',
                                                            hue_order=['Broderick et al. V1',
                                                                       'NSD V1',
                                                                       'NSD V2',
                                                                       'NSD V3'],
                                                            lgd_title='ROI',
                                                            width=6,
                                                            pal=params.roi_pal,
                                                            dodge=0.26,
                                                            dot_scale=1,
                                                            ylim_list=params.ylim_list,
                                                            ytick_list=params.ytick_list,
                                                            save_path=output[0])

def Pv_projection(xaxis):
    if xaxis == "angle":
        return 'polar'
    else:
        return None

rule plot_preferred_period_2D:
    input:
        model_prediction = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","{{dset}}",'prediction_frame-{frame}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list(wildcards.dset)),
        precision = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],'dataframes','{{dset}}', 'precision', 'precision-v_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.csv'), subj=make_subj_list(wildcards.dset))
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-pperiod-prediction_frame-{frame}_xaxis-{xaxis}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-avg_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        projection = lambda wildcards: Pv_projection(wildcards.xaxis)
    run:
        precision_v = utils.load_dataframes(input.precision)
        precision_s = precision_v.groupby(['sub','vroinames']).mean().reset_index()
        precision_s['precision'] = 1/precision_s['sigma_v_squared']
        df = utils.load_dataframes(input.model_prediction,*ARGS_2D)
        df = df.merge(precision_s, on=['sub','vroinames'])
        if wildcards.xaxis == "angle":
            df = df.query('eccentricity == 5')
        df = df.groupby(['sub', 'names', wildcards.xaxis]).mean().reset_index()
        vis2D.plot_preferred_period(df, x=wildcards.xaxis, y='Pv', precision='precision',
                                    hue=None, hue_order=None,
                                    col=None, col_wrap=None,
                                    lgd_title=None, height=6,
                                    projection=params.projection, save_path=output[0])

rule plot_preferred_period_comparison:
    input:
        broderick_model_prediction=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","broderick",'prediction_frame-{{frame}}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-0.0005_eph-30000_dset-broderick_sub-{subj}_roi-V1_vs-pRFsize.h5'), subj=make_subj_list('broderick')),
        broderick_precision_v=expand(os.path.join(config['OUTPUT_DIR'],'dataframes','broderick','precision','precision-v_dset-broderick_sub-{subj}_roi-V1_vs-{{vs}}.csv'), subj=make_subj_list('broderick')),
        nsd_model_params=expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","prediction_2D","nsdsyn",'prediction_frame-{{frame}}_eccentricity-{{ecc1}}-{{ecc2}}-{{n_ecc}}_angle-{{ang1}}-{{ang2}}-{{n_ang}}_lr-{{lr}}_eph-{{max_epoch}}_dset-nsdsyn_sub-{subj}_roi-{{roi}}_vs-{{vs}}.h5'), subj=make_subj_list('nsdsyn')),
        nsd_precision_v=expand(os.path.join(config['OUTPUT_DIR'],'dataframes','nsdsyn','precision','precision-v_dset-nsdsyn_sub-{subj}_roi-V1_vs-{{vs}}.csv'), subj=make_subj_list('nsdsyn'))
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D","dset_comparison",'fig-pperiod-prediction_frame-{frame}_col-{col}_xaxis-{xaxis}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-all_sub-avg_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        col = lambda wildcards: None if wildcards.col == 'None' else wildcards.col
    run:
        broderick_model_df = utils.load_dataframes(input.broderick_model_prediction,*PARAMS_2D)
        broderick_precision_v = utils.load_dataframes(input.broderick_precision_v)
        broderick_precision_s = broderick_precision_v.groupby(['sub','vroinames'], group_keys=False).mean().reset_index()
        broderick_df = broderick_model_df.merge(broderick_precision_s[['sub','vroinames','sigma_v_squared']],
                                                on=['sub', 'vroinames'])

        nsd_model_df = utils.load_dataframes(input.nsd_model_params,*ARGS_2D)
        nsd_precision_v = utils.load_dataframes(input.nsd_precision_v)
        nsd_precision_s = nsd_precision_v.groupby(['sub','vroinames'], group_keys=False).mean().reset_index()
        nsd_df = nsd_model_df.merge(nsd_precision_s, on=['sub', 'vroinames'])

        all_df = broderick_df.append(nsd_df)
        all_df['precision'] = 1 / all_df['sigma_v_squared']
        vis2D.plot_preferred_period(all_df, x=wildcards.xaxis,
                                    hue='dset', hue_order=['broderick', 'nsdsyn'],
                                    col=params.col,
                                    lgd_title='Dataset', xlim=(0,10), save_path=output[0])

rule plot_pp:
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D","dset_comparison",'fig-pperiod-prediction_frame-{frame}_col-{col}_xaxis-{xaxis}_eccentricity-{ecc1}-{ecc2}-{n_ecc}_angle-{ang1}-{ang2}-{n_ang}_lr-{lr}_eph-{max_epoch}_dset-all_sub-avg_roi-{roi}_vs-{vs}.{fig_format}'),
        frame=['relative', 'absolute'], col=['None', 'names'], xaxis='eccentricity', ecc1='0', ecc2='10', n_ecc='3', ang1='0', ang2='360', n_ang='360', dset='nsdsyn', vs='pRFsize', lr=LR_2D, max_epoch=MAX_EPOCH_2D, roi='V1', fig_format='svg')

rule plot_individual_model_parameters:
    input:
        model_params = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D", "{{dset}}", 'model-params_lr-{{lr}}_eph-{{max_epoch}}_dset-{{dset}}_sub-{subj}_roi-{{roi}}_vs-{{vs}}.pt'), subj=make_subj_list(wildcards.dset)),
        precision_s = os.path.join(config['OUTPUT_DIR'],'dataframes','{dset}','precision','precision-s_dset-{dset}_vs-{vs}.csv')
    output:
        os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-params_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}')
    params:
        params_list = PARAMS_2D,
        groups = PARAMS_GROUP_2D,
        subj_list = lambda wildcards: make_subj_list(wildcards.dset)
    run:
        df = model.load_all_models(input.model_params, *ARGS_2D)
        precision_s = pd.read_csv(input.precision_s)
        final_params = df.merge(precision_s, on=['sub'])
        vis2D.plot_precision_weighted_avg_parameters(final_params,
                                                     params.params_list, params.groups,
                                                     weight='precision',
                                                     hue = 'sub', hue_order= params.subj_list,
                                                     lgd_title='Subjects',
                                                     height=7, pal=utils.subject_color_palettes(wildcards.dset, params.subj_list),
                                                     save_path=output[0], suptitle=wildcards.roi)


rule plot_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D", "{dset}", 'fig-{d_type}-history_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}'), d_type=['model','loss'], roi=['V1','V2','V3'], dset='nsdsyn', lr=LR_2D, max_epoch=MAX_EPOCH_2D, vs=['pRFsize'], fig_format=['svg']),
        expand(os.path.join(config['OUTPUT_DIR'],'figures',"sfp_model","results_2D","{dset}",'fig-params_lr-{lr}_eph-{max_epoch}_dset-{dset}_sub-individual_roi-{roi}_vs-{vs}.{fig_format}'), roi=['V1','V2','V3'], dset='nsdsyn', lr=LR_2D, max_epoch=MAX_EPOCH_2D, vs=['pRFsize'], fig_format=['svg'])

rule generate_synthetic_data:
    input:
        stim_info_path=os.path.join(config['NSD_DIR'], 'nsdsyn_stim_description_corrected.csv'),
        params_path = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "nsdsyn", "summary", "precision_weighted_params.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "roi-{roi}_grating-{grating_type}_nvox-{n_voxels}_noise-{noise_lvl}.csv")
    log:
        os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "roi-{roi}_grating-{grating_type}_nvox-{n_voxels}_noise-{noise_lvl}.log")
    params:
        subj_df_dir = os.path.join(config['OUTPUT_DIR'], "dataframes"),
        random_state = 42
    run:
        params_info = pd.read_csv(input.params_path)
        params_info = params_info[params_info['vroinames'] == wildcards.roi]
        params_dict = params_info.to_dict(orient='records')[0]
        syn = sim.SynthesizeData(roi=wildcards.roi, 
                                 n_voxels=int(wildcards.n_voxels), 
                                 precision_weight=True,
                                 sample_subj_list="all",
                                 stim_info_path=input.stim_info_path,
                                 dataframe_dir=params.subj_df_dir,
                                 random_state=params.random_state)
        syn_data = syn.synthesize_BOLD_2d(params_dict,
                                          grating_type=wildcards.grating_type,
                                          model=7)
        syn_data['normed_betas'] = syn.add_noise(syn_data, 
                                                 beta_col='normed_betas', 
                                                 noise_sd=MEASURED_NOISE_SD*int(wildcards.noise_lvl))
        syn_data.to_csv(output[0])

rule save_subj_cov_matrix:
    input:
        subj_df_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn", "model", "dset-nsdsyn_sub-{subj}_roi-{roi}_vs-pRFsize_tavg-False.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "cov-matrix","roi-{roi}_sub-{subj}.npy")
    run:
        subj_data = pd.read_csv(input.subj_df_path)
        subj_data['trial'] = subj_data.groupby(['sub','voxel','class_idx']).cumcount()
        cov_matrix = sim.measure_noise_covariance(subj_data, 'betas', 
                                                  new_shape=['voxel', 'class_idx', 'trial'], 
                                                  return_all=False)
        np.save(output[0], cov_matrix)

rule generate_synthetic_data_with_noise_covariance:
    input:
        subj_df_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn", "model", "dset-nsdsyn_sub-{sub}_roi-{roi}_vs-pRFsize_tavg-False.csv"),
        cov_matrix_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "cov-matrix", "roi-{roi}_sub-{sub}.npy"),
        subj_precision_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn", "precision", "precision-v_sub-{sub}_roi-{roi}_vs-pRFsize.csv"),
        original_params_path = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "nsdsyn", "summary", "precision_weighted_params.csv"),
        zero_params_path = os.path.join(config['OUTPUT_DIR'], "sfp_model", "simulation", "params_slopezero.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.csv")
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "dataframes", "simulation", "roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.txt")
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "dataframes", "simulation", "roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.log")
    run:
        subj_data = pd.read_csv(input.subj_df_path)
        subj_data['trial'] = subj_data.groupby(['sub','voxel','class_idx']).cumcount()
        
        subj_precision = pd.read_csv(input.subj_precision_path)
        print(subj_precision.head())
        subj_data = subj_data.merge(subj_precision, on=['sub','voxel','vroinames'])
        if wildcards.slope == "original":
            params_info = pd.read_csv(input.original_params_path)
        else:
            params_info = pd.read_csv(input.zero_params_path)
        params_info = params_info[params_info['vroinames'] == wildcards.roi]
        params_dict = params_info.to_dict(orient='records')[0]
        syn = sim.SynthesizeData(roi=wildcards.roi, 
                                 df=subj_data,
                                 random_state=wildcards.seed)                                 
        syn_data = syn.synthesize_BOLD_2d(params_dict,
                                          grating_type=wildcards.grating_type,
                                          model=7, phase_info=True)
        syn_data = syn.add_noise_from_covariance(syn_data, 
                                                 beta_col='betas', 
                                                 n_trials=8, 
                                                 noise_cov=np.load(input.cov_matrix_path), 
                                                 noise_level=int(wildcards.noise_lvl))
        syn_data.to_csv(output[0], index=False)

        
rule debug_simulation:
    input:
        subj_df_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn", "model", "dset-nsdsyn_sub-{sub}_roi-{roi}_vs-pRFsize_tavg-False.csv"),
        cov_matrix_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "cov-matrix", "roi-{roi}_sub-{sub}.npy"),
        subj_precision_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "nsdsyn", "precision", "precision-v_sub-{sub}_roi-{roi}_vs-pRFsize.csv"),
        original_params_path = os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_2D", "nsdsyn", "summary", "precision_weighted_params.csv"),
        zero_params_path = os.path.join(config['OUTPUT_DIR'], "sfp_model", "simulation", "params_slopezero.csv")
    output:
        os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "debug_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.csv")
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "dataframes", "simulation", "debug_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.txt")
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "dataframes", "simulation", "debug_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.log")
    run:
        subj_data = pd.read_csv(input.subj_df_path)
        subj_data['trial'] = subj_data.groupby(['sub','voxel','class_idx']).cumcount()
        
        subj_precision = pd.read_csv(input.subj_precision_path)
        print(subj_precision.head())
        subj_data = subj_data.merge(subj_precision, on=['sub','voxel','vroinames'])
        if wildcards.slope == "original":
            params_info = pd.read_csv(input.original_params_path)
        else:
            params_info = pd.read_csv(input.zero_params_path)
        params_info = params_info[params_info['vroinames'] == wildcards.roi]
        params_dict = params_info.to_dict(orient='records')[0]
        print(params_dict)
        params_info.to_csv(output[0], index=False)
        # syn = sim.SynthesizeData(roi=wildcards.roi, 
        #                          df=subj_data,
        #                          random_state=wildcards.seed)                                 
        # syn_data = syn.synthesize_BOLD_2d(params_dict,
        #                                   grating_type=wildcards.grating_type,
        #                                   model=7, phase_info=True)
        # syn_data = syn.add_noise_from_covariance(syn_data, 
        #                                          beta_col='betas', 
        #                                          n_trials=8, 
        #                                          noise_cov=np.load(input.cov_matrix_path), 
        #                                          noise_level=int(wildcards.noise_lvl))
        # syn_data.to_csv(output[0], index=False)


rule run_simulation:
    input:
        syn_data_path = os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_basesub-{sub}_slope-{slope}_rnseed-{seed}.csv")
    output:
        model_results = os.path.join(config['OUTPUT_DIR'], "sfp_model", "simulation", 'model-params_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_lr-{lr}_eph-{max_epoch}_basesub-{sub}_slope-{slope}_rnseed-{seed}.pt'),
        model_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "simulation", 'model-history_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_lr-{lr}_eph-{max_epoch}_basesub-{sub}_slope-{slope}_rnseed-{seed}.h5'),
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_model", "simulation", 'loss-history_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_lr-{lr}_eph-{max_epoch}_basesub-{sub}_slope-{slope}_rnseed-{seed}.h5')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs", "sfp_model", "simulation", 'model-history_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_lr-{lr}_eph-{max_epoch}_basesub-{sub}_slope-{slope}_rnseed-{seed}.log')
    params:
        random_state = 42
    benchmark:
        os.path.join(config['OUTPUT_DIR'], "benchmark", "sfp_model", "simulation", 'model-history_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_lr-{lr}_eph-{max_epoch}_basesub-{sub}_slope-{slope}_rnseed-{seed}.txt')
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        syn_data = pd.read_csv(input.syn_data_path)
        syn_data = syn_data.groupby(['sub','voxel','class_idx']).mean().reset_index()
        my_dataset = model.SpatialFrequencyDataset(syn_data, beta_col='noisy_betas')
        my_model = model.SpatialFrequencyModel(random_state=params.random_state)
                
        loss_history, model_history, _ = model.fit_model(my_model, my_dataset,
                                                         save_path=output.model_results,
                                                         learning_rate=float(wildcards.lr), 
                                                         max_epoch=int(wildcards.max_epoch),
                                                         print_every=1000, loss_all_voxels=False)

        model_history.to_hdf(output.model_history, key='stage', mode='w')
        loss_history.to_hdf(output.loss_history, key='stage', mode='w')   

rule run_simulation_single:
    input:
        os.path.join(config['OUTPUT_DIR'], "dataframes", "simulation", "debug_roi-V1_grating-constant_cov-True_noise-1_basesub-subj01_slope-original_rnseed-111.csv")

rule run_simulation_all:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "simulation", 'model-params_roi-{roi}_grating-{grating_type}_cov-True_noise-{noise_lvl}_lr-{lr}_eph-{max_epoch}_basesub-{sub}_slope-{slope}_rnseed-{seed}.pt'),
               roi=['V1'],
               grating_type=['scaled', 'constant'],
               noise_lvl=[1,3],
               lr=LR_2D,
               slope=["original", "zero"],
               seed=np.arange(0,100),
               sub=['subj01'],
               max_epoch=MAX_EPOCH_2D)

rule run_Broderick_subj:
    input:
        input_path = os.path.join(config['BD_DIR'],  "dataframes", "{subj}_stim_voxel_info_df_vs_md.csv")
    output:
        log_file = os.path.join(config['BD_DIR'],"logs","sfp_model","results_2D",'log_dset-Broderick_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.txt'),
        model_history = os.path.join(config['BD_DIR'],"sfp_model","results_2D",'model_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.h5'),
        loss_history = os.path.join(config['BD_DIR'],"sfp_model","results_2D",'loss_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.h5'),
#        losses_history = os.path.join(config['BD_DIR'],"sfp_model","results_2D",'losses_history_dset-Broderick_bts-md_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}.h5')
    log:
        os.path.join(config['BD_DIR'],"logs","sfp_model","results_2D",'loss_history_dset-Broderick_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}.log')
    benchmark:
        os.path.join(config['BD_DIR'],"benchmark","sfp_model","results_2D",'loss_history_dset-Broderick_full_ver-{full_ver}_{subj}_lr-{lr}_eph-{max_epoch}_benchmark_{roi}.txt')
    resources:
        cpus_per_task = 1,
        mem_mb = 4000
    run:
        subj_df = pd.read_csv(input.input_path)
        subj_dataset = model.SpatialFrequencyDataset(subj_df, beta_col='betas')
        subj_model = model.SpatialFrequencyModel(full_ver=(wildcards.full_ver=="True"))
        loss_history, model_history, elapsed_time, losses = model.fit_model(subj_model,subj_dataset,output.log_file,max_epoch=int(wildcards.max_epoch),print_every=100,loss_all_voxels=False,anomaly_detection=False,amsgrad=False,eps=1e-8)
        model_history.to_hdf(output.model_history, key='stage',mode='w')
        loss_history.to_hdf(output.loss_history, key='stage',mode='w')




def get_df_type_and_subj_list(dset):
    if dset == "broderick":
        stat = "median"
        subj_list = [utils.sub_number_to_string(sn, dataset="broderick") for sn in [1, 6, 7, 45, 46, 62, 64, 81, 95, 114, 115, 121]]
    elif dset == "nsdsyn":
        stat = "mean"
        subj_list = [utils.sub_number_to_string(sn, dataset="nsd") for sn in np.arange(1,9)]
    #file_name = [f'{wildcards.df_type}_dset-{wildcards.dset}_bts-{stat}_full_ver-{wildcards.full_ver}_{subj}_lr-{wildcards.lr}_eph-{wildcards.max_epoch}_{wildcards.roi}.h5' for subj in subj_list]
    return stat, subj_list


rule plot_avg_subj_parameter_history:
    input:
        subj_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D")
    output:
        history_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'model_history_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png')
    run:
        params = pd.read_csv(os.path.join(config['OUTPUT_DIR'], "dataframes", config['PARAMS']))
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir, wildcards.dset, wildcards.stat, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "model", [wildcards.roi])
        subj_list = [utils.sub_number_to_string(sn, dataset=wildcards.dset) for sn in sn_list]
        if wildcards.dset == "broderick":
            new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
            subj_replace_dict = dict(zip(subj_list, new_subj))
            model_history = model_history.replace({'subj': subj_replace_dict})
            subj_list = new_subj
        model_history = sim.add_ground_truth_to_df(params, model_history, id_val='ground_truth')
        params_col, params_group = sim.get_params_name_and_group(params,full_ver=(wildcards.full_ver=="True"))
        sfp_nsdsyn.visualization.plot_2D_model_results.plot_param_history_horizontal(model_history, params=params_col, group=np.arange(0,9), to_label='subj',
            label_order=subj_list, ground_truth=True, lgd_title="Subjects", save_fig=True, save_path=output.history_fig,
            height=8, col_wrap=3, ci="sd", n_boot=100, log_y=False)


rule plot_avg_subj_loss_history:
    input:
        subj_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D",'loss_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D")
    output:
        history_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'loss_history_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.png')
    run:
        sn_list = get_sn_list(wildcards.dset)
        loss_history = model.load_history_df_subj(input.df_dir, wildcards.dset, wildcards.stat, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "loss", [wildcards.roi])
        sfp_nsdsyn.visualization.plot_2D_model_results.plot_loss_history(loss_history,hue=None,lgd_title=None,col='lr_rate',height=5,save_path=output.history_fig,log_y=True,sharey=True)


rule plot_scatterplot_subj:
    input:
        bd_file = os.path.join(config['OUTPUT_DIR'], 'dataframes','Broderick_individual_subject_params_median_across_bootstraps.csv'),
        my_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'), subj=make_subj_list(wildcards)),
        df_dir= os.path.join(config['OUTPUT_DIR'], "sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_subj_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.eps')
    log:
        os.path.join(config['OUTPUT_DIR'], "logs","figures", "sfp_model","results_2D",'scatterplot_subj_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.log')
    run:
        bd_df = pd.read_csv(input.bd_file)
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir, wildcards.dset, wildcards.stat, [wildcards.full_ver], sn_list, [float(wildcards.lr)], [int(wildcards.max_epoch)], "model", [wildcards.roi])
        max_epoch = model_history.epoch.max()
        fnl_df = model_history.query('epoch == @max_epoch')
        subj_list = [utils.sub_number_to_string(sn, dataset=wildcards.dset) for sn in sn_list]
        if wildcards.dset == "broderick":
            new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
            subj_replace_dict = dict(zip(subj_list, new_subj))
            fnl_df = fnl_df.replace({'subj': subj_replace_dict})
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        fnl_df = pd.melt(fnl_df,id_vars=['subj'],value_vars=params_col,var_name='params',value_name='My_value')
        df = fnl_df.merge(bd_df, on=['subj','params'])
        f_name = 'scatter_comparison.png'
        grid = sfp_nsdsyn.visualization.plot_2D_model_results.scatter_comparison(df.query('params in @params_col'),
            x="Broderick_value", y="My_value", col="params",
            col_order=params_col,label_order=new_subj,
            to_label='subj',lgd_title="Subjects",height=6,
            save_fig=True, save_path=output.scatter_fig)

rule plot_scatterplot_avgparams:
    input:
        bd_file = os.path.join(config['OUTPUT_DIR'],'dataframes','Broderick_individual_subject_params_allbootstraps.csv'),
        my_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_avgparams_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_{roi}.eps')
    run:
        bd_df = pd.read_csv(input.bd_file)
        bd_fnl_df = bd_df.groupby(['subj', 'params']).median().reset_index()
        m_bd_fnl_df = model.get_mean_and_error_for_each_param(bd_fnl_df, err="sem")
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir,wildcards.dset,wildcards.stat,[
            wildcards.full_ver],sn_list,[float(wildcards.lr)],[int(wildcards.max_epoch)],"model",[wildcards.roi])
        m_epoch = model_history.epoch.max()
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        params_group = [0,1,1,2,2,2,2,3,3]
        fnl_df = model_history.query('epoch == @m_epoch')[params_col]
        m_fnl_df = model.get_mean_and_error_for_each_param(fnl_df, err="sem")
        sfp_nsdsyn.visualization.plot_2D_model_results.scatterplot_two_avg_params(m_bd_fnl_df, m_fnl_df, params_col, params_group, x_label='Broderick et al.(2022) values', y_label=f'My values: {wildcards.dset}', save_fig=True, save_path=output.scatter_fig)



rule plot_scatterplot_subj_betweenVareas:
    input:
        roi_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_subj_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_V1-vs-{roi}.eps')
    run:
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir,wildcards.dset,wildcards.stat,[wildcards.full_ver],sn_list,[float(wildcards.lr)],[int(wildcards.max_epoch)],"model",["V1", wildcards.roi])
        m_epoch = model_history.epoch.max()
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        params_group = [0,1,1,2,2,2,2,3,3]
        V1_df = model_history.query('epoch == @m_epoch & vroinames == "V1"')[params_col]
        roi_df = model_history.query('epoch == @m_epoch & vroinames == @wildcards.roi')[params_col]
        subj_list = [utils.sub_number_to_string(sn,dataset=wildcards.dset) for sn in sn_list]
        if wildcards.dset == "broderick":
            new_subj = ["sub-{:02d}".format(sn) for sn in np.arange(1,13)]
            subj_replace_dict = dict(zip(subj_list,new_subj))
            V1_df = V1_df.replace({'subj': subj_replace_dict})
            roi_df = roi_df.replace({'subj': subj_replace_dict})
        else:
            new_subj = subj_list
        long_V1 = utils.melt_params(V1_df, value_name='V1_value')
        long_roi = utils.melt_params(roi_df, value_name=f'{wildcards.roi}_value')
        df = pd.concat((long_V1, long_roi),axis=0)
        sfp_nsdsyn.visualization.plot_2D_model_results.scatter_comparison(df.query('params in @params_col'),
            x="V1_value",y=f"{wildcards.roi}_value",col="params",
            col_order=params_col,label_order=new_subj,
            to_label='subj',lgd_title="Subjects",height=7,
            save_fig=True, save_path=output.scatter_fig)


rule plot_scatterplot_avgparams_betweenVareas:
    input:
        roi_files = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-{{full_ver}}_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D")
    output:
        scatter_fig = os.path.join(config['OUTPUT_DIR'], "figures", "sfp_model", "results_2D",'scatterplot_avgparams_dset-{dset}_bts-{stat}_full_ver-{full_ver}_allsubj_lr-{lr}_eph-{max_epoch}_V1-vs-{roi}.eps')
    run:
        sn_list = get_sn_list(wildcards.dset)
        model_history = model.load_history_df_subj(input.df_dir,wildcards.dset,wildcards.stat,[
            wildcards.full_ver],sn_list,[float(wildcards.lr)],[int(wildcards.max_epoch)],"model",["V1", wildcards.roi])
        m_epoch = model_history.epoch.max()
        params_col = ['sigma', 'slope', 'intercept', 'p_1', 'p_2', 'p_3', 'p_4', 'A_1', 'A_2']
        params_group = [0,1,1,2,2,2,2,3,3]
        V1_df = model_history.query('epoch == @m_epoch & vroinames == "V1"')[params_col]
        roi_df = model_history.query('epoch == @m_epoch & vroinames == @wildcards.roi')[params_col]

        fnl_V1_df = model.get_mean_and_error_for_each_param(V1_df, err="sem")
        fnl_roi_df = model.get_mean_and_error_for_each_param(roi_df, err="sem")
        sfp_nsdsyn.visualization.plot_2D_model_results.scatterplot_two_avg_params(fnl_V1_df, fnl_roi_df, params_col, params_group, x_label=f'{wildcards.dset}: V1', y_label=f'{wildcards.dset}: {wildcards.roi}', save_fig=True, save_path=output.scatter_fig)

def make_file_name_list(wildcards, stim_list):
    f_names = [f'{wildcards.df_type}_history_dset-{wildcards.dset}_bts-{wildcards.stat}_{wildcards.subj}_lr-{wildcards.lr}_eph-{wildcards.max_epoch}_{wildcards.roi}_{stim}_vs-pRFcenter_e{wildcards.e1}-{wildcards.e2}_nbin-{wildcards.enum}.h5' for stim in stim_list]
    return f_names

def make_info_columns(wildcards, df):
    df['dset'] = wildcards.dset
    df['subj'] = wildcards.subj
    df['vroinames'] = wildcards.roi
    df['lr_rate'] = float(wildcards.lr)
    df['max_epoch'] = int(wildcards.max_epoch)
    return df

rule combine_all_stim:
    input:
        file_names = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'], "sfp_model", "results_1D", "{f_name}"), f_name=make_file_name_list(wildcards, stim_list))
    output:
        allstim = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_1D",'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.h5')
    log:
        os.path.join(config['OUTPUT_DIR'],"logs", "sfp_model","results_1D",'allstim_{df_type}_history_dset-{dset}_bts-{stat}_{subj}_lr-{lr}_eph-{max_epoch}_{roi}_vs-pRFcenter_e1-{e1}_e2-{e2}_nbin-{enum}.log')
    run:
        all_df = pd.DataFrame({})
        for stim, f in zip(stim_list, input.file_names):
            df = pd.read_hdf(f)
            df['names'] = stim
            df = make_info_columns(wildcards, df)
            all_df = pd.concat((all_df, df), axis=0)
        all_df.to_hdf(output.allstim, key='stage', mode='w')
        # for f in input.file_names:
        #     os.remove(f)

rule save_precision_s:
    input:
        subj_df = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"dataframes","{{dset}}","{subj}_stim_voxel_info_df_vs-pRFsigma_{{roi}}.csv"),subj=make_subj_list(wildcards))
    output:
        os.path.join(config['OUTPUT_DIR'],"dataframes","{dset}","precision_s_{dset}_{roi}.csv")
    run:
        from sfp_nsdsyn import bts
        precision_df = pd.DataFrame({})
        for df in input.subj_df:
            tmp = pd.read_csv(df)
            tmp = bts.get_precision_s(tmp, subset=['subj','vroinames'])
            precision_df = precision_df.append(tmp)
            precision_df.to_csv(output[0], index=False)


rule plot_2D_parameters_individual:
    input:
        model_history=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-True_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        #subj_df=lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"dataframes","{{dset}}","{subj}_stim_voxel_info_df_vs-pRFsigma_{{roi}}.csv"),subj=make_subj_list(wildcards)),
        precision_df=os.path.join(config['OUTPUT_DIR'],"dataframes","{dset}","precision_s_{dset}_{roi}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'pointplot-params_avg-False_dset-{dset}_bts-{stat}_lr-{lr}_eph-{max_epoch}_vs-pRFsigma_roi-{roi}.{fig_format}'),
    params:
        df_dir = os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D"),
        params_order = PARAMS_2D,
        params_group = [1,2,3,4,4,4,4,4,4]
    run:
        from sfp_nsdsyn import model
        from sfp_nsdsyn import bts
        from sfp_nsdsyn import vis
        model_history = model.load_history_files(input.model_history)
        final_params = model_history[model_history.epoch == int(wildcards.max_epoch) - 1]
        precision_df = pd.read_csv(input.precision_df)
        final_params_with_precision = pd.merge(final_params, precision_df, on='subj')
        pal = vis.make_dset_palettes(wildcards.dset)
        grid = vis.plot_individual_parameters(final_params_with_precision, params.params_order, subplot_group=params.params_group, palette=pal,
            roi='all')
        utils.save_fig(save_fig=True, save_path=output[0])

def get_projection(x):
    if x == 'angle':
        projection = 'polar'
        despine = False
    else:
        projection = None
        despine = True
    return projection, despine

def get_stim_class(stim):
    stim_dict = {'a': 'annulus', 'f': 'forward spiral', 'p': 'pinwheel', 'r':'reverse spiral'}
    if stim == 'all':
        return stim_dict.values()
    else:
        return [v for k, v in stim_dict.items() if k in stim]

rule preferred_period:
    input:
        stim_info = '/Volumes/server/Projects/sfp_nsd/natural-scenes-dataset/nsdsynthetic_sf_stim_description.csv',
        model_history = lambda wildcards: expand(os.path.join(config['OUTPUT_DIR'],"sfp_model","results_2D",'model_history_dset-{{dset}}_bts-{{stat}}_full_ver-True_{subj}_lr-{{lr}}_eph-{{max_epoch}}_{{roi}}.h5'),subj=make_subj_list(wildcards)),
        precision_df = os.path.join(config['OUTPUT_DIR'],"dataframes","{dset}","precision_s_{dset}_{roi}.csv")
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'y-preferred-period_x-{x}_stim-{stim}_avg-True_dset-{dset}_bts-{stat}_lr-{lr}_eph-{max_epoch}_vs-pRFsigma_roi-{roi}.{fig_format}')
    params:
        sn_list = lambda wildcards: get_sn_list(wildcards.dset),
        stim_class = lambda wildcards: get_stim_class(wildcards.stim),
        properties = lambda wildcards: get_projection(wildcards.x)
    run:
        from sfp_nsdsyn import prep
        from sfp_nsdsyn import vis
        from sfp_nsdsyn import model
        stim_info = model.stim_w_r_and_w_a(input.stim_info, params.stim_class)
        synthetic_voxels = model.create_synthetic_cols(stim_info,
                                                       var_list=['eccentricity', 'angle'],
                                                       val_range_list=[(0, 10), (0, np.pi * 2)],
                                                       n_val_list=[3, 360])
        synthetic_voxels['local_ori'] = prep.calculate_local_orientation(w_a=synthetic_voxels.w_a,w_r=synthetic_voxels.w_r,retinotopic_angle=synthetic_voxels.angle,angle_in_radians=True)

        model_history = model.load_history_files(input.model_history)
        final_params = model_history[model_history.epoch == int(wildcards.max_epoch) - 1]
        synthetic_voxels_all_subj = pd.DataFrame({})
        for sn in params.sn_list:
            tmp = synthetic_voxels.copy()
            subj = utils.sub_number_to_string(sn, wildcards.dset)
            subj_params = final_params.query('subj == @subj')
            tmp['subj'] = subj
            tmp['vroinames'] = wildcards.roi
            tmp['Pv'] = tmp.apply(model.get_Pv_row, params=subj_params, axis=1)
            synthetic_voxels_all_subj = synthetic_voxels_all_subj.append(tmp, ignore_index=True)

        precision_s = pd.read_csv(input.precision_df)
        synthetic_voxels_all_subj = pd.merge(synthetic_voxels_all_subj, precision_s, on=['subj','vroinames'])
        if wildcards.x == 'angle':
            synthetic_voxels_all_subj = synthetic_voxels_all_subj.query('eccentricity == 5')
        vis.plot_preferred_period(synthetic_voxels_all_subj,
                                  x=wildcards.x,
                                  col=None, col_wrap=None,
                                  hue='names', hue_order=params.stim_class,
                                  projection=params.properties[0],
                                  despine=params.properties[1])
        utils.save_fig(save_fig=True, save_path=output[0])


rule svg_all:
    input:
        a = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'pointplot-precision-weighted-params_avg-True_dset-nsdsyn_bts-mean_lr-0.0005_eph-20000_vs-pRFsigma_roi-{roi}.png'), roi=['V1V2V3']),
        b = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_model","results_2D",'y-preferred-period_x-{x}_stim-{stim}_avg-True_dset-nsdsyn_bts-mean_lr-0.0005_eph-20000_vs-pRFsigma_roi-{roi}.png'), x=['eccentricity','angle'], stim=['all'], roi=['V1', 'V2', 'V3'])

rule sfp_anova_test:
    input:
        design_mat=os.path.join(config['NSD_DIR'],'nsddata','experiments','nsdsynthetic','nsdsynthetic_expdesign.mat'),
        stim_info=os.path.join(config['NSD_DIR'],'nsdsyn_stim_description.csv'),
        template=os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{sub}','label','{hemi}.prfeccentricity.mgz'),
        betas=os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{sub}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','{hemi}.betas_nsdsynthetic.hdf5'),
    output:
        F_map = os.path.join(config['OUTPUT_DIR'], "sfp_anova", "brain_maps","{dset}","{hemi}.sub-{sub}_stat-anova_value-F.mgz"),
        p_map = os.path.join(config['OUTPUT_DIR'],"sfp_anova","brain_maps","{dset}","{hemi}.sub-{sub}_stat-anova_value-p.mgz")

    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        betas_df = sfm.get_whole_brain_betas(betas_path=input.betas,
                                             design_mat_path=input.design_mat,
                                             stim_info_path=input.stim_info,
                                             task_keys=['fixation_task', 'memory_task'],
                                             task_average=True,
                                             x_axis='voxel', y_axis='stim_idx', long_format=True)
        stim_list = [s.replace('-', ' ') for s in STIM_LIST]
        F, p, identifiers = sfm.sf_multiple_one_way_anova(betas_df.query('names in @stim_list'),
                                                          to_test='freq_lvl',
                                                          values='betas',
                                                          on='voxel',
                                                          identifier_list=['names', 'phase'],
                                                          test_unique=None)
        map_values_as_mgz(input.template, F, save_path=output.F_map)
        map_values_as_mgz(input.template, p, save_path=output.p_map)

rule save_all_voxel_df_into_different_bins:
    input:
        design_mat = os.path.join(config['NSD_DIR'],'nsddata','experiments','nsdsynthetic','nsdsynthetic_expdesign.mat'),
        stim_info = os.path.join(config['NSD_DIR'],'nsdsyn_stim_description.csv'),
        eccentricity = os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{sub}','label','{hemi}.prfeccentricity.mgz'),
        betas = os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{sub}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','{hemi}.betas_nsdsynthetic.hdf5'),
    output:
        os.path.join(config['OUTPUT_DIR'], 'dataframes', 'nsdsyn', 'all_voxels', 'binned_hemi-{hemi}_nbin-{nbin}_curbin-{curbin}_dset-nsdsyn_sub-{sub}_roi-wholebrain.h5')
    run:
        betas_df = sfm.get_whole_brain_betas(betas_path=input.betas,
                                             design_mat_path=input.design_mat,
                                             stim_info_path=input.stim_info,
                                             task_keys=['fixation_task', 'memory_task'],
                                             task_average=True, eccentricity_path=input.eccentricity,
                                             x_axis='voxel',y_axis='stim_idx',long_format=True)
        stim_list = [s.replace('-',' ') for s in STIM_LIST]
        betas_df = betas_df.query('names in @stim_list')
        avg_betas_df = betas_df.groupby(['voxel','freq_lvl']).mean().reset_index()
        total_n_voxels = avg_betas_df.voxel.nunique()
        avg_betas_df['bins'], step = sfm.divide_df_into_n_bins(avg_betas_df,
                                                               to_bins='voxel',
                                                               n_bins=int(wildcards.nbin),
                                                               return_step=True)
        avg_betas_df = avg_betas_df[avg_betas_df['bins'] == int(wildcards.curbin)]
        avg_betas_df.to_hdf(output[0], key='stage', mode='w')
        print(f'{wildcards.sub} - total n voxels: {total_n_voxels}, step: {step}')

rule save_all_voxels:
    input:
        expand(os.path.join(config['OUTPUT_DIR'], 'dataframes', 'nsdsyn', 'all_voxels', 'binned_hemi-{hemi}_nbin-{nbin}_curbin-{curbin}_dset-nsdsyn_sub-{sub}_roi-wholebrain.h5'), hemi='lh', sub=['subj01'], nbin=20, curbin=np.arange(0,20))

rule sfp_voxel_wise_log_gaussian_tuning:
    input:
        design_mat = os.path.join(config['NSD_DIR'],'nsddata','experiments','nsdsynthetic','nsdsynthetic_expdesign.mat'),
        stim_info = os.path.join(config['NSD_DIR'],'nsdsyn_stim_description.csv'),
        eccentricity = os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{sub}','label','{hemi}.prfeccentricity.mgz'),
        betas = os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{sub}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','{hemi}.betas_nsdsynthetic.hdf5'),
    output:
        loss_history = os.path.join(config['OUTPUT_DIR'], "sfp_anova", "voxel-tuning", "train_history", "{dset}","loss-history_voxel-test100_hemi-{hemi}_sub-{sub}.hdf"),
        model_history= os.path.join(config['OUTPUT_DIR'],"sfp_anova","voxel-tuning","train_history","{dset}","model-history_voxel-test100_hemi-{hemi}_sub-{sub}.hdf"),
    #model= os.path.join(config['OUTPUT_DIR'],"sfp_anova","voxel-tuning", "model", "{dset}", "model-logG_voxel-test100_hemi-{hemi}_sub-{sub}.pt")
    run:
        betas_df = sfm.get_whole_brain_betas(betas_path=input.betas,
                                             design_mat_path=input.design_mat,
                                             stim_info_path=input.stim_info,
                                             task_keys=['fixation_task', 'memory_task'],
                                             task_average=True, eccentricity_path=input.eccentricity,
                                             x_axis='voxel',y_axis='stim_idx',long_format=True)
        stim_list = [s.replace('-',' ') for s in STIM_LIST]
        random_voxels = utils.pick_random_voxels(betas_df.voxel.unique(), 100)
        betas_df = betas_df.query('names in @stim_list & voxel in @random_voxels')
        avg_betas_df = betas_df.groupby(['voxel','freq_lvl']).mean().reset_index()
        all_models_history = pd.DataFrame({})
        all_losses_history = pd.DataFrame({})
        for v in random_voxels:
            print(f'sub {wildcards.sub}, voxel num: {v}\n')
            tmp = avg_betas_df.query('voxel == @v')
            my_model = tuning.LogGaussianTuningModel()
            my_dataset = tuning.LogGaussianTuningDataset(tmp['local_sf'], tmp['betas'])
            loss_history, model_history = tuning.fit_tuning_curves(my_model, my_dataset,
                                                       learning_rate=0.001,
                                                       max_epoch=10000,
                                                       print_every=10000,
                                                       save_path=None)
            loss_history['voxel'] = v
            model_history['voxel'] = v
            all_models_history = pd.concat((all_models_history, model_history), axis=0)
            all_losses_history = pd.concat((all_losses_history, loss_history), axis=0)

        all_losses_history.to_hdf(output.loss_history, key='stage', mode='w')
        all_models_history.to_hdf(output.model_history,key='stage',mode='w')

rule voxel_wise_tuning_all:
    input:
        #expand(os.path.join(config['OUTPUT_DIR'], "sfp_anova", "voxel-tuning", "train_history", "{dset}","loss-history_voxel-test100_hemi-{hemi}_sub-{sub}.hdf"), dset='nsdsyn', hemi=['lh', 'rh'], sub=make_subj_list('nsdsyn'))
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","voxel-tuning", "{dset}","method-curvefit_hemi-{hemi}_sub-{sub}.hdf"), dset='nsdsyn', hemi=['lh', 'rh'], sub=make_subj_list('nsdsyn'))

rule voxel_wise_tuning:
    input:
        design_mat=os.path.join(config['NSD_DIR'],'nsddata','experiments','nsdsynthetic','nsdsynthetic_expdesign.mat'),
        stim_info=os.path.join(config['NSD_DIR'],'nsdsyn_stim_description.csv'),
        eccentricity=os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{sub}','label','{hemi}.prfeccentricity.mgz'),
        betas=os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{sub}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','{hemi}.betas_nsdsynthetic.hdf5'),
        template=os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{sub}','label','{hemi}.prfeccentricity.mgz'),
    output:
        df = os.path.join(config['OUTPUT_DIR'],"dataframes", "sfp_maps","mgzs", "{dset}","hemi-{hemi}_sub-{sub}_frame-{ref_frame}.hdf"),
        amp_map = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}", "{hemi}.sub-{sub}_value-amp_frame-{ref_frame}.mgz"),
        mode_map = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}", "{hemi}.sub-{sub}_value-mode_frame-{ref_frame}.mgz"),
        sigma_map = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}", "{hemi}.sub-{sub}_value-sigma_frame-{ref_frame}.mgz"),
        r2_map= os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.sub-{sub}_value-r2_frame-{ref_frame}.mgz"),
        rmse_map= os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.sub-{sub}_value-rmse_frame-{ref_frame}.mgz"),
    params:
        p0 = np.random.random(3) + [0, 0.5, 0.5]
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        betas_df = sfm.get_whole_brain_betas(betas_path=input.betas,
                                             design_mat_path=input.design_mat,
                                             stim_info_path=input.stim_info,
                                             task_keys=['fixation_task', 'memory_task'],
                                             task_average=True,eccentricity_path=input.eccentricity,
                                             x_axis='voxel',y_axis='stim_idx', long_format=True, reference_frame={wildcards.ref_frame})
        stim_list = [s.replace('-',' ') for s in STIM_LIST]
        betas_df = betas_df.query('names in @stim_list')
        avg_betas_df = betas_df.groupby(['voxel', 'freq_lvl']).mean().reset_index()
        avg_betas_df = avg_betas_df[['voxel', 'freq_lvl', 'betas', 'local_sf']]
        p_opt = pd.DataFrame({})
        print(f'subject {wildcards.sub} {wildcards.hemi}')
        for v in avg_betas_df.voxel.unique().tolist():
            v_tmp = avg_betas_df.query('voxel == @v')
            tmp, _ = tuning.fit_logGaussian_curves(v_tmp, x='local_sf', y='betas', goodness_of_fit=True,
                                                   initial_params=params.p0)
            tmp['voxel'] = v
            p_opt = pd.concat((tmp, p_opt))
        p_opt = p_opt.sort_values('voxel').set_index('voxel').reset_index()
        p_opt['sub'] = wildcards.sub
        p_opt.to_hdf(output.df, key='stage',mode='w')
        map_values_as_mgz(input.template, p_opt['amp'].to_numpy(), save_path=output.amp_map)
        map_values_as_mgz(input.template, p_opt['mode'].to_numpy(), save_path=output.mode_map)
        map_values_as_mgz(input.template, p_opt['sigma'].to_numpy(), save_path=output.sigma_map)
        map_values_as_mgz(input.template, p_opt['r2'].to_numpy(), save_path=output.r2_map)
        map_values_as_mgz(input.template, p_opt['rmse'].to_numpy(), save_path=output.rmse_map)

def breakdown_dfs(value_dict, voxel_dict, sn_list, hemi):
    hemi_df = pd.DataFrame({})
    for sn in sn_list:
        for roi in value_dict[sn].keys():
            tmp = pd.DataFrame({})
            tmp['value'] = value_dict[sn][roi]
            tmp['voxel'] = voxel_dict[sn][roi]
            tmp['ROI'] = roi
            tmp['hemi'] = hemi
            tmp['sub'] = sn
            hemi_df = pd.concat((hemi_df, tmp))
    return hemi_df

rule save_value_for_each_roi_as_a_dataframe:
    output:
        os.path.join(config['OUTPUT_DIR'], "dataframes","sfp_maps", "mgzs", "{dset}", "sub-all_value-{val}_frame-{ref_frame}.hdf")
    input:
        lh_mgzs = expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","lh.sub-{sub}_value-{{val}}_frame-{{ref_frame}}.mgz"), sub=make_subj_list('nsdsyn')),
        rh_mgzs= expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","rh.sub-{sub}_value-{{val}}_frame-{{ref_frame}}.mgz"),sub=make_subj_list('nsdsyn'))
    params:
        SUBJECTS_DIR=os.path.join(config['NSD_DIR'],"nsddata","freesurfer"),
        roi_list=['V1v', 'V1d', 'V2v','V2d', 'V3v','V3d', 'hV4', 'pFFA','aFFA','PPA']
    run:
        from pysurfer.mgz_helper import extract_info_from_filename, get_vertices_in_labels, get_existing_labels_only
        lh_rois, lh_voxels = {}, {}
        rh_rois, rh_voxels = {}, {}
        sn_list = []
        for lh_mgz in input.lh_mgzs:
            info = extract_info_from_filename(lh_mgz, *['sub'])
            sn=info['sub']
            label_dir = os.path.join(params.SUBJECTS_DIR, sn, 'label')
            lh_labels, lh_label_paths = get_existing_labels_only(params.roi_list, label_dir, 'lh',return_paths=True)
            lh_rois[sn], lh_voxels[sn] = get_vertices_in_labels(lh_mgz, lh_label_paths, lh_labels, load_mgz=True,return_label=True)
            sn_list.append(sn)
        for rh_mgz in input.rh_mgzs:
            info = extract_info_from_filename(rh_mgz, *['sub'])
            sn=info['sub']
            label_dir = os.path.join(params.SUBJECTS_DIR, sn, 'label')
            rh_labels, rh_label_paths = get_existing_labels_only(params.roi_list, label_dir, 'rh',return_paths=True)
            rh_rois[sn], rh_voxels[sn] = get_vertices_in_labels(rh_mgz, rh_label_paths, rh_labels, load_mgz=True,return_label=True)

        lh_df = breakdown_dfs(lh_rois, lh_voxels, sn_list, hemi='lh')
        rh_df = breakdown_dfs(rh_rois, rh_voxels, sn_list, hemi='rh')
        all_df = pd.concat((lh_df, rh_df))
        all_df.to_hdf(output[0], key='stage',mode='w')

def combine_ventral_and_dorsal_rois(df, roi):
    # Replace 'V1v' and 'V1d' with 'V1', and 'V2v' and 'V2d' with 'V2'
    new_col = df[roi].replace({'V1v': 'V1', 'V1d': 'V1',
                                  'V2v': 'V2', 'V2d': 'V2',
                                  'V3v': 'V3', 'V3d': 'V3',
                                  'pFFA': 'FFA-1', 'aFFA': 'FFA-2'})
    return new_col

rule quantify_value_dataframe:
    output:
        sub_hue=os.path.join(config['OUTPUT_DIR'], "figures","sfp_maps", "mgzs", "{dset}", "fig-medianplot_hue-sub_sub-all_value-{val}_frame-{ref_frame}.png"),
        roi_hue=os.path.join(config['OUTPUT_DIR'], "figures","sfp_maps", "mgzs", "{dset}", "fig-medianplot_hue-roi_sub-all_value-{val}_frame-{ref_frame}.png")
    input:
        os.path.join(config['OUTPUT_DIR'], "dataframes","sfp_maps", "mgzs", "{dset}", "sub-all_value-{val}_frame-{ref_frame}.hdf")
    params:
        roi_list=['V1', 'V2', 'V3', 'hV4', 'FFA-1', 'FFA-2', 'PPA']
    run:
        # Calculate median values for each subject and ROI
        all_df = pd.read_hdf(input[0], key='stage')
        all_df['ROI'] = combine_ventral_and_dorsal_rois(all_df, 'ROI')
        medians = all_df.groupby(['sub', 'ROI'])['value'].median().reset_index()
        y_label = r"$R^2$"if wildcards.val == 'r2' else wildcards.val
        g = vis1D.plot_median_for_each_sub_and_roi(medians,'ROI','value',
                                                   x_order=params.roi_list,
                                                   hue='sub',
                                                   hue_order=make_subj_list('nsdsyn'),
                                                   height=5,
                                                   y_label=y_label,
                                                   lgd_title='Subject',
                                                   save_path=output.sub_hue)
        g = vis1D.plot_median_for_each_sub_and_roi(medians,'ROI','value',
                                                   x_order=params.roi_list,
                                                   hue='ROI',
                                                   hue_order=params.roi_list,
                                                   height=5,
                                                   palette=retinotopy_colors(to_seaborn=True),
                                                   save_path=output.roi_hue)



rule save_precision_for_each_roi_as_a_dataframe:
    output:
        os.path.join(config['OUTPUT_DIR'], "dataframes","sfp_maps", "mgzs", "{dset}", "sub-all_value-{val}.hdf")
    input:
        lh_mgzs = expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","lh.sub-{sub}_value-{{val}}.mgz"), sub=make_subj_list('nsdsyn')),
        rh_mgzs= expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","rh.sub-{sub}_value-{{val}}.mgz"),sub=make_subj_list('nsdsyn'))
    params:
        SUBJECTS_DIR=os.path.join(config['NSD_DIR'],"nsddata","freesurfer"),
        roi_list=['V1v', 'V1d', 'V2v','V2d', 'V3v','V3d', 'hV4', 'pFFA','aFFA','PPA']
    run:
        from pysurfer.mgz_helper import extract_info_from_filename, get_vertices_in_labels, get_existing_labels_only
        lh_rois, lh_voxels = {}, {}
        rh_rois, rh_voxels = {}, {}
        sn_list = []
        for lh_mgz in input.lh_mgzs:
            info = extract_info_from_filename(lh_mgz, *['sub'])
            sn=info['sub']
            label_dir = os.path.join(params.SUBJECTS_DIR, sn, 'label')
            lh_labels, lh_label_paths = get_existing_labels_only(params.roi_list, label_dir, 'lh',return_paths=True)
            lh_rois[sn], lh_voxels[sn] = get_vertices_in_labels(lh_mgz, lh_label_paths, lh_labels, load_mgz=True,return_label=True)
            sn_list.append(sn)
        for rh_mgz in input.rh_mgzs:
            info = extract_info_from_filename(rh_mgz, *['sub'])
            sn=info['sub']
            label_dir = os.path.join(params.SUBJECTS_DIR, sn, 'label')
            rh_labels, rh_label_paths = get_existing_labels_only(params.roi_list, label_dir, 'rh',return_paths=True)
            rh_rois[sn], rh_voxels[sn] = get_vertices_in_labels(rh_mgz, rh_label_paths, rh_labels, load_mgz=True,return_label=True)

        lh_df = breakdown_dfs(lh_rois, lh_voxels, sn_list, hemi='lh')
        rh_df = breakdown_dfs(rh_rois, rh_voxels, sn_list, hemi='rh')
        all_df = pd.concat((lh_df, rh_df))
        all_df.to_hdf(output[0], key='stage',mode='w')

rule quantify_precision_dataframe:
    output:
        sub_hue=os.path.join(config['OUTPUT_DIR'], "figures","sfp_maps", "mgzs", "{dset}", "fig-medianplot_hue-sub_sub-all_value-{val}.png"),
        roi_hue=os.path.join(config['OUTPUT_DIR'], "figures","sfp_maps", "mgzs", "{dset}", "fig-medianplot_hue-roi_sub-all_value-{val}.png")
    input:
        os.path.join(config['OUTPUT_DIR'], "dataframes","sfp_maps", "mgzs", "{dset}", "sub-all_value-{val}.hdf")
    params:
        roi_list=['V1', 'V2', 'V3', 'hV4', 'FFA-1', 'FFA-2', 'PPA']
    run:
        # Calculate median values for each subject and ROI
        all_df = pd.read_hdf(input[0], key='stage')
        all_df['ROI'] = combine_ventral_and_dorsal_rois(all_df, 'ROI')
        medians = all_df.groupby(['sub', 'ROI'])['value'].median().reset_index()
        y_label = r"$R^2$"if wildcards.val == 'r2' else wildcards.val.title()
        g = vis1D.plot_median_for_each_sub_and_roi(medians,'ROI','value',
                                                   x_order=params.roi_list,
                                                   hue='sub',
                                                   hue_order=make_subj_list('nsdsyn'),
                                                   height=5,
                                                   y_label=y_label,
                                                   lgd_title='Subject',
                                                   save_path=output.sub_hue)
        g = vis1D.plot_median_for_each_sub_and_roi(medians,'ROI','value',
                                                   x_order=params.roi_list,
                                                   hue='ROI',
                                                   hue_order=params.roi_list,
                                                   height=5,
                                                   palette=retinotopy_colors(to_seaborn=True),
                                                   save_path=output.roi_hue)


rule precision_v_map:
    input:
        design_mat=os.path.join(config['NSD_DIR'],'nsddata','experiments','nsdsynthetic','nsdsynthetic_expdesign.mat'),
        stim_info=os.path.join(config['NSD_DIR'],'nsdsyn_stim_description.csv'),
        eccentricity=os.path.join(config['NSD_DIR'],'nsddata','freesurfer','{sub}','label','{hemi}.prfeccentricity.mgz'),
        betas=os.path.join(config['NSD_DIR'],'nsddata_betas','ppdata','{sub}','nativesurface','nsdsyntheticbetas_fithrf_GLMdenoise_RR','{hemi}.betas_nsdsynthetic.hdf5'),
    output:
        os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}", "{hemi}.sub-{sub}_value-precision.mgz"),
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        betas_df = sfm.get_whole_brain_betas(betas_path=input.betas,
                                            design_mat_path=input.design_mat,
                                            stim_info_path=input.stim_info,
                                            task_keys=['fixation_task', 'memory_task'],
                                            task_average=True,eccentricity_path=input.eccentricity,
                                            x_axis='voxel',y_axis='stim_idx',long_format=True)
        stim_list = [s.replace('-',' ') for s in STIM_LIST]
        betas_df = betas_df.query('names in @stim_list')
        betas_df = bts.normalize_betas_by_frequency_magnitude(betas_df,betas='betas',freq_lvl='freq_lvl')
        sigma_v = bts.get_sigma_v_for_whole_brain(betas_df, betas='normed_betas', class_list=None, sigma_power=2)
        map_values_as_mgz(input.eccentricity, 1/sigma_v, save_path=output[0])

def custom_colormap():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Define the colors and their positions along the color gradient
    colors = [(1, 0, 0), (0.9, 0.1, 0), (0.9, 0.2, 0), (0.8, 0.3, 0), (0.9, 0.6, 0), (0.9, 0.8, 0),
              (1, 1, 0)]  # Red, Orange, Yellow
    positions = [0, 0.1, 0.2, 0.5, 0.8, 1]  # Positions corresponding to each color

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("RedOrangeYellow",list(zip(positions,colors)))
    return custom_cmap

rule freeview_precision_mgz:
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss", "precision", "roi-wang_view-{view}_thres-{downthres}_sub-{sn}_value-precision.png"),
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","nsdsyn","{hemi}.sub-{{sn}}_value-precision.mgz"), hemi=['lh','rh'])
    params:
        freesurfer_dir=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer"),
        #rois=['V1v','V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4','FFA-1','FFA-2','PPA'],
        rois=['V1v','V1d','V2v','V2d','V3v','V3d','hV4','wangVO1','wangVO2','wangPHC1','wangPHC2','PPA']
    run:
        from pysurfer.freeview_helper import make_custom_color_palettes_for_overlay, plot_mgz
        from pysurfer.mgz_helper import extract_info_from_filename
        from matplotlib.pyplot import get_cmap
        label_colors = [np.asarray([255,255,255])] * len(params.rois)
        overlay_custom=make_custom_color_palettes_for_overlay(get_cmap('plasma'), val_range=(float(wildcards.downthres), 70), n=100, log_scale=False)
        #overlay_custom=make_custom_color_palettes_for_overlay(get_cmap('custom_cmap'), val_range=(float(wildcards.downthres), 200), n=300, log_scale=False)
        kwargs = {'label_opacity': 1, 'label_outline': True, 'overlay_custom': overlay_custom, 'overlay_opacity': 0.95}

        labels = [f'{roi}.label' for roi in params.rois]
        label_dir = os.path.join(params.freesurfer_dir, wildcards.sn, 'label')

        info = extract_info_from_filename(input[0])
        plot_mgz(params.freesurfer_dir, sn=wildcards.sn,
                 overlay=info['overlay'], overlay_dir=info['folder'],
                 labels=labels, label_dir=label_dir, label_colors=label_colors,
                 colorscale=True, view=wildcards.view,
                 surf='inflated', save_path=output[0], **kwargs)

rule freeview_val_mgz:
    output:
        #os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss", "{val}", "view-{view}_thres-{thres}_sub-{sn}_value-{val}_frame-{ref_frame}.png"),
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","{val}","roi-wang_view-{view}_thres-{thres}_sub-{sn}_value-{val}_frame-{ref_frame}.png"),
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","nsdsyn","{hemi}.sub-{{sn}}_value-{{val}}_frame-{{ref_frame}}.mgz"), hemi=['lh','rh'])
    params:
        freesurfer_dir=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer"),
        #rois=['V1v','V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4','FFA-1','FFA-2','PPA'],
        rois=['V1v','V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'wangVO1','wangVO2','wangPHC1', 'wangPHC2', 'PPA']
    run:
        from pysurfer.freeview_helper import make_custom_color_palettes_for_overlay, plot_mgz
        from pysurfer.mgz_helper import extract_info_from_filename
        from matplotlib.pyplot import get_cmap
        label_colors = retinotopy_colors(params.rois, all_black=True)
        overlay_custom=make_custom_color_palettes_for_overlay(get_cmap('autumn'), val_range=(float(wildcards.thres), 1), n=100, log_scale=False)
        kwargs = {'label_opacity': 1, 'label_outline': True, 'overlay_custom': overlay_custom}

        labels = [f'{roi}.label' for roi in params.rois]
        label_dir = os.path.join(params.freesurfer_dir, wildcards.sn, 'label')

        info = extract_info_from_filename(input[0])
        plot_mgz(params.freesurfer_dir, sn=wildcards.sn,
                 overlay=info['overlay'], overlay_dir=info['folder'],
                 labels=labels, label_dir=label_dir, label_colors=label_colors,
                 colorscale=True, view=wildcards.view,
                 surf='inflated', save_path=output[0], **kwargs)

rule visualize_all:
    input:
        a = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss", "{val}", "view-{view}_thres-{thres}_sub-{sn}_value-{val}_frame-{ref_frame}.png"), view=['posterior','inferior'], thres=[0.3, 0.2], val=['r2'], ref_frame=['absolute'], sn=make_subj_list('nsdsyn')),
        b = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss", "precision", "view-{view}_thres-{downthres}_sub-{sn}_value-precision.png"), view=['posterior','inferior'], downthres=[2], sn=make_subj_list('nsdsyn')),
        #c = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss", "masks", "view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_frame-{ref_frame}.png"), view=['posterior','inferior'], r2thres=[0.3, 0.2], precisionthres=[2], ref_frame=['absolute','relative'], sn=make_subj_list('nsdsyn')),
        #d = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss", "masking_process", "view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_frame-{ref_frame}.png"), view=['posterior','inferior'], r2thres=[0.3, 0.2], precisionthres=[2], ref_frame=['absolute'], sn=make_subj_list('nsdsyn'))

rule visualize_intersection_mask:
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss","masks","view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_frame-{ref_frame}.png"),
    input:
        expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","{hemi}.mask-intersection_r2thres-{{r2thres}}_precisionthres-{{precisionthres}}_sub-{{sn}}_frame-{{ref_frame}}.mgz"), hemi=['lh','rh'])
    params:
        freesurfer_dir=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer"),
        rois=['V1v','V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4','FFA-1','FFA-2','PPA'],
    run:
        from pysurfer.freeview_helper import make_custom_color_palettes_for_overlay, plot_mgz, retinotopy_colors
        from pysurfer.mgz_helper import extract_info_from_filename
        #label_colors =  [np.asarray([0, 0, 0])] * 7 + retinotopy_colors(['FFA-1','FFA-2','PPA']) +[np.asarray([30,30,30])] * 4
        label_colors = retinotopy_colors(params.rois[:-3], all_black=True)
        label_colors += retinotopy_colors(params.rois[-3:], all_black=False)
        kwargs = {'label_opacity': 1, 'label_outline': True, 'overlay_opacity': 0.8}
        labels = [f'{roi}.label' for roi in params.rois]
        label_dir = os.path.join(params.freesurfer_dir, wildcards.sn, 'label')

        info = extract_info_from_filename(input[0])
        plot_mgz(params.freesurfer_dir, sn=wildcards.sn,
                 overlay=info['overlay'], overlay_dir=info['folder'],
                 labels=labels, label_dir=label_dir, label_colors=label_colors,
                 colorscale=False, view=wildcards.view,
                 surf='inflated', save_path=output[0], **kwargs)
#
rule visualize_masking_process:
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss", "masking_process", "view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_frame-{ref_frame}.png"),
    input:
        precision_ss=os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss","precision", "view-{view}_thres-{precisionthres}_sub-{sn}_value-precision.png"),
        r2_ss=os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss", "r2", "view-{view}_thres-{r2thres}_sub-{sn}_value-r2_frame-{ref_frame}.png"),
        intersection_ss=os.path.join(config['OUTPUT_DIR'], "figures", "sfp_maps","mgzs","{dset}","ss", "masks", "view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_frame-{ref_frame}.png")
    run:
        from pysurfer.freeview_helper import plot_freeview_ss
        n_cols=3
        cols=[input.precision_ss, input.r2_ss, input.intersection_ss]
        col_titles=['Precision', r"$R^2$", 'Intersection mask']
        plot_freeview_ss(3,cols, col_titles, suptitle=wildcards.sn, dpi=500, save_path=output[0])

rule make_a_precision_mask:
    input:
        precision=os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}", "{hemi}.sub-{sub}_value-{mask}.mgz"),
    output:
        mask_mgz=os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "{hemi}.mask-{mask}_sub-{sub}_thres-{thres}.mgz"),
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs
        varexp_mask = load_mgzs(input.precision, fdata_only=True,squeeze=False)
        varexp_mask[varexp_mask < float(wildcards.thres)] = 0
        varexp_mask[varexp_mask > float(wildcards.thres)] = 1
        map_values_as_mgz(template=input.precision, data=varexp_mask, save_path=output.mask_mgz)

rule make_a_val_mask:
    output:
        mask_mgz=os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-{val}_sub-{sub}_thres-{thres}_frame-{ref_frame}.mgz"),
    input:
        varexp = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.sub-{sub}_value-{val}_frame-{ref_frame}.mgz"),
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs
        varexp_mask = load_mgzs(input.varexp, fdata_only=True,squeeze=False)
        varexp_mask[varexp_mask < float(wildcards.thres)] = 0
        varexp_mask[varexp_mask > float(wildcards.thres)] = 1
        map_values_as_mgz(template=input.varexp, data=varexp_mask, save_path=output.mask_mgz)

rule make_a_intersection_mask:
    input:
        r2 = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-r2_sub-{sub}_thres-{r2thres}_frame-{ref_frame}.mgz"),
        precision = os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "{hemi}.mask-precision_sub-{sub}_thres-{precisionthres}.mgz"),
    output:
        intersection_mask = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sub}_frame-{ref_frame}.mgz"),
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs
        r2_mask = load_mgzs(input.r2, fdata_only=True, squeeze=False).astype(int)
        precision_mask = load_mgzs(input.precision, fdata_only=True, squeeze=False).astype(int)
        intersection_mask = r2_mask & precision_mask
        map_values_as_mgz(template=input.r2, data=intersection_mask, save_path=output.intersection_mask)

rule save_all_subject_r2_figures:
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","allsub_view-{view}_thres-{thres}_value-{val}_frame-{ref_frame}.pdf")
    input:
        ss = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","view-{{view}}_thres-{{thres}}_sub-{sn}_value-{{val}}_frame-{{ref_frame}}.png"), sn=make_subj_list('nsdsyn'))
    run:
        from pysurfer.freeview_helper import plot_freeview_ss_two_rows
        from pysurfer.mgz_helper import extract_info_from_filename
        sn_list = [extract_info_from_filename(sn, 'sub')['sub'] for sn in input.ss]
        plot_freeview_ss_two_rows(input.ss,
                                  sn_list,
                                  suptitle=wildcards.val,
                                  save_path=output[0], dpi=500)

rule save_all_subject_precision_figures:
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","allsub_view-{view}_upthres-{thres}_value-{val}.pdf")
    input:
        ss = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","view-{{view}}_upthres-{{thres}}_sub-{sn}_value-{{val}}.png"), sn=make_subj_list('nsdsyn'))
    run:
        from pysurfer.freeview_helper import plot_freeview_ss_two_rows
        from pysurfer.mgz_helper import extract_info_from_filename
        sn_list = [extract_info_from_filename(sn, 'sub')['sub'] for sn in input.ss]
        plot_freeview_ss_two_rows(input.ss,
                                  sn_list,
                                  suptitle=wildcards.val,
                                  save_path=output[0], dpi=500)


rule mask_val_map:
    input:
        mask_mgz=os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sub}_frame-{ref_frame}.mgz"),
        val_map=os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}", "{hemi}.sub-{sub}_value-{val}_frame-{ref_frame}.mgz"),
    output:
        masked_val_map=os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "{hemi}.mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sub}_value-{val}_frame-{ref_frame}.mgz"),
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs
        varexp_mask = load_mgzs(input.mask_mgz, fdata_only=True, squeeze=False)
        val_map = load_mgzs(input.val_map, fdata_only=True,squeeze=False)
        val_map[varexp_mask == 0] = np.nan
        map_values_as_mgz(template=input.val_map, data=val_map, save_path=output.masked_val_map)

rule visualize_masked_mgz:
    output:
        #wang = os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss","base-sfp","roi-wang_view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_value-{val}_frame-{ref_frame}.png"),
        cat = os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss","base-sfp","roi-cat_view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_value-{val}_frame-{ref_frame}.png"),
    input:
        expand(os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{{dset}}", "{hemi}.mask-intersection_r2thres-{{r2thres}}_precisionthres-{{precisionthres}}_sub-{{sn}}_value-{{val}}_frame-{{ref_frame}}.mgz"), hemi=['lh','rh'])
    params:
        freesurfer_dir=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer"),
        rois=['V1v','V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4','FFA-1','FFA-2','PPA'],
        #rois=['V1v','V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4','wangVO1','wangVO2','wangPHC1','wangPHC2'],
        label_colors = [np.asarray([0,0,0])]*10
    run:
        from pysurfer.freeview_helper import make_custom_color_palettes_for_overlay, plot_mgz
        from pysurfer.mgz_helper import extract_info_from_filename
        from matplotlib.pyplot import get_cmap

        overlay_custom=make_custom_color_palettes_for_overlay(get_cmap('jet'), val_range=(0.1, 130), n=150, log_scale=True)
        kwargs = {'label_opacity': 1, 'label_outline': True, 'overlay_custom': overlay_custom}

        labels = [f'{roi}.label' for roi in params.rois]
        label_dir = os.path.join(params.freesurfer_dir, wildcards.sn, 'label')
        colorscale = False
        info = extract_info_from_filename(input[0])
        plot_mgz(params.freesurfer_dir, sn=wildcards.sn,
                 overlay=info['overlay'], overlay_dir=info['folder'],
                 labels=labels, label_dir=label_dir, label_colors=params.label_colors,
                 view=wildcards.view, colorscale=colorscale,
                 surf='inflated', save_path=output[0], **kwargs)

rule save_all_subject_sfp_figures:
    output:
        os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{dset}","ss", "base-sfp", "allsub_roi-cat_view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_value-{val}_frame-{ref_frame}.pdf")
    input:
        ss = expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","{{dset}}","ss","base-sfp","roi-cat_view-{{view}}_mask-intersection_r2thres-{{r2thres}}_precisionthres-{{precisionthres}}_sub-{sn}_value-{{val}}_frame-{{ref_frame}}.png"), sn=make_subj_list('nsdsyn'))
    run:
        from pysurfer.freeview_helper import plot_freeview_ss_two_rows
        from pysurfer.mgz_helper import extract_info_from_filename
        sn_list = [extract_info_from_filename(sn, 'sub')['sub'] for sn in input.ss]
        plot_freeview_ss_two_rows(input.ss,
                                  sn_list,
                                  suptitle=wildcards.val,
                                  save_path=output[0], dpi=500)
rule visualize_sfp:
    input:
        #expand(os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "{hemi}.mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sub}_value-{val}_frame-{ref_frame}.mgz"), dset='nsdsyn', hemi=['lh','rh'], sub=make_subj_list('nsdsyn'), val=['mode'], r2thres=[0.3], precisionthres=[2], ref_frame=['absolute']),
        expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","base-sfp","allsub_roi-cat_view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_value-{val}_frame-{ref_frame}.pdf"), view=['inferior','posterior'], r2thres=[0.3], precisionthres=[2], val=['mode'],  ref_frame=['absolute'])
        #expand(os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","base-sfp","view-{view}_mask-intersection_r2thres-{r2thres}_precisionthres-{precisionthres}_sub-{sn}_value-{val}_frame-{ref_frame}.png"), r2thres=[0.3], precisionthres=[2], view=['inferior','posterior'], sn=make_subj_list('nsdsyn'), val=['mode'],  ref_frame=['absolute']),


rule dataframes_for_each_ROI:
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_maps", "{dset}", "{hemi}.histogram_mask-precision_sub-{sub}_value-{val}_thres-{thres}_frame-{ref_frame}.png"),
    input:
        lh_masked_val_map=os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "lh.mask-precision_sub-{sub}_value-{val}_thres-{thres}_frame-{ref_frame}.mgz"),
        rh_masked_val_map=os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "rh.mask-precision_sub-{sub}_value-{val}_thres-{thres}_frame-{ref_frame}.mgz"),
    params:
        SUBJECTS_DIR=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer"),
        labels = ['V1v', 'V1d', 'V2v','V2d', 'V3v','V3d', 'hV4', 'aFFA', 'pFFA', 'PPA']
    run:
        from pysurfer.mgz_helper import get_vertices_in_labels, read_label, get_existing_labels_only
        lh_rois, rh_rois = {}, {}
        label_dir = os.path.join(params.SUBJECTS_DIR, wildcards.sub,'label')
        lh_labels, lh_label_paths = get_existing_labels_only(params.labels, label_dir, 'lh',return_paths=True)
        rh_labels, rh_label_paths = get_existing_labels_only(params.labels, label_dir, 'rh',return_paths=True)

        lh_rois[wildcards.sub] = get_vertices_in_labels(input.lh_masked_val_map,lh_label_paths,lh_labels,load_mgz=True,return_label=True)
        rh_rois[wildcards.sub] = get_vertices_in_labels(input.rh_masked_val_map,rh_label_paths,rh_labels,load_mgz=True,return_label=True)


#TODO: visualize goodness of fit maps

rule fsaverage_masked_val_map:
    input:
        masked_val_map = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-precision_sub-{sub}_value-{val}_thres-{thres}_frame-{ref_frame}.mgz"),
    output:
        fsaverage_masked_val_map = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-precision_space-fsaverage_sub-{sub}_value-{val}_thres-{thres}_frame-{ref_frame}.mgz"),
    params:
        SUBJECTS_DIR=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer")
    shell:
        """
        export SUBJECTS_DIR={params.SUBJECTS_DIR}
        mri_surf2surf --srcsubject {wildcards.sub} --trgsubject fsaverage --hemi {wildcards.hemi} --sval {input.masked_val_map} --tval {output.fsaverage_masked_val_map}
        """

rule average_subjects_brain_map:
    input:
        masked_mgzs=expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","{{hemi}}.mask-precision_space-fsaverage_sub-{sub}_value-{{val}}_thres-{{thres}}_frame-{{ref_frame}}.mgz"), sub=make_subj_list('nsdsyn')),
    output:
        avg_masked_mgz=os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "{dset}", "{hemi}.mask-precision_space-fsaverage_sub-fsaverage_value-{val}_thres-{thres}_frame-{ref_frame}.mgz")
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs
        avg_mgz = []
        for sub_mgz in input.masked_mgzs:
            tmp_mgz = load_mgzs(sub_mgz, fdata_only=True, squeeze=True)
            avg_mgz.append(tmp_mgz)
        avg_mgz = np.asarray(avg_mgz)
        avg_mgz = np.nanmean(avg_mgz, axis=0)
        map_values_as_mgz(template=sub_mgz, data=avg_mgz, save_path=output.avg_masked_mgz)

rule visualize_average_subjects_brain_map:
    input:
        lh_avg_masked_mgz = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","lh.mask-precision_space-fsaverage_sub-fsaverage_value-{val}_thres-{thres}_frame-{ref_frame}.mgz"),
        rh_avg_masked_mgz = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","rh.mask-precision_space-fsaverage_sub-fsaverage_value-{val}_thres-{thres}_frame-{ref_frame}.mgz")
    output:
        avg_masked_png = os.path.join(config['OUTPUT_DIR'],"figures", "sfp_maps","mgzs","{dset}","mask-precision_space-fsaverage_sub-fsaverage_value-{val}_thres-{thres}_frame-{ref_frame}.png")

rule visualize_precision_r2_mask:
    output:
        os.path.join(config['OUTPUT_DIR'], "figures", "sfp_maps", "mgzs", "{dset}", "mask-precision_sub-{sub}_value-{val}_thres-{thres}_frame-{ref_frame}.png")
    input:
        precision=os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","view-{{view}}_log-True_mask-{{mask}}_sub-{sn}_value-{{val}}_thres-{{thres}}_frame-{{ref_frame}}.png"),
        r2=os.path.join(config['OUTPUT_DIR'],"figures","sfp_maps","mgzs","nsdsyn","ss","view-{{view}}_log-True_mask-{{mask}}_sub-{sn}_value-{{val}}_thres-{{thres}}_frame-{{ref_frame}}.png")

rule map_to_fsaverage:
    input:
        mgz_path = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.sub-{sub}_value-{val}_frame-{ref_frame}.mgz"),
    output:
        os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.space-fsaverage_sub-{sub}_value-{val}_frame-{ref_frame}.mgz"),
    params:
        SUBJECTS_DIR=os.path.join(config['NSD_DIR'],"nsddata","freesurfer")
    shell:
        """
        export SUBJECTS_DIR={params.SUBJECTS_DIR}
        mri_surf2surf --srcsubject {wildcards.sub} --trgsubject fsaverage --hemi {wildcards.hemi} --sval {input.mgz_path} --tval {output}
        """

rule average_subjects:
    output:
        avg_mgz=os.path.join(config[
            'OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.avg_space-fsaverage_sub-fsaverage_value-{val}_frame-{ref_frame}.mgz"),
    input:
        subj_mgzs=expand(os.path.join(config[
            'OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","{{hemi}}.space-fsaverage_sub-{sub}_value-{{val}}_frame-{{ref_frame}}.mgz"),sub=make_subj_list('nsdsyn'))
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs

        avg_mgz = []
        for sub_mgz in input.subj_mgzs:
            tmp_mgz = load_mgzs(sub_mgz,fdata_only=True,squeeze=True)
            avg_mgz.append(tmp_mgz)
        avg_mgz = np.asarray(avg_mgz)
        avg_mgz = np.nanmean(avg_mgz,axis=0)
        map_values_as_mgz(template=sub_mgz,data=avg_mgz,save_path=output.avg_mgz)


rule fsaverage_all:
    input:
        expand(os.path.join(config[
            'OUTPUT_DIR'],"sfp_maps","mgzs","nsdsyn","{hemi}.avg_space-fsaverage_sub-fsaverage_value-{val}_frame-{ref_frame}.mgz"),hemi=[
            'lh', 'rh'],sub=make_subj_list('nsdsyn'),val=['r2'],ref_frame=['absolute']),
        #expand(os.path.join(config['OUTPUT_DIR'], "sfp_maps", "mgzs", "nsdsyn", "{hemi}.mask-precision_space-fsaverage_sub-fsaverage_value-{val}_thres-{thres}_frame-{ref_frame}.mgz"), hemi=['lh','rh'], sub=make_subj_list('nsdsyn'), thres=[0,2,4,6,8], val=['mode','r2','rmse'], ref_frame=['absolute', 'relative'])


rule r2_mask_to_fsaverage:
    output:
        fsaverage_mask_mgz=os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-r2_space-fsaverage_sub-{sub}_thres-{thres}_frame-{ref_frame}.mgz"),
    input:
        mask_mgz = os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.mask-r2_sub-{sub}_thres-{thres}_frame-{ref_frame}.mgz"),
    params:
        SUBJECTS_DIR=os.path.join(config['NSD_DIR'], "nsddata", "freesurfer")
    shell:
        """
        export SUBJECTS_DIR={params.SUBJECTS_DIR}
        mri_surf2surf --srcsubject {wildcards.sub} --trgsubject fsaverage --hemi {wildcards.hemi} --sval {input.mask_mgz} --tval {output.fsaverage_mask_mgz}
        """

rule average_r2_mask:
    output:
        avg_mask_mgz=os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{dset}","{hemi}.avg_mask-r2_space-fsaverage_sub-fsaverage_thres-{thres}_frame-{ref_frame}.mgz"),
    input:
        masks = expand(os.path.join(config['OUTPUT_DIR'],"sfp_maps","mgzs","{{dset}}","{{hemi}}.mask-r2_space-fsaverage_sub-{sub}_thres-{{thres}}_frame-{{ref_frame}}.mgz"), sub=make_subj_list('nsdsyn'))
    run:
        from pysurfer.mgz_helper import map_values_as_mgz
        from pysurfer.mgz_helper import load_mgzs
        avg_mask = []
        for mask_mgz in input.masks:
            tmp_mask = load_mgzs(mask_mgz,fdata_only=True,squeeze=True)
            avg_mask.append(tmp_mask)
        avg_mask = np.asarray(avg_mask)
        avg_mask = np.nanmean(avg_mask, axis=0)
        map_values_as_mgz(template=mask_mgz,data=avg_mask,save_path=output.avg_mask_mgz)