import os
import argparse 
from datetime import datetime
from unittest import result
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import scanpy as sc
from data._utils import pearson_mean, pearson_list, r2_mean, mse_mean, contribution_df, computecs, condition_fc_groups_by_cov


def parse_args():
    parse = argparse.ArgumentParser(description='perturbation-conditioned generative model')  
    parse.add_argument('--split_key', default='random_split_0', type=str, help='split key of data')  
    args = parse.parse_args()  
    return args



if __name__ == "__main__":
    args_train = parse_args()
    start_time = datetime.now()


    config_kwargs = {
        'save_dir' : './results/lincs/',
        'results_dir' : './results/lincs/',
        'split_key' : args_train.split_key,
        'x_dimension' : 978,
        'obs_key' : 'cov_drug_name',
    } 

    results_dict = {}
    results_dict['split_key'] = config_kwargs['split_key']

    print('Loading data...')
    pre_array = np.genfromtxt(config_kwargs['save_dir']+config_kwargs['split_key']+'_y_pre_array.csv', delimiter=',')
    true_array = np.genfromtxt(config_kwargs['save_dir']+config_kwargs['split_key']+'_y_true_array.csv', delimiter=',')
    control_array = np.genfromtxt(config_kwargs['save_dir']+config_kwargs['split_key']+'_x_true_array.csv', delimiter=',')

    f = open(config_kwargs['save_dir']+config_kwargs['split_key']+'cov_drug_array.csv',"r")
    lines = f.readlines()
    cov_drug = [x.strip() for x in lines]
    print('********Loading data complete**************')

    pre_df = pd.DataFrame(pre_array, index=cov_drug)
    true_df = pd.DataFrame(true_array, index=cov_drug)
    control_df = pd.DataFrame(control_array, index=cov_drug)

    pre_df = contribution_df(pre_df)
    true_df = contribution_df(true_df)
    control_df = contribution_df(control_df)
    control_df['cov_drug_name'] = control_df.cell_type.astype(str) + '_' + 'DMSO'
    control_df['condition'] = 'DMSO'

    all_pre_df = pd.concat([pre_df, control_df], ignore_index=True)
    all_true_df = pd.concat([true_df, control_df], ignore_index=True)


    print('********calculating condition_exp_mean, fold change**************')
    condition_exp_mean_pre, control_exp_mean_pre, fold_change_pre = condition_fc_groups_by_cov(all_pre_df, groupby='cov_drug_name', covariate='cell_type', control_group='DMSO')

    condition_exp_mean_t, control_exp_mean_t, fold_change_t = condition_fc_groups_by_cov(all_true_df, groupby='cov_drug_name', covariate='cell_type', control_group='DMSO')

    condition_exp_mean_pre_df = pd.DataFrame.from_dict(condition_exp_mean_pre, orient='index')
    condition_exp_mean_t_df = pd.DataFrame.from_dict(condition_exp_mean_t, orient='index')

    control_exp_mean_t_df = pd.DataFrame.from_dict(control_exp_mean_t, orient='index')

    fold_change_pre_df = pd.DataFrame.from_dict(fold_change_pre, orient='index')
    fold_change_t_df = pd.DataFrame.from_dict(fold_change_t, orient='index')

    print('********calculating condition pearson or r2**************')

    results_dict['condition_r2_score'] = r2_mean(condition_exp_mean_pre_df.to_numpy(), condition_exp_mean_t_df.to_numpy())

    results_dict['foldchange_pearson'] = pearson_mean(fold_change_pre_df.to_numpy(), fold_change_t_df.to_numpy())

    print('********saving results**************')
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(config_kwargs['results_dir']+config_kwargs['split_key']+"_results.csv")
  