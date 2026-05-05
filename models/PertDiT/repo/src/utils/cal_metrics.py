import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import scanpy as sc
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
from torchmetrics.regression import R2Score

# keep the same as PRNet, calculate PCC of logFC and R2 grouped by cov_drug or cov_drug_dose with common cell type control 
def calculate_correlation_coefficients_celltype_control(obs, obs_key, x, y_true, y_pred, FC=False):
    res_dict = {}
    control_dict = defaultdict(list)
    column_name = obs[obs_key]  
    for index, key in tqdm(enumerate(column_name)):
        if key not in res_dict:
            res_dict[key] = {'y_true': [], 'y_pred': []}
        control_dict[str(key).split('_')[0]].append(x[index])
        if FC:
            res_dict[key]['y_true'].append(y_true[index]+x[index])
            res_dict[key]['y_pred'].append(y_pred[index]+x[index])
        else:
            res_dict[key]['y_true'].append(y_true[index])
            res_dict[key]['y_pred'].append(y_pred[index])
    control_dict_mean = {}
    for key, values in control_dict.items():
        control_dict_mean[key] = np.array(values).mean(axis=0)
    fc_pearson = []
    mean_r2 = []
    for key, values in tqdm(res_dict.items()):
        cell_type = str(key).split('_')[0]
        x_list = control_dict_mean[cell_type]
        y_true_list = np.array(values['y_true']).mean(axis=0)
        y_pred_list = np.array(values['y_pred']).mean(axis=0)
        y_true_over_x = y_true_list - x_list
        y_pred_over_x = y_pred_list - x_list
        correlation = np.corrcoef(y_true_over_x, y_pred_over_x)[0, 1]
        r2 = r2_score(y_true_list, y_pred_list)
        fc_pearson.append(correlation)
        mean_r2.append(r2)
    return np.array(fc_pearson), np.array(mean_r2)

# calculate PCC of logFC and R2 grouped by cov_drug, cov_drug_dose, drug or drug_dose with 1-to-1 control
def calculate_correlation_coefficients(obs, obs_key, x, y_true, y_pred, FC=False, cal_R2=True):
    res_dict = {}
    column_name = obs[obs_key]  

    for index, key in tqdm(enumerate(column_name)):
        if key not in res_dict:
            res_dict[key] = {'x': [], 'y_true': [], 'y_pred': []}
        res_dict[key]['x'].append(x[index])
        if FC:
            res_dict[key]['y_true'].append(y_true[index]+x[index])
            res_dict[key]['y_pred'].append(y_pred[index]+x[index])
        else:
            res_dict[key]['y_true'].append(y_true[index])
            res_dict[key]['y_pred'].append(y_pred[index])

    correlation_coefficients = []
    mean_r2 = []
    for key, values in tqdm(res_dict.items()):
        x_list = np.array(values['x']).mean(axis=0)
        y_true_list = np.array(values['y_true']).mean(axis=0)
        y_pred_list = np.array(values['y_pred']).mean(axis=0)
        y_true_over_x = y_true_list - x_list
        y_pred_over_x = y_pred_list - x_list
        correlation = np.corrcoef(y_true_over_x, y_pred_over_x)[0, 1]
        correlation_coefficients.append(correlation)
        if cal_R2:
            r2 = r2_score(y_true_list, y_pred_list)
            mean_r2.append(r2)

    return np.array(correlation_coefficients), np.array(mean_r2)

def cal_r2(y_true,y_pred):
    dim=y_true.shape[0]
    metric = R2Score(num_outputs=dim, multioutput='raw_values')
    if dim==1:
        return metric(y_pred.permute(1,0), y_true.permute(1,0)).unsqueeze(0)
    return np.array(metric(y_pred.permute(1,0), y_true.permute(1,0))).mean()

def cal_pearson(a,b):
    ab = np.concatenate((a, b),axis=1)
    dim=ab.shape[1]//2
    s=np.apply_along_axis(lambda x:np.corrcoef(x[0:dim],x[dim:2*dim])[0,1],axis=1,arr=ab)
    return s.mean()

# calculate all 12 metrics including R2 and PCC Fold-change grouped by drug, drug_dose, cov_drug, cod_drug_dose 
def cal_r2_pearsonfc_all_record(split_key, folder, FC=False, common_path = 'data/result'):
    total_y_pred = torch.load(folder +'/total_y_pred.pkl')
    total_y_true = torch.load(common_path + '/' + split_key +'/common/total_y_true.pkl')
    total_x = torch.load(common_path + '/' + split_key+ '/common/total_x.pkl')
    lincs_adata = sc.read('data/lincs_adata.h5ad')
    test_adata = lincs_adata[(lincs_adata.obs[split_key] == 'test') & (lincs_adata.obs['dose']!= 0.0)]
    obs = test_adata.obs.copy()
    obs['condition_dose'] = obs.cov_drug_dose_name.apply(lambda x: '_'.join(str(x).split('_')[1:]))

    if FC:
        r2 = cal_r2(total_x+total_y_true, total_x+total_y_pred)
        pearson = cal_pearson(total_x+total_y_true, total_x+total_y_pred)
    else:
        r2 = cal_r2(total_y_true, total_y_pred)
        pearson = cal_pearson(total_y_true, total_y_pred)
    res=[r2, pearson]
    print(r2, pearson)

    p,r = calculate_correlation_coefficients_celltype_control(obs, 'cov_drug_dose_name',  total_x, total_y_true, total_y_pred, FC=FC)
    res = res+[p.max(), p.min(), p.mean(), np.median(p),r.max(), r.min(), r.mean(), np.median(r)]
    print(p.max(), p.min(), p.mean(), np.median(p))
    print(r.max(), r.min(), r.mean(), np.median(r))

    p,r = calculate_correlation_coefficients_celltype_control(obs, 'cov_drug_name',  total_x, total_y_true, total_y_pred, FC=FC)
    res = res+[p.max(), p.min(), p.mean(), np.median(p),r.max(), r.min(), r.mean(), np.median(r)]
    print(p.max(), p.min(), p.mean(), np.median(p))
    print(r.max(), r.min(), r.mean(), np.median(r))

    p, r = calculate_correlation_coefficients(obs, 'condition_dose', total_x, total_y_true, total_y_pred, FC=FC)
    res = res+[p.max(), p.min(), p.mean(), np.median(p),r.max(), r.min(), r.mean(), np.median(r)]
    print(p.max(), p.min(), p.mean(), np.median(p))
    print(r.max(), r.min(), r.mean(), np.median(r))

    p, r = calculate_correlation_coefficients(obs, 'cov_drug_dose_name', total_x, total_y_true, total_y_pred, FC=FC, cal_R2=False)
    res = res+[p.max(), p.min(), p.mean(), np.median(p)]
    print(p.max(), p.min(), p.mean(), np.median(p))

    p, r = calculate_correlation_coefficients(obs, 'condition', total_x, total_y_true, total_y_pred, FC=FC)
    res = res+[p.max(), p.min(), p.mean(), np.median(p),r.max(), r.min(), r.mean(), np.median(r)]
    print(p.max(), p.min(), p.mean(), np.median(p))
    print(r.max(), r.min(), r.mean(), np.median(r))

    p, r = calculate_correlation_coefficients(obs, 'cov_drug_name', total_x, total_y_true, total_y_pred, FC=FC, cal_R2=False)
    res = res+[p.max(), p.min(), p.mean(), np.median(p)]
    print(p.max(), p.min(), p.mean(), np.median(p))

    if FC:
        r2 = cal_r2(total_y_true, total_y_pred)
        pearson = cal_pearson(total_y_true, total_y_pred)
    else:
        r2 = cal_r2(total_y_true-total_x, total_y_pred-total_x)
        pearson = cal_pearson(total_y_true-total_x, total_y_pred-total_x)
    res = res+[r2, pearson]
    print(r2, pearson)

    return  total_x, total_y_true, total_y_pred, obs, res

def cal_fc_only(split_key, folder, FC=False, common_path = 'data/result'):
    total_y_pred = torch.load(folder +'/total_y_pred.pkl')
    total_y_true = torch.load(common_path + '/' + split_key +'/common/total_y_true.pkl')
    total_x = torch.load(common_path + '/' + split_key+ '/common/total_x.pkl')
    lincs_adata = sc.read('data/lincs_adata.h5ad')
    test_adata = lincs_adata[(lincs_adata.obs[split_key] == 'test') & (lincs_adata.obs['dose']!= 0.0)]
    obs = test_adata.obs.copy()
    obs['condition_dose'] = obs.cov_drug_dose_name.apply(lambda x: '_'.join(str(x).split('_')[1:]))
    if FC:
        r2 = cal_r2(total_y_true, total_y_pred)
        pearson = cal_pearson(total_y_true, total_y_pred)
    else:
        r2 = cal_r2(total_y_true-total_x, total_y_pred-total_x)
        pearson = cal_pearson(total_y_true-total_x, total_y_pred-total_x)
    print(r2, pearson)
    return [r2, pearson]

# calculate all metrics of a method in all 15 splits
def cal_metrics_of_one_method_all(method_name):
    splits = ['random_', 'drug_', 'cell_type_']
    splits_num = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4']
    res_dict = {}
    for split_key in splits:
        for split_num in splits_num:
            split = split_key+split_num
            folder_path = "data/result/" + split + "/" + method_name
            pkl_path = os.path.join(folder_path, "total_y_pred.pkl")
            if os.path.exists(pkl_path):
                print(split)
                _, _, _, _, res = cal_r2_pearsonfc_all_record(split, folder_path)
                res_dict[split] = np.array(res)[[0,1,4,8,12,16,20,24,28,32,36,40,42,43]]
            else:
                print(f"{folder_path} has no test results")
            print('------------------------------------')
    return res_dict

def cal_metrics_of_one_method(method_name):
    if os.path.exists('data/res_tables/'+method_name+'_all_splits.csv'):
        res_pd = pd.read_csv('data/res_tables/'+method_name+'_all_splits.csv',index_col=0)
    else:
        return cal_metrics_of_one_method_all(method_name)
    splits = ['random_', 'drug_', 'cell_type_']
    splits_num = ['split_0', 'split_1', 'split_2', 'split_3', 'split_4']
    res_dict = {}
    for split_key in splits:
        for split_num in splits_num:
            split = split_key+split_num
            folder_path = "data/result/" + split + "/" + method_name
            pkl_path = os.path.join(folder_path, "total_y_pred.pkl")
            if os.path.exists(pkl_path) and split not in res_pd.index:
                print(split)
                _, _, _, _, res = cal_r2_pearsonfc_all_record(split, folder_path)
                res_dict[split] = np.array(res)[[0,1,4,8,12,16,20,24,28,32,36,40,42,43]]
            elif not os.path.exists(pkl_path):
                print(f"{folder_path} has no test results")
            else:
                print(f"{folder_path} has already been recorded")
                res_dict[split] = np.array(res_pd.loc[split,:])
                if res_dict[split].shape[0]<14:
                    res_dict[split] = np.append(res_dict[split], cal_fc_only(split, folder_path))
            print('------------------------------------')
    return res_dict

# calculate all metrics of a method in my_split
def cal_metrics_of_one_method_mysplit_all(method_name):
    splits = ['Drug_unseen', 'Cell_line_unseen', 'Both_unseen']
    res_dict = {}
    for split in splits:
        folder_path = "data/result/" + split + "/" + method_name
        pkl_path = os.path.join(folder_path, "total_y_pred.pkl")
        if os.path.exists(pkl_path):
            print(split)
            _, _, _, _, res = cal_r2_pearsonfc_all_record(split, folder_path)
            res_dict[split] = np.array(res)[[0,1,4,8,12,16,20,24,28,32,36,40,42,43]]
        else:
            print(f"{folder_path} has no test results")
        print('------------------------------------')
    return res_dict

def cal_metrics_of_one_method_mysplit(method_name):
    if os.path.exists('data/res_tables/'+method_name+'_mysplit.csv'):
        res_pd = pd.read_csv('data/res_tables/'+method_name+'_mysplit.csv',index_col=0)
    else:
        return cal_metrics_of_one_method_mysplit_all(method_name)
    splits = ['Drug_unseen', 'Cell_line_unseen', 'Both_unseen']
    res_dict = {}
    for split in splits:
        folder_path = "data/result/" + split + "/" + method_name
        pkl_path = os.path.join(folder_path, "total_y_pred.pkl")
        if os.path.exists(pkl_path) and split not in res_pd.index:
            print(split)
            _, _, _, _, res = cal_r2_pearsonfc_all_record(split, folder_path)
            res_dict[split] = np.array(res)[[0,1,4,8,12,16,20,24,28,32,36,40,42,43]]
        elif not os.path.exists(pkl_path):
            print(f"{folder_path} has no test results")
        else:
            print(f"{folder_path} has already been recorded")
            res_dict[split] = np.array(res_pd.loc[split,:])
            if res_dict[split].shape[0]<14:
                res_dict[split] = np.append(res_dict[split], cal_fc_only(split, folder_path))
        print('------------------------------------')
    return res_dict

# calculate all metrics of methods in a split folder
def cal_metrics_in_folders_all(split_name):
    parent_folder = 'data/result/'+split_name
    res_dict = {}
    for root, dirs, _ in os.walk(parent_folder):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            pkl_path = os.path.join(folder_path, "total_y_pred.pkl")
            if os.path.exists(pkl_path):
                print(dir_name)
                if dir_name.endswith('FC'):
                    _, _, _, _, res = cal_r2_pearsonfc_all_record(split_name, folder_path, FC=True)
                else:
                    _, _, _, _, res = cal_r2_pearsonfc_all_record(split_name, folder_path)
                res_dict[dir_name] = np.array(res)[[0,1,4,8,12,16,20,24,28,32,36,40,42,43]]
            else:
                print(f"{folder_path} has no test results")
    return res_dict

def cal_metrics_in_folders(split_name, parent_folder=None):
    if os.path.exists('data/res_tables/'+split_name+'_all_methods.csv'):
        res_pd = pd.read_csv('data/res_tables/'+split_name+'_all_methods.csv',index_col=0)
    else:
        return cal_metrics_in_folders_all(split_name)
    if parent_folder is None:
        parent_folder = 'data/result/'+split_name
    res_dict = {}
    for root, dirs, _ in os.walk(parent_folder):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            pkl_path = os.path.join(folder_path, "total_y_pred.pkl")
            if os.path.exists(pkl_path) and dir_name not in res_pd.index:
                print(dir_name)
                if dir_name.endswith('FC'):
                    _, _, _, _, res = cal_r2_pearsonfc_all_record(split_name, folder_path, FC=True)
                else:
                    _, _, _, _, res = cal_r2_pearsonfc_all_record(split_name, folder_path)
                res_dict[dir_name] = np.array(res)[[0,1,4,8,12,16,20,24,28,32,36,40,42,43]]
            elif not os.path.exists(pkl_path):
                print(f"{folder_path} has no test results")
            else:
                print(f"{folder_path} has already been recorded")
                res_dict[dir_name] = np.array(res_pd.loc[dir_name,:])
                if res_dict[dir_name].shape[0]<14:
                    if dir_name.endswith('FC'):
                        FC=True
                    else:
                        FC=False
                    res_dict[dir_name] = np.append(res_dict[dir_name], cal_fc_only(split_name, folder_path, FC=FC))
    return res_dict 