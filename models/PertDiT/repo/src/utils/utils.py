import os
import numpy as np 
import pandas as pd
import torch
import scanpy as sc
from tqdm import tqdm
from torchmetrics.regression import R2Score

def cal_r2(y_true,y_pred):
    dim=y_true.shape[0]
    try:
        metric = R2Score(num_outputs=dim, multioutput='raw_values')
    except (TypeError, ValueError):
        metric = R2Score(multioutput='raw_values')
    if dim==1:
        return metric(y_pred.permute(1,0), y_true.permute(1,0)).unsqueeze(0)
    return metric(y_pred.permute(1,0), y_true.permute(1,0))

def mse(y,f):
    mse = ((y - f)**2).mean(axis=1)
    return mse

def pearson(a,b):
    ab = np.concatenate((a, b),axis=1)
    dim=ab.shape[1]//2
    s=np.apply_along_axis(lambda x:np.corrcoef(x[0:dim],x[dim:2*dim])[0,1],axis=1,arr=ab)
    return torch.from_numpy(s)

def logfc_metric(true,y_pred,y_ctrl):
    eps = 1e-5
    pred = torch.log2((y_pred + eps) / (y_ctrl + eps))
    r2 = cal_r2(true.unsqueeze(0), pred.unsqueeze(0))
    cpp = pearson(true.unsqueeze(0), pred.unsqueeze(0))
    acc_signs = ((pred * true) > 0).sum() / len(true)
    return r2,cpp,acc_signs

def calculate_correlation_coefficients(obs, obs_key, x, y_true, y_pred):
    res_dict = {}
    column_name = obs[obs_key]  

    for index, key in tqdm(enumerate(column_name)):
        if index>=x.shape[0]:
            break
        if key not in res_dict:
            res_dict[key] = {'x': [], 'y_true': [], 'y_pred': []}
        res_dict[key]['x'].append(x[index])
        res_dict[key]['y_true'].append(y_true[index])
        res_dict[key]['y_pred'].append(y_pred[index])

    correlation_coefficients = []
    for key, values in tqdm(res_dict.items()):
        x_list = np.array(values['x']).mean(axis=0)+1e-6
        y_true_list = np.array(values['y_true']).mean(axis=0)+1e-6
        y_pred_list = np.array(values['y_pred']).mean(axis=0)+1e-6

        y_true_over_x = y_true_list - x_list
        y_pred_over_x = y_pred_list - x_list

        correlation = np.corrcoef(y_true_over_x, y_pred_over_x)[0, 1]
        correlation_coefficients.append(correlation)

    return np.array(correlation_coefficients), res_dict

def train_valid_test(adata, split_key = 'cell_type_split_0'):
    '''
    Get train_valid_test dataset
    '''
    train_index = adata.obs[(adata.obs[split_key] == 'train') & (adata.obs['dose']!= 0.0)].index.tolist()
    valid_index = adata.obs[(adata.obs[split_key] == 'valid') & (adata.obs['dose']!= 0.0)].index.tolist()
    test_index = adata.obs[(adata.obs[split_key] == 'test') & (adata.obs['dose']!= 0.0)].index.tolist()
    control_index = adata.obs[adata.obs['dose'] == 0.0].index.tolist()
    # print(len(train_index)+len(valid_index)+len(test_index)+len(control_index))

    if len(train_index)>0:
        train_index = train_index + control_index
        train_adata = adata[train_index, :]
    else:
        train_adata = None
    if len(valid_index)>0:
        valid_index = valid_index + control_index
        valid_adata = adata[valid_index, :]
    else:
        valid_adata=None
    if len(test_index)>0:
        test_index = test_index + control_index
        test_adata = adata[test_index, :]
    else:
        test_adata=None
    print(f'The size of train_data: {len(train_index)}; valid_data: {len(valid_index)}; test_data: {len(test_index)}; control_data: {len(control_index)}')
    return train_adata, valid_adata, test_adata