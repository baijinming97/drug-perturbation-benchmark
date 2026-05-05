import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(res_pd, title="Random_split_0"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(res_pd, annot=True, cmap='coolwarm')
    plt.title("Heatmap of " + title)
    plt.show()

def plot_heatmap_change(res_pd, fig_size = (8, 6), title = "Heatmap of Random_split_0"):
    plt.figure(figsize=fig_size)
    sns.heatmap(res_pd.sort_index(), annot=False, cmap='RdBu_r', center=0)
    plt.title(title)
    plt.show()

# select rows if method3 is not complete
def plot_res_partial_seperate(col, method1_data, method2_data, method3_data, method_names = ['PRNet', 'AdaDiT', 'CrossDiT']):
    all_data = np.column_stack([np.array(method1_data.iloc[:,col]).reshape(3,5), np.array(method2_data.iloc[:,col]).reshape(3,5)]).reshape(3,2,5)
    part_data = []
    part_data.append(np.row_stack((all_data[0],method3_data.iloc[:,col].values[0:5])))
    part_data.append(np.row_stack((all_data[1],method3_data.iloc[:,col].values[5:10])))
    part_data.append(np.row_stack((all_data[2][:,0:3], method3_data.iloc[:,col].values[10:13])))
    print(part_data)
    method_names = ['PRNet', 'AdaDiT', 'CrossDiT']
    experiments = ['Random Split', 'Unseen drugs', 'Unseen cell lines']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    max_val=-1
    for i, ax in enumerate(axes):
        data = part_data[i]
        max_val = max(max_val, data.max())
        for j in range(len(data)):
            color = plt.cm.tab10(j)
            ax.bar(j, data[j].mean(), width=0.5, label=method_names[j], yerr=data[j].std(), capsize=5, color=color)
            for _, value in enumerate(data[j]):
                ax.scatter(j, value, color=color, edgecolors='k')
        ax.set_title(experiments[i])
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(method_names)
    fig.text(0.05, 0.5, method1_data.columns[col], va='center', rotation='vertical', fontsize=12)
    plt.show()

def plot_res_seperate(col, method1_data, method2_data, method3_data, method_names = ['PRNet', 'AdaDiT', 'CrossDiT']):
    all_data = np.column_stack([np.array(method1_data.iloc[:,col]).reshape(3,5), 
                                np.array(method2_data.iloc[:,col]).reshape(3,5), 
                                np.array(method3_data.iloc[:,col]).reshape(3,5)]).reshape(3,3,5)
    experiments = ['Random Split', 'Unseen drugs', 'Unseen cell lines']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    max_val=-1
    for i, ax in enumerate(axes):
        data = all_data[i]
        max_val = max(max_val, data.max())
        for j in range(len(data)):
            color = plt.cm.tab10(j)
            ax.bar(j, data[j].mean(), width=0.5, label=method_names[j], yerr=data[j].std(), capsize=5, color=color)
            for _, value in enumerate(data[j]):
                ax.scatter(j, value, color=color, edgecolors='k')
        ax.set_title(experiments[i])
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(method_names)
    fig.text(0.05, 0.5, method1_data.columns[col], va='center', rotation='vertical', fontsize=12)
    plt.show()

def plot_res(col, method1_data, method2_data, method3_data, method_names=['PRNet', 'CatCrossDiT', 'CrossDiT'],ax = None):
    all_data = np.column_stack([np.array(method1_data.iloc[:, col]).reshape(3, 5),
                                np.array(method2_data.iloc[:, col]).reshape(3, 5),
                                np.array(method3_data.iloc[:, col]).reshape(3, 5)]).reshape(3, 3, 5)
    experiments = ['Random Split', 'Unseen drugs', 'Unseen cell lines']
    bar_width = 0.2
    bar_gap = 0.05
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))
    else:
        fig = ax.get_figure()
    max_val = -1
    for i in range(len(all_data)):
        data = all_data[i]
        max_val = max(max_val, data.max())
        for j in range(len(data)):
            color = plt.cm.tab10(j)
            bar_positions = np.arange(len(experiments)) + (j - 1) * bar_width + bar_gap
            ax.bar(bar_positions[i], data[j].mean(), width=bar_width, label=method_names[j], yerr=data[j].std(),
                   capsize=5, color=color)
            for _, value in enumerate(data[j]):
                ax.scatter(bar_positions[i], value, color=color, edgecolors='k')

    ax.set_xticks(np.arange(len(experiments))  + bar_gap)
    ax.set_xticklabels(experiments)
    ax.set_title(method1_data.columns[col])
    ax.set_ylim(0, max_val * 1.1)
    if col==9:
        handles, labels = ax.get_legend_handles_labels()
        print(handles, labels)
        fig.legend(handles[:3], labels[:3], loc='center right',bbox_to_anchor=(0.98, 0.5))
        fig.text(0.1, 0.5, 'Value of Metrics', va='center', rotation='vertical', fontsize=12)

def plot_res_mysplit_seperate(col, method1_data, method2_data, method3_data, method_names = ['PRNet', 'AdaDiT', 'CrossDiT']):
    all_data = np.column_stack([np.array(method1_data.iloc[:,col]).reshape(3,1), 
                                np.array(method2_data.iloc[:,col]).reshape(3,1), 
                                np.array(method3_data.iloc[:,col]).reshape(3,1)])
    experiments = ['Unseen drugs', 'Unseen cell lines', 'Both unseen']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    max_val=-1
    for i, ax in enumerate(axes):
        data = all_data[i]
        max_val = max(max_val, data.max())
        for j in range(len(data)):
            color = plt.cm.tab10(j)
            ax.bar(j, data[j], width=0.5, label=method_names[j], capsize=5, color=color)
        ax.set_title(experiments[i])
        ax.set_xticks(np.arange(len(data)))
        ax.set_xticklabels(method_names)
    fig.text(0.05, 0.5, method1_data.columns[col], va='center', rotation='vertical', fontsize=12)
    plt.show()

def plot_res_mysplit(col, method_datas, method_names=['PRNet', 'ChemCPA', 'CatCrossDiT', 'CrossDiT'],ax = None):
    num_methods = len(method_datas)
    all_data = np.column_stack([np.array(method_datas[i].iloc[:,col]).reshape(3,1) for i in range(num_methods)])
    experiments = ['Unseen drugs', 'Unseen cell lines', 'Both unseen']
    bar_width = 0.2
    bar_gap = 0.05
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))
    else:
        fig = ax.get_figure()
    max_val = -1
    colors = [plt.cm.tab10(3),plt.cm.tab10(0),plt.cm.tab10(1),plt.cm.tab10(2)]
    for i in range(len(all_data)):
        data = all_data[i]
        max_val = max(max_val, data.max())
        for j in range(len(data)):
            color = colors[j]
            bar_positions = np.arange(len(experiments)) + (j - 1) * bar_width + bar_gap
            ax.bar(bar_positions[i], data[j], width=bar_width, label=method_names[j],
                   capsize=5, color=color)
    ax.set_xticks(np.arange(len(experiments))+ bar_gap+np.array([(j - 1) * bar_width for j in range(len(data))]).mean())
    ax.set_xticklabels(experiments)
    ax.set_title(method_datas[0].columns[col])
    ax.set_ylim(0, max_val * 1.1)
    if col==9:
        handles, labels = ax.get_legend_handles_labels()
        print(handles, labels)
        fig.legend(handles[:num_methods], labels[:num_methods], loc='center right',bbox_to_anchor=(0.98, 0.5))
        fig.text(0.1, 0.5, 'Value of Metrics', va='center', rotation='vertical', fontsize=12)

def cmp_methods(method_name, base_name):
    title = method_name+'_vs_'+base_name
    new_data = pd.read_csv('data/res_tables/'+method_name+'_all_splits.csv',index_col=0)
    base_data = pd.read_csv('data/res_tables/'+base_name+'_all_splits.csv',index_col=0)
    common_indices = new_data.index.intersection(base_data.index)
    result_df = new_data.loc[common_indices].sub(base_data.loc[common_indices])
    print(f"Sum of changes is {result_df.sum()}")
    plot_heatmap_change(result_df.iloc[:,[0,3,5,7,10,13,8,11,6,9]],title=title)

def cmp_methods_mysplit(method_name, base_name, title=None, print_changes=False):
    if title is None:
        title = method_name+'_vs_'+base_name
    new_data = pd.read_csv('data/res_tables/'+method_name+'_mysplit.csv',index_col=0)
    base_data = pd.read_csv('data/res_tables/'+base_name+'_mysplit.csv',index_col=0)
    common_indices = new_data.index.intersection(base_data.index)
    result_df = new_data.loc[common_indices].sub(base_data.loc[common_indices])
    if print_changes:
        print(f"Sum of changes is {result_df.sum()}")
    plot_heatmap_change(result_df.iloc[:,[0,3,5,7,10,13,8,11,6,9]],title=title)

def get_cmp_methods_mysplit_res(method_name, base_name, title=None):
    if title is None:
        title = method_name+'_vs_'+base_name
    new_data = pd.read_csv('data/res_tables/'+method_name+'_mysplit.csv',index_col=0)
    base_data = pd.read_csv('data/res_tables/'+base_name+'_mysplit.csv',index_col=0)
    common_indices = new_data.index.intersection(base_data.index)
    result_df = new_data.loc[common_indices].sub(base_data.loc[common_indices])
    return result_df.iloc[:,[0,3,5,7,10,13,8,11,6,9]], title