import os
import sys
import time
import json
from datetime import datetime
import torch
import numpy as np
import argparse
from models import DeepCE
from utils import DataReader
from utils import rmse, correlation, precision_k

start_time = datetime.now()

parser = argparse.ArgumentParser(description='DeepCE Training')
parser.add_argument('--drug_file', required=True)
parser.add_argument('--gene_file', required=True)
parser.add_argument('--train_file', required=True)
parser.add_argument('--dev_file', required=True)
parser.add_argument('--test_file', required=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--ablation', type=str, default='none', choices=['none', 'zero', 'shuffle'],
                    help='Drug feature ablation: none (default), zero, or shuffle')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory. Auto-generated with timestamp if not specified.')
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

drug_file = args.drug_file
gene_file = args.gene_file
dropout = args.dropout
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
batch_size = args.batch_size
max_epoch = args.max_epoch
patience = args.patience
ablation_mode = args.ablation

# output directory
if args.output_dir is None:
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results',
                              f'{run_timestamp}_{ablation_mode}')
else:
    output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 4
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse'
intitializer = torch.nn.init.xavier_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

data = DataReader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                  gene_expression_file_test, filter, device)
print('#Train: %d' % len(data.train_feature['drug']))
print('#Dev: %d' % len(data.dev_feature['drug']))
print('#Test: %d' % len(data.test_feature['drug']))

# model creation
model = DeepCE(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=len(filter['cell_id']),
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)
model.to(device)
model = model.double()

# print config
config_info = {
    'ablation': ablation_mode,
    'max_epoch': max_epoch,
    'patience': patience,
    'batch_size': batch_size,
    'dropout': dropout,
    'lr': 0.0002,
    'loss_type': loss_type,
    'filter': filter,
    'train_samples': len(data.train_feature['drug']),
    'dev_samples': len(data.dev_feature['drug']),
    'test_samples': len(data.test_feature['drug']),
    'output_dir': output_dir,
}
print('Config:', json.dumps(config_info, indent=2))
with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config_info, f, indent=2)

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
best_dev_pearson = float("-inf")
best_dev_epoch = -1
epochs_no_improve = 0
save_best_predictions = False

pearson_list_dev = []
pearson_list_test = []
spearman_list_dev = []
spearman_list_test = []
rmse_list_dev = []
rmse_list_test = []
precisionk_list_dev = []
precisionk_list_test = []

for epoch in range(max_epoch):
    print("Iteration %d:" % (epoch+1))
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True)):
        ft, lb = batch
        drug = ft['drug']
        mask = ft['mask']
        if data.use_pert_type:
            pert_type = ft['pert_type']
        else:
            pert_type = None
        if data.use_cell_id:
            cell_id = ft['cell_id']
        else:
            cell_id = None
        if data.use_pert_idose:
            pert_idose = ft['pert_idose']
        else:
            pert_idose = None
        optimizer.zero_grad()
        predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
                        ablation_mode=ablation_mode)
        loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Train loss: %.6f' % (epoch_loss/(i+1)))

    model.eval()

    # --- Dev evaluation ---
    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='dev', batch_size=batch_size, shuffle=False)):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
                            ablation_mode=ablation_mode)
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        print('Dev loss: %.6f' % (epoch_loss / (i + 1)))
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_dev.append(rmse_score)
        print('RMSE: %.4f' % rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_dev.append(pearson)
        print('Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_dev.append(spearman)
        print('Spearman\'s correlation: %.4f' % spearman)
        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_list_dev.append(precision)

        # Early stopping check
        if pearson > best_dev_pearson:
            best_dev_pearson = pearson
            best_dev_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            save_best_predictions = True
            print('*** New best dev Pearson: %.4f, model saved ***' % pearson)
        else:
            epochs_no_improve += 1
            print('No improvement for %d epoch(s) (best: %.4f at epoch %d)' %
                  (epochs_no_improve, best_dev_pearson, best_dev_epoch + 1))

    # --- Test evaluation ---
    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose,
                            ablation_mode=ablation_mode)
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
        print('Test loss: %.6f' % (epoch_loss / (i + 1)))
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_test.append(rmse_score)
        print('RMSE: %.4f' % rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_test.append(pearson)
        print('Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_test.append(spearman)
        print('Spearman\'s correlation: %.4f' % spearman)
        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_list_test.append(precision)

        # Save test predictions at best dev epoch
        if save_best_predictions:
            np.save(os.path.join(output_dir, 'test_labels.npy'), lb_np)
            np.save(os.path.join(output_dir, 'test_predictions.npy'), predict_np)
            save_best_predictions = False

    # Early stopping break
    if epochs_no_improve >= patience:
        print('Early stopping at epoch %d (patience=%d)' % (epoch + 1, patience))
        break

# --- Final summary ---
print('\n' + '='*60)
print('Training finished. Best dev epoch: %d' % (best_dev_epoch + 1))
print('='*60)

print("Epoch %d got best Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, pearson_list_dev[best_dev_epoch]))
print("Epoch %d got Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, spearman_list_dev[best_dev_epoch]))
print("Epoch %d got RMSE on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_dev[best_dev_epoch]))
print("Epoch %d got P@100 POS and NEG on dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_dev[best_dev_epoch][-1][0],
                                                                  precisionk_list_dev[best_dev_epoch][-1][1]))

print("Epoch %d got Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_list_test[best_dev_epoch]))
print("Epoch %d got Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_list_test[best_dev_epoch]))
print("Epoch %d got RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_test[best_dev_epoch]))
print("Epoch %d got P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_test[best_dev_epoch][-1][0],
                                                                  precisionk_list_test[best_dev_epoch][-1][1]))

best_test_epoch = np.argmax(pearson_list_test)
print("Epoch %d got best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, pearson_list_test[best_test_epoch]))
print("Epoch %d got Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, spearman_list_test[best_test_epoch]))
print("Epoch %d got RMSE on test set: %.4f" % (best_test_epoch + 1, rmse_list_test[best_test_epoch]))
print("Epoch %d got P@100 POS and NEG on test set: %.4f, %.4f" % (best_test_epoch + 1,
                                                                  precisionk_list_test[best_test_epoch][-1][0],
                                                                  precisionk_list_test[best_test_epoch][-1][1]))

end_time = datetime.now()
elapsed = end_time - start_time
print('Total time: %s' % str(elapsed))

# Save results
results = {
    'ablation': ablation_mode,
    'best_dev_epoch': best_dev_epoch + 1,
    'total_epochs': epoch + 1,
    'elapsed_time': str(elapsed),
    'best_dev': {
        'pearson': pearson_list_dev[best_dev_epoch],
        'spearman': spearman_list_dev[best_dev_epoch],
        'rmse': rmse_list_dev[best_dev_epoch],
        'precision_at_100_pos': precisionk_list_dev[best_dev_epoch][-1][0],
        'precision_at_100_neg': precisionk_list_dev[best_dev_epoch][-1][1],
    },
    'test_at_best_dev': {
        'pearson': pearson_list_test[best_dev_epoch],
        'spearman': spearman_list_test[best_dev_epoch],
        'rmse': rmse_list_test[best_dev_epoch],
        'precision_at_100_pos': precisionk_list_test[best_dev_epoch][-1][0],
        'precision_at_100_neg': precisionk_list_test[best_dev_epoch][-1][1],
    },
    'best_test': {
        'epoch': int(best_test_epoch + 1),
        'pearson': pearson_list_test[best_test_epoch],
        'spearman': spearman_list_test[best_test_epoch],
        'rmse': rmse_list_test[best_test_epoch],
        'precision_at_100_pos': precisionk_list_test[best_test_epoch][-1][0],
        'precision_at_100_neg': precisionk_list_test[best_test_epoch][-1][1],
    },
    'history': {
        'pearson_dev': [float(x) for x in pearson_list_dev],
        'pearson_test': [float(x) for x in pearson_list_test],
        'spearman_dev': [float(x) for x in spearman_list_dev],
        'spearman_test': [float(x) for x in spearman_list_test],
        'rmse_dev': [float(x) for x in rmse_list_dev],
        'rmse_test': [float(x) for x in rmse_list_test],
    }
}
with open(os.path.join(output_dir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to %s' % output_dir)
