import os
import torch
import yaml
import logging
import argparse
from datetime import datetime
from trainer.Trainer import PertDit_Trainer
from utils.seed_everything import seed_everything

def parse_args():
    parse = argparse.ArgumentParser(description='PertDit training and testing')
    parse.add_argument('--cfg', default='Ada', type=str, help='Select in [Ada, Cross, CrossUNet, DirectAda, DirectCross, PRNet]')
    parse.add_argument('--train', default='train', type=str, help='Select in [train, test]')
    parse.add_argument('--folder', default=None, type=str, help='a result folder that contains ckpt and config.yaml used for training or testing')
    parse.add_argument('--ablation', default='none', type=str, choices=['none', 'zero', 'shuffle'],
                        help='Drug feature ablation: none (baseline), zero, or shuffle')
    parse.add_argument('--split', default=None, type=str,
                        help='Override split in config (e.g. random_split_0, drug_split_0)')
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args_train = parse_args()
    seed_everything(117)
    if args_train.folder is not None:
        config_path = args_train.folder + '/' + 'config.yaml'
        if args_train.train == 'train':
            ckpt = args_train.folder + '/' + 'PertDit_latest.pth'
        else:
            ckpt = args_train.folder + '/' + 'PertDit_best.pth'
    else:
        config_path = "config/" + args_train.cfg + ".yaml"
        ckpt = None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override split if specified
    if args_train.split is not None:
        config['train']['split'] = args_train.split

    # Append ablation suffix to result_name to avoid overwriting baseline results
    ablation_mode = args_train.ablation
    if ablation_mode != 'none':
        config['result_name'] = config['result_name'] + '_ablation_' + ablation_mode

    # avoid cover the original results
    if os.path.exists('data/result/'+config['train']['split']+'/'+config['result_name'] + '/PertDit_best.pth') and args_train.train == 'train' and ckpt is None:
        print('Results already exist, please correct the result name in the config yaml file')
        os._exit(0)
    # initialize trainer
    now = datetime.now()
    time_str = now.strftime('%H_%M_%S')
    log_name = f'{args_train.train}_at_{time_str}'
    my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt, ablation_mode=ablation_mode)
    torch.set_num_threads(4)
    my_trainer.train()

    # After training, test with best model
    ckpt = 'data/result/'+config['train']['split']+'/'+config['result_name'] + '/PertDit_best.pth'
    now = datetime.now()
    time_str = now.strftime('%H_%M_%S')
    log_name = f'test_at_{time_str}'
    my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt, ablation_mode=ablation_mode)
    my_trainer.eval(dataset='test')
