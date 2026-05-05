import os
import yaml
import shutil
import argparse 
from datetime import datetime
from trainer.Trainer import PertDit_Trainer
from utils.seed_everything import seed_everything

def parse_args():
    parse = argparse.ArgumentParser(description='PertDit training and testing')  
    parse.add_argument('--cfg', default='Ada', type=str, help='Select in [Ada, Cross, CrossUNet, DirectAda, DirectCross, PRNet]') 
    parse.add_argument('--new', default=False, type=bool)  
    parse.add_argument('--folder', default=None, type=str, help='a result folder that contains ckpt and config.yaml used for training or testing')  
    args = parse.parse_args()  
    return args

if __name__ == "__main__":
    args_test = parse_args()
    seed_everything(117)
    if args_test.folder is not None:
        config_path = args_test.folder + '/' + 'config.yaml'
        ckpt = args_test.folder + '/' + 'PertDit_best.pth'
    else:
        config_path = "config/" + args_test.cfg + ".yaml"
        ckpt = None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    ckpt = 'data/result/'+config['train']['split']+'/'+config['result_name'] + '/PertDit_best.pth'

    if not os.path.exists('data/result/'+config['train']['split']+'/'+config['result_name'] + '/PertDit_best.pth'):
        print('No checkpoint')
        os._exit(0)

    if args_test.new:
        now = datetime.now()
        time_str = now.strftime('%H_%M_%S')
        log_name = f'_test_at_{time_str}'
        config['result_name'] = config['result_name'] + log_name
        print('New test in folder ' + config['result_name'])
        res_folder = 'data/result/'+config['train']['split']+'/'+config['result_name']
        os.makedirs(res_folder) 
        try:
            shutil.copy(ckpt, res_folder)
            print("Copy success")
        except FileNotFoundError:
            print("Copy failure")
    
    if not os.path.exists('data/result/'+config['train']['split']+'/'+config['result_name'] + '/total_y_pred.pkl'):
        now = datetime.now()
        time_str = now.strftime('%H_%M_%S')
        log_name = f'test_at_{time_str}'
        my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt)
        my_trainer.eval(dataset='test')
    
    config['train']['split'] = "Drug_unseen"
    if not os.path.exists('data/result/'+config['train']['split']+'/'+config['result_name'] + '/total_y_pred.pkl'):
        now = datetime.now()
        time_str = now.strftime('%H_%M_%S')
        log_name = f'test_at_{time_str}'
        my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt)
        my_trainer.eval(dataset='test')

    config['train']['split'] = "Cell_line_unseen"
    if not os.path.exists('data/result/'+config['train']['split']+'/'+config['result_name'] + '/total_y_pred.pkl'):
        now = datetime.now()
        time_str = now.strftime('%H_%M_%S')
        log_name = f'test_at_{time_str}'
        my_trainer = PertDit_Trainer(config, log_name=log_name, ckpt=ckpt)
        my_trainer.eval(dataset='test')
