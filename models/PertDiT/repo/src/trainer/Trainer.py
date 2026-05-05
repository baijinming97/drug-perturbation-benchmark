import os
import numpy as np 
import torch
import torch.nn as nn
import yaml
import datetime
import scanpy as sc
from tqdm import tqdm
import logging
from utils.seed_everything import seed_everything
from utils.utils import *
from trainer.optimizer import get_optimizer_scheduler
from trainer.lossfunc_and_generator import choose_loss_generator
from sampler.Sampler import Diffusion_Sampler
from sampler.edm import EDM_Sampler
from dataset.my_Dataset import Choose_dataset_loader
from model.model_factory import Choose_model


class PertDit_Trainer:
    def __init__(self, config, ckpt=None, log_name = 'my_log', ablation_mode='none'):
        self.config = config
        self.ablation_mode = ablation_mode
        result_name = config['result_name']
        self.result_path = 'data/result/'+config['train']['split']+'/'+result_name
        self.split_path = 'data/result/'+config['train']['split']
        if not os.path.exists(self.result_path): 
            os.makedirs(self.result_path)  
        with open(self.result_path+'/config.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # logging setting
        logging.basicConfig(
            filename = self.result_path + '/' + log_name + '.log',  
            level = logging.DEBUG,  
            format = '%(asctime)s - %(levelname)s - %(message)s'  
        )

        # load data
        print("Loading data")
        pert_smiles_emb = torch.load('data/pert_smiles_emb.pkl')
        dosage_prompt_emb = torch.load('data/dosage_prompt_emb_lincs.pkl')
        lincs_adata = sc.read('data/lincs_adata.h5ad')
        lincs_adata.X = np.clip(lincs_adata.X, 0, 1e3)
        sc.pp.normalize_total(lincs_adata)
        sc.pp.log1p(lincs_adata)
        
        # build dataset and dataloader
        print("preparing dataset and dataloader")
        self.scale = True if result_name.startswith('EDM') else False
        if self.scale:
            print('start scaling')
            train_adata, _, _ = train_valid_test(lincs_adata, split_key = config['train']['split'])
            self.mean_value = np.mean(train_adata.X)
            self.std_value = np.std(train_adata.X)
            print(f"data mean: {self.mean_value} and data std: {self.std_value}")
            lincs_adata.X = (lincs_adata.X - self.mean_value) * (0.5 /  self.std_value)
            print('Finish scaling')

        train_adata, valid_adata, test_adata = train_valid_test(lincs_adata, split_key = config['train']['split'])
        self.device = torch.device(config['device'])
        myDataset_dosage, myDataloader = Choose_dataset_loader(config['drug_encoder'], self.device)
        if config['using_cfg']:
            print("Using classifier-free guidance")
            logging.info("Using classifier-free guidance")
        self.train_dataset = myDataset_dosage(train_adata, pert_smiles_emb, dosage_prompt_emb, self.device, cfg=config['using_cfg'], cfg_prob=config['diffusion']['cfg_prob'], FC = config['using_FC'])
        self.valid_dataset = myDataset_dosage(valid_adata, pert_smiles_emb, dosage_prompt_emb, self.device, FC = config['using_FC'])
        self.test_dataset = myDataset_dosage(test_adata, pert_smiles_emb, dosage_prompt_emb, self.device, FC = config['using_FC'])
        if config['valid_sampling_step']!=1:
            indices = [i for i in range(0, len(self.valid_dataset), config['valid_sampling_step'])]
            sampled_dataset = torch.utils.data.Subset(self.valid_dataset, indices)
            self.valid_dataset = sampled_dataset
        print(f"Length of train_dataset: {len(self.train_dataset)}, valid_dataset: {len(self.valid_dataset)}, test_dataset: {len(self.test_dataset)}")
        self.epochs = config['train']['epochs']
        self.batch_size = config['train']['batch_size']
        self.learning_rate = config['train']['lr']
        self.train_loader = myDataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = myDataloader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = myDataloader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # build model
        print("Initializing model")
        self.model = Choose_model(config)
        # Set drug feature ablation mode
        self.model.ablation_mode = self.ablation_mode
        if self.ablation_mode != 'none':
            print(f"[ABLATION] Drug feature ablation mode: {self.ablation_mode}")
            logging.info(f"Drug feature ablation mode: {self.ablation_mode}")
        # if ckpt is not None:
        #     self.model.load_state_dict(torch.load(ckpt), map_location = self.device)
        # else:
        self.model.init_weights()
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f'Total number of parameters: {total_params}')
        logging.info(self.model)
        self.model=self.model.to(self.device)

        # build optimizer and scheduler
        self.optimizer, self.scheduler = get_optimizer_scheduler(
            self.model.parameters(),
            config['scheduler']['lr_max'],
            config['scheduler']['warmup_n_steps'],
            config['scheduler']['lr_start'],
            config['scheduler']['T_max'],
            config['scheduler']['lr_min']
        )

        # build diffusion sampler
        if config['model']['model_type'].startswith("EDM"):
            self.sampler = EDM_Sampler(
                num_steps = config['diffusion']['num_steps'],
                sigma_min = config['diffusion']['sigma_min'],
                sigma_max = config['diffusion']['sigma_max'],
                rho = config['diffusion']['rho'],
                S_max = config['diffusion']['S_max'],
                cls_loss = config['diffusion']['cls_loss'],
                device=self.device
            )
        else:
            uncond = pert_smiles_emb['negative_ctrl'].mean(dim=0) if config['using_cfg'] else None
            loss_ratio = config['diffusion']['loss_ratio'] if 'loss_ratio' in config['diffusion'] else None
            if 'using_recons' in config and config['using_recons']:
                print(f"Using reconstruction loss and the loss ratio of classification is {loss_ratio}")
            self.sampler = Diffusion_Sampler(
                sampler_type=config['diffusion']['sampler_type'],
                num_train_timesteps=config['diffusion']['train_steps'],
                timesteps=config['diffusion']['num_steps'],
                start=config['diffusion']['beta_start'],
                end=config['diffusion']['beta_end'],
                beta_schedule=config['diffusion']['beta_schedule'],
                device=self.device,
                guidance_scale=config['diffusion']['guidance_scale'],
                uncond = uncond,
                loss_ratio = loss_ratio
            )
            logging.info(self.sampler.scheduler)

        self.train_batchs = min(config['train_batchs'],len(self.train_loader)) if config['train_batchs']!=-1 else len(self.train_loader)
        self.valid_batchs = min(config['valid_batchs'],len(self.valid_loader)) if config['valid_batchs']!=-1 else len(self.valid_loader)
        self.test_batchs = min(config['test_batchs'],len(self.test_loader)) if config['test_batchs']!=-1 else len(self.test_loader)
        self.relu = nn.ReLU(inplace=True)
        self.using_FC = config.get('using_FC', False)
        self.loss_func, self.generator_func = choose_loss_generator(config)

        # load checkpoint and resume training
        if ckpt is not None:
            print("Load checkpoint in: "+ ckpt)
            logging.info("Load checkpoint in: "+ ckpt)
            self.load_checkpoint(ckpt)
            self.skip = self.epoch+1
        else:
            self.skip = 0
            self.mse_predict = []
            self.train_losses = []
            self.epoch = 0
            self.best_epoch = -1
            self.best_mse = float('inf')
            self.best_r2 = float('-inf')
        

    def train(self):
        # skip trained epochs to keep reproducibility
        if self.skip>0:
            for _ in range(self.skip):
                for batch in self.train_loader:
                    break
        for self.epoch in range(self.skip, self.epochs):
            avg_loss = []
            logging.info('epoch {} start at: {}'.format(self.epoch, datetime.datetime.now()))
            print(self.result_path)
            self.model.train()
            loop = tqdm(enumerate(self.train_loader),total=self.train_batchs)
            for i, batch in loop:
                if i==self.train_batchs:
                    loop.set_description(f'Epoch [{self.epoch}/{self.epochs}] [{i}/{self.train_batchs}]')
                    break
                loss = self.loss_func(self.sampler, self.model, batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                self.scheduler.step()
                avg_loss.append(loss.item())
                if len(avg_loss)%100==0 and len(avg_loss)>0:
                    logging.info('epoch: {}, step: {}, loss: {}, time: {}, lr: {}'.format(
                        self.epoch,
                        len(avg_loss),
                        sum(avg_loss[-100:]) / 100,
                        datetime.datetime.now(),
                        self.optimizer.param_groups[0]['lr']
                    ))
                loop.set_description(f'Epoch [{self.epoch}/{self.epochs}] [{i}/{self.train_batchs}]')
                loop.set_postfix(Loss=loss.item())
            train_loss=sum(avg_loss)/len(avg_loss)
            self.train_losses.append(train_loss)
            logging.info('train loss = {}'.format(train_loss))
        
            # validation
            if self.epoch%1==0:
                self.eval()
                if self.epoch-self.best_epoch > self.config['train']['early_stopping_patience']-1 and self.epoch>self.config['train']['patience']:
                    logging.info(f"No improvement in last {self.config['train']['early_stopping_patience']} epoches triggering early stopping")
                    break
        
        # test
        logging.info('Load best model and start testing')
        self.eval(dataset='test', ckpt=self.result_path+'/PertDit_best.pth')

    @torch.no_grad()
    def eval(self, dataset='valid', ckpt=None):
        self.model.eval()
        if dataset=='valid':
            loader = self.valid_loader
            limit = self.valid_batchs
            logging.info('epoch {} sampling start at: {}'.format(self.epoch, datetime.datetime.now()))
        elif dataset=='test':
            if ckpt is not None:
                logging.info("Load checkpoint in: "+ ckpt)
                self.load_checkpoint(ckpt)
            loader = self.test_loader
            limit = self.test_batchs
            total_y_pred = torch.Tensor()
            total_y_true = torch.Tensor()
            total_x = torch.Tensor()
        else:
            raise NotImplementedError
        total_preds=torch.Tensor()
        total_cors=torch.Tensor()
        eval_loop = tqdm(enumerate(loader),total=limit)
        for i,batch in eval_loop:
            if i==limit:
                break
            y_pred = self.generator_func(self.sampler, self.model, batch)
            y_true = batch[0].cpu()
            x = batch[1].cpu()
            if self.scale:
                y_pred = y_pred*2*self.std_value+self.mean_value
                y_true = y_true*2*self.std_value+self.mean_value
                x = x*2*self.std_value+self.mean_value
            if not self.using_FC:
                y_pred = self.relu(y_pred)
            MSE_Loss = mse(y_true,y_pred)
            R2 = cal_r2(y_true,y_pred)
            total_preds = torch.cat((total_preds, MSE_Loss), 0)
            total_cors = torch.cat((total_cors, R2), 0)
            if dataset=='test':
                total_y_pred = torch.cat((total_y_pred, y_pred), 0)
                total_y_true = torch.cat((total_y_true, y_true), 0)
                total_x = torch.cat((total_x, x), 0)
            eval_loop.set_postfix(MSE_Loss=MSE_Loss.mean(), R2 = R2.mean())
        mse_now=total_preds.mean()
        cor_now=total_cors.mean()
        if dataset=='valid':
            self.save_checkpoint(self.result_path+'/PertDit_latest.pth')
            if mse_now < self.best_mse:
                self.save_checkpoint(self.result_path+'/PertDit_best.pth')
                self.best_epoch = self.epoch+1
                self.best_mse = mse_now
                self.best_r2 = cor_now
                logging.info('rmse improved at epoch {}; best_mse: {}; best_R2: {}'.format(self.best_epoch, self.best_mse, cor_now))
            else:
                logging.info('no improvement since epoch {}; best_mse: {}; best_R2: {}; now_mse: {}; R2: {}'.format(self.best_epoch, self.best_mse, self.best_r2, mse_now, cor_now))
            self.mse_predict.append(total_preds)
        else:
            # For FC mode: convert back to absolute expression before computing FC metrics
            if self.using_FC:
                eval_y_true = total_y_true + total_x
                eval_y_pred = total_y_pred + total_x
            else:
                eval_y_true = total_y_true
                eval_y_pred = total_y_pred
            coeff, _ = calculate_correlation_coefficients(self.test_dataset.drug_adata.obs, 'condition', total_x, eval_y_true, eval_y_pred)
            drug_mean_fc = coeff.mean()
            coeff, _ = calculate_correlation_coefficients(self.test_dataset.drug_adata.obs, 'cov_drug_name', total_x, eval_y_true, eval_y_pred)
            cov_mean_fc = coeff.mean()
            logging.info('Test metrics: Test MSE: {}; Test R2: {}; Test Drug_FC_PCC: {}; Test Cov_FC_PCC: {};'.format(mse_now, cor_now, drug_mean_fc, cov_mean_fc))
            torch.save(total_y_pred, self.result_path+'/total_y_pred.pkl')
            if not os.path.exists(self.split_path+'/common/total_x.pkl'):
                os.makedirs(self.split_path+'/common', exist_ok=True)
                torch.save(total_x, self.split_path+'/common/total_x.pkl')
                torch.save(total_y_true, self.split_path+'/common/total_y_true.pkl')
            
    def save_checkpoint(self, path):
        checkpoint = {
            'mse_predict': self.mse_predict,
            'train_losses': self.train_losses,
            'epoch': self.epoch,
            'best_epoch': self.best_epoch,
            'best_mse': self.best_mse,
            'best_r2': self.best_r2,
            'model_state_dict': self.model.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.mse_predict = checkpoint['mse_predict']
        self.train_losses = checkpoint['train_losses']
        self.epoch = checkpoint['epoch']
        self.best_epoch = checkpoint['best_epoch']
        self.best_mse = checkpoint['best_mse']
        self.best_r2 = checkpoint['best_r2']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model=self.model.to(self.device)
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def reset_exp2exp(self, t_add):
        self.sampler.t_add = t_add
        self.config['model']['model_type'] = 'exp2exp'
        self.loss_func, self.generator_func = choose_loss_generator(self.config)




