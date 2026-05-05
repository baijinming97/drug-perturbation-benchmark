"""
PertDiT reproduction runner.
Uses original unmodified code with configurable seed and split.

Usage:
    python run_repro.py --split random_split_0 --seed 117 --run_name CrossDiT_repro_s117
"""
import os
import sys
import yaml
import torch
import argparse
from datetime import datetime
from utils.seed_everything import seed_everything

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--seed', type=int, default=117)
    parser.add_argument('--run_name', required=True)
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    # Load base config
    with open('config/Cross.yaml') as f:
        config = yaml.safe_load(f)

    config['train']['split'] = args.split
    config['result_name'] = args.run_name

    # Check existing results
    result_path = f"data/result/{args.split}/{args.run_name}/PertDit_best.pth"
    if os.path.exists(result_path):
        print(f'[SKIP] Results already exist: {result_path}')
        return

    print(f'Split: {args.split}, Seed: {args.seed}, Run: {args.run_name}')

    # Import trainer (deferred to after seed is set)
    from trainer.Trainer import PertDit_Trainer

    torch.set_num_threads(4)
    now = datetime.now()
    log_name = f'train_at_{now.strftime("%H_%M_%S")}'

    trainer = PertDit_Trainer(config, log_name=log_name, ckpt=None)
    print('Start training')
    trainer.train()  # includes test at end

    print(f'[DONE] {args.split} seed={args.seed}')

if __name__ == '__main__':
    main()
