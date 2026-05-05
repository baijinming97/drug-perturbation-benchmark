import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(lr_max, warmup_n_steps, lr_start, T_max, lr_min, optimizer):
    if warmup_n_steps > 0:
        wu_b = lr_start / lr_max
        wu_k = (1-wu_b) / warmup_n_steps
    else:
        wu_k, wu_b = 0, 1
    def lr_lambda(cur_iter):
        if cur_iter < warmup_n_steps:
            return wu_k * cur_iter + wu_b
        elif cur_iter >= warmup_n_steps and cur_iter <= T_max:
            return (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - warmup_n_steps) / (T_max - warmup_n_steps) * math.pi))) / lr_max
        else:
            return lr_min / lr_max
    return LambdaLR(optimizer, lr_lambda = lr_lambda)

def get_optimizer_scheduler(trainable_params, lr_max, warmup_n_steps, lr_start, T_max, lr_min):
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_params,
        lr=lr_max,
        betas=(0.9, 0.999),
        weight_decay=1.0e-2,
        eps=1.0e-8,
    )
    scheduler = get_warmup_scheduler(lr_max, warmup_n_steps, lr_start, T_max, lr_min, optimizer)
    return optimizer, scheduler