from model.EDM_AdaDit import EDM_Adaformer
from model.EDM_Cross import EDM_Crossformer
from model.EDM_BasicCross import EDM_BasicCrossformer
from model.CrossDit import Crossformer, CatCrossformer
from model.DirectCrossDit import DirectCrossformer
from model.AdaDit import DirectAdaformer, Adaformer, Adaformer_wide_deep
from model.Cross_UNet import cross_attention_unet
from model.PRNet import PRNet

def Choose_model(config):
    #["EDM", "EDMCross", "Ada", "DirectAda", "CatCross", "Cross", "DirectCross", "CrossU", "PR"]
    model_type = config['model']['model_type']
    if model_type == "EDM":
        model = EDM_Adaformer(
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "EDMCross":
        model = EDM_Crossformer(
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "EDMBasicCross":
        model = EDM_BasicCrossformer(
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "Ada":
        model = Adaformer(
            num_steps = config['diffusion']['train_steps'],
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "Adawd":
        model = Adaformer_wide_deep(
            num_steps = config['diffusion']['train_steps'],
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "DirectAda":
        model = DirectAdaformer(
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "CatCross":
        model = CatCrossformer(
            num_steps = config['diffusion']['train_steps'],
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
            block_type = config['model']['block_type'],
        )
    elif model_type == "Cross":
        if 'no_norm_pre_x' in config['model'] and config['model']['no_norm_pre_x']:
            print('no pre_x norm')
            nonorm = True
        else:
            nonorm = False
        model = Crossformer(
            num_steps = config['diffusion']['train_steps'],
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
            nonorm = nonorm
        )
    elif model_type == "DirectCross":
        model = DirectCrossformer(
            num_layers = config['model']['num_layers'],
            d_model = config['model']['d_model'],
            d_cond = config['model']['d_cond'],
            d_pre = config['model']['d_pre'],
            mlp_hidden_dim = config['model']['mlp_hidden_dim'],
            num_heads = config['model']['num_heads'],
            dropout = config['model']['dropout'],
        )
    elif model_type == "CrossU":
        model = cross_attention_unet(
            input_dim = config['model']['d_pre'],
            channel_dim = config['model']['channels'],
            strides = config['model']['strides'],
            drug_input_dim = config['model']['d_cond'],
            n_heads = config['model']['num_heads'],
            num_steps = config['diffusion']['train_steps']
        )
    elif model_type == "PR":
        model = PRNet()
    else:
        raise NotImplementedError
    return model
