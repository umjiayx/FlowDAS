r"""Lorenz experiment helpers"""

import os
import json
from pathlib import Path
from typing import *
import numpy as np
import random
import torch
import yaml
from mcs import *
from datetime import datetime
import argparse
import math

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/lorenz'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)

# From SDA
def to(x: Any, **kwargs) -> Any:
    if torch.is_tensor(x):
        return x.to(**kwargs)
    elif type(x) is list:
        return [to(y, **kwargs) for y in x]
    elif type(x) is tuple:
        return tuple(to(y, **kwargs) for y in x)
    elif type(x) is dict:
        return {k: to(v, **kwargs) for k, v in x.items()}
    else:
        return x

def get_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def make_chain() -> MarkovChain:
    return NoisyLorenz63(dt=0.025)

def make_chain_generalize(sigma, rho, beta) -> MarkovChain:
    return NoisyLorenz63Generalize(dt=0.025, sigma=sigma, rho=rho, beta=beta)

def get_Lorenz_parameters(coeff: float):
    sigma0_mean = 10.0
    rho0_mean = 28.0
    beta0_mean = 8 / 3

    # Physical parameters: Gaussian
    sigma0_variance = coeff * sigma0_mean  # User-controlled variance parameter
    rho0_variance = coeff * rho0_mean
    beta0_variance = coeff * beta0_mean

    sigma = np.random.normal(sigma0_mean, sigma0_variance)
    rho = np.random.normal(rho0_mean, rho0_variance)
    beta = np.random.normal(beta0_mean, beta0_variance)

    return sigma, rho, beta

def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path / 'config.json', mode='x') as f:
        json.dump(config, f)

def load_checkpoint(model,file_path="checkpoint.pth"):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def compute_nrmse_LT(gt, est, LT, window):
    '''
    Compute the normalized root mean square error (NRMSE) of the estimated trajectory
    with respect to the ground truth trajectory (the first LT states).
    gt: (L+w, 3)
    est: (L+w, 3)
    '''
    gt_LT = gt[window:LT+window]
    est_LT = est[window:LT+window]
    rmse = torch.sqrt(torch.mean((gt_LT - est_LT) ** 2))
    denominator = torch.sqrt(torch.mean(gt_LT ** 2))
    nrmse = rmse / denominator
    return nrmse


def compute_mse(gt, est, LT):
    mse = torch.sum((gt - est) ** 2) / LT
    return mse


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def prepare_generate():
    parser = argparse.ArgumentParser(description='Generate Lorenz datasets')
    parser.add_argument('--config', type=str, default='generate_Lorenz_data',
                        help='Name of the config file in the config directory')
    parser.add_argument('--num_datasets', type=int, 
                        help='Number of datasets to generate')
    parser.add_argument('--num_particles', type=int, 
                        help='Number of particles to generate')
    args = parser.parse_args()
    
    config_path = PATH / 'config' / f'{args.config}.yml'
    config = get_config(config_path)

    # Only update config if the arguments were explicitly provided
    if args.num_datasets is not None:
        config['num_datasets'] = args.num_datasets
    if args.num_particles is not None:
        config['num_particles'] = args.num_particles

    # Studying generalization of FlowDAS
    if config['study_generalizability']:
        config['data_dir'] = PATH / 'data_gen'
    else:
        config['data_dir'] = PATH / 'data'

    # Studying memorization and generalization of FlowDAS
    if config['study_Mem_Gen']:
        dataset_size = int(math.log2(config['num_datasets'])) + int(math.log2(config['num_particles']))
        config['data_dir'] = PATH / f'data_gen_memgen_datasize{dataset_size}'

    config['log_file_path'] = config['data_dir'] / config['log_file_name']

    # Create dataset directory if it doesn't exist
    if not config['data_dir'].exists():
        config['data_dir'].mkdir(parents=True, exist_ok=True)

    # Log the start time
    log_entry = f"Dataset generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"    
    with open(config['log_file_path'], 'a') as f:
        f.write(log_entry)

    return config