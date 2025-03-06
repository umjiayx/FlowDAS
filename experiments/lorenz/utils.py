r"""Lorenz experiment helpers"""

import os
import json
from pathlib import Path
from typing import *
import numpy as np
import random
import torch

from mcs import *
from datetime import datetime


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


def compute_nrmse_LT(gt, est, LT):
    '''
    Compute the normalized root mean square error (NRMSE) of the estimated trajectory
    with respect to the ground truth trajectory (the first LT states).
    '''
    gt_LT = gt[:LT]
    est_LT = est[:LT]
    rmse = torch.sqrt(torch.sum((gt_LT - est_LT) ** 2) / LT)
    denominator = torch.sqrt(torch.sum(gt_LT ** 2) / LT)
    nrmse = rmse / denominator
    return nrmse


def compute_mse(gt, est, LT):
    mse = torch.sum((gt - est) ** 2) / LT
    return mse


def set_seed(seed=427):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)