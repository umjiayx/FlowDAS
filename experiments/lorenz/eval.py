import sys
import os
import functools
import h5py
import numpy as np
import torch
import logging
import random
from datetime import datetime
from tqdm import tqdm

from utils import *
from flowdas import ScoreNet, marginal_prob_std, Euler_Maruyama_sampler

path_dataset = './data/dataset'

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
runpath = PATH / 'eval' / f'eval_run_{timestamp}'
runpath.mkdir(parents=True, exist_ok=True)

log_filename = f"eval_log_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"

log_filepath = runpath / log_filename
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(message)s",  # Format for the log messages
    handlers=[
        logging.FileHandler(log_filepath),  # Log to a file
        logging.StreamHandler()  # Also log to the terminal
    ]
)

# Add the parent directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, parent_dir)


def set_seed(seed=427):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_nrmse(gt, est, N_T):
    rmse = torch.sqrt(torch.sum((gt - est) ** 2) / N_T)
    denominator = torch.sqrt(torch.sum(gt ** 2) / N_T)
    nrmse = rmse / denominator

    return nrmse


def compute_mse(gt, est, N_T):
    mse = torch.sum((gt - est) ** 2) / N_T
    return mse


def create_observations():
    """
    Create the observations for the combined-para dataset.
    The obs.h5 contains the first (L+1) steps of the testing trajectory.
    """
    # Read input data
    L = 64 # length of the testing trajectory
    with h5py.File(f'{path_dataset}/test.h5', mode='r') as f:
        x = f['x'][:,:L+1]
    
    # Delete existing obs.h5 if it exists
    if os.path.exists(f'{path_dataset}/obs.h5'):
        os.remove(f'{path_dataset}/obs.h5')
        
    # Create new obs.h5 file
    with h5py.File(f'{path_dataset}/obs.h5', mode='w') as f:
        f.create_dataset('gt', data=x)


def run_evaluation():
    n_mc = 21 # Number of Monte Carlo samples
    step_size = 0.0002 # Step size
    freq = 'hi'
    N_trajectory = 32 # Number of testing trajectories
    l2_flowdas = []
    nrmse_all = []
    N_T = 15 # Number of testing states of each trajectory

    for i in range(0, N_trajectory):
        logging.info(f"Trajectory #: {1+i}/{N_trajectory}")

        # Observation
        with h5py.File(f'{path_dataset}/obs.h5', mode='r') as f:
            gt = torch.from_numpy(f['gt'][i]) # shape: (L+1, 3)

        y = torch.atan(gt)[:, :1] # shape: (L+1, 1)
        y = y + torch.normal(0, 0.25, size=y.shape) # shape: (L+1, 1)

        if freq == 'lo':
            sigma_obs, step = 0.05, 8
        else:
            sigma_obs, step = 0.25, 1
        
        #### Load and prepare the score network (FlowDAS) ####
        device = 'cuda:0'
        sigma = 25.0
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        x_dim = 3
        hidden_depth = 6
        embed_dim = 512
        use_bn = False
        extra_dim = 3

        flow_prior = ScoreNet(
            marginal_prob_std=marginal_prob_std_fn,
            x_dim=x_dim,
            extra_dim=extra_dim,
            hidden_depth=hidden_depth,
            embed_dim=embed_dim,
            use_bn=use_bn
        ).to(device)
        
        # Load the checkpoint
        #checkpoint_path = '../../rose-firefly-28_a5jxth6jcheckpoint29000.pth'
        #score_prior_old = load_checkpoint(score_prior_old, checkpoint_path)

        checkpoint_path = './runs_stochastic_gen_jiayx_right_3/training_runbest_model.pth'
        flow_prior = load_checkpoint(flow_prior, checkpoint_path)

        # Monte Carlo sampling
        #est_all = []
        est_all = []
        #est_all.append(gt[0, :].to(device))
        est_all.append(gt[0, :].to(device))

        for i in tqdm(range(N_T - 1), desc="Monte Carlo sampling"):
            x_t_gen = Euler_Maruyama_sampler(
                flow_prior,
                marginal_prob_std_fn,
                num_steps=600,
                device=device,
                cond=est_all[i],
                measurement=y[i + 1, :],
                noisy_level=sigma_obs,
                MC_times=n_mc, 
                batch_size=1, 
                step_size=step_size 
            )

            #est_all.append(x_t)
            est_all.append(x_t_gen)

        # Compute metrics
        est_all[0] = est_all[0].unsqueeze(0)
        est_all_tensor = torch.cat(est_all, dim=0)
        #l2_generalize = compute_nrmse(gt[:N_T], est_all_tensor.detach().cpu()[:N_T], N_T)

        nrmse = compute_nrmse(gt[:N_T], est_all_tensor.detach().cpu()[:N_T], N_T)
        #logging.info(f"FlowDAS - L2: {l2_separate.item()}")
        #l2_flowdas.append(l2_separate)
        logging.info(f"FlowDAS - NRMSE: {nrmse.item()}")
        nrmse_all.append(nrmse)
    
    # averaged_l2_flowdas = sum(l2_flowdas)/N_trajectory
    averaged_nrmse = sum(nrmse_all)/N_trajectory
    # logging.info(f"Averaged L2_FlowDAS for dataset {idx+1}: {averaged_l2_flowdas}")
    return averaged_nrmse # , averaged_l2_flowdas



if __name__ == "__main__":
    """
    For separate parameter settings.
    """
    # random.seed(427)
    set_seed(427)
    logging.info(f"############### Creating Observations and Running Evaluation ################")
    create_observations()
    averaged_nrmse = run_evaluation()
    # logging.info(f"Averaged L2_FlowDAS for combined dataset: {l2_combined}")
    logging.info(f"Averaged NRMSE of FlowDAS for the dataset: {averaged_nrmse}")

