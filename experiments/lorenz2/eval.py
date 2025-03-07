import sys
import os
import functools
import h5py
import numpy as np
import torch
import logging
import argparse

from datetime import datetime
from tqdm import tqdm
from utils import *
from flowdas import ScoreNet, marginal_prob_std, Euler_Maruyama_sampler


def setup_evaluation_logging(runpath):
    log_filename = f"eval_log_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    log_filepath = runpath / log_filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )


def prepare_evaluation():
    set_seed(427)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runpath = PATH / 'runs_eval' / f'eval_run_{timestamp}'
    runpath.mkdir(parents=True, exist_ok=True)
    setup_evaluation_logging(runpath)
    logging.info(f"############### Running Evaluation ################")

    # Add the parent directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.insert(0, parent_dir)


def get_local_config():
    config = {
        'path_dataset' : './data/dataset',
        'checkpoint_path_win1' : '',
        'checkpoint_path_win2' : '',
        'checkpoint_path_win3' : '',
        'marginal_prob_std_fn': functools.partial(marginal_prob_std, sigma=25.0), # 25?
        'device': 'cuda:0',
        'window': 2,
        'x_dim': 3,
        'extra_dim': 3,
        'hidden_depth': 5,
        'embed_dim': 384,
        'use_bn': False,
        'N_MC': 21, # 21
        'step_size': 0.0002,
        'num_steps': 600,
        'freq': 'hi',
        'N_trajectory': 1, #32,
        'LT': 15, # 15, # Number of testing states of each trajectory
        'sigma_obs_hi': 0.25,
        'sigma_obs_lo': 0.05,
        'prev_stats_as_cond': True
    }
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='FlowDAS Evaluation')
    # parser.add_argument('--checkpoint_path', type=str, default='./runs_train/training_run_250304_205913/best_model.pth', help='Path to model checkpoint')
    # parser.add_argument('--device', type=str, default='cuda:0', help='Device to run evaluation on')
    # parser.add_argument('--N_trajectory', type=int, default=32, help='Number of trajectories to evaluate')
    # parser.add_argument('--LT', type=int, default=15, help='Number of testing states of each trajectory')
    # parser.add_argument('--path_dataset', type=str, default='./data/dataset', help='Path to dataset')
    parser.add_argument('--window', type=int, required=True, help='Window size')
    return parser.parse_args()


def create_observations(config):
    """
    Create the observations for the combined-para dataset.
    The obs.h5 contains the first (L+1) steps of the testing trajectory.
    Input: 
        x.shape: (N, L+1, 3)
    Output: 
        obs.shape: (N, L+1, 1)
    """
    # Read input data
    with h5py.File(f'{config["path_dataset"]}/test.h5', mode='r') as f:
        x = f['x'][:,:config['LT']+config['window']] # shape: (N, L+w, 3)
    
    # Delete existing obs.h5 if it exists
    if os.path.exists(f'{config["path_dataset"]}/obs.h5'):
        os.remove(f'{config["path_dataset"]}/obs.h5')
        
    # Create new obs.h5 file (no need to deal with window size)
    with h5py.File(f'{config["path_dataset"]}/obs.h5', mode='w') as f:
        x = torch.from_numpy(x) # shape: (N, L+w, 3)
        obs = observation_generator(x, config['sigma_obs_hi']) # shape: (N, L+w, 1)
        # print('creating obs.shape: ',obs.shape)
        f.create_dataset('obs', data=obs)


def get_flow_prior(config):
    flow_prior = ScoreNet(
        marginal_prob_std=config['marginal_prob_std_fn'],
        x_dim=config['x_dim']*config['window'],
        extra_dim=config['extra_dim']*config['window'],
        hidden_depth=config['hidden_depth'],
        embed_dim=config['embed_dim'],
        use_bn=config['use_bn']
    ).to(config['device'])

    try:
        ckp_path = config[f'checkpoint_path_win{config["window"]}']
    except KeyError:
        raise ValueError(f"Window size {config['window']} not supported")

    flow_prior = load_checkpoint(flow_prior, ckp_path)
    return flow_prior


def construct_new_cond(current_cond, x_t_gen):
    """
    Input:
        current_cond.shape: (1, 3*window)
        x_t_gen.shape: (1, 3)
    Output:
        new_cond.shape: (1, 3*window)
    """
    tmp = current_cond[:, 3:].clone()
    new_cond = torch.cat([tmp, x_t_gen], dim=1)
    return new_cond

def run_evaluation(config):
    nrmse_all = []

    for n in range(0, config['N_trajectory']):
        logging.info(f"Trajectory #: {1+n}/{config['N_trajectory']}")

        # Ground truth
        with h5py.File(f'{config["path_dataset"]}/test.h5', mode='r') as f:
            gt = torch.from_numpy(f['x'][n]).to(config['device']) 
            gt = gt[:config['LT']+config['window']] # shape: (L+1, 3)

        # Observation
        with h5py.File(f'{config["path_dataset"]}/obs.h5', mode='r') as f:
            # TODO: deal with window size.
            obs = torch.from_numpy(f['obs'][n]).to(config['device']) # shape: (L+1, 1)

        flow_prior = get_flow_prior(config)

        # Deal with window size.
        # gt_win.shape: (L+1-window+1, 3*window)
        # obs_win.shape: (L+1-window+1, 1)
        # gt_win, obs_win = get_obs_win(gt, obs, config['window']) 
        gt_win, obs_win = gt, obs
        initial_cond = gt[:config['window']].reshape(1, -1)

        # Monte Carlo sampling
        cond_win = []
        cond_win.append(initial_cond)
        est_all_win = [gt[l, :].unsqueeze(0) for l in range(config['window'])]
        
        # Generating the trajectory
        for i in tqdm(range(config['LT'] - 1), desc="Monte Carlo sampling"):
            x_t_gen = Euler_Maruyama_sampler(
                flow_prior,
                num_steps=config['num_steps'],
                device=config['device'],
                base=est_all_win[i+config['window']-1],
                cond=cond_win[i],
                measurement=obs_win[i+1, :],
                noisy_level=config['sigma_obs_hi'],
                MC_times=config['N_MC'], 
                batch_size=1, # TODO: why batch size is 1?
                step_size=config['step_size'] 
            ) # shape: (B, 3*window)

            est_all_win.append(x_t_gen)
            new_cond = construct_new_cond(cond_win[i], x_t_gen)
            cond_win.append(new_cond)

        # Deal with window size: (B, 3*window) -> (B, 3)
        # est_all = [est_all_win[i][:, -3:] for i in range(len(est_all_win))]
        est_all = est_all_win

        # Compute metrics
        est_all_tensor = torch.cat(est_all, dim=0)
        nrmse = compute_nrmse_LT(gt, est_all_tensor.detach(), config['LT'])
        logging.info(f"FlowDAS - NRMSE: {nrmse.item()}")
        nrmse_all.append(nrmse)
    
    averaged_nrmse = sum(nrmse_all)/config['N_trajectory']
    logging.info(f"Averaged NRMSE of FlowDAS across {config['N_trajectory']} testing trajectories: {averaged_nrmse}")


if __name__ == "__main__":
    prepare_evaluation()
    
    # Get default config and update with command line arguments
    config = get_local_config()
    args = parse_args()
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            # Only update if the user explicitly specified this argument
            if key in args.__dict__ and args.__dict__[key] is not None:
                config[key] = value
            # logging.info(f"Setting {key} to {value} from command line arguments")
    
    # Log the configuration
    logging.info("Evaluation configuration:")
    for key, value in config.items():
        #logging.info(f"  {key}: {value}")
        pass
    
    create_observations(config)
    run_evaluation(config)