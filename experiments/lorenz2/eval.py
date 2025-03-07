import sys
import os
import functools
import h5py
import torch
import logging
import argparse
import shutil

from datetime import datetime
from tqdm import tqdm
from utils import *
from flowdas import ScoreNet, marginal_prob_std, Euler_Maruyama_sampler

import matplotlib.pyplot as plt
from glob import glob


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


def get_local_config():
    config = {
        'path_dataset1' : './data/dataset',
        'path_dataset2' : './data/dataset',
        'path_dataset3' : './data/dataset',
        'checkpoint_path_win1' : './runs_train/training_run_20250306_212902/best_model.pth',
        'checkpoint_path_win2' : './runs_train/training_run_20250306_212216/best_model.pth',
        'checkpoint_path_win3' : './runs_train/training_run_20250306_212948/best_model.pth',
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
        'N_trajectory': 32, #32,
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


def prepare_evaluation():
    set_seed(427)
    # Add the parent directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.insert(0, parent_dir)
    
    # Get default config and update with command line arguments
    config = get_local_config()
    args = parse_args()

    assert config['LT'] >= config['window'], "LT must be greater than or equal to window"
    
    # Update config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key in config:
            # Only update if the user explicitly specified this argument
            if key in args.__dict__ and args.__dict__[key] is not None:
                config[key] = value
            # logging.info(f"Setting {key} to {value} from command line arguments")
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    runpath = PATH / 'runs_eval' / f'run_{timestamp}_win={config["window"]}_N={config["N_trajectory"]}_L={config["LT"]}'
    runpath.mkdir(parents=True, exist_ok=True)
    config['runpath'] = runpath
    
    setup_evaluation_logging(runpath)

    # Log the configuration
    logging.info("Evaluation configuration:")
    for key, value in config.items():
        logging.info(f"  {key}: {value}")
    
    logging.info("\n\n")

    return config


def visualize(config):
    """
    Visualize the ground truth and estimated trajectories from h5 files.
    
    Args:
        config: Configuration dictionary containing runpath
    """
    logging.info("Visualizing the ground truth and estimated trajectories...")
    root_path = config['runpath']
    h5files = os.path.join(root_path, 'trajectory_*.h5')

    # Find all trajectory files in the specified directory
    trajectory_files = sorted(glob(h5files))

    if not trajectory_files:
        logging.info(f"No trajectory files found in {root_path}.")
        return

    # Create subfolders for trajectory plots and data
    plot_dir = os.path.join(root_path, 'trajectory_plot')
    data_dir = os.path.join(root_path, 'trajectory_data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    logging.info(f"Created directory for trajectory plots: {plot_dir}")
    logging.info(f"Created directory for trajectory data: {data_dir}")
    
    # Move all h5 files to the trajectory_data subfolder
    for h5_file in trajectory_files:
        filename = os.path.basename(h5_file)
        new_path = os.path.join(data_dir, filename)
        shutil.move(h5_file, new_path)
        trajectory_files[trajectory_files.index(h5_file)] = new_path
    logging.info(f"Moved all trajectory h5 files to {data_dir}")

    logging.info(f"Found {len(trajectory_files)} trajectory files.")

    # Process each trajectory file
    for traj_idx, traj_file in enumerate(trajectory_files):
        logging.info(f"Processing file: {traj_file}")
        
        with h5py.File(traj_file, 'r') as f:
            ground_truth = f['gt'][:]
            estimated = f['est'][:]
        
        # Create a figure with 3 subplots (one for each coordinate x, y, z)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Trajectory Comparison - {os.path.basename(traj_file)}', fontsize=16)
        
        coordinate_names = ['x', 'y', 'z']
        
        # Plot each coordinate in its own subplot
        for i in range(3):
            axes[i].plot(ground_truth[:, i], label='Ground Truth', color='blue')
            axes[i].plot(estimated[:, i], label='Estimated', color='red', linestyle='--')
            axes[i].set_ylabel(f'{coordinate_names[i]} coordinate')
            axes[i].legend()
            axes[i].grid(True)
        
        axes[2].set_xlabel('Time step')
        
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f'trajectory_comparison_{traj_idx+1}.png')
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
        
        logging.info(f"Plot saved as '{save_path}'")

    logging.info("All trajectory plots have been generated.")


def create_observations(config):
    """
    Create the observations for the combined-para dataset.
    The obs.h5 contains the first (L+1) steps of the testing trajectory.
    Input: 
        x.shape: (N, L+1, 3)
    Output: 
        obs.shape: (N, L+1, 1)
    """
    logging.info("Creating observations...")
    path_dataset = config[f'path_dataset{config["window"]}']
    
    # Read input data
    with h5py.File(f'{path_dataset}/test.h5', mode='r') as f:
        x = f['x'][:,:config['LT']+config['window']] # shape: (N, L+w, 3)
    
    # Delete existing obs.h5 if it exists
    if os.path.exists(f'{path_dataset}/obs_win={config["window"]}.h5'):
        os.remove(f'{path_dataset}/obs_win={config["window"]}.h5')
        
    # Create new obs.h5 file (no need to deal with window size)
    with h5py.File(f'{path_dataset}/obs_win={config["window"]}.h5', mode='w') as f:
        x = torch.from_numpy(x) # shape: (N, L+w, 3)
        obs = observation_generator(x, config['sigma_obs_hi']) # shape: (N, L+w, 1)
        # print('creating obs.shape: ',obs.shape)
        f.create_dataset('obs', data=obs)

    logging.info("Observations created successfully.\n\n")


def get_flow_prior(config):
    flow_prior = ScoreNet(
        marginal_prob_std=config['marginal_prob_std_fn'],
        x_dim=config['x_dim'],
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
    logging.info(f"Running Evaluation...")
    nrmse_all = []
    path_dataset = config[f'path_dataset{config["window"]}']
    for n in range(0, config['N_trajectory']):
        logging.info(f"Trajectory #: {1+n}/{config['N_trajectory']}")

        # Ground truth
        with h5py.File(f'{path_dataset}/test.h5', mode='r') as f:
            gt = torch.from_numpy(f['x'][n]).to(config['device']) 
            gt = gt[:config['LT']+config['window']] # shape: (L+1, 3)

        # Observation
        with h5py.File(f'{path_dataset}/obs_win={config["window"]}.h5', mode='r') as f:
            # TODO: deal with window size.
            obs = torch.from_numpy(f['obs'][n]).to(config['device']) # shape: (L+1, 1)

        flow_prior = get_flow_prior(config)

        gt_win, obs_win = gt, obs
        initial_cond = gt[:config['window']].reshape(1, -1)

        # Monte Carlo sampling
        cond_win = []
        cond_win.append(initial_cond)
        est_all_win = [gt[l, :].unsqueeze(0) for l in range(config['window'])]
        
        # Generating the trajectory
        for i in tqdm(range(config['LT']), desc="Monte Carlo sampling"): # why LT-1?
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
        est_all = est_all_win

        # Compute metrics
        est_all_tensor = torch.cat(est_all, dim=0)
        nrmse = compute_nrmse_LT(gt, est_all_tensor.detach(), config['LT'], config['window'])
        logging.info(f"FlowDAS - NRMSE: {nrmse.item()}")
        nrmse_all.append(nrmse)
        
        # Save ground truth and estimated trajectory to h5 file
        log_dir = os.path.dirname(logging.getLogger().handlers[0].baseFilename)
        h5_path = os.path.join(log_dir, f'trajectory_{n+1}.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('gt', data=gt.cpu().numpy())
            f.create_dataset('est', data=est_all_tensor.detach().cpu().numpy())
        logging.info(f"Saved trajectory data to {h5_path}.")
    
    averaged_nrmse = sum(nrmse_all)/config['N_trajectory']
    std_nrmse = torch.std(torch.stack(nrmse_all))
    logging.info(f"Averaged NRMSE of FlowDAS across {config['N_trajectory']} testing trajectories: {averaged_nrmse:.4f} ± {std_nrmse:.4f}\n\n")


if __name__ == "__main__":
    config = prepare_evaluation()
    
    create_observations(config)
    run_evaluation(config)
    visualize(config)