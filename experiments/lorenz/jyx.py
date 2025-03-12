
import os
import sys
import h5py
import torch
import numpy as np
import logging
from utils import *


config = {
    'path_dataset': '/home/jiayx/projects/FlowDAS/experiments/lorenz/data_gen/dataset',
    'LT': 15,
    'window': 1,
    'sigma_obs_hi': 0.25,
}





def create_observations(config):
    """
    Create observation files for the combined-para dataset.
    
    The obs.h5 file contains the first (L+1) steps of the testing trajectory.
    
    Args:
        config: Configuration dictionary with parameters
        
    Input data shape: 
        x: (N, L+1, 3) - N trajectories with L+1 timesteps of 3D coordinates
        
    Output data shape:
        obs: (N, L+1, 1) - Observations for each trajectory and timestep
    """
    path_dataset = config[f'path_dataset']
    
    # Define the observation file path
    obs_file_path = f'{path_dataset}/obs_L{config["LT"]}_win{config["window"]}.h5'
    
    # Read input data
    with h5py.File(f'{path_dataset}/test.h5', mode='r') as f:
        x = f['data'][:, :config['LT']+config['window']]  # shape: (N, L+w, 3)
    
    # Check if the observation file already exists
    if os.path.exists(obs_file_path):
        # Generate what the new observations would be
        x_tensor = torch.from_numpy(x)
        new_obs = observation_generator(x_tensor, config['sigma_obs_hi'])
        
        # Load existing observations
        with h5py.File(obs_file_path, mode='r') as f:
            existing_obs = f['obs'][:]
        
        # Compare existing and new observations
        if np.allclose(existing_obs, new_obs.numpy(), rtol=1e-3, atol=1e-3):
            print(f"Existing observations file is identical. Skipping creation.")
        else:
            new_obs_np = new_obs.numpy()
            print(f"Existing observations file contains different data!")
            print(f"This requires attention. Exiting evaluation.")
            print(f"New observation.shape: {new_obs_np[0,1:10,:]}")
            print(f"Existing observation.shape: {existing_obs[0,1:10,:]}")
            print(f"Difference: {np.sum(np.abs(new_obs_np - existing_obs))}")
            sys.exit(1)
    else:
        # Create new observations file
        with h5py.File(obs_file_path, mode='w') as f:
            x_tensor = torch.from_numpy(x)
            obs = observation_generator(x_tensor, config['sigma_obs_hi'])
            f.create_dataset('obs', data=obs)
            print(f"Created new observations file at {obs_file_path}")

    print("Observations created successfully.\n\n")

if __name__ == "__main__":
    set_seed(427)

    create_observations(config)