import h5py
from utils import *
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import yaml
import argparse

PATH = Path(__file__).parent.absolute()


def simulate(dataset_idx: int, config: dict):
    '''
    dataset_idx:
        Generate/Simulate the dataset_idx th dataset.
    N:
        Number of simulated Lorenz Particles.
    L:
        Trajectory length.
    '''

    N = config['num_particles']
    L = config['len_trajectory']

    sigma, rho, beta = get_Lorenz_parameters(coeff=config['coeff'])

    chain = make_chain_generalize(sigma=sigma, rho=rho, beta=beta) # When coeff = 0, using standard parameters.

    print(f"Dataset {dataset_idx} - Chain parameters - sigma: {chain.sigma:.3f}, rho: {chain.rho:.3f}, beta: {chain.beta:.3f}")

    x = chain.prior((N,)) # sample from a Gaussian for N initial points
    x = chain.trajectory(x, length=config['L0'], last=True) # Now, x is a (N, 3) tensor, containing N final states
    x = chain.trajectory(x, length=L) # Now, x is (L, N, 3), containing length-L trajectories of N particles
    x = chain.preprocess(x) # Normalize x to about zero-centered
    x = x.transpose(0, 1) # x is (N, L, spatial coordinates)

    i = int(0.8 * len(x))
    j = int(0.9 * len(x))

    splits = {
        'train': x[:i],
        'valid': x[i:j],
        'test': x[j:],
    }

    # Create dataset directory
    dataset_dir = config['data_dir'] / str(dataset_idx)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        with h5py.File(dataset_dir / f'{name}.h5', mode='w') as f:
            f.create_dataset('data', data=data, dtype=np.float32)
    
    # Log the parameters
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"Time: {current_time}, Dataset index: {dataset_idx}, Parameters: sigma={sigma:.3f}, rho={rho:.3f}, beta={beta:.3f}\n"
    
    with open(config['log_file_path'], 'a') as f:
        f.write(log_entry)
    
    print(f'Dataset {dataset_idx} generated!')


def combine_datasets(config: dict):
    num_datasets = config['num_datasets']

    # Combine all datasets into single files
    print("Combining datasets...")

    # Create combined directory
    combined_dir = config['data_dir'] / 'dataset'
    if combined_dir.exists():
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # For each split (train/valid/test)
    for split in ['train', 'valid', 'test']:
        # Collect data from all dataset folders
        all_data = []
        for dataset_idx in range(num_datasets):
            dataset_path = config['data_dir'] / str(dataset_idx) / f'{split}.h5'
            with h5py.File(dataset_path, 'r') as f:
                data = f['data'][:]
                all_data.append(data)
        
        # Concatenate along first dimension
        combined_data = np.concatenate(all_data, axis=0)
        
        # Use different but deterministic seeds for different splits and shuffle the combined data!
        split_seeds = {'train': 42, 'valid': 43, 'test': 44}
        rng = np.random.default_rng(seed=split_seeds[split])
        rng.shuffle(combined_data)

        # Save combined data
        with h5py.File(combined_dir / f'{split}.h5', 'w') as f:
            f.create_dataset('data', data=combined_data, dtype=np.float32)
        
        print(f'Combined {split} dataset shape: {combined_data.shape}')

    print("Dataset generation complete!")

    # Reorganize and relocate datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = config['data_dir'] / timestamp    
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_datasets):
        dataset_path = config['data_dir'] / str(i)
        if dataset_path.exists():
            # Move individual dataset folder to timestamp directory
            shutil.move(dataset_path, timestamp_dir / str(i))

    combined_path = config['data_dir'] / 'dataset'
    if combined_path.exists():
        shutil.copytree(combined_path, timestamp_dir / 'dataset')

    print(f"Moved dataset folders to {timestamp_dir}")

    # Log the end time
    log_entry = f"Dataset generation complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    with open(config['log_file_path'], 'a') as f:
        f.write(log_entry)
        f.write(f"Path to dataset folders: {timestamp_dir}\n")
        f.write('\n')



if __name__ == "__main__":
    """
    This script generates multiple datasets (with different parameters) and then combines them into a single dataset. 
    The user can specify the number of datasets to generate. A naive choice is to generate only one dataset.
    """
    set_seed(42)
    config = prepare_generate()

    for i in range(config['num_datasets']):
        simulate(dataset_idx=i, config=config)
    
    combine_datasets(config=config)