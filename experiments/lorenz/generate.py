import h5py
from utils import *
import numpy as np
from datetime import datetime
from pathlib import Path

PATH = Path(__file__).parent.absolute()
log_file = PATH / 'data' / f'data_generation_log.txt'


def simulate(dataset_idx: int, N: int, L: int):
    '''
    dataset_idx:
        Generate/Simulate the dataset_idx th dataset.
    N:
        Number of simulated Lorenz Particles.
    L:
        Trajectory length.
    '''

    L0 = 1024 # Warm-up
    coeff = 0

    sigma, rho, beta = get_Lorenz_parameters(coeff=coeff)

    chain = make_chain_generalize(sigma=sigma, rho=rho, beta=beta) # When coeff = 0, using standard parameters.

    print(f"Dataset {dataset_idx} - Chain parameters - sigma: {chain.sigma:.3f}, rho: {chain.rho:.3f}, beta: {chain.beta:.3f}")

    x = chain.prior((N,)) # sample from a Gaussian for N initial points
    x = chain.trajectory(x, length=L0, last=True) # Now, x is a (N, 3) tensor, containing N final states
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
    dataset_dir = PATH / 'data' / str(dataset_idx)
    if dataset_dir.exists():
        import shutil
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for name, x in splits.items():
        with h5py.File(dataset_dir / f'{name}.h5', mode='w') as f:
            f.create_dataset('x', data=x, dtype=np.float32)
    
    # Log the parameters
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"Time: {current_time}, Index: {dataset_idx}, Parameters: sigma={sigma:.3f}, rho={rho:.3f}, beta={beta:.3f}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    print(f'Dataset {dataset_idx} generated!')


def combine_datasets(num_datasets):
    # Combine all datasets into single files
    print("Combining datasets...")

    # Create combined directory
    combined_dir = PATH / 'data' / 'dataset'
    if combined_dir.exists():
        import shutil
        shutil.rmtree(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # For each split (train/valid/test)
    for split in ['train', 'valid', 'test']:
        # Collect data from all dataset folders
        all_data = []
        for dataset_idx in range(num_datasets):
            dataset_path = PATH / 'data' / str(dataset_idx) / f'{split}.h5'
            with h5py.File(dataset_path, 'r') as f:
                data = f['x'][:]
                all_data.append(data)
        
        # Concatenate along first dimension
        combined_data = np.concatenate(all_data, axis=0)
        
        # Save combined data
        with h5py.File(combined_dir / f'{split}.h5', 'w') as f:
            f.create_dataset('x', data=combined_data, dtype=np.float32)
        
        print(f'Combined {split} dataset shape: {combined_data.shape}')

    print("Dataset generation complete!")


if __name__ == "__main__":
    """
    This script generates multiple datasets (with different parameters) and then combines them into a single dataset. 
    The user can specify the number of datasets to generate. A naive choice is to generate only one dataset.
    """
    # Create dataset directory
    data_dir = PATH / 'data'
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log_entry = f"Dataset generation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)

    num_datasets = 25
    num_particles = 256
    len_trajectory = 1024
    
    for i in range(num_datasets):
        simulate(dataset_idx=i, N=num_particles, L=len_trajectory)
    
    combine_datasets(num_datasets)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = PATH / 'data' / timestamp    
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    # Move all dataset_idx folders to timestamp directory
    for i in range(num_datasets):
        dataset_path = PATH / 'data' / str(i)
        if dataset_path.exists():
            dataset_path.rename(timestamp_dir / str(i))
            
    print(f"Moved dataset folders to {timestamp_dir}")

    with open(log_file, 'a') as f:
        f.write('\n')