import functools
import torch
import logging
from datetime import datetime

from mcs import *
from utils import *
from flowdas import ScoreNet, marginal_prob_std, train_model

import subprocess # to determine the number of workers
import argparse
import yaml
from pathlib import Path

PATH = Path(__file__).parent.absolute()


"""
Suppose you have B batches of data, This B depends on the batch_size and number of your total data.
The num_workers should not be larger than B! If larger, some workers will be idle.
If batch_size is too large, the memory can be too large to handle efficiently, which then becomes the bottleneck.
If batch_size is small, the GPU can keep receiving data from the CPU, though GPU memory is not fully utilized.

Here, the Lorenz dataset is pretty small, the main bottleneck is the data preprocessing on CPU memory.
So, it's better to set num_workers to a large number.
"""


def initialize_model(config, device):
    """Initialize and return the score model"""
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=config['sigma'])
    flow_prior = ScoreNet(marginal_prob_std=marginal_prob_std_fn, 
                         x_dim=config['x_dim'], 
                         extra_dim=config['extra_dim']*config['window'], # extra_dim is x0
                         hidden_depth=config['depth'], 
                         embed_dim=config['width'], 
                         use_bn=config['use_bn']).to(device)
    
    # Print model architecture
    logging.info("Model Architecture:")
    logging.info("------------------")
    logging.info(str(flow_prior))
    logging.info("------------------")

    # Calculate and print parameter counts
    total_params = sum(p.numel() for p in flow_prior.parameters())
    trainable_params = sum(p.numel() for p in flow_prior.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info("Model initialized and moved to device: %s", config['device'])
    return flow_prior


def load_datasets(config):
    # Determine dataset paths based on generalizability study flag
    data_dir = 'data_gen' if config['study_generalizability'] else 'data'
    data_dir = f'data_gen_memgen_datasize{config["dataset_size"]}' if config['study_Mem_Gen'] else data_dir
    dataset_class = TrajectoryDatasetV2 # if config['study_generalizability'] else TrajectoryDataset
    
    # Load datasets
    train_path = PATH / f'{data_dir}/dataset/train.h5'
    valid_path = PATH / f'{data_dir}/dataset/valid.h5'
    trainset = dataset_class(train_path, window=config['window'])
    validset = dataset_class(valid_path, window=config['window'])
    
    # Log dataset information
    logging.info(f"Loaded training dataset from {train_path} with {len(trainset)} samples")
    logging.info(f"Loaded validation dataset from {valid_path} with {len(validset)} samples")
    
    return trainset, validset


def train(model, config, trainset, validset, runpath):
    """Train the model and save checkpoints"""
    logging.info("Starting model training...")
    model = train_model(score_model=model, 
                data=trainset, 
                val_data=validset,
                lr=config['learning_rate'], 
                batch_size=config['batch_size'],
                n_epochs=config['epochs'], 
                checkpoint_path=runpath, 
                best_model_path=config['best_model_path'],
                num_workers=config['num_workers'],
                save_interval=config['save_interval'],
                config=config)
    logging.info("Model training completed.")
    # Save final model
    final_model_path = config['runpath'] / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")    
    return model


def setup_training_logging(runpath):
    log_filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_filepath = runpath / log_filename
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )


def get_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare():
    parser = argparse.ArgumentParser(description='Train Lorenz model')
    parser.add_argument('--config', type=str, # default='train_win1_G',
                        help='Name of the config file in the config directory')
    parser.add_argument('--epochs', type=int, # default=5000,
                        help='Number of epochs to train')
    args = parser.parse_args()
    
    config_path = PATH / 'config' / f'{args.config}.yml'
    config = get_config(config_path)
    
    # Override epochs in config if specified in command line
    if args.epochs is not None:
        config['epochs'] = args.epochs
    
    # Create a path that includes both timestamp and config name (without extension)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    runpath = PATH / 'runs_train' / f'run_{timestamp}_{args.config}'
    runpath.mkdir(parents=True, exist_ok=True)

    # Update config with runpath and num_workers
    config['runpath'] = runpath
    config['num_workers'] = int(subprocess.check_output(['nproc']).strip())

    best_model_path = PATH / 'runs_train' / 'best_models' / f'{args.config}'
    config['best_model_path'] = best_model_path
    best_model_path.mkdir(parents=True, exist_ok=True)

    # Initialize logging and config
    setup_training_logging(config['runpath'])
    
    logging.info(f"Created run directory at {config['runpath']}")
    logging.info(f"Configuration: {config}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config['device'] = device

    return config
    

if __name__ == "__main__":
    # Prepare config
    config = prepare()

    # Initialize model
    flow_prior = initialize_model(config, config['device'])
    
    # Load datasets
    trainset, validset = load_datasets(config)

    # Train model
    flow_prior = train(flow_prior, config, trainset, validset, config['runpath'])

    logging.info(f"Training complete. Model and logs saved in: {config['runpath']}")