import functools
import torch
import logging
from datetime import datetime

from mcs import *
from utils import *
from flowdas import ScoreNet, marginal_prob_std, train_model

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


def get_config():
    config = {
        'window': 3,
        'width': 384, # 256
        'depth': 5, 
        'epochs': 1500, # 10000
        'batch_size': 256,
        'optimizer': 'Adam',
        'learning_rate': 5e-3,
        'scheduler': 'linear',
        'sigma': 25.0,
        'extra_dim': 3,
        'x_dim': 3,
        'use_bn': False
    }
    return config


def initialize_model(config, device):
    """Initialize and return the score model"""
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=config['sigma'])
    flow_prior = ScoreNet(marginal_prob_std=marginal_prob_std_fn, 
                         x_dim=config['x_dim']*config['window'], 
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
    return flow_prior


def load_datasets():
    """Load and return training and validation datasets"""
    trainset = TrajectoryDataset(PATH / 'data/dataset/train.h5', window=config['window'], flatten=True)
    validset = TrajectoryDataset(PATH / 'data/dataset/valid.h5', window=config['window'], flatten=True)
    return trainset, validset


def load_datasets_V2():
    """Load and return training and validation datasets"""
    trainset = TrajectoryDatasetV2(PATH / 'data/dataset/train.h5', window=config['window'])
    validset = TrajectoryDatasetV2(PATH / 'data/dataset/valid.h5', window=config['window'])
    return trainset, validset


def train(model, config, trainset, validset, runpath):
    """Train the model and save checkpoints"""
    loss_train = train_model(score_model=model, 
                            data=trainset, 
                            val_data=validset,
                            lr=config['learning_rate'], 
                            batch_size=config['batch_size'],
                            n_epochs=config['epochs'], 
                            print_interval=100, 
                            checkpoint_path=runpath, 
                            best_model_path=runpath,
                            save_interval=1000)
    return loss_train


if __name__ == "__main__":
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runpath = PATH / 'runs_train' / f'training_run_{timestamp}'
    runpath.mkdir(parents=True, exist_ok=True)

    # Initialize logging and config
    setup_training_logging(runpath)
    config = get_config()
    logging.info(f"Created run directory at {runpath}")
    logging.info(f"Configuration: {config}")

    # Setup device and model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    flow_prior = initialize_model(config, device)
    logging.info("Model initialized and moved to device: %s", device)

    # Load datasets
    # trainset, validset = load_datasets()
    trainset, validset = load_datasets_V2()
    logging.info(f"Loaded training dataset with {len(trainset)} samples")
    logging.info(f"Loaded validation dataset with {len(validset)} samples")

    # Train model
    logging.info("Starting model training...")
    loss_train = train(flow_prior, config, trainset, validset, runpath)
    logging.info(f"Training completed with final loss: {loss_train[-1]}")

    # Save final model
    final_model_path = runpath / 'final_model.pth'
    torch.save(flow_prior.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    logging.info(f"Training complete. Model and logs saved in: {runpath}")