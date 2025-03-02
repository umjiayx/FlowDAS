import functools
import torch
import logging
from datetime import datetime

from mcs import *
from utils import *
from flowdas import ScoreNet, marginal_prob_std, train_model

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
runpath = PATH / 'runs' / f'training_run_{timestamp}'
runpath.mkdir(parents=True, exist_ok=True)

# Set up logging
log_filename = f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Set up logging to save to runpath
log_filepath = runpath / log_filename
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)

# Configuration
LOCAL_CONFIG = {
    'window': 2,
    # 'embedding': 32,
    'width': 256,
    'depth': 5,
    # 'activation': 'SiLU',
    'epochs': 10000,
    'batch_size': 64,
    'optimizer': 'Adam',
    'learning_rate': 5e-3,
    # 'weight_decay': 1e-3,
    'scheduler': 'linear',
    'sigma': 25.0,
    'extra_dim': 3,
    'x_dim': 3,
    'use_bn': False
}

if __name__ == "__main__":
    # save_config(LOCAL_CONFIG, runpath)
    logging.info(f"Created run directory at {runpath}")
    logging.info(f"Configuration: {LOCAL_CONFIG}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=LOCAL_CONFIG['sigma'])
    x_dim = LOCAL_CONFIG['x_dim']
    hidden_depth = LOCAL_CONFIG['depth'] #5
    embed_dim = LOCAL_CONFIG['width'] #256
    use_bn = LOCAL_CONFIG['use_bn']
    extra_dim = LOCAL_CONFIG['extra_dim']

    # Initialize Score Model
    flow_prior = ScoreNet(marginal_prob_std=marginal_prob_std_fn, 
                          x_dim=x_dim, 
                          extra_dim=extra_dim,
                          hidden_depth=hidden_depth, 
                          embed_dim=embed_dim, 
                          use_bn=use_bn).to(device)
    logging.info("Model initialized and moved to device: %s", device)

    # Load Data
    trainset = TrajectoryDataset(PATH / 'data/dataset/train.h5', window=2, flatten=True)
    logging.info(f"Loaded training dataset with {len(trainset)} samples")
    validset = TrajectoryDataset(PATH / 'data/dataset/valid.h5', window=2, flatten=True)
    logging.info(f"Loaded validation dataset with {len(validset)} samples")

    # Train Model
    logging.info("Starting model training...")
    loss_train = train_model(score_model=flow_prior, 
                            data=trainset, 
                            val_data=validset,
                            lr=LOCAL_CONFIG['learning_rate'], 
                            n_epochs=LOCAL_CONFIG['epochs'], 
                            print_interval=100, 
                            checkpoint_path=runpath, 
                            best_model_path=runpath,
                            save_interval=1000)
    logging.info(f"Training completed with final loss: {loss_train[-1]}")

    # Save Final Model
    final_model_path = runpath / 'final_model.pth'
    torch.save(flow_prior.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    logging.info(f"Training complete. Model and logs saved in: {runpath}")