import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import tqdm

from torch.utils.data import DataLoader, Dataset
from utils import to


class MultiGaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding multiple inputs (e.g., time and extra elements)."""
    def __init__(self, input_dim, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights for each input dimension
        self.W = nn.Parameter(torch.randn(input_dim, embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x_proj = x[..., None] * self.W[None, :, :] * 2 * np.pi  # Broadcasting over batch and input_dim
        x_proj = x_proj.view(x.shape[0], -1)  # Flatten the last two dimensions
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon a feedforward architecture with extra elements."""
    def __init__(self, marginal_prob_std, x_dim, extra_dim=0, hidden_depth=2, embed_dim=128, use_bn=True):
        super().__init__()
        self.x_dim = x_dim
        self.extra_dim = extra_dim  # Number of extra elements
        self.hidden_depth = hidden_depth
        self.embed_dim = embed_dim
        self.use_bn = use_bn

        # Adjusted embedding layer to handle time and extra elements
        input_dim = 1 + extra_dim  # 1 for time t, plus extra elements
        self.embed = nn.Sequential(
            MultiGaussianFourierProjection(input_dim=input_dim, embed_dim=embed_dim),
            nn.Linear(input_dim * embed_dim, embed_dim)
        )

        # Input layer
        self.input = nn.Linear(x_dim, embed_dim)

        # Hidden layers
        self.fc_all = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(self.hidden_depth)])

        # Batch normalization layers
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(num_features=embed_dim) for _ in range(self.hidden_depth)])

        # Output layer
        self.output = nn.Linear(embed_dim, x_dim)

        # Activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, extra_elements=None):
        # Combine time t and extra elements
        if extra_elements is not None:
            # Ensure that t and extra_elements have the same batch size
            if extra_elements.dim() == 1:
                extra_elements = extra_elements.unsqueeze(-1)
            if t.dim() ==1 :
                t = t.unsqueeze(-1)
            te = torch.cat([t, extra_elements], dim=-1)  # Shape: [batch_size, input_dim]
        else:
            te = t.unsqueeze(-1)  # Shape: [batch_size, 1]

        # Obtain the Gaussian random feature embedding for t and extra elements
        embed = self.act(self.embed(te))  # Shape: [batch_size, embed_dim]

        # Process input x
        h = self.input(x)  # Shape: [batch_size, embed_dim]

        # Residual connections with embedding
        for i in range(self.hidden_depth):
            h = h + self.act(self.fc_all[i](h)) + embed
            if self.use_bn:
                h = self.bn[i](h)

        # Output layer
        h = self.output(h)

        return h


def loss_fn(model, x, marginal_prob_std=None, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """

    zt_squeezed = x['zt']
    cond_squeezed = x['cond']

    score = model(zt_squeezed.to('cuda:0'), x['t'],extra_elements = cond_squeezed.to('cuda:0'))
    target = x['drift_target']
    loss = (score - target).pow(2).sum(-1).mean()

    return loss,score.shape[0]


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The standard deviation.
    """
    # t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

    Returns:
    The vector of diffusion coefficients.
    """
    return sigma**t


def get_time(D):
    D['t'] = torch.distributions.Uniform(low=0, high=1).sample(sample_shape = (D['z0'].shape[0],1)).to('cuda:0')
    return D


def prepare_batch_nse(batch = None, for_sampling = False,sampling_batch_size = None, device = 'cuda:0' ):
    xlo, xhi = batch[:,:,0:3].view(-1, 3),batch[:,:,3:].view(-1, 3)
    xlo, xhi = xlo.to(device), xhi.to(device)
    N = xlo.shape[0]
    y = None
    D = {'z0': xlo, 'z1': xhi, 'label': y, 'N': N}
    return D


def prepare_batch(batch = None, for_sampling = False):
    D = prepare_batch_nse(batch, for_sampling = for_sampling)
    D = get_time(D)
    sigma_coef = 1
    def wide(t):
        return t

    def alpha(t):
        return wide(1-t) 

    def alpha_dot(t):
        return wide(-1.0 * torch.ones_like(t))

    def beta(t):
        return wide(t.pow(2))

    def beta_dot(t):
        return wide(2.0 * t)

    def sigma(t):
        return sigma_coef * wide(1-t) 

    def sigma_dot( t):
        return sigma_coef * wide(-torch.ones_like(t)) 
    
    def gamma(t):
        return wide(t.sqrt()) * sigma(t)

    def compute_zt( D):
        return D['at'] * D['z0'] + D['bt'] * D['z1'] + D['gamma_t'] * D['noise']

    def compute_target(D):
        return D['adot'] * D['z0'] + D['bdot'] * D['z1'] +  (D['sdot'] * D['root_t']) * D['noise']
    
    D['cond'] = D['z0']
    D['noise'] = torch.randn_like(D['z0'])
    D['at'] = alpha(D['t'])
    D['bt'] = beta(D['t'])
    D['adot'] = alpha_dot(D['t'])
    D['bdot'] = beta_dot(D['t'])
    D['root_t'] = wide(D['t'].sqrt())
    D['gamma_t'] = gamma(D['t'])
    D['st'] = sigma(D['t'])
    D['sdot'] = sigma_dot(D['t'])
    D['zt'] = compute_zt(D)
    D['drift_target'] = compute_target(D)
    return D


def save_checkpoint(epoch, model, optimizer, loss, file_path="checkpoint.pth"):
    print(f"Saving Checkpoint at epoch {epoch}.")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, str(file_path) + "checkpoint" + str(epoch) + ".pth")


def save_best_checkpoint(epoch, model, optimizer, loss, best_model_path="best_model.pth"):
    print(f"Saving best model at epoch {epoch}.")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, str(best_model_path) + "best_model" + ".pth")


def train_model(score_model, data=None, val_data=None, lr=1e-4, batch_size=1000, n_epochs=5000, 
                print_interval=100, checkpoint_path="checkpoint.pth", save_interval=500, best_model_path="best_model.pth"):
    
    logger = logging.getLogger(__name__)

    logger.info("Initializing training...")
    logger.info(f"Training for {n_epochs} epochs with learning rate {lr}")
    logger.info(f"Checkpoints will be saved every {save_interval} epochs to {checkpoint_path}")

    # Load training and validation data
    data = DataLoader(data, batch_size=256, shuffle=True)
    val_data = DataLoader(val_data, batch_size=256, shuffle=False) if val_data is not None else None
    
    logger.info(f"Training Data Loaded: {len(data)} batches of size 256")
    if val_data:
        logger.info(f"Validation Data Loaded: {len(val_data)} batches of size 256")

    optimizer = Adam(score_model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: 1 - (t / n_epochs))
    
    logger.info("Optimizer and scheduler initialized")

    train_loss = []
    val_loss_history = []
    best_val_loss = float("inf")  # Track best validation loss

    # Progress bar for epochs
    pbar = tqdm(total=n_epochs, position=0, leave=True, desc="Training Progress")
    logger.info("Starting training loop...")

    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch+1}/{n_epochs}")
        logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        avg_loss = 0.
        num_items = 0
        
        # Progress bar for batches
        batch_pbar = tqdm(total=len(data), position=1, leave=True, 
                          desc=f"Processing batches for epoch {epoch+1}")

        # ---------- TRAINING PHASE ----------
        if data is not None:
            # Training Loop
            score_model.train()
            for idx, x in enumerate(data):
                x, kwargs = to(x, device='cuda')
                x = prepare_batch(batch=x)
                loss, N_ba = loss_fn(score_model, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * N_ba
                num_items += N_ba
                
                batch_pbar.update(1)
                batch_pbar.set_description(f"Batch {idx+1}/{len(data)} - Loss: {loss.item():.4f}")

            scheduler.step()
            epoch_avg_loss = avg_loss / num_items
            train_loss.append(epoch_avg_loss)

            # Update epoch progress bar
            pbar.update(1)
            pbar.set_description(
                f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {epoch_avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # Log training loss
            logger.info(f"Epoch {epoch+1} Training Loss: {epoch_avg_loss:.4f}")

        # Close batch progress bar
        batch_pbar.close()

        # ---------- VALIDATION PHASE ----------
        if val_data is not None:
            score_model.eval()  # Set model to evaluation mode
            val_loss = 0.
            val_items = 0

            with torch.no_grad():  # Disable gradient computation
                for val_x in val_data:
                    val_x, val_kwargs = to(val_x, device='cuda')
                    val_x = prepare_batch(batch=val_x)
                    val_loss_batch, N_ba_val = loss_fn(score_model, val_x)
                    val_loss += val_loss_batch.item() * N_ba_val
                    val_items += N_ba_val

            epoch_val_loss = val_loss / val_items
            val_loss_history.append(epoch_val_loss)

            # Log validation loss
            logger.info(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}")

            # Update tqdm bar with validation loss
            pbar.set_description(
                f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_avg_loss:.4f} - Val Loss: {epoch_val_loss:.4f}"
            )

            # Save the best model based on validation loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving best model...")
                model_to_save = score_model.module if hasattr(score_model, 'module') else score_model
                save_best_checkpoint(epoch, model_to_save, optimizer, best_val_loss, best_model_path)
                logger.info("Best model saved successfully.")

        # ---------- PERIODIC CHECKPOINT SAVING ----------
        if epoch % save_interval == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            model_to_save = score_model.module if hasattr(score_model, 'module') else score_model
            save_checkpoint(epoch, model_to_save, optimizer, epoch_avg_loss, checkpoint_path)
            logger.info("Checkpoint saved successfully.")

    # Close epoch progress bar
    pbar.close()
    logger.info("Training completed!")
    logger.info(f"Final Training Loss: {train_loss[-1]:.4f}")
    if val_data:
        logger.info(f"Final Validation Loss: {val_loss_history[-1]:.4f}")

    return train_loss, val_loss_history


