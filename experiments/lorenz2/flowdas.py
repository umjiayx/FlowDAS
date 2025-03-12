import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
from utils import to

from mcs import observe

import time

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


def loss_fn(model, x):
    """
    The loss function for training score-based generative models.

    Args:
    - model: 
        A PyTorch model instance that represents a time-dependent score-based model.
    - x: 
        A mini-batch of training data (according to the prepare_batch function).
    - marginal_prob_std: 
        A function that gives the standard deviation of the perturbation kernel.
    - eps: 
        A tolerance value for numerical stability.
    """

    zt_squeezed = x['zt'] # Stochastic Interpolant
    cond_squeezed = x['cond']
    target = x['drift_target']
    # device = zt_squeezed.device
    score = model(zt_squeezed, x['t'], extra_elements=cond_squeezed)
    loss = (score - target).pow(2).sum(-1).mean() # Mean Squared Error

    return loss, score.shape[0]


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


def prepare_batch(batch=None, device='cuda:0', config=None):
    """Process batch data and prepare for training/sampling.
    
    Args:
        batch: Input batch data with shape (N, L-1, 6)
        device: Device to put tensors on
        
    Returns:
        Dictionary containing processed batch data
    """
    # Process input batch
    assert batch.shape[-1] % 2 == 0, f"Last dimension of batch must be even, got {batch.shape[-1]}"
    half_dim = batch.shape[-1] // 2
    # batch = batch.to(device)
    
    xlo = batch[:,:,0:half_dim].view(-1, half_dim) # (N*(L-1), 3*w), i.e., (NL, 3) for simplicity
    xhi = batch[:,:,half_dim:].view(-1, half_dim) # (N*(L-1), 3*w), i.e., (NL, 3) for simplicity
    N = xlo.shape[0]

    if config['prev_stats_as_cond']:
        z0 = xlo[:, -3:]
        z1 = xhi[:, -3:]
    else:
        z0 = xlo
        z1 = xhi
    
    # Initialize
    sigma_coef = 1
    D = {
        'z0': z0, 
        'z1': z1, 
        'label': None,
        'N': N
    }

    # Helper functions
    def get_time(D):
        D['t'] = torch.distributions.Uniform(low=0, high=1).sample(sample_shape=(D['z0'].shape[0],1)).to(device)
        return D

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

    def sigma_dot(t):
        return sigma_coef * wide(-torch.ones_like(t))
    
    def gamma(t):
        return wide(t.sqrt()) * sigma(t)

    def compute_zt(D):
        return D['at'] * D['z0'] + D['bt'] * D['z1'] + D['gamma_t'] * D['noise']

    def compute_target(D):
        return D['adot'] * D['z0'] + D['bdot'] * D['z1'] + (D['sdot'] * D['root_t']) * D['noise']

    # Compute additional quantities
    D = get_time(D) # (NL, 1)
    D['cond'] = xlo # (NL, 3)
    D['noise'] = torch.randn_like(D['z0']) # (NL, 3)
    D['at'] = alpha(D['t']) # (NL, 1)
    D['bt'] = beta(D['t']) # (NL, 1)
    D['adot'] = alpha_dot(D['t']) # (NL, 1)
    D['bdot'] = beta_dot(D['t']) # (NL, 1)
    D['root_t'] = wide(D['t'].sqrt()) # (NL, 1)
    D['gamma_t'] = gamma(D['t']) # (NL, 1)
    D['st'] = sigma(D['t']) # (NL, 1)
    D['sdot'] = sigma_dot(D['t']) # (NL, 1)
    D['zt'] = compute_zt(D) # (NL, 3)
    D['drift_target'] = compute_target(D) # (NL, 3)

    return D


def save_checkpoint(epoch, model, optimizer, loss, file_path="checkpoint.pth"):
    print(f"Saving Checkpoint at epoch {epoch}.")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    checkpoint_filename = f"checkpoint_{epoch}.pth"
    save_path = file_path / checkpoint_filename  # This is the correct way to join paths
    torch.save(checkpoint, str(save_path))


def save_best_checkpoint(epoch, model, optimizer, loss, best_model_path, checkpoint_path):
    print(f"Saving best model at epoch {epoch}.")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    checkpoint_filename = f"best_model.pth"
    save_path_best = best_model_path / checkpoint_filename
    save_path_checkpoint = checkpoint_path / checkpoint_filename
    torch.save(checkpoint, str(save_path_best))
    torch.save(checkpoint, str(save_path_checkpoint))


def train_model(score_model, data=None, val_data=None, lr=1e-4, batch_size=1024, n_epochs=5000, num_workers=16,
                checkpoint_path="checkpoint.pth", save_interval=500, best_model_path="best_model.pth", config=None):
    
    logger = logging.getLogger(__name__)

    logger.info("Initializing training...")
    logger.info(f"Training for {n_epochs} epochs with learning rate {lr}")
    logger.info(f"Checkpoints will be saved every {save_interval} epochs to {checkpoint_path}")

    # Load training and validation data
    # Yixuan: num_workers and pin_memory=True is important for speed
    data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True) if val_data is not None else None

    # data = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    # val_data = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=2, persistent_workers=True) if val_data is not None else None
    
    logger.info(f"Training Data Loaded: {len(data)} batches of size {batch_size}")
    if val_data:
        logger.info(f"Validation Data Loaded: {len(val_data)} batches of size {batch_size}")

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
                x, kwargs = to(x, device='cuda:0') # x: (B, L-1, 6)
                x = prepare_batch(batch=x, config=config) 
                # logger.info(f"shapes: {x['z0'].shape}, {x['z1'].shape}, {x['drift_target'].shape}")
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
                    val_x = prepare_batch(batch=val_x, config=config)
                    # logger.info(f"shapes: {val_x['z0'].shape}, {val_x['z1'].shape}, {val_x['drift_target'].shape}")
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
                save_best_checkpoint(epoch, model_to_save, optimizer, best_val_loss, best_model_path, checkpoint_path)
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

    # return train_loss, val_loss_history
    return score_model


def MC_taylor_est2rd_x1(model ,xt, t, bF, g, label = None, cond = None,MC_times = 1, use_original_sigma = True, analytical = True):
    """
    xt.shape: (B, 3*window)
    """
    def clip_x1(x):
        return torch.clamp(x, min=-3, max=3)
    if use_original_sigma == True and analytical == False:
        hat_x1 = xt + bF * (1-t) + g * torch.randn_like(xt) * (1-t).sqrt()
    elif use_original_sigma == True and analytical == True and MC_times == 1:
        assert 1==0
        hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
        t1 = torch.FloatTensor([1-1e-5])
        bF2 = model(xt,t1.to(xt.device),extra_elements = cond.to('cuda:0')).requires_grad_(True)
        hat_x1 =  xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
        return hat_x1.requires_grad_(True)
    elif use_original_sigma == True and analytical == True and MC_times != 1: # Yixuan: We are using this!
        hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
        t1 = torch.FloatTensor([1-1e-5])
        bF2 = model(xt,t1.to(xt.device),extra_elements = cond.to('cuda:0')).requires_grad_(True)
        hat_x1_list = []
        for _ in range(MC_times):
            hat_x1 =  xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            # hat_x1 = clip_x1(hat_x1)
            hat_x1_list.append(hat_x1.requires_grad_(True))
        return hat_x1_list # A list, with MC_times elements, each shape is (B, 3*window).


def grad_and_value_NOEST(x_prev, x1_hat, measurement, **kwargs):
    # x1_hat should be a list of tensors, not a single tensor
    '''
    The input x1_hat is a list of MC_times tensors, each tensor is the estimated x_0 at a different time step.
    x_prev is the xt at the previous time step of SI. x_prev.shape: (B, 3*window)
    '''
    if not isinstance(x1_hat, list):
        raise ValueError("x1_hat must be a list of tensors, not a single tensor")
    else:
        x1_hat = torch.cat(x1_hat, dim=0).requires_grad_(True) # (B*MC_times, 3*window)
        differences = torch.linalg.norm(measurement - observe(x1_hat)[:,-3], dim=0) # -3 means the x of the last xyz
        # print(f"x1_hat.shape: {x1_hat.shape}, measurement.shape: {measurement.shape}")
        
        # Compute the weights
        weights = -differences / (2 * (0.25)**2) # Yixuan: 0.25 is the std of the Gaussian noise!!! should be changed to sigma_obs_hi later. 
        # assert 1==0

        # Detach the weights to prevent gradients from flowing through them
        weights_detached = weights.detach()

        # Apply softmax to the detached weights
        softmax_weights = torch.softmax(weights_detached, dim=0)

        # Perform element-wise multiplication
        result = softmax_weights * differences

        # Sum up the results
        final_result = result.sum()
        # print('difference norm',final_result)
        norm_grad_tuple = torch.autograd.grad(outputs=final_result, inputs=x_prev, allow_unused=True)
        norm_grad = norm_grad_tuple[0] # shape: (B, 3*window)
    
    return norm_grad, final_result


def EM(model, base=None, label=None, cond=None, diffusion_fn=None, num_steps=500, 
       measurement=None, noisy_level=None, MC_times=1, step_size=None):
    """
    cond.shape: (B, 3*window)
    """
    steps = num_steps
    tmin, tmax = 0, 1
    ts = torch.linspace(tmin, tmax, steps).type_as(base)
    dt = ts[1] - ts[0]
    ones = torch.ones(base.shape[0]).type_as(base)
    xt = base.requires_grad_(True) # (B, 3*window)
    cycle = True

    def step_fn(model, xt, t, label, cond, measurement, device, diff_list, 
                noisy_level=None, MC_times=1, step_size=step_size): 
        if t[0] ==0:
            t+=1e-5
        if t[0] ==1:
            t-=1e-5

        bF = model(xt, t.to(xt.device), extra_elements=cond.to('cuda:0')).requires_grad_(True)
        def sigma(t):
            return  1-t 
        sigma = sigma(t)
        
        f = bF
        g = sigma
        scale = step_size

        es_x1 = MC_taylor_est2rd_x1(model, xt, t.to(xt.device), bF ,g.to(xt.device), cond=cond, MC_times=MC_times)
        norm_grad, diff_ele = grad_and_value_NOEST(x_prev=xt, x1_hat=es_x1, measurement=measurement.to(xt.device))
        
        diff_list.append(diff_ele.detach())
        if norm_grad is None:
            norm_grad = 0
            print('no grad!')

        xt = xt + f*dt + g*torch.randn_like(xt)*dt.sqrt() - scale*norm_grad # Yixuan: norm_grad is the DPS gradient wrt xsn
        return xt, diff_list # return sample and its mean
    
    cycle_times = 0
    
    while cycle and cycle_times < 3:
        diff_list = []
        for i, tscalar in enumerate(ts):
            if tscalar == 1:
                break
            if i == 0 and (diffusion_fn is not None) :
                # only need to do this when using other diffusion coefficients that you didn't train with
                # because the drift-to-score conversion has a denominator that features 0 at time 0
                # if just sampling with "sigma" (the diffusion coefficient you trained with) you
                # can skip this
                tscalar = ts[1] # 0 + (1/500)

            xt, diff_list = step_fn(model, xt, tscalar*ones, label=None, cond=cond, 
                                    measurement=measurement, device='cuda:0', 
                                    diff_list=diff_list, noisy_level=noisy_level, 
                                    MC_times=MC_times, step_size=step_size)

        if abs(diff_list[-1].sum()) < 3: # TODO: why a '3' here?
            cycle = False

        cycle_times += 1
    
    return xt # shape: (B, 3*window)


def Euler_Maruyama_sampler(score_prior, num_steps=1000, device='cuda:0',
                           base=None,
                           cond=None, measurement=None,
                           noisy_level=None, MC_times=1, 
                           batch_size=64, step_size=None):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
    the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
    Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples. shape: (B, 3*window)
    """
    batch_size = batch_size
    score_prior.eval()
    cond = cond.repeat(batch_size, 1) # (B, 3*window)
    base = base.repeat(batch_size, 1) # (B, 3*window)
    EM_args = {'base': base, 'label': None, 'cond': cond}
    
    sample = EM(score_prior,diffusion_fn=None, 
                **EM_args, num_steps=num_steps,
                measurement=measurement, noisy_level=noisy_level, 
                MC_times=MC_times, step_size=step_size)
    return sample # shape: (B, 3*window)
