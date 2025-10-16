import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.functional import interpolate
from torchvision import transforms as T
from torchvision.utils import make_grid

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import seaborn as sns
import math
import wandb
import argparse
import datetime
from time import time
import os
import numpy as np
import random


# unet.py
from unet import Unet


def set_random_seed(seed):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value, or None to skip seeding
    """
    if seed is None:
        print("No random seed set - results will be non-deterministic")
        return
    
    print(f"Setting random seed: {seed}")
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (for all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("  ✓ Python random seed set")
    print("  ✓ NumPy random seed set")
    print("  ✓ PyTorch random seed set")
    if torch.cuda.is_available():
        print("  ✓ PyTorch CUDA random seed set")
    print("  ✓ Deterministic mode enabled")


class Config:
    def __init__(self, config_dict=None, **kwargs):
        """
        Initialize Config from a dictionary (loaded from YAML) and optional overrides.
        
        Args:
            config_dict: Dictionary loaded from YAML config file
            **kwargs: Command-line argument overrides
        """
        # If no config_dict provided, use defaults
        if config_dict is None:
            config_dict = {}
        
        # Helper function to get nested config values
        def get_config(keys, default=None):
            """Get value from nested config dict, e.g., get_config(['dataset', 'name'])"""
            val = config_dict
            for key in keys:
                if isinstance(val, dict):
                    val = val.get(key, default)
                else:
                    return default
            return val if val is not None else default
        
        # Dataset configuration
        self.dataset = kwargs.get('dataset', get_config(['dataset', 'name'], 'nse'))
        self.sample_only = kwargs.get('sample_only', get_config(['sampling', 'sample_only'], False))
        self.debug = kwargs.get('debug', get_config(['sampling', 'debug_mode'], False))
        print("SELF DEBUG IS", self.debug)
        
        # Device configuration
        self.device = kwargs.get('device', get_config(['system', 'device'], 'cuda:0'))
        
        # Random seed for reproducibility
        self.random_seed = kwargs.get('random_seed', get_config(['system', 'random_seed'], None))
        
        # Interpolant parameters
        self.sigma_coef = kwargs.get('sigma_coef', get_config(['interpolant', 'sigma_coef'], 1.0))
        self.beta_fn = kwargs.get('beta_fn', get_config(['interpolant', 'beta_fn'], 't^2'))
        self.noise_strength = get_config(['interpolant', 'noise_strength'], 1.0)
        self.t_min_train = get_config(['interpolant', 't_min_train'], 0.0)
        self.t_max_train = get_config(['interpolant', 't_max_train'], 1.0)
        self.t_min_sampling = get_config(['interpolant', 't_min_sampling'], 0.0)
        self.t_max_sampling = get_config(['interpolant', 't_max_sampling'], 0.999)
        
        # FlowDAS parameters
        self.time_window = kwargs.get('time_window', get_config(['flowdas', 'time_window'], 1))
        self.auto_step = kwargs.get('auto_step', get_config(['flowdas', 'auto_step'], 1))
        self.MC_times = kwargs.get('MC_times', get_config(['flowdas', 'MC_times'], 1))
        self.exp_times = kwargs.get('exp_times', get_config(['flowdas', 'exp_times'], 1))
        self.grad_scale = kwargs.get('grad_scale', get_config(['flowdas', 'grad_scale'], 1.0))
        
        # Sampling parameters
        self.EM_sample_steps = get_config(['sampling', 'EM_sample_steps'], 500)
        self.load_path = kwargs.get('load_path', get_config(['sampling', 'load_path'], None))
        self.trajectory_index = kwargs.get('trajectory_index', get_config(['sampling', 'trajectory_index'], None))
        self.time_index = kwargs.get('time_index', get_config(['sampling', 'time_index'], None))
        
        # Paths
        self.home = kwargs.get('savedir', get_config(['logging', 'output_dir'], './tmp_images/'))
        self.ckpt_save_dir = kwargs.get('ckpt_save_dir', get_config(['logging', 'ckpt_save_dir'], './ckpts'))
        # add window_size, dataset name, and timestamp to the ckpt_save_dir
        self.ckpt_save_dir = self.ckpt_save_dir + '_'  + self.dataset + '_win=' + str(self.time_window) + '_T' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Dataset-specific configuration
        if self.dataset == 'cifar':
            self.center_data = get_config(['dataset', 'center_data'], True)
            self.C = 3
            self.H = get_config(['dataset', 'cifar', 'image_size'], 32)
            self.W = self.H
            self.num_classes = get_config(['dataset', 'cifar', 'num_classes'], 10)
            self.data_path = get_config(['dataset', 'cifar', 'data_path'], '../data/')
            self.grid_kwargs = {'normalize': True, 'value_range': (-1, 1)}

        elif self.dataset == 'nse':
            self.center_data = get_config(['dataset', 'center_data'], False)
            
            maybe_create_dir(self.home)
            
            # NSE data path with command-line override support
            # Priority: command-line arg > YAML config > default
            nse_path = kwargs.get('nse_datapath')  # Command-line override (highest priority)
            if nse_path is not None:
                self.data_fname = nse_path
            else:
                self.data_fname = get_config(['dataset', 'nse', 'nse_datapath'], './nse_data_tiny.pt')  # YAML or default
            
            self.num_classes = 1
            self.lo_size = get_config(['dataset', 'nse', 'lo_size'], 64)
            self.hi_size = get_config(['dataset', 'nse', 'hi_size'], 128)
            self.time_lag = get_config(['dataset', 'nse', 'time_lag'], 2)
            self.subsampling_ratio = get_config(['dataset', 'nse', 'subsampling_ratio'], 1.0)
            self.avg_pixel_norm = get_config(['dataset', 'nse', 'avg_pixel_norm'], 3.0679163932800293)
            self.temporal_downsampling = get_config(['dataset', 'nse', 'temporal_downsampling'], 2)
            self.train_val_split = get_config(['dataset', 'nse', 'train_val_split'], 0.95)
            self.grid_kwargs = get_config(['logging', 'grid_kwargs'], {'normalize': False})
            self.C = 1
            self.H = self.hi_size
            self.W = self.hi_size

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Training parameters
        self.overfit = kwargs.get('overfit', get_config(['training', 'overfit'], False))
        print(f"OVERFIT MODE (USEFUL FOR DEBUGGING) IS {self.overfit}")
        
        self.num_workers = get_config(['training', 'num_workers'], 4)
        self.delta_t = get_config(['system', 'delta_t'], 0.5)
        self.batch_size = get_config(['training', 'batch_size'], 128 if self.dataset == 'cifar' else 32)
        self.sampling_batch_size = get_config(['training', 'sampling_batch_size'], 
                                               self.batch_size if self.dataset == 'cifar' else 1)
        self.max_grad_norm = get_config(['training', 'max_grad_norm'], 1.0)
        self.base_lr = get_config(['training', 'base_lr'], 2e-5)
        self.max_steps = get_config(['training', 'max_steps'], 1_000_000)
        
        # Logging parameters
        self.wandb_project = get_config(['logging', 'wandb_project'], 'FlowDAS_NSE')
        self.wandb_entity = get_config(['logging', 'wandb_entity'], 'jiayx18')
        self.use_wandb = kwargs.get('use_wandb', get_config(['logging', 'use_wandb'], True))
        
        # Debug mode overrides
        if self.debug:
            self.EM_sample_steps = get_config(['sampling', 'debug', 'EM_sample_steps'], 10)
            self.sample_every = get_config(['sampling', 'debug', 'sample_every'], 10)
            self.print_loss_every = get_config(['sampling', 'debug', 'print_loss_every'], 10)
            self.save_every = 10000000
        else:
            self.sample_every = get_config(['training', 'sample_every'], 1000)
            self.print_loss_every = get_config(['training', 'print_loss_every'], 100)
            self.save_every = get_config(['training', 'save_every'], 1000)
        
        # Model architecture
        self.unet_use_classes = get_config(['model', 'use_classes'], 
                                           True if self.dataset == 'cifar' else False)
        self.unet_channels = get_config(['model', 'channels'], 128)
        dim_mults_list = get_config(['model', 'dim_mults'], [1, 2, 2, 2])
        self.unet_dim_mults = tuple(dim_mults_list) if isinstance(dim_mults_list, list) else dim_mults_list
        self.unet_resnet_block_groups = get_config(['model', 'resnet_block_groups'], 8)
        self.unet_learned_sinusoidal_dim = get_config(['model', 'learned_sinusoidal_dim'], 32)
        self.unet_attn_dim_head = get_config(['model', 'attn_dim_head'], 64)
        self.unet_attn_heads = get_config(['model', 'attn_heads'], 4)
        self.unet_learned_sinusoidal_cond = get_config(['model', 'learned_sinusoidal_cond'], True)
        self.unet_random_fourier_features = get_config(['model', 'random_fourier_features'], False)
        
        # Measurement configuration (for operator and noiser)
        self.measurement_config = get_config(['measurement'], {})


def get_measurement_operator_noiser(config):
    """
    Create measurement operator and noiser from config.
    
    Args:
        config: Config object (contains measurement_config, hi_size, device, etc.)
    
    Returns:
        operator: Measurement operator function
        noiser: Noise operator
    """
    from measurements import get_noise
    
    # Get measurement config from Config object
    operator_config = config.measurement_config.get('operator', {})
    noise_config = config.measurement_config.get('noise', {})
    
    # Create measurement operator based on config
    operator_name = operator_config.get('name', 'super_resolution')
    
    print(f"\n### MEASUREMENT OPERATOR: {operator_name} ###")
    
    if operator_name == 'sparse_observation':
        # Sparse observation operator: randomly mask pixels
        ratio = operator_config.get('ratio', 0.05)
        print(f"Sparse observation with ratio: {ratio} ({ratio*100:.1f}% of pixels observed)")
        
        # Create random mask (fixed for reproducibility during training/testing)
        torch.manual_seed(42)  # For reproducibility
        mask_shape = (1, 1, config.hi_size, config.hi_size)
        mask = (torch.rand(mask_shape) < ratio).float().to(config.device)
        print(f"Mask shape: {mask.shape}, Observed pixels: {mask.sum().item()}/{mask.numel()}")
        
        operator = lambda x: x * mask
        
    elif operator_name == 'super_resolution':
        # Super resolution operator: downsample to lower resolution
        if 'target_size' in operator_config:
            # Use explicit target size if provided
            target_size = operator_config['target_size']
            if isinstance(target_size, list):
                target_size = tuple(target_size)
            print(f"Super resolution with target size: {target_size}")
        elif 'scale_factor' in operator_config:
            # Compute target size from scale factor
            scale_factor = operator_config['scale_factor']
            target_size = (config.hi_size // scale_factor, config.hi_size // scale_factor)
            print(f"Super resolution with scale factor: {scale_factor} (target size: {target_size})")
        else:
            # Default: scale factor of 4
            scale_factor = 4
            target_size = (config.hi_size // scale_factor, config.hi_size // scale_factor)
            print(f"Super resolution with default scale factor: {scale_factor} (target size: {target_size})")
        
        operator = lambda x: interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
    else:
        raise ValueError(f"Unknown operator name: {operator_name}. "
                        f"Supported options: 'super_resolution', 'sparse_observation'")
    
    # Create noise operator
    print(f"Noise type: {noise_config.get('name', 'gaussian')}")
    noiser = get_noise(**noise_config)

    return operator, noiser

def maybe_create_dir(f):
    if not os.path.exists(f):
        print("making", f)
        os.makedirs(f)

def bad(x):
    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))                                                                                        

def is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False

## if you want to make a grid of images
def to_grid(x, grid_kwargs):
    # nrow = int(np.floor(np.sqrt(x.shape[0])))
    return None

def clip_grad_norm(model, max_norm):
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm = max_norm, 
        norm_type= 2.0, 
        error_if_nonfinite = False
    )

def get_cifar_dataloader(config):
    Flip = T.RandomHorizontalFlip()
    Tens = T.ToTensor()
    transform = T.Compose([Flip, Tens])
    ds = datasets.CIFAR10(
        config.data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    batch_size = config.batch_size

    return DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = True, 
        num_workers = config.num_workers,
        pin_memory = True,
        drop_last = True, 
    )

def setup_wandb(config):
    if not config.use_wandb:
        return
    
    config.wandb_run = wandb.init(
        project = config.wandb_project,
        entity = config.wandb_entity,
        resume = None,
        id = None,
    )

    config.wandb_run_id = config.wandb_run.id

    for key in vars(config):
        item = getattr(config, key)
        if is_type_for_logging(item):
            setattr(wandb.config, key, item)
    print("finished wandb setup")


class DriftModel(nn.Module):
    def __init__(self, config):
        super(DriftModel, self).__init__()
        self.config = config
        c = config
        self._arch = Unet(
            num_classes = c.num_classes,
            # 1
            in_channels = c.C * (c.time_window + 1), 
            # +1 for the initial condition
            out_channels= c.C,
            # c.C =1
            dim = c.unet_channels,
            ## 128
            dim_mults = c.unet_dim_mults ,
            # (1, 2, 2, 2) 
            resnet_block_groups = c.unet_resnet_block_groups,
            # unet_resnet_block_groups = 8
            learned_sinusoidal_cond = c.unet_learned_sinusoidal_cond,
            # unet_learned_sinusoidal_cond = True
            random_fourier_features = c.unet_random_fourier_features,
            # unet_random_fourier_features = False
            learned_sinusoidal_dim = c.unet_learned_sinusoidal_dim,
            # unet_learned_sinusoidal_dim = 32
            attn_dim_head = c.unet_attn_dim_head,
            # unet_attn_dim_head = 64
            attn_heads = c.unet_attn_heads,
            # unet_attn_heads = 4
            use_classes = c.unet_use_classes,
            # unet_use_classes = False 
        )
        num_params = np.sum([int(np.prod(p.shape)) for p in self._arch.parameters()])
        print("Num params in main arch for drift is", f"{num_params:,}")

    def forward(self, zt, t, y, cond=None):
        
        if not self.config.unet_use_classes:
            y = None

        if cond is not None:
            zt = torch.cat([zt, cond], dim = 1)

        out = self._arch(zt, t, y)
        return out


def maybe_subsample(x, subsampling_ratio):
    if subsampling_ratio:
        x = x[ : int(subsampling_ratio * x.shape[0]), ...]
    return x

def maybe_lag(data, time_lag):
    if time_lag > 0:
        inputs = data[:, :-time_lag, ...]
        outputs = data[:, time_lag:, ...]
    else:
        inputs, outputs = data, data
    return inputs, outputs

def maybe_downsample(inputs, outputs, lo_size, hi_size):    
    upsampler = nn.Upsample(scale_factor=int(hi_size/lo_size), mode='nearest')
    hi = interpolate(outputs, size=(hi_size,hi_size),mode='bilinear').reshape([-1,hi_size,hi_size])
    lo = upsampler(interpolate(inputs, size=(lo_size,lo_size),mode='bilinear'))
    return lo, hi

def flatten_time(lo, hi, hi_size):
    hi = hi.reshape([-1,hi_size,hi_size])
    lo = lo.reshape([-1,hi_size,hi_size])
    # make the data N C H W
    hi = hi[:,None,:,:] 
    lo = lo[:,None,:,:] 
    return lo, hi

def loader_from_tensor(lo, hi, batch_size, shuffle):
    return DataLoader(TensorDataset(lo, hi), batch_size = batch_size, shuffle = shuffle)


####################################################################
# NOTE: this is for the flowdas dataset/dataloader
####################################################################

class AE_3D_Dataset(Dataset):
    def __init__(self, input, time_window=7, transform=None, sample_only = False, auto_step = 1):
        """
        Args:
            input: Input data of shape [batch_size, time_index, height, width]
            time_window: Number of previous frames to use for prediction
            transform: Optional transform to be applied
        """
        self.input = input  # Shape: [batch_size, time_index, height, width], jiayx: (B, T, H, W))
        self.time_window = time_window
        self.transform = transform
        self.sample_only = sample_only
        self.auto_step = auto_step

        # Calculate the total number of valid input-target pairs
        self.num_samples = self.input.shape[0] * (self.input.shape[1] - time_window)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Returns:
            ip: Input sequence of shape [time_window, height, width]
            op: Target frame of shape [height, width]
        """
        # Calculate batch index and time index
        # index = 27
        if self.sample_only == False:
            batch_idx = index // (self.input.shape[1] - self.time_window)
            time_idx = index % (self.input.shape[1] - self.time_window - self.auto_step - 1)
        else: 
            batch_idx = index // (self.input.shape[1] - self.time_window)
            time_idx = index % (self.input.shape[1] - self.time_window - self.auto_step - 1)

        # Extract input sequence (previous 3 frames)
        ip = self.input[batch_idx, time_idx : time_idx + self.time_window]  # Shape: [time_window, height, width]

        # Extract target frame (4th frame)
        op = self.input[batch_idx, time_idx + self.time_window : time_idx + self.time_window + self.auto_step]

        # Apply transforms if provided
        if self.transform:
            ip = self.transform(ip)
            op = self.transform(op)

        return ip, op


def get_forecasting_dataloader_flowdas(config, shuffle = False):    
    data_raw, time_raw = torch.load(config.data_fname)
    del time_raw
    
    # Get normalization value from config or use precomputed default
    avg_pixel_norm = getattr(config, 'avg_pixel_norm', 3.0679163932800293)
    data_raw = data_raw / avg_pixel_norm
    new_avg_pixel_norm = 1.0
    
    # Downsample the data along the time dimension
    temporal_ds = getattr(config, 'temporal_downsampling', 2)
    if temporal_ds > 1:
        data_raw = data_raw[:, ::temporal_ds]
    
    # Spatial interpolation to target size
    target_size = (config.hi_size, config.hi_size)
    data_raw = interpolate(
        data_raw, 
        size=target_size,
        mode='bilinear',
        align_corners=False
        )

    u = data_raw
    print('u', u.shape)
    
    ## train dataset
    train_to_val = getattr(config, 'train_val_split', 0.95)

    if not config.sample_only:
        u_train = u[:int(train_to_val * u.shape[0]), ...]
        train_dataset = AE_3D_Dataset(u_train, time_window=config.time_window, transform=None, 
                                     sample_only=config.sample_only, auto_step=config.auto_step)
        train_loader_args = dict(batch_size=config.batch_size, shuffle=True, 
                                num_workers=config.num_workers)
        train_loader = DataLoader(train_dataset, **train_loader_args)
        flowdas_loader = train_loader
    else:
        u_test = u[int(train_to_val * u.shape[0]):, ...]
        test_dataset = AE_3D_Dataset(u_test, time_window=config.time_window, transform=None, 
                                     sample_only=config.sample_only, auto_step=config.auto_step)
        test_loader_args = dict(batch_size=config.sampling_batch_size, shuffle=True, 
                                num_workers=config.num_workers)
        test_loader = DataLoader(test_dataset, **test_loader_args)
        flowdas_loader = test_loader
        
    return flowdas_loader, avg_pixel_norm, new_avg_pixel_norm

####################################################################
# NOTE: plotting functions
####################################################################

def make_one_redblue_plot(x, fname=None):
    """
    Create a red-blue colormap plot.
    
    Args:
        x: 2D numpy array or tensor [H, W]
        fname: Optional filename to save the plot
    
    Returns:
        RGB image as numpy array [H, W, 3] with values in [0, 1]
    """
    plt.ioff()
    fig = plt.figure(figsize=(3, 3), dpi=100) # why dpi=100?
    plt.imshow(x, cmap=sns.cm.icefire, vmin=-2, vmax=2.)
    plt.axis('off')
    
    # Save to file if filename provided
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    
    # Render to RGB array
    fig.canvas.draw()
    # Get RGB buffer from the figure
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB by removing alpha channel
    img_array = img_array[:, :, :3]
    
    plt.close(fig)
    
    # Normalize to [0, 1]
    return img_array.astype(np.float32) / 255.0


def open_redblue_plot_as_tensor(fname):
    return T.ToTensor()(Image.open(fname))


def make_redblue_plots(x, config, name, return_images=False):
    """
    Create red-blue colormap plots for a batch of images.
    
    Args:
        x: Tensor of shape [batch_size, time_frames, H, W]
        config: Config object with home directory
        name: Name prefix for saved files
        return_images: If True, return list of RGB images instead of None
    
    Returns:
        If return_images=True: List of RGB images [H, W, 3] with values in [0, 1]
        If return_images=False: None (legacy behavior)
    """
    plt.ioff()
    x = x.cpu()
    bsz = x.size()[0]
    bsz_2 = x.size()[1]
    
    images = []
    
    if bsz_2 > 1:
        # Multiple time frames
        for i in range(bsz_2):
            fname = config.home + name + f'tmp{i}.jpg' if config.home else None
            img = make_one_redblue_plot(x[0, i, ...].numpy(), fname=fname)
            if return_images:
                images.append(img)
    else:
        # Single time frame
        for i in range(bsz):
            fname = config.home + name + f'tmp{i}_flow.jpg' if config.home else None
            img = make_one_redblue_plot(x[i, 0, ...].numpy(), fname=fname)
            if return_images:
                images.append(img)
    
    if return_images:
        return images
    return None


