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

# local
from unet import Unet

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
            ## 1
            in_channels = c.C * 4, # times two/(four for lookback condition 03/21/2025 chensiyi) for conditioning
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

        # print('input', zt.shape)
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

def get_forecasting_dataloader(config, shuffle = False):
    data_raw, time_raw = torch.load(config.data_fname)
    del time_raw
    

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    #Ntj, Nts, Nx, Ny = data_raw.size() 
    #avg_pixel_norm = torch.norm(data_raw,dim=(2,3),p='fro').mean() / np.sqrt(Nx*Ny)
    print('data_raw',data_raw.shape)
    avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    data_raw = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    
    lo, hi = maybe_lag(data_raw, config.time_lag)
    print('lo1, hi1',lo.shape, hi.shape)
    lo=lo[190:]
    hi=hi[190:]
    lo, hi = maybe_downsample(lo, hi, config.lo_size, config.hi_size)
    print('lo2, hi2',lo.shape, hi.shape)
    lo, hi = flatten_time(lo, hi, config.hi_size)
    print('lo3, hi3',lo.shape, hi.shape)
    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)
    print('lo4, hi4',lo.shape, hi.shape)
    # assert 1==0
    # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)
    return loader, avg_pixel_norm, new_avg_pixel_norm

def new_get_forecasting_dataloader_4train(config, shuffle = False):
    # data_raw, time_raw = torch.load(config.data_fname)
    # del time_raw
    

    # # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    # #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    
    
    # #Ntj, Nts, Nx, Ny = data_raw.size() 
    # #avg_pixel_norm = torch.norm(data_raw,dim=(2,3),p='fro').mean() / np.sqrt(Nx*Ny)
    # print('data_raw',data_raw.shape)
    # avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori
    # data_raw = data_raw/avg_pixel_norm
    # new_avg_pixel_norm = 1.0

    # # here "lo" will be the conditioning info (initial condition) and "hi" will be the target
    # # lo is x_t and hi is x_{t+tau}, and lo might be lower res than hi

    
    # lo, hi = maybe_lag(data_raw, config.time_lag)
    # print('lo1, hi1',lo.shape, hi.shape)
    # lo=lo[190:]
    # hi=hi[190:]
    # lo, hi = maybe_downsample(lo, hi, config.lo_size, config.hi_size)
    # print('lo2, hi2',lo.shape, hi.shape)
    # lo, hi = flatten_time(lo, hi, config.hi_size)
    # print('lo3, hi3',lo.shape, hi.shape)
    # lo = maybe_subsample(lo, config.subsampling_ratio)
    # hi = maybe_subsample(hi, config.subsampling_ratio)
    # print('lo4, hi4',lo.shape, hi.shape)
    # # assert 1==0
    # # now they are image shaped. Be sure to shuffle to de-correlate neighboring samples when training. 
    # loader = loader_from_tensor(lo, hi, config.batch_size, shuffle = shuffle)
    data_raw, time_raw = torch.load('/scratch/qingqu_root/qingqu/jiayx/FlowDAS/data_file.pt')
    del time_raw
    avg_pixel_norm = 3.0679163932800293
    data_raw = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0
    # data_raw = normalize_slices(data_raw)
    # first skipping
    data_raw = data_raw[:,::2]
    data_raw =interpolate(
    data_raw, 
    size=(128, 128),  # Target spatial dimensions
    mode='bilinear',   # Interpolation mode
    align_corners=False  # Adjust based on your use case (see note below)
                        )
    # data_raw = data_raw[:,:48]
    # data_raw = data_raw.reshape(-1, 6, 128, 128)
    ## 3 to 3
    u = data_raw
    print('u', u.shape)
    train_to_val = 0.95
    # rand_array = np.random.permutation(1500)
    # print(rand_array)
    u_train = u[int(train_to_val*u.shape[0]):, ...]
    train_dataset = new_AE_3D_Dataset(u_train,time_window=3,transform=None,sample_only = config.sample_only, auto_step = config.auto_step)
    train_loader_args = dict(batch_size=config.batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(train_dataset, **train_loader_args)
     
        
    return train_loader, avg_pixel_norm, new_avg_pixel_norm

def make_one_redblue_plot(x, fname):
    plt.ioff()
    fig = plt.figure(figsize=(3,3))
    plt.imshow(x, cmap=sns.cm.icefire, vmin=-2, vmax=2.)
    plt.axis('off')
    plt.savefig(fname, bbox_inches = 'tight')
    plt.close("all")         

def open_redblue_plot_as_tensor(fname):
    return T.ToTensor()(Image.open(fname))

def make_redblue_plots(x, config, name):
    plt.ioff()
    x = x.cpu()
    bsz = x.size()[0]
    bsz_2 = x.size()[1]
    if bsz_2 >1:
        for i in range(bsz):
            make_one_redblue_plot(x[i,0,...], fname = config.home + name +  f'tmp{i}_score.jpg')
            make_one_redblue_plot(x[i,1,...], fname = config.home + name +  f'second_tmp{i}_score.jpg')
    else:
        for i in range(bsz):
            make_one_redblue_plot(x[i,0,...], fname = config.home + name +  f'frame{i}_tmp{i}_flow.jpg')
    # tensor_img = T.ToTensor()(Image.open(config.home + name +f'tmp0_flow.jpg'))
    # C, H, W = tensor_img.size()
    # out = torch.zeros((bsz,C,H,W))
    # for i in range(bsz):
    #     out[i,...] = open_redblue_plot_as_tensor(config.home + name + f'tmp{i}_flow.jpg')
    return None

class new_AE_3D_Dataset(Dataset):
    def __init__(self, input, time_window=7, transform=None, sample_only = False, auto_step = 1):
        """
        Args:
            input: Input data of shape [batch_size, time_index, height, width]
            time_window: Number of previous frames to use for prediction
            transform: Optional transform to be applied
        """
        self.input = input  # Shape: [batch_size, time_index, height, width]
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
        if self.sample_only == False:
            batch_idx = index // (self.input.shape[1] - self.time_window)
            time_idx = index % (self.input.shape[1] - self.time_window)
        else: 
            batch_idx = index // (self.input.shape[1] - self.time_window)
            time_idx = index % (self.input.shape[1] - self.time_window - self.auto_step - 1)

        # Extract input sequence (previous 3 frames)
        ip = self.input[batch_idx, time_idx:time_idx + self.time_window]  # Shape: [time_window, height, width]

        # Extract target frame (4th frame)
        if self.sample_only == False:
            op = self.input[batch_idx, time_idx + self.time_window]  # Shape: [height, width]
        else:
            op = self.input[batch_idx, time_idx + self.time_window:time_idx + self.time_window + self.auto_step]

        # Apply transforms if provided
        if self.transform:
            ip = self.transform(ip)
            op = self.transform(op)

        return ip, op
