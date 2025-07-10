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

from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange
from lightning import LightningDataModule, seed_everything
from sevir_dataloader import SEVIRDataLoader

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sevir_cmap import get_cmap, VIL_COLORS, VIL_LEVELS
# from ...utils.path import default_dataset_sevir_dir, default_dataset_sevirlr_dir
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
default_exps_dir = os.path.abspath(os.path.join(root_dir, "experiments"))
default_dataset_dir = os.path.abspath(os.path.join(root_dir, "datasets"))
default_dataset_sevir_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevir"))
default_dataset_sevirlr_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevirlr"))
from augmentation import TransformsFixRotation
from typing import Union, Dict, Sequence, Tuple, List, Optional
import pandas as pd

# local
from unet import Unet
def save_vis_step_end(
            data_idx: int,
            context_seq: np.ndarray,
            target_seq: np.ndarray,
            pred_seq: Union[np.ndarray, Sequence[np.ndarray]],
            pred_label: Union[str, Sequence[str]] = None,
            label_mode: str = "name",
            mode: str = "test",
            prefix: str = "",
            suffix: str = "", ):
        r"""
        Parameters
        ----------
        data_idx
        context_seq, target_seq, pred_seq:   np.ndarray
            layout should not include batch
        mode:   str
        """
        if mode == "train":
            example_data_idx_list = self.train_example_data_idx_list
        elif mode == "val":
            example_data_idx_list = self.val_example_data_idx_list
        elif mode == "test":
            example_data_idx_list = self.test_example_data_idx_list
        else:
            raise ValueError(f"Wrong mode {mode}! Must be in ['train', 'val', 'test'].")
        if label_mode == "name":
            # use the given label
            context_label = "context"
            target_label = "target"
        elif label_mode == "avg_int":
            context_label = f"context\navg_int={np.mean(context_seq):.4f}"
            target_label = f"target\navg_int={np.mean(target_seq):.4f}"
            if isinstance(pred_label, Sequence):
                pred_label = [f"{label}\navg_int={np.mean(seq):.4f}" for label, seq in zip(pred_label, pred_seq)]
            elif isinstance(pred_label, str):
                pred_label = f"{pred_label}\navg_int={np.mean(pred_seq):.4f}"
            else:
                raise TypeError(f"Wrong pred_label type {type(pred_label)}! must be in [str, Sequence[str]].")
        else:
            raise NotImplementedError
        if isinstance(pred_seq, Sequence):
            seq_list = [context_seq, target_seq] + list(pred_seq)
            label_list = [context_label, target_label] + pred_label
        else:
            seq_list = [context_seq, target_seq, pred_seq]
            label_list = [context_label, target_label, pred_label]
        if data_idx in example_data_idx_list:
            png_save_name = f"{prefix}{mode}_epoch_{self.current_epoch}_data_{data_idx}{suffix}.png"
            print(os.path.join(self.example_save_dir, png_save_name))
            for i in seq_list:
                print(i.shape)
            for j in label_list:
                print(j)
            """
            /gpfs/accounts/qingqu_root/qingqu1/siyiche/PreDiff/experiments/tmp_sevirlr_prediff2/examples/test_epoch_0_data_0_rank0.png
            (7, 128, 128, 1)
            (6, 128, 128, 1)
            (6, 128, 128, 1)
            (6, 128, 128, 1)
            context
            target
            PreDiff_aligned_pred_0
            PreDiff_pred_0
            """
            vis_sevir_seq(
                save_path=os.path.join(self.example_save_dir, png_save_name),
                seq=seq_list,
                label=label_list,
                interval_real_time=10,
                plot_stride=1, fs=self.oc.eval.fs,
                label_offset=self.oc.eval.label_offset,
                label_avg_int=self.oc.eval.label_avg_int, )


def vis_sevir_seq(
        save_path,
        seq: Union[np.ndarray, Sequence[np.ndarray]],
        label: Union[str, Sequence[str]] = "pred",
        norm: Optional[Dict[str, float]] = None,
        interval_real_time: float = 10.0,  plot_stride=2,
        label_rotation=0,
        label_offset=(-0.06, 0.4),
        label_avg_int=False,
        fs=10,
        max_cols=10, ):
    """
    Parameters
    ----------
    seq:    Union[np.ndarray, Sequence[np.ndarray]]
        shape = (T, H, W). Float value 0-1 after `norm`.
    label:  Union[str, Sequence[str]]
        label for each sequence.
    norm:   Union[str, Dict[str, float]]
        seq_show = seq * norm['scale'] + norm['shift']
    interval_real_time: float
        The minutes of each plot interval
    max_cols: int
        The maximum number of columns in the figure.
    """

    def cmap_dict(s):
        return {'cmap': get_cmap(s, encoded=True)[0],
                'norm': get_cmap(s, encoded=True)[1],
                'vmin': get_cmap(s, encoded=True)[2],
                'vmax': get_cmap(s, encoded=True)[3]}

    # cmap_dict = lambda s: {'cmap': get_cmap(s, encoded=True)[0],
    #                        'norm': get_cmap(s, encoded=True)[1],
    #                        'vmin': get_cmap(s, encoded=True)[2],
    #                        'vmax': get_cmap(s, encoded=True)[3]}

    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    # font.set_name('Times New Roman')
    fontproperties.set_size(fs)
    # font.set_weight("bold")

    if isinstance(seq, Sequence):
        seq_list = [ele.astype(np.float32) for ele in seq]
        assert isinstance(label, Sequence) and len(label) == len(seq)
        label_list = label
    elif isinstance(seq, np.ndarray):
        seq_list = [seq.astype(np.float32), ]
        assert isinstance(label, str)
        label_list = [label, ]
    else:
        raise NotImplementedError
    if label_avg_int:
        label_list = [f"{ele1}\nAvgInt = {np.mean(ele2): .3f}"
                      for ele1, ele2 in zip(label_list, seq_list)]
    # plot_stride
    seq_list = [ele[::plot_stride, ...] for ele in seq_list]
    seq_len_list = [len(ele) for ele in seq_list]

    max_len = max(seq_len_list)

    max_len = min(max_len, max_cols)
    seq_list_wrap = []
    label_list_wrap = []
    seq_len_list_wrap = []
    for i, (seq, label, seq_len) in enumerate(zip(seq_list, label_list, seq_len_list)):
        num_row = math.ceil(seq_len / max_len)
        for j in range(num_row):
            slice_end = min(seq_len, (j + 1) * max_len)
            seq_list_wrap.append(seq[j * max_len: slice_end])
            if j == 0:
                label_list_wrap.append(label)
            else:
                label_list_wrap.append("")
            seq_len_list_wrap.append(min(seq_len - j * max_len, max_len))

    if norm is None:
        norm = {'scale': 255,
                'shift': 0}
    nrows = len(seq_list_wrap)
    fig, ax = plt.subplots(nrows=nrows,
                           ncols=max_len,
                           figsize=(3 * max_len, 3 * nrows))

    for i, (seq, label, seq_len) in enumerate(zip(seq_list_wrap, label_list_wrap, seq_len_list_wrap)):
        ax[i][0].set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=label_rotation)
        ax[i][0].yaxis.set_label_coords(label_offset[0], label_offset[1])
        for j in range(0, max_len):
            if j < seq_len:
                x = seq[j] * norm['scale'] + norm['shift']
                ax[i][j].imshow(x, **cmap_dict('vil'))
                if i == len(seq_list) - 1 and i > 0:  # the last row which is not the `in_seq`.
                    ax[-1][j].set_title(f"Min {int(interval_real_time * (j + 1) * plot_stride)}",
                                        y=-0.25, fontproperties=fontproperties)
            else:
                ax[i][j].axis('off')

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])

    # Legend of thresholds
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                             label=f'{int(VIL_LEVELS[i - 1])}-{int(VIL_LEVELS[i])}')
                       for i in range(1, num_thresh_legend + 1)]
    ax[0][0].legend(handles=legend_elements, loc='center left',
                    bbox_to_anchor=(-1.2, -0.),
                    borderaxespad=0, frameon=False, fontsize='10')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path)
    plt.close(fig)


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
            in_channels = c.C * 7, # times two/(four for lookback condition 03/21/2025 chensiyi) for conditioning
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
    data_raw, time_raw = torch.load('/scratch/qingqu_root/qingqu1/siyiche/data_file.pt')
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
    ## train dataset
    train_to_val = 0.95
    # rand_array = np.random.permutation(1500)
    # print(rand_array)
    if config.sample_only == False:
        u_train = u[:int(train_to_val*u.shape[0]), ...]
        train_dataset = new_AE_3D_Dataset(u_train,time_window=1,transform=None,sample_only = config.sample_only, auto_step = config.auto_step)
        train_loader_args = dict(batch_size=config.batch_size, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_dataset, **train_loader_args)
    else:
        u_test = u[int(train_to_val*u.shape[0]):, ...]
        train_dataset = new_AE_3D_Dataset(u_test,time_window=10,transform=None,sample_only = config.sample_only, auto_step = config.auto_step)
        train_loader_args = dict(batch_size = 1, shuffle=True, num_workers=4)
        train_loader = DataLoader(train_dataset, **train_loader_args)
     
        
    return train_loader, avg_pixel_norm, new_avg_pixel_norm

def new_get_forecasting_dataloader_4train_sevir(config, shuffle = False):
    def get_sevir_datamodule(dataset_cfg,
                             micro_batch_size: int = 1,
                             num_workers: int = 4):
        dm = SEVIRLightningDataModule(
            seq_len=dataset_cfg["seq_len"],
            sample_mode=dataset_cfg["sample_mode"],
            stride=dataset_cfg["stride"],
            batch_size=dataset_cfg["batch_size"],
            layout=dataset_cfg["layout"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            aug_mode=dataset_cfg["aug_mode"],
            ret_contiguous=False,
            # datamodule_only
            dataset_name=dataset_cfg["dataset_name"],
            start_date=dataset_cfg["start_date"],
            train_test_split_date=dataset_cfg["train_test_split_date"],
            end_date=dataset_cfg["end_date"],
            val_ratio=dataset_cfg["val_ratio"],
            num_workers=dataset_cfg['num_workers'],
            sevir_dir = dataset_cfg["sevir_dir"],
             )
        return dm
    dm = get_sevir_datamodule(config)
    dm.prepare_data()
    dm.setup()
    # train_dataloader = dm.train_dataloader()
    # total_sum = 0.0
    # total_count = 0
    # for batch in train_dataloader:
    #             print(batch.shape)
    #             # Assuming batch contains your numerical data
    #             # If batch is a tensor:
    #             total_sum += batch.sum().item()  # .item() converts to Python float
    #             total_count += batch.numel()     # Number of elements in the tensor
                
    #             # If batch is a tuple/list containing the data tensor:
    #             # data = batch[0]  # Adjust index as needed
    #             # total_sum += data.sum().item()
    #             # total_count += data.numel()

    #         # Compute final mean
    # dataset_mean = total_sum / total_count
    # print(f"Dataset mean: {dataset_mean}")
    # total_var = 0.0
    # for batch in train_dataloader:
    #     data = batch
    #     total_var += ((data - dataset_mean)**2).sum().item()

    # dataset_std = (total_var / total_count)**0.5
    # print(f"Dataset std: {dataset_std}")
    # assert 1==0
    if config["sample_only"] == False:
        return dm.train_dataloader(), 1, 1
    else:
        return dm.test_dataloader(), 1, 1

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




class SEVIRTorchDataset(TorchDataset):

    orig_dataloader_layout = "NHWT"
    orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
    aug_layout = "THW"

    def __init__(self,
                 seq_len: int = 25,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "THWC",
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=1,
            layout=self.orig_dataloader_layout,
            num_shard=1,
            rank=0,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x:x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"].squeeze(0)
        if self.aug_mode != "0":
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
            data = self.aug(data)
            data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
        else:
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
        # print('data', data.shape)
        
        # print('self.ret_contigous', self.ret_contiguous)
        # assert 1==0
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data

    def __len__(self):
        return self.sevir_dataloader.__len__()
class SEVIRLightningDataModule(LightningDataModule):

    def __init__(self,
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "NTHWC",
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True,
                 # datamodule_only
                 dataset_name: str = "sevir",
                 sevir_dir: str = None,
                 start_date: Tuple[int] = None,
                 train_test_split_date: Tuple[int] = (2019, 6, 1),
                 end_date: Tuple[int] = None,
                 val_ratio: float = 0.1,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 ):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        assert layout[0] == "N"
        self.layout = layout.replace("N", "")
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.batch_size = batch_size
        print('batch_size', batch_size)
        # assert 1==0
        self.num_workers = num_workers
        self.seed = seed
        if sevir_dir is not None:
            sevir_dir = os.path.abspath(sevir_dir)
        if dataset_name == "sevir":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevir_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevirlr":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevirlr_dir
            print('sevir_dir', sevir_dir)
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir' or 'sevirlr'.")
        self.dataset_name = dataset_name
        self.sevir_dir = sevir_dir
        print(' self.sevir_dir',  self.sevir_dir)
        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width
        # train val test split
        self.start_date = datetime.datetime(*start_date) \
            if start_date is not None else None
        self.train_test_split_date = datetime.datetime(*train_test_split_date) \
            if train_test_split_date is not None else None
        self.end_date = datetime.datetime(*end_date) \
            if end_date is not None else None
        self.val_ratio = val_ratio

    def prepare_data(self) -> None:
        if os.path.exists(self.sevir_dir):
            # Further check
            assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
            assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
        else:
            if self.dataset_name == "sevir":
                download_SEVIR(save_dir=os.path.dirname(self.sevir_dir))
            elif self.dataset_name == "sevirlr":
                download_SEVIRLR(save_dir=os.path.dirname(self.sevir_dir))
            else:
                raise NotImplementedError

    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if stage in (None, "fit"):
            sevir_train_val = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=True,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.start_date,
                end_date=self.train_test_split_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode=self.aug_mode,
                ret_contiguous=self.ret_contiguous,)
            

            # Iterate through the dataset
            

            self.sevir_train, self.sevir_val = random_split(
                dataset=sevir_train_val,
                lengths=[1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(self.seed))
        if stage in (None, "test"):
            self.sevir_test = SEVIRTorchDataset(
                sevir_catalog=self.catalog_path,
                sevir_data_dir=self.raw_data_dir,
                raw_seq_len=self.raw_seq_len,
                split_mode="uneven",
                shuffle=False,
                seq_len=self.seq_len,
                stride=self.stride,
                sample_mode=self.sample_mode,
                layout=self.layout,
                start_date=self.train_test_split_date,
                end_date=self.end_date,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                verbose=self.verbose,
                aug_mode="0",
                ret_contiguous=self.ret_contiguous,)

    def train_dataloader(self):
        return DataLoader(self.sevir_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.sevir_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.sevir_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)






