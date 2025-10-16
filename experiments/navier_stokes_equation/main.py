import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader 
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from measurements import get_noise
from PIL import Image
import math
import yaml
import argparse
import os
import random
from torch.nn.functional import interpolate
from scipy.fftpack import fft2, ifft2, fftshift
from matplotlib import pyplot as plt


os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'

# interpolant.py
from interpolant import Interpolant

# utils.py
from utils import (
    is_type_for_logging, 
    to_grid, 
    maybe_create_dir,
    clip_grad_norm, 
    get_cifar_dataloader, 
    make_redblue_plots,
    setup_wandb, 
    bad,
    get_forecasting_dataloader_flowdas,
    DriftModel,
    AE_3D_Dataset,
    Config,
    get_measurement_operator_noiser,
    set_random_seed,
)

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class Trainer:

    def __init__(self, config, 
                 load_path = None, sample_only = False, use_wandb = True, 
                 operator = None, noiser = None, 
                 MC_times = 1, exp_times = 1):
        self.config = config
        c = config
        self.operator = operator
        self.noiser = noiser
        self.device = c.device
        self.MC_times = MC_times
        self.exp_times = exp_times
        self.auto_step = c.auto_step

        if sample_only:
            assert load_path is not None

        self.sample_only = sample_only

        c.use_wandb = use_wandb

        self.I = Interpolant(c)

        self.load_path = load_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        if c.dataset == 'cifar':
            self.dataloader = get_cifar_dataloader(c)

        elif c.dataset == 'nse':
            self.dataloader, old_pixel_norm, new_pixel_norm = get_forecasting_dataloader_flowdas(c)
            c.old_pixel_norm = old_pixel_norm
            c.new_pixel_norm = new_pixel_norm
            # NOTE: if doing anything with the samples other than wandb plotting,
            # e.g. if computing metrics like spectra
            # must scale the output by old_pixel_norm to put it back into data space
            # we model the data divided by old_pixel_norm
            
            # Store the dataset for direct access
            self.dataset = self.dataloader.dataset

        self.overfit_batch = next(iter(self.dataloader))

        self.model = DriftModel(c)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=c.base_lr)
        self.step = 0
      
        if self.load_path is not None:
            self.load()

        self.U = torch.distributions.Uniform(low=c.t_min_train, high=c.t_max_train)
        # print('self.U',self.U)
        setup_wandb(c)

        print("\n\n### CONFIG ###\n\n")
        self.print_config()
        print("\n\n")

    def save(self,):
        D = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        maybe_create_dir(f'{self.config.ckpt_save_dir}')
        path = f"{self.config.ckpt_save_dir}/latest.pt"
        torch.save(D, path)
        print(f"saved ckpt at {path}")

    def load(self,):
        D = torch.load(self.load_path)
        self.model.load_state_dict(D['model_state_dict'])
        self.optimizer.load_state_dict(D['optimizer_state_dict'])
        self.step = D['step']
        print(f"loaded! step is {self.step}")

    def print_config(self,):
        c = self.config
        for key in vars(c):
            val = getattr(c, key)
            if is_type_for_logging(val):
                print(key, val)

    def get_time(self, D):
        D['t'] = self.U.sample(sample_shape = (D['N'],)).to(self.device)
        # print('Dt',D['t'])
        return D       

    def wide(self, t):
        return t[:, None, None, None] 

    def drift_to_score(self, D):
        z0 = D['z0']
        zt = D['zt']
        at, bt, adot, bdot, bF = D['at'], D['bt'], D['adot'], D['bdot'], D['bF']
        st, sdot = D['st'], D['sdot']
        numer = (-bt * bF) + (adot * bt * z0) + (bdot * zt) - (bdot * at * z0)
        denom = (sdot * bt - bdot * st) * st * self.wide(D['t'])
        assert not bad(numer)
        assert not bad(denom)
        return numer / denom

    def compute_kinetic_energy_spectrum(self, vorticity, dx):
        # Fourier transform of the vorticity
        vorticity_hat = fft2(vorticity)
        ny, nx = vorticity.shape
        kx = np.fft.fftfreq(nx, dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, dx) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)
        k2 = kx**2 + ky**2
        k2[0, 0] = np.inf  # Avoid division by zero at the zero frequency

        # Stream function in Fourier space
        psi_hat = -vorticity_hat / k2

        # Velocity components in Fourier space
        u_hat = -1j * ky * psi_hat
        v_hat = 1j * kx * psi_hat

        # Kinetic energy in Fourier space
        E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

        # Radial wavenumber
        k = np.sqrt(kx**2 + ky**2)
        k = fftshift(k)
        E_hat = fftshift(E_hat)

        # Bin energy by wavenumber magnitude
        k_bins = np.arange(0.5, np.max(k), 1.0)  # Adjust bin width as needed
        energy_spectrum = np.zeros(len(k_bins) - 1)
        for i in range(len(k_bins) - 1):
            indices = (k >= k_bins[i]) & (k < k_bins[i + 1])
            energy_spectrum[i] = np.sum(E_hat[indices])

        return k_bins[:-1], energy_spectrum

    def taylor_est_x1(self, xt, t, bF, g, use_original_sigma = True, analytical = True):
        if use_original_sigma == True and analytical == False:
            hat_x1 = xt + bF * (1-t) + g * torch.randn_like(xt) * (1-t).sqrt()
        elif use_original_sigma == True and analytical == True:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
        return hat_x1.requires_grad_(True)

    def taylor_est2rd_x1(self, xt, t, bF, g, label, cond,use_original_sigma = True, analytical = True):
        MC_times = self.MC_times
        if use_original_sigma == True and analytical == False:
            hat_x1 = xt + bF * (1-t) + g * torch.randn_like(xt) * (1-t).sqrt()
        elif use_original_sigma == True and analytical == True and MC_times == 1:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            t1 = torch.FloatTensor([1])
            bF2 = self.model(hat_x1,t1.to(hat_x1.device),label,cond=cond).requires_grad_(True)
            hat_x1 =  xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            return hat_x1.requires_grad_(True)
        elif use_original_sigma == True and analytical == True and MC_times != 1:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
            t1 = torch.FloatTensor([1])
            bF2 = self.model(hat_x1,t1.to(hat_x1.device),label,cond=cond).requires_grad_(True)
            hat_x1_list = []
            for i in range(MC_times):
                hat_x1 =  xt + (bF + bF2)/2 * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
                hat_x1_list.append(hat_x1.requires_grad_(True))
            return hat_x1_list

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
            # print('if require grad',x_prev.requires_grad,x_0_hat.requires_grad)
        if isinstance(x_0_hat, torch.Tensor):
            assert 1==0
            difference = (measurement - self.noiser(self.operator(x_0_hat))).requires_grad_(True)
            norm = torch.linalg.norm(difference).requires_grad_(True)
            print('diff',norm)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev,allow_unused=True)[0]
        else:
            difference = 0
            for i in range(len(x_0_hat)):
                difference +=(measurement - self.operator(x_0_hat[i])).requires_grad_(True)
            difference = difference/len(x_0_hat)
            # print('difference',difference)
            norm = torch.linalg.norm(difference).requires_grad_(True)
            # print('difference norm',norm)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev,allow_unused=True)[0]
        return norm_grad, norm
    
    
    def EM(self, base = None, label= None, cond = None, diffusion_fn = None, measurement = None):
        c = self.config
        steps = c.EM_sample_steps
        ## steps == 500
        tmin, tmax = c.t_min_sampling, c.t_max_sampling
        ts = torch.linspace(tmin, tmax, steps)
        dt = ts[1] - ts[0]
        ones = torch.ones(base.shape[0])
 
        # initial condition
        # print('base',base)
        xt = base.requires_grad_(True)
        # diffusion_fn = None means use the diffusion function that you trained with
        # otherwise, for a desired diffusion coefficient, do the model surgery to define
        # the correct drift coefficient

        def step_fn(xt, t, label, measurement,device):
            D = self.I.interpolant_coefs({'t': t, 'zt': xt, 'z0': base})
            t = t.numpy()
            t = torch.FloatTensor(t)
            t = t.to(device)
            bF = self.model(xt, t.to(xt.device), label, cond = cond).requires_grad_(True)
            D['bF'] = bF
            sigma = self.I.sigma(t)
           ##TODO: here sigma = 1-t
            # specified diffusion func
            if diffusion_fn is not None:
                g = diffusion_fn(t)
                s = self.drift_to_score(D)
                f = bF + .5 *  (g.pow(2) - sigma.pow(2)) * s

            # default diffusion func
            else:
                f = bF
                g = sigma

            grad_scale = self.config.grad_scale
            # es_x1 = self.taylor_est_x1(xt,t,bF,g)
            es_x1 = self.taylor_est2rd_x1(xt,t,bF,g,label,cond)
            norm_grad, norm = self.grad_and_value(x_prev=xt, x_0_hat=es_x1, measurement=measurement)
            mu = xt + f * dt
            if norm_grad is None:
                norm_grad = 0
                print('no grad!')
            xt = mu + g * torch.randn_like(mu) * dt.sqrt() - grad_scale * norm_grad
            # xt = mu + g * torch.randn_like(mu) * dt.sqrt()
            return xt, mu # return sample and its mean

        for i, tscalar in enumerate(ts):
            
            if i == 0 and (diffusion_fn is not None):
                # only need to do this when using other diffusion coefficients that you didn't train with
                # because the drift-to-score conversion has a denominator that features 0 at time 0
                # if just sampling with "sigma" (the diffusion coefficient you trained with) you
                # can skip this
                tscalar = ts[1] # 0 + (1/500)

            if (i+1) % 100 == 0:
                print("100 sample steps")
            xt, mu = step_fn(xt, tscalar * ones, label = label, measurement = measurement,device = self.device)
        assert not bad(mu)
        return mu

    def definitely_sample(self,):
      
        c = self.config

        print("\n SAMPLING")

        self.model.eval().to(self.device)
        
        D = self.prepare_batch(batch = None, for_sampling = True)

        EM_args = {'base': D['z0'], 'label': D['label'], 'cond': D['cond']}
        # list diffusion funcs
        # None means use the one you trained with
        diffusion_fns = {
            'g_sigma': None
        }
       


        y = self.operator(D['z1'])
        y = self.noiser(y)

        plotD = {}

        # make samples
        for k in diffusion_fns.keys():
            # Storage for samples across time steps
            samples_per_step = []  # Will store: [(z0_t, measurement_t, sample_t, z1_t), ...]
            
            ## here is the auto-regressive setting：
            for step in range(self.auto_step):
                print(f'Sampling for auto_step {step+1}/{self.auto_step}')
                if step >= 1:
                    ## starting the auto-regressive
                    D['cond'] = torch.cat([D['cond'][:,1:], sample], dim = 1)
                    D['z0'] = D['cond']
                
                # Run all experiments, but only save the last one for visualization
                for id_exp in range(self.exp_times):
                    sample = self.EM(diffusion_fn=diffusion_fns[k], measurement = y[:,step], 
                                    base=D['z0'][:,-1].unsqueeze(1), label=D['label'], cond=D['cond'])

                    # Save numpy arrays and red-blue plots for all experiments (legacy behavior)
                    make_redblue_plots(sample.detach().cpu(), c, f'frame{step}_results_expid_{id_exp}', return_images=False)
                    np.save(c.home+f'results_expid{id_exp}_step{step}.npy', sample.detach().cpu().numpy())
                    
                    make_redblue_plots(D['z1'][:,step:step+1].detach().cpu(), c, f'frame{step}_z1_expid_{id_exp}', return_images=False)
                    np.save(c.home+f'z1_expid{id_exp}_step{step}.npy', D['z1'][:,step].detach().cpu().unsqueeze(1).numpy())

                    # Compute energy spectrum for last experiment
                    if id_exp == self.exp_times - 1:
                        dx = 2 * np.pi / c.hi_size
                        sample_k, sample_spect = self.compute_kinetic_energy_spectrum(
                            sample[0, 0].detach().cpu().numpy(), dx)
                        gt_k, gt_spect = self.compute_kinetic_energy_spectrum(
                            D['z1'][0, step].detach().cpu().numpy(), dx)

                        np.savez(c.home+f'energy_spectrum_gt_step{step}.npz', 
                                k_bins=gt_k, energy_spectrum=gt_spect)
                        np.savez(c.home+f'energy_spectrum_flow_step{step}.npz', 
                                k_bins=sample_k, energy_spectrum=sample_spect)
                    # end for id_exp in range(self.exp_times)
                
                # Store tensors for visualization (only from last experiment)
                # Create colorful images using red-blue colormap
                z0_t = D['z0'][:,-1:].detach().cpu()  # [B, 1, H, W]
                measurement_t = y[:,step:step+1].detach().cpu()  # [B, 1, H, W]
                sample_t = sample.detach().cpu()       # [B, 1, H, W]
                z1_t = D['z1'][:,step:step+1].detach().cpu()  # [B, 1, H, W]
                
                samples_per_step.append((z0_t, measurement_t, sample_t, z1_t))
                print(f'Finished sampling for auto_step {step+1}/{self.auto_step}')
                # end for step in range(self.auto_step)
            
            #########################################################
            # Now create colorful visualizations after all samples are generated
            #########################################################
            print(f"\nCreating visualizations for {self.auto_step} time steps...")
            
            # Convert each tensor to colorful red-blue images
            # NEW LAYOUT: Each column = one time step, each row = z0/measurement/sample/z1
            z0_columns = []
            measurement_columns = []
            sample_columns = []
            z1_columns = []
            
            for step_idx, (z0_t, measurement_t, sample_t, z1_t) in enumerate(samples_per_step):
                # Get RGB images [H, W, 3] for each component
                z0_rgb = make_redblue_plots(z0_t, c, f'wandb_z0_step{step_idx}', return_images=True)[0]
                measurement_rgb = make_redblue_plots(measurement_t, c, f'wandb_measurement_step{step_idx}', return_images=True)[0]
                sample_rgb = make_redblue_plots(sample_t, c, f'wandb_sample_step{step_idx}', return_images=True)[0]
                z1_rgb = make_redblue_plots(z1_t, c, f'wandb_z1_step{step_idx}', return_images=True)[0]
                
                # Collect columns for each type
                z0_columns.append(z0_rgb)
                measurement_columns.append(measurement_rgb)
                sample_columns.append(sample_rgb)
                z1_columns.append(z1_rgb)
            
            # Concatenate each row horizontally: [H, auto_step*W, 3]
            z0_row = np.concatenate(z0_columns, axis=1)
            measurement_row = np.concatenate(measurement_columns, axis=1)
            sample_row = np.concatenate(sample_columns, axis=1)
            z1_row = np.concatenate(z1_columns, axis=1)
            
            # Stack the four rows vertically: [4*H, auto_step*W, 3]
            final_image = np.concatenate([z0_row, measurement_row, sample_row, z1_row], axis=0)
            
            print(f"Created comparison image with shape: {final_image.shape}")
            print(f"  Layout: 4 rows (z0|y|sample|z1) × {self.auto_step} columns (time steps)")
            
            # Convert to wandb.Image
            # final_image is already in [0, 1] range and RGB format [H, W, 3]
            plotD["Comparison: z0|y|sample|z1"] = wandb.Image(
                final_image, 
                caption=f"Step {self.step}: Rows=z0|y|sample|z1, Cols=time steps 0-{self.auto_step-1}"
            )
        
        # end for k in diffusion_fns.keys()

        if self.config.use_wandb:
            wandb.log(plotD, step = self.step)
            print(f"Logged {len(plotD)} images to wandb")


    # @torch.no_grad() # need to remove this because FlowDAS requires grads for sampling!
    def maybe_sample(self,):
        is_time = self.step % self.config.sample_every == 0
        is_logging = self.config.use_wandb
        if is_time and is_logging:
            self.definitely_sample()

    def optimizer_step(self,):
        norm = clip_grad_norm(
            self.model, 
            max_norm = self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return norm

    def image_sq_norm(self, x):
        return x.pow(2).sum(-1).sum(-1).sum(-1)

    def training_step(self, D):
        assert self.model.training
        # print('input1', D['zt'].shape, D['cond'].shape)
        model_out = self.model(D['zt'], D['t'], D['label'], cond = D['cond'])
        target = D['drift_target']
        # print('model output', model_out.shape)
        # print('target', target.shape)
        return self.image_sq_norm(model_out - target).mean()

    def center(self, x):
        return (x * 2.0) - 1.0

    @torch.no_grad()
    def prepare_batch_nse(self, batch = None, for_sampling = False):

        assert not self.config.center_data

        # Special handling for inference with specific trajectory/time indices
        if for_sampling and self.sample_only and self.config.trajectory_index is not None:
            # Get sample directly from dataset
            traj_idx = self.config.trajectory_index
            time_idx = self.config.time_index if self.config.time_index is not None else 0
            
            # Get dataset properties
            T = self.dataset.input.shape[1]  # Total time steps
            N_traj = self.dataset.input.shape[0]  # Number of trajectories
            max_time_idx = T - self.config.time_window - self.config.auto_step - 1
            
            # Validate indices
            if traj_idx >= N_traj:
                raise ValueError(f"trajectory_index={traj_idx} exceeds number of trajectories ({N_traj})")
            if time_idx > max_time_idx:
                raise ValueError(f"time_index={time_idx} too large. Max valid: {max_time_idx} "
                               f"(T={T}, time_window={self.config.time_window}, auto_step={self.config.auto_step})")
            
            print(f"Using trajectory {traj_idx}, starting at time {time_idx} (out of {T} time steps)")
            print(f"  Valid time range: 0 to {max_time_idx}, {max_time_idx}=T-time_window-auto_step-1={T}-{self.config.time_window}-{self.config.auto_step}-1")
            
            # Compute dataset index
            dataset_index = traj_idx * (T - self.config.time_window) + time_idx
            
            # Get sample from dataset
            xlo, xhi = self.dataset[dataset_index]
            
            # Add batch dimension
            xlo = xlo.unsqueeze(0)  # [time_window, H, W] -> [1, time_window, H, W]
            xhi = xhi.unsqueeze(0)  # [auto_step, H, W] -> [1, auto_step, H, W]
        else:
            # Original logic: use batch from dataloader
            xlo, xhi = batch

            # Sample selection logic for training
            if for_sampling and not self.sample_only:
                # Training mode: randomly choose a test sample
                chose = random.randint(0, xlo.shape[0] - self.config.sampling_batch_size - 2)
                xlo = xlo[chose:chose+self.config.sampling_batch_size]
                xhi = xhi[chose:chose+self.config.sampling_batch_size]

        xlo, xhi = xlo.to(self.device), xhi.to(self.device)

        N = xlo.shape[0]
        y = None
        D = {'z0': xlo, 'z1': xhi, 'label': y, 'N': N}
        return D

    @torch.no_grad()
    def prepare_batch_cifar(self, batch = None, for_sampling = False):

        x, y = batch

        if for_sampling:
            x = x[:self.config.sampling_batch_size]
            y = y[:self.config.sampling_batch_size]

        x, y = x.to(self.device), y.to(self.device)

        # possibly center the data, e.g., for images, from [0,1] to [-1,1]
        z1 = self.center(x) if self.config.center_data else x

        D = {'N': z1.shape[0], 'label': y, 'z1': z1}
       
        # point mass base density 
        # since we don't have any conditioning info for this cifar test
        # for PDEs, could set z0 to the previous known condition.
        D['z0'] = torch.zeros_like(D['z1'])

        return D
   
    def prepare_batch(self, batch = None, for_sampling = False):

        
        if batch is None or self.config.overfit:
            batch = self.overfit_batch
        
        # print('batch.shape', batch[0].shape)
        if self.config.dataset == 'cifar':
            D = self.prepare_batch_cifar(batch, for_sampling = for_sampling) 
        else:
            D = self.prepare_batch_nse(batch, for_sampling = for_sampling)

        # get random batch of times
        D = self.get_time(D)

        # conditioning in the model is the initial condition
        D['cond'] = D['z0']

        # interpolant noise
        D['noise'] = torch.randn_like(D['z0'][:,-1].unsqueeze(1))

        # get alpha, beta, etc
        D = self.I.interpolant_coefs(D)
       
        D['zt'] = self.I.compute_zt_flowdas(D)
        
        D['drift_target'] = self.I.compute_target_flowdas(D)
   
        return D

    def sample_ckpt(self,):
        print("not training. just sampling a checkpoint")
        # assert self.config.use_wandb
        self.definitely_sample()
        print("DONE")
        ##TODO: here is the sampling process

    def do_step(self, batch_idx, batch):
        D = self.prepare_batch(batch)
        ## preproccess
        self.model.train()
        loss = self.training_step(D)
        loss.backward()
        grad_norm = self.optimizer_step() # updates self.step 
        self.maybe_sample()

        if self.step % self.config.print_loss_every == 0:
            print("="*40)
            print(f"[Step {self.step:>5}] | Loss: {loss.item():.3f}")
            print("="*40)
            if self.config.use_wandb:
                wandb.log({'loss': loss.item(), 'grad_norm': grad_norm}, step = self.step)

        if self.step % self.config.save_every == 0:
            print(f"{'='*40}\n[Step {self.step}] Saving checkpoint to: {self.config.ckpt_save_dir}\n{'='*40}")
            self.save()

    def fit(self,):
        print("\n\n### STARTING TRAINING ###\n")

        while self.step < self.config.max_steps:

            for batch_idx, batch in enumerate(self.dataloader):
 
                if self.step >= self.config.max_steps:
                    return

                self.do_step(batch_idx, batch)



def main():
    """
    Main entry point for FlowDAS training/sampling.
    
    Configuration Flow:
    1. Load YAML config file (provides defaults for all parameters)
    2. Parse command-line arguments (overrides YAML values if specified)
    3. Create Config object: Config(config_dict=yaml_config, **cmd_overrides)
       - YAML values are used as defaults
       - Command-line args override YAML values
       - All final values stored in conf object
    4. Everything downstream uses conf object (single source of truth)
    
    Note: load_path is now in the config under sampling section
    """
    parser = argparse.ArgumentParser(description='FlowDAS for Navier-Stokes Equations')
    
    # Config file
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                       help='Path to main config YAML file')
    parser.add_argument('--task_config', type=str, default=None,
                       help='Path to task-specific config (deprecated, use --config instead)')
    
    # Override arguments (these will override config file values)
    parser.add_argument('--dataset', type=str, choices=['cifar', 'nse'], default=None,
                       help='Dataset name')
    parser.add_argument('--load_path', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--use_wandb', type=int, default=None,
                       help='Use Weights & Biases logging (1=True, 0=False)')
    parser.add_argument('--sigma_coef', type=float, default=None,
                       help='Diffusion coefficient')
    parser.add_argument('--beta_fn', type=str, default=None, choices=['t', 't^2'],
                       help='Beta function type')
    parser.add_argument('--debug', type=int, default=None,
                       help='Debug mode (1=True, 0=False)')
    parser.add_argument('--sample_only', type=int, default=None,
                       help='Only run sampling, no training (1=True, 0=False)')
    parser.add_argument('--overfit', type=int, default=None,
                       help='Overfit to single batch (1=True, 0=False)')
    parser.add_argument('--ckpt_save_dir', type=str, default=None,
                       help='Directory to save checkpoints')
    parser.add_argument('--savedir', type=str, default=None,
                       help='Directory to save output images')
    parser.add_argument('--nse_datapath', type=str, default=None,
                       help='Path to NSE data file')
    parser.add_argument('--exp_times', type=int, default=None,
                       help='Number of experiments for energy spectrum computation')
    parser.add_argument('--MC_times', type=int, default=None,
                       help='Number of Monte Carlo samples to estimate X1')
    parser.add_argument('--auto_step', type=int, default=None,
                       help='Number of auto-regressive steps')
    parser.add_argument('--time_window', type=int, default=None,
                       help='Number of previous frames for conditioning')
    parser.add_argument('--grad_scale', type=float, default=None,
                       help='Gradient scaling factor for data assimilation')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for reproducibility (None for non-deterministic)')
    parser.add_argument('--trajectory_index', type=int, default=None,
                       help='Which trajectory to use for inference (0 to N_trajectories-1)')
    parser.add_argument('--time_index', type=int, default=None,
                       help='Starting time index in trajectory (0 to T-time_window-auto_step-1)')
    
    args = parser.parse_args()
    
    # Load main config file
    config_path = args.config
    if args.task_config is not None:
        # Support legacy task_config argument
        config_path = args.task_config
        print(f"Warning: --task_config is deprecated, please use --config instead")
    
    print(f"\nLoading config from: {config_path}")
    full_config = load_yaml(config_path)
    
    # Device setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    print("\n\n### COMMAND LINE ARGS ###\n")
    for k in vars(args):
        val = getattr(args, k)
        if val is not None:
            print(f"{k}: {val}")
    
    # Build kwargs for Config, only including non-None overrides
    config_kwargs = {
        'device': device,
    }
    
    # Add command-line overrides (only if specified)
    if args.dataset is not None:
        config_kwargs['dataset'] = args.dataset
    if args.debug is not None:
        config_kwargs['debug'] = bool(args.debug)
    if args.overfit is not None:
        config_kwargs['overfit'] = bool(args.overfit)
    if args.sigma_coef is not None:
        config_kwargs['sigma_coef'] = args.sigma_coef
    if args.beta_fn is not None:
        config_kwargs['beta_fn'] = args.beta_fn
    if args.savedir is not None:
        config_kwargs['savedir'] = args.savedir
    if args.nse_datapath is not None:
        config_kwargs['nse_datapath'] = args.nse_datapath
    if args.auto_step is not None:
        config_kwargs['auto_step'] = args.auto_step
    if args.sample_only is not None:
        config_kwargs['sample_only'] = bool(args.sample_only)
    if args.time_window is not None:
        config_kwargs['time_window'] = args.time_window
    if args.ckpt_save_dir is not None:
        config_kwargs['ckpt_save_dir'] = args.ckpt_save_dir
    if args.use_wandb is not None:
        config_kwargs['use_wandb'] = bool(args.use_wandb)
    if args.MC_times is not None:
        config_kwargs['MC_times'] = args.MC_times
    if args.exp_times is not None:
        config_kwargs['exp_times'] = args.exp_times
    if args.load_path is not None:
        config_kwargs['load_path'] = args.load_path
    if args.grad_scale is not None:
        config_kwargs['grad_scale'] = args.grad_scale
    if args.random_seed is not None:
        config_kwargs['random_seed'] = args.random_seed
    if args.trajectory_index is not None:
        config_kwargs['trajectory_index'] = args.trajectory_index
    if args.time_index is not None:
        config_kwargs['time_index'] = args.time_index
    
    print("\n\n### CREATING CONFIG ###\n")
    conf = Config(config_dict=full_config, **config_kwargs)
    
    # Set random seed for reproducibility
    print("\n### RANDOM SEED SETUP ###\n")
    set_random_seed(conf.random_seed)
    print()

    # Create measurement operator and noiser from config
    # Config object contains all needed info (measurement_config, hi_size, device)
    operator, noiser = get_measurement_operator_noiser(conf)

    ##TODO reminder: 
    # to add more condition: 
    # 1. use new checkpoint 
    # 2. change the time_window para in dataset setting'train_dataset = new_AE_3D_Dataset(u_train, time_window=7,transform=None)' 
    # 3. change the model para: in_channels = c.C * 8

    # Create trainer with config and operator/noiser
    # All parameters come from conf (which contains YAML defaults + command-line overrides)
    trainer = Trainer(
        conf, 
        load_path=conf.load_path,      # From config (can be overridden by command-line)
        sample_only=conf.sample_only,  # From config
        use_wandb=conf.use_wandb,      # From config
        operator=operator,
        noiser=noiser,
        MC_times=conf.MC_times,        # From config
        exp_times=conf.exp_times       # From config
    )

    if conf.sample_only:
        trainer.sample_ckpt()
    else:
        ## TODO: remember you should always change the train/test dataset 
        trainer.fit()

if __name__ == '__main__':
    main()
