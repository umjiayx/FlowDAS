import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import LinearLR
import torchvision.utils
from torchvision import transforms, datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.utils import make_grid
from dps.measurements import get_noise
from PIL import Image
import math
import yaml
import argparse
import os
import random
from torch.nn.functional import interpolate
os.environ['WANDB_DIR'] = os.getcwd() + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = os.getcwd() + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = os.getcwd() + '/wandb/.config/'
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# local
from utils import (
    is_type_for_logging, 
    to_grid, 
    maybe_create_dir,
    clip_grad_norm, 
    get_cifar_dataloader, 
    get_forecasting_dataloader,
    make_redblue_plots,
    setup_wandb, 
    bad,
)
from utils_forlookback_sevir_vil import new_get_forecasting_dataloader_4train_sevir, new_AE_3D_Dataset, DriftModel
from interpolant_new import Interpolant

class Trainer:

    def __init__(self, config, load_path = None, sample_only = False, use_wandb = True, operator = None, noiser = None):

        self.config = config
        c = config
        self.operator = operator
        self.noiser = noiser
        self.device = c.device

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
            config_sevir = {'dataset_name': 'sevirlr', 'img_height': 128, 'img_width': 128, 'in_len': 1, 'out_len': 1, 'seq_len': 2, 'plot_stride': 1, 'interval_real_time': 10, 'sample_mode': 'sequent', 'stride': 6, 'layout': 'NTHWC', 'start_date': None, 'train_test_split_date': [2019, 6, 1], 'end_date': None, 'val_ratio': 0.1, 'metrics_mode': '0', 'metrics_list': ['csi', 'pod', 'sucr', 'bias'], 'threshold_list': [16, 74, 133, 160, 181, 219], 'aug_mode': '2', 'sevir_dir':'/scratch/qingqu_root/qingqu1/siyiche/PreDiff/datasets/sevirlr/', 'batch_size':50, 'sample_only':c.sample_only, 'num_workers': 16,}
            self.dataloader, old_pixel_norm, new_pixel_norm = new_get_forecasting_dataloader_4train_sevir(config_sevir)
            
            c.old_pixel_norm = old_pixel_norm
            c.new_pixel_norm = new_pixel_norm
            # NOTE: if doing anything with the samples other than wandb plotting,
            # e.g. if computing metrics like spectra
            # must scale the output by old_pixel_norm to put it back into data space
            # we model the data divided by old_pixel_norm

        self.overfit_batch = next(iter(self.dataloader))

        self.model = DriftModel(c)

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=c.base_lr)
        self.scheduler = LinearLR(self.optimizer, start_factor=1, end_factor=0.001, total_iters=c.max_steps)
        self.step = 0
      
        if self.load_path is not None:
            self.load()

        self.U = torch.distributions.Uniform(low=c.t_min_train, high=c.t_max_train)
        # print('self.U',self.U)
        setup_wandb(c)
        self.print_config()

    def save(self,):
        D = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }
        maybe_create_dir('./ckpts_condition1_sevir_0506_l1_2en5')
        path = f"./ckpts_condition1_sevir_0506_l1_2en5/latest.pt"
        torch.save(D, path)
        print("saved ckpt at ", path)

    def load(self,):
        D = torch.load(self.load_path)
        self.model.load_state_dict(D['model_state_dict'])
        self.optimizer.load_state_dict(D['optimizer_state_dict'])
        self.step = D['step']
        print("loaded! step is", self.step)

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

    def taylor_est_x1(self, xt, t, bF, g, use_original_sigma = True, analytical = True):
        if use_original_sigma == True and analytical == False:
            hat_x1 = xt + bF * (1-t) + g * torch.randn_like(xt) * (1-t).sqrt()
        elif use_original_sigma == True and analytical == True:
            hat_x1 = xt + bF * (1-t) + torch.randn_like(xt) * (2/3 - t.sqrt()+(1/3) * (t.sqrt())**3)
        return hat_x1.requires_grad_(True)

    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
            print('if require grad',x_prev.requires_grad,x_0_hat.requires_grad)
            difference = (measurement - self.noiser(self.operator(x_0_hat))).requires_grad_(True)
            norm = torch.linalg.norm(difference).requires_grad_(True)
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
        print('base',base)
        xt = base.requires_grad_(True)
        # assert 1==0
        # diffusion_fn = None means use the diffusion function that you trained with
        # otherwise, for a desired diffusion coefficient, do the model surgery to define
        # the correct drift coefficient

        def step_fn(xt, t, label, measurement,device):
            D = self.I.interpolant_coefs({'t': t, 'zt': xt, 'z0': base})
            print('DIVICE',xt.device)
            print(t.device)
            print(t.dtype)
            t = t.numpy()
            print(t)
            t = torch.FloatTensor(t)
            print('xt',xt)
            print('t',t)
            t = t.to(device)
            print(label)
            print(cond)
            print('before bf',xt.requires_grad)
            bF = self.model(xt, t.to(xt.device), label, cond = cond).requires_grad_(True)
            print('after bf',xt.requires_grad,bF.requires_grad)
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

            scale = 1
            es_x1 = self.taylor_est_x1(xt,t,bF,g)
            norm_grad, norm = self.grad_and_value(x_prev=xt, x_0_hat=es_x1, measurement=measurement)
            mu = xt + f * dt
            if norm_grad ==None:
                norm_grad = 0
                print('no grad!')
            xt = mu + g * torch.randn_like(mu) * dt.sqrt() - scale * norm_grad
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

        print("SAMPLING")

        self.model.eval().to(self.device)
        
        D = self.prepare_batch(batch = None, for_sampling = True)
        for i in D:
            print(i,D[i])

        EM_args = {'base': D['z0'], 'label': D['label'], 'cond': D['cond']}
        print('EM_args',D['z0'])
        # list diffusion funcs
        # None means use the one you trained with
        diffusion_fns = {
            'g_sigma': None,
            'g_other': lambda t: c.sigma_coef * self.wide(1-t).pow(4),
        }
       
        if c.dataset == 'cifar':
            preprocess_fn = lambda x : to_grid(x, c.grid_kwargs)

        else:
            assert c.dataset == 'nse'
            preprocess_fn = lambda x, name: to_grid(make_redblue_plots(x, c, name), c.grid_kwargs)


        print('preprocess',D['z0'])
        z1 = preprocess_fn(D['z1'], 'z1')
        z0 = preprocess_fn(D['z0'], 'z0')
        print('after process',z0,D['z0'])
        cond = preprocess_fn(D['cond'], 'cond')
        # print('DZ0DIVICE',D['z0'].device)
        print(D['z0'].shape)
       
        y = self.operator(D['z1'])
        y = self.noiser(y)
        ## FIXME : what's wrong with the observation making code, this will harm the code?!
        print('observation',y)
        y_ob = preprocess_fn(y, 'observation')
        plotD = {}

        # make samples
        for k in diffusion_fns.keys():
           
            print('sampling for diffusion function:', k)
            print('base_input',D['z0'])
            sample = self.EM(diffusion_fn=diffusion_fns[k], measurement = y, base=D['z0'], label=D['label'], cond=D['cond'])

            sample = preprocess_fn(sample.detach().cpu(), 'results')

            all_tensors = torch.cat([z0, sample, z1], dim=-1) 
            
            # plotD[k + "(cond, sample, real)"] = wandb.Image(all_tensors)


        if self.config.use_wandb:
            wandb.log(plotD, step = self.step)

    @torch.no_grad()
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

    def image_l1_norm(self, x):
        return x.abs().sum(-1).sum(-1).sum(-1)

    def training_step(self, D):
        assert self.model.training
        # print('input1', D['zt'].shape, D['cond'].shape)
        model_out = self.model(D['zt'], D['t'], D['label'], cond = D['cond'])
        target = D['drift_target']
        return self.image_l1_norm(model_out - target).mean()

    def center(self, x):
        return (x * 2.0) - 1.0

    @torch.no_grad()
    def prepare_batch_nse(self, batch = None, for_sampling = False):

        assert not self.config.center_data

        xlo, xhi = batch[:,:1].squeeze(-1), batch[:,1:].squeeze(-1)
        # print('xlo shape', xlo.shape)

        if for_sampling:
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
        # print('sampling_diver',batch)
        # print('sampling_diver',batch[0].shape)
        # # get (z0, z1, label, N)
        # print('batch',batch)
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
       
        D['zt'] = self.I.compute_zt_new(D)
        # print('after compute', D['zt'].shape)
        
        D['drift_target'] = self.I.compute_target_new(D)
   
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
        print('l', loss)
        loss.backward()
        grad_norm = self.optimizer_step() # updates self.step 
        self.scheduler.step()
        self.maybe_sample()

        if self.step % self.config.print_loss_every == 0:
            print(f"Grad step {self.step}. Loss:{loss.item()}")
            # if self.config.use_wandb:
            #     wandb.log({'loss': loss.item(), 'grad_norm': grad_norm}, step = self.step)

        if self.step % self.config.save_every == 0:
            print("saving!")
            self.save()

    def fit(self,):

        print('starting fit')
        print("starting training")
        while self.step < self.config.max_steps:

            for batch_idx, batch in enumerate(self.dataloader):
 
                if self.step >= self.config.max_steps:
                    return

                self.do_step(batch_idx, batch)

class Config:
    
    def __init__(self, dataset, debug, overfit, sigma_coef, beta_fn,device, home,nse_path = None, auto_step = 1, sample_only = False):

        self.dataset = dataset

        self.debug = debug
        print("SELF DEBUG IS", self.debug)
        self.device = device
        # interpolant + sampling
        self.sigma_coef = sigma_coef
        self.beta_fn = beta_fn
        self.EM_sample_steps = 500
        self.t_min_sampling = 0.0  # no min time needed
        self.t_max_sampling = .999
        self.auto_step = auto_step
        self.sample_only = sample_only

        # data
        if self.dataset == 'cifar':

            self.center_data = True
            self.C = 3
            self.H = 32
            self.W = 32
            self.num_classes = 10
            self.data_path = '../data/'
            self.grid_kwargs = {'normalize' : True, 'value_range' : (-1, 1)}

        elif self.dataset == 'nse':


            self.center_data = False
            self.home = home

            maybe_create_dir(self.home)

            if nse_path == None:
                self.data_fname = 'nse_data_tiny.pt'
            else:
                self.data_fname = nse_path
            self.num_classes = 1
            self.lo_size = 64
            self.hi_size = 128
            self.time_lag = 2
            self.subsampling_ratio = 1.0 
            self.grid_kwargs = {'normalize': False}
            self.C = 1
            self.H = self.hi_size
            self.W = self.hi_size

        else:
            assert False


        # shared
        self.num_workers = 4
        self.delta_t = 0.5
        self.wandb_project = 'nse'
        self.wandb_entity = 'siyiche'
        self.use_wandb = True
        self.noise_strength = 1.0

        self.overfit = overfit
        print(f"OVERFIT MODE (USEFUL FOR DEBUGGING) IS {self.overfit}")

        if self.debug:
            self.EM_sample_steps = 10
            self.sample_every = 10
            self.print_loss_every = 10
            self.save_every = 10000000
        else:
            self.sample_every = 1000
            self.print_loss_every = 100 #1000 
            self.save_every = 1000
        
        # some training hparams
        self.batch_size = 128 if self.dataset == 'cifar' else 32 
        self.sampling_batch_size = self.batch_size if self.dataset=='cifar' else 1
        self.num_workers = 4
        self.t_min_train = 0.0
        self.t_max_train = 1.0
        self.max_grad_norm = 1.0
        self.base_lr = 2e-5
        self.max_steps = 1_000_000
        
        # arch
        self.unet_use_classes = True if self.dataset == 'cifar' else False
        self.unet_channels = 128
        self.unet_dim_mults = (1, 2, 2, 2) 
        self.unet_resnet_block_groups = 8
        self.unet_learned_sinusoidal_dim = 32
        self.unet_attn_dim_head = 64
        self.unet_attn_heads = 4
        self.unet_learned_sinusoidal_cond = True
        self.unet_random_fourier_features = False


def main():

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--dataset', type = str, choices = ['cifar', 'nse'], default = 'nse')
    parser.add_argument('--load_path', type = str, default = None)
    parser.add_argument('--use_wandb', type = int, default = 1)
    parser.add_argument('--sigma_coef', type = float, default = 1.0) 
    parser.add_argument('--beta_fn', type = str, default = 't^2', choices=['t','t^2'])
    parser.add_argument('--debug', type = int, default = 0)
    parser.add_argument('--sample_only', type = int, default = 0)
    parser.add_argument('--overfit', type = int, default = 0)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--savedir', type=str, default = './tmp_images/')
    parser.add_argument('--nse_datapath', type=str, default = None)
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    for k in vars(args):
        print(k, getattr(args, k))
    conf = Config(
        dataset = args.dataset, 
        debug = bool(args.debug), # use as desired 
        overfit = bool(args.overfit),
        sigma_coef = args.sigma_coef, 
        beta_fn = args.beta_fn,
        device = device,
        home = args.savedir,
        nse_path = args.nse_datapath,
        sample_only = bool(args.sample_only)
    )
    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']
    operator = lambda x :  interpolate(x, size=(16,16),mode='bilinear',align_corners=False)
    noiser = get_noise(**measure_config['noise'])
    trainer = Trainer(
        conf, 
        load_path = args.load_path, # none trains from scratch 
        sample_only = bool(args.sample_only), 
        use_wandb = bool(args.use_wandb),
        operator = operator,
        noiser = noiser 
    )

    if bool(args.sample_only):
        trainer.sample_ckpt()
    else:
        trainer.fit()

if __name__ == '__main__':
    main()
