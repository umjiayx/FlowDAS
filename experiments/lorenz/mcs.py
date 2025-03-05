r"""Markov chains"""

"""
This code is based on the Score-based Data Assimilation (SDA) framework.

References:
    @inproceedings{rozet2023sda,
      title={Score-based Data Assimilation},
      author={Fran{\c{c}}ois Rozet and Gilles Louppe},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=VUvLSnMZdX},
    }

    @article{rozet2023sda-2lqg,
      title={Score-based Data Assimilation for a Two-Layer Quasi-Geostrophic Model},
      author={Fran{\c{c}}ois Rozet and Gilles Louppe},
      booktitle={Machine Learning and the Physical Sciences Workshop (NeurIPS)},
      year={2023},
      url={https://arxiv.org/abs/2310.01853},
    }
"""

import abc
import torch
from torch import Tensor, Size
from torch.distributions import Normal, MultivariateNormal
from typing import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py

class MarkovChain(abc.ABC):
    r"""Abstract first-order time-invariant Markov chain class

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
        https://wikipedia.org/wiki/Time-invariant_system
    """

    @abc.abstractmethod
    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ p(x_0) """

        pass

    @abc.abstractmethod
    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ p(x_i | x_{i-1}) """

        pass

    def trajectory(self, x: Tensor, length: int, last: bool = False) -> Tensor:
        r""" (x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1}) """

        if last:
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)


class DampedSpring(MarkovChain):
    r"""Linearized dynamics of a mass attached to a spring, subject to wind and drag."""

    def __init__(self, dt: float = 0.01):
        super().__init__()

        self.mu_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.Sigma_0 = torch.tensor([1.0, 1.0, 1.0, 1.0]).diag()

        self.A = torch.tensor([
            [1.0, dt, dt**2 / 2, 0.0],
            [0.0, 1.0, dt, 0.0],
            [-0.5, -0.1, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.99],
        ])
        self.b = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.Sigma_x = torch.tensor([0.1, 0.1, 0.1, 1.0]).diag() * dt

    def prior(self, shape: Size = ()) -> Tensor:
        return MultivariateNormal(self.mu_0, self.Sigma_0).sample(shape)

    def transition(self, x: Tensor) -> Tensor:
        return MultivariateNormal(x @ self.A.T + self.b, self.Sigma_x).sample()


class DiscreteODE(MarkovChain):
    r"""Discretized ordinary differential equation (ODE)

    Wikipedia:
        https://wikipedia.org/wiki/Ordinary_differential_equation
    """

    def __init__(self, dt: float = 0.01, steps: int = 1):
        super().__init__()

        self.dt, self.steps = dt, steps

    @staticmethod
    def rk4(f: Callable[[Tensor], Tensor], x: Tensor, dt: float) -> Tensor:
        r"""Performs a step of the fourth-order Runge-Kutta integration scheme.

        Wikipedia:
            https://wikipedia.org/wiki/Runge-Kutta_methods
        """

        k1 = f(x)
        k2 = f(x + dt * k1 / 2)
        k3 = f(x + dt * k2 / 2)
        k4 = f(x + dt * k3)

        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @abc.abstractmethod
    def f(self, x: Tensor) -> Tensor:
        r""" f(x) = \frac{dx}{dt} """

        pass

    def transition(self, x: Tensor) -> Tensor:
        for _ in range(self.steps):
            x = self.rk4(self.f, x, self.dt / self.steps)

        return x


class Lorenz63(DiscreteODE):
    r"""Lorenz 1963 dynamics

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_system
    """

    def __init__(
        self,
        sigma: float = 10.0,  # [9, 13]
        rho: float = 28.0,  # [28, 40]
        beta: float = 8 / 3,  # [1, 3]
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sigma, self.rho, self.beta = sigma, rho, beta

    def prior(self, shape: Size = ()) -> Tensor:
        mu = torch.tensor([0.0, 0.0, 25.0])
        sigma = torch.tensor([
            [64.0, 50.0,  0.0],
            [50.0, 81.0,  0.0],
            [ 0.0,  0.0, 75.0],
        ])

        return MultivariateNormal(mu, sigma).sample(shape)

    def f(self, x: Tensor) -> Tensor:
        return torch.stack((
            self.sigma * (x[..., 1] - x[..., 0]),
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],
            x[..., 0] * x[..., 1] - self.beta * x[..., 2],
        ), dim=-1)

    @staticmethod
    def preprocess(x: Tensor) -> Tensor:
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return (x - mu) / sigma

    @staticmethod
    def postprocess(x: Tensor) -> Tensor:
        mu = x.new_tensor([0.0, 0.0, 25.0])
        sigma = x.new_tensor([8.0, 9.0, 8.6])

        return mu + sigma * x


class NoisyLorenz63(Lorenz63):
    r"""Noisy Lorenz 1963 dynamics"""

    def moments(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return super().transition(x), self.dt ** 0.5

    def transition(self, x: Tensor) -> Tensor:
        return Normal(*self.moments(x)).sample()

    def log_prob(self, x1: Tensor, x2: Tensor) -> Tensor:
        return Normal(*self.moments(x1)).log_prob(x2).sum(dim=-1)
    

class NoisyLorenz63Generalize(Lorenz63):
    r"""Noisy Lorenz 1963 dynamics for generalizability"""
        
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8/3, **kwargs):
        # Pass these parameters to the parent Lorenz63 class
        super().__init__(sigma=sigma, rho=rho, beta=beta, **kwargs)

    def moments(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return super().transition(x), self.dt ** 0.5

    def transition(self, x: Tensor) -> Tensor:
        return Normal(*self.moments(x)).sample()

    def log_prob(self, x1: Tensor, x2: Tensor) -> Tensor:
        return Normal(*self.moments(x1)).log_prob(x2).sum(dim=-1)


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        file: Path,
        window: int = None,
        flatten: bool = False,
    ):
        super().__init__()

        with h5py.File(file, mode='r') as f:
            # load data from h5 file into the memory
            self.data = f['x'][:]

        self.window = window
        # self.flatten = flatten

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = torch.from_numpy(self.data[i]) # (L, 3)

        if self.window is not None:
            previous = x[:-1]  # All except the last time step
            current = x[1:]    # All except the first time step

        x_pairs = torch.cat((previous, current), dim=1) # (L-1, 6)
        return x_pairs, {}
    

class TrajectoryDatasetV2(Dataset):
    """
    Compared to the TrajectoryDataset, this dataset is designed for using the information
    from 'several' previous time steps to predict the next time step.
    The original TrajectoryDataset is when the window is 1.
    """
    def __init__(
        self,
        file: Path,
        window: int = None
    ):
        super().__init__()
        self.window = window
        with h5py.File(file, mode='r') as f:
            self.data = f['x'][:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x = torch.from_numpy(self.data[i]) # (L, 3)

        if self.window is not None:
            assert isinstance(self.window, int), "window must be an integer"
            assert 0 < self.window < x.shape[0], "window must be within range (1, L)"

            # Create sliding windows of size self.window
            x_pairs = []
            for i in range(x.shape[0] - self.window):
                # Concatenate self.window consecutive timesteps
                window_slice_previous = x[i:i+self.window].reshape(-1)  # Flatten the window into a single vector
                window_slice_current = x[i+1:i+self.window+1].reshape(-1)
                x_pairs.append(torch.cat((window_slice_previous, window_slice_current), dim=0))
            
            # Stack all windows into a tensor
            x_pairs = torch.stack(x_pairs)
            return x_pairs, {} # Shape: (L-window+1, 3*window)