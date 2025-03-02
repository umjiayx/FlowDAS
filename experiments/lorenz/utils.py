r"""Lorenz experiment helpers"""

import os

from pathlib import Path
from typing import *

from mcs import *

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/lorenz'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)

# From SDA
def to(x: Any, **kwargs) -> Any:
    if torch.is_tensor(x):
        return x.to(**kwargs)
    elif type(x) is list:
        return [to(y, **kwargs) for y in x]
    elif type(x) is tuple:
        return tuple(to(y, **kwargs) for y in x)
    elif type(x) is dict:
        return {k: to(v, **kwargs) for k, v in x.items()}
    else:
        return x


def make_chain() -> MarkovChain:
    return NoisyLorenz63(dt=0.025)

def make_chain_generalize(sigma, rho, beta) -> MarkovChain:
    return NoisyLorenz63Generalize(dt=0.025, sigma=sigma, rho=rho, beta=beta)