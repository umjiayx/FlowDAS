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

def make_chain() -> MarkovChain:
    return NoisyLorenz63(dt=0.025)

def make_chain_generalize(sigma, rho, beta) -> MarkovChain:
    return NoisyLorenz63Generalize(dt=0.025, sigma=sigma, rho=rho, beta=beta)