# FlowDAS: A Flow-Based Framework for Data Assimilation

This repository contains the official implementation of [FlowDAS: A Flow-Based Framework for Data Assimilation](https://arxiv.org/abs/2501.16642).

This is the first version of the code and is currently under development for extension. At this stage, we provide the implementation for the Lorenz 1963 experiment. The code for Navier-Stokes, PIV, weather forecasting (and more!) will be released soon.


## Code

The majority of the code is written in [Python](https://www.python.org). Neural networks are built and trained using the [PyTorch](https://pytorch.org/) automatic differentiation framework.

## Environment Setup

To set up the environment for FlowDAS, we provide an `environment.yml` file that contains all the necessary dependencies. You can create and activate the conda environment using the commands below. Note that after creating the environment, you'll need to install PyTorch separately to ensure compatibility with your CUDA version.


```
conda env create --file environment.yml --prefix /path/to/your/conda/env
conda activate flowdas
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

## Getting Started

This codebase is organized to make it easy to reproduce our experiments and extend the framework. Here's how to get started:

### Reproducing Results

To reproduce the results from our paper for the Lorenz 1963 experiment, run:
```
./scripts/run.sh
```

### Generalizability of FlowDAS

Under development. Coming soon! :)


## Citation

If you find this work useful, please cite it as follows:

```bib
@article{chen2025flowdas,
  title={FlowDAS: A Flow-Based Framework for Data Assimilation},
  author={Chen, Siyi and Jia, Yixuan and Qu, Qing and Sun, He and Fessler, Jeffrey A},
  journal={arXiv preprint arXiv:2501.16642},
  year={2025}
}
```

