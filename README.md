# FlowDAS: A Flow-Based Framework for Data Assimilation

This repository contains the official implementation of [FlowDAS: A Flow-Based Framework for Data Assimilation](https://arxiv.org/abs/2501.16642).

This is the first version of the code and is currently under development for extension. At this stage, we provide the implementation for the Lorenz 1963 and weather forecasting experiment. The code for Navier-Stokes, PIV (and more!) will be released soon.

This codebase is organized to make it easy to reproduce our experiments and extend the framework. Here's how to get started:

## Environment Setup

To set up the environment for FlowDAS, we provide an `environment.yml` file (in each experiment folder) that contains all the necessary dependencies. You can create and activate the conda environment using the commands below. Note that after creating the environment, you'll need to install PyTorch separately to ensure compatibility with your CUDA version.


```
conda env create --file environment.yml --prefix /path/to/your/conda/env
conda activate flowdas
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```


## Experiments

### 1. Lorenz 1963

To reproduce the results from our paper for the Lorenz 1963 experiment, run: ``./scripts/run.sh``.

### 2. Navier-Stokes Equation

The dataset used in our Navier-Stokes Equation experiment can be downloaded [here](https://www.dropbox.com/scl/fi/rqpv4pgujhtzswo6xhpzt/data_file.pt?rlkey=sn0ethjnf6b9mw3koybd3v0dh&st=czi9kcz1&dl=0). The checkpoint of our pre-trained model can be accessed [here](https://www.dropbox.com/scl/fi/jhqtf5yag7ithzrv2yxgs/latest.pt?rlkey=j4t7zhdu3uypnhqsrv1cx9io5&st=5dw9opdx&dl=0). 

### 3. Weather Forecasting

The dataset used in our weather forecasting experiment can be downloaded [here](https://www.dropbox.com/scl/fi/h83pp33jx5gz62gk0gncs/sevir_lr.zip?rlkey=dtnnk6x4af0hhrneugijhq60s&st=ux0ud8pz&dl=0). The checkpoint of our pre-trained model can be accessed [here](https://www.dropbox.com/scl/fi/5z1bwfdvbztnums9deqhe/latest.pt?rlkey=o5izt721am3hzkcwjmmn7joym&st=0o0nvqsr&dl=0). 


## Acknowledgements

We built our repo on the [Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes](https://github.com/interpolants/forecasting) repo that is publicly available.



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

