# FlowDAS



## Code

The majority of the code is written in [Python](https://www.python.org). Neural networks are built and trained using the [PyTorch](https://pytorch.org/) automatic differentiation framework.

```
conda env create -f environment.yml
conda activate flowdas
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

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

