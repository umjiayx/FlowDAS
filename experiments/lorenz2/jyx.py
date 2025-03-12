import torch
from torch import Tensor
import h5py


def compute_nrmse_LT(gt, est, LT, window):
    '''
    Compute the normalized root mean square error (NRMSE) of the estimated trajectory
    with respect to the ground truth trajectory (the first LT states).
    gt: (L+w, 3)
    est: (L+w, 3)
    '''
    gt_LT = gt[window:LT+window]
    est_LT = est[window:LT+window]
    rmse = torch.sqrt(torch.mean((gt_LT - est_LT) ** 2))
    denominator = torch.sqrt(torch.mean(gt_LT ** 2))
    nrmse = rmse / denominator
    return nrmse

if __name__ == "__main__":
    gt = torch.ones(10, 3)
    est = torch.zeros(10, 3) + 0.5
    print(compute_nrmse_LT(gt, est, 8, 0))
    err = gt - est
    print(err[2:10])
    print(torch.mean(err[2:10] ** 2))


