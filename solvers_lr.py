# pip install matplotlib
import csv
import pandas as pd
import os
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time


from sketches import  (hadamard, sjlt, lev_approx, rrs_uniform,rrs_uniform_debia,rrs_lev_scores,
                       rrs_lev_scores_debia,rrs_shrinkage,rrs_shrinkage_debia,srht_opt,srht_opt_debia,less )


from sklearn.kernel_approximation import RBFSampler



SKETCH_FN = { 'rrs_uniform': rrs_uniform,'rrs_uniform_debia': rrs_uniform_debia,'rrs_lev_scores': rrs_lev_scores,
                       'rrs_lev_scores_debia': rrs_lev_scores_debia,'rrs_shrinkage': rrs_shrinkage,
              'rrs_shrinkage_debia': rrs_shrinkage_debia, 'srht_opt': srht_opt,'srht_opt_debia': srht_opt_debia,'less': less}


torch.set_default_dtype(torch.float64)



class OLSRegression:
    
    def __init__(self, A, b):
        self.A = A
        self.b = b
        if self.b.ndim == 1:
            self.b = self.b.reshape((-1,1))
        self.n, self.d = A.shape
        self.c = self.b.shape[1]
        # self.lambd = lambd
        self.device = A.device

        # self.n_trials=n_trials
        

    # "solve_exactly" is use to compute the optimal x
    def solve_exactly(self,sketch):

        x_opt=  torch.linalg.pinv(self.A)@ self.b
        
        return x_opt
        

    def sols_(self,  sketch_size, sketch, nnz, alpha):
        
        start = time()

        # hsqrt = self.sqrt_hess(x).reshape((-1,1))
        sa,sb = SKETCH_FN[sketch](self.A,self.b,sketch_size, nnz=nnz,  alpha=alpha)

        check_beta =torch.linalg.pinv(sa)@ sb
        #  torch.linalg.lstsq(sa, sb).solution



        
        return check_beta, time()-start

    def ols_vary_sketch_size_nonava(self, sketch_size, sketch, nnz, n_trials, alpha):

        losses = []
        times = []

        # time_sketch = profile_times(A, sketch, sketch_size, nnz)

        # x = 1. / np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        for ii in range(n_trials):
            beta_ols=self.solve_exactly(sketch)
            check_beta, time_ = self.sols_( sketch_size, sketch, nnz, alpha)
            time_=torch.tensor(time_,dtype=torch.float64)
            num=((self.A @ (check_beta - beta_ols)) ** 2).sum()
            den=( (self.A @ beta_ols- self.b) **
                        2).sum()
            loss_check = num/den

            losses.append( loss_check.cpu().numpy().item())
            # self.ols_(x, sketch_size_value, sketch, lambda_ridge, nnz,alpha)
            times.append(time_.cpu().numpy().item())

        losses = np.array(losses)
        # losses /= losses[0]

        return losses, times  # ,x,

    def ols_vary_sketch_size_nonava_all(self, sketch_size, sketch,  nnz,  n_trials, alpha):
        losses_all = []
        times_all = []

        # losses /= losses[0]

        for ii, sketch_size_value in enumerate(sketch_size):
            losses_ii, times_ii = self.ols_vary_sketch_size_nonava(sketch_size_value, sketch,  nnz,
                                                               n_trials, alpha)

            losses_ii = torch.tensor(losses_ii)
            times_ii = torch.tensor(times_ii)

            losses_all.append(losses_ii.cpu().numpy().tolist())
            # self.ols_(x, sketch_size_value, sketch, lambda_ridge, nnz,alpha)
            times_all.append(times_ii.cpu().numpy().tolist())

        losses_all = np.array(losses_all)

        return losses_all, times_all  # ,x,

        #