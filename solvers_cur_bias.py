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


from sketches_cur import  (hadamard, sjlt, lev_approx, rrs_uniform,rrs_uniform_debia,rrs_lev_scores,
                       rrs_lev_scores_debia,rrs_shrinkage,rrs_shrinkage_debia,srht_opt,srht_opt_debia )


from sklearn.kernel_approximation import RBFSampler


SKETCH_FN = { 'rrs_uniform': rrs_uniform,'rrs_lev_scores': rrs_lev_scores, 'rrs_shrinkage': rrs_shrinkage,
              'srht_opt': srht_opt}

SKETCH_FN_debia = { 'rrs_uniform':  rrs_uniform_debia, 'rrs_lev_scores': rrs_lev_scores_debia,'rrs_shrinkage': rrs_shrinkage_debia, 'srht_opt': srht_opt_debia}
              #'rrs_uniform_debia': rrs_uniform_debia,
# 'rrs_rlev_scores': rrs_rlev_scores,
# SKETCH_FN = {'gaussian': gaussian, 'less': less, 'less_sparse': sparse_rademacher,
#              'srht': srht, 'rrs': rrs, 'rrs_lev_scores': rrs_lev_scores,'rrs_rlev_scores':rrs_rlev_scores}
#

torch.set_default_dtype(torch.float64)



class CURdecomposition:
    
    def __init__(self, A,sa_c, sa_r):
        self.A = A
        self.sa_c=sa_c
        self.sa_r=sa_r
        # self.b = b
        # if self.b.ndim == 1:
        #     self.b = self.b.reshape((-1,1))
        self.n, self.d = A.shape
        # self.c = self.b.shape[1]
        # self.lambd = lambd
        self.device = A.device

        # self.n_trials=n_trials

    def _c_r_computation_random(self,c_col_size,r_row_size):
        

        # compute the A.^2
        A_ele_squ= self.A.pow(2)

        """
        Return C= A@S^T
        Column sampling probabilities proportional to ||A[:,j]||_2^2.
        """

        # shape: (p,)  sum(dim=0)the sum of column
        probs_col = (A_ele_squ.sum(dim=0)) #.clamp_min(eps)
        prob_col=probs_col / probs_col.sum()

        s_index_col=np.random.choice(self.d, c_col_size, replace=True, p=prob_col)
        sa_matrix_c  =self.A[ :,s_index_col]
        # sy_response = response[s_index, ::]
        s_prob_col=prob_col[s_index_col]
        weight_c=torch.sqrt(torch.tensor(c_col_size*s_prob_col.reshape((1, -1)),dtype=torch.float64))
        # print("size_sa_matrix_c:", sa_matrix_c.shape)
        # print("size_weight_c:", weight_c.shape)
        sa_c=sa_matrix_c /weight_c 
        """
        Return R=S @ A
        Row sampling probabilities proportional to ||A[i,:]||_2^2.
        """
        #Row sampling probabilities proportional to ||A[i,:]||_2^2.
        # shape: (n,)  sum(dim=1) the sum of row
        probs_row = (A_ele_squ.sum(dim=1))   #clamp_min(eps)
        prob_row= probs_row  / probs_row.sum()


        s_index_row=np.random.choice(self.n, r_row_size, replace=True, p=prob_row)
        sa_matrix_r  =self.A[s_index_row, ::]
        # sy_response = response[s_index, ::]
        s_prob_row=prob_row[s_index_row]
        weight_r =torch.sqrt(torch.tensor(r_row_size*s_prob_row.reshape((-1, 1)),dtype=torch.float64))
        sa_r=sa_matrix_r  /weight_r 
        # sy = sy_response / weight

        return sa_c, sa_r


    def fast_cur(self,c_col_size,r_row_size,c_sketch_size,r_sketch_size,sketch, nnz,   alpha_c,alpha_r):

        start = time()
        # self.sa_c,self.sa_r= self._c_r_computation_random(c_col_size,r_row_size)

        sc, sr, sa_cr = SKETCH_FN[sketch](self.A,self.sa_c, self.sa_r, c_sketch_size,r_sketch_size, nnz=nnz,  alpha_c=alpha_c,alpha_r=alpha_r)

        
        u_fast=torch.linalg.pinv(sc)@ sa_cr@ torch.linalg.pinv(sr)
        cur_fast=self.sa_c@u_fast@self.sa_r


        
        return cur_fast, time()-start
    

    
    def cur_vary_sketch_size_nonava(self,c_col_size,r_row_size,c_sketch_size,r_sketch_size,  sketch, nnz, n_trials,  alpha_c,alpha_r):

        dem=( (self.A ) **  2).sum()
        # dem=( (self.A ) **
        #                 2).sum()

        div_mean_ii = 0
        div_sta_mean_ii = 0
        times_mean_ii = 0
        for ii in range(n_trials):

            cur_fast,time_=self.fast_cur(c_col_size,r_row_size,c_sketch_size,r_sketch_size,sketch, nnz,   alpha_c,alpha_r)
            time_=torch.tensor(time_,dtype=torch.float64)

            div_sta_=self.A -cur_fast
             
            div_sta_mean_ii += div_sta_ / n_trials
            times_mean_ii += time_ / n_trials
        
         # dem=(torch.linalg.norm(self.A_H_inv-self.b, ord=2)) **2
        div_sta_norm_ii = ((div_sta_mean_ii) ** 2 ).sum()/ dem
      
        times_ii = times_mean_ii
        
        losses_sta_ii =div_sta_norm_ii
        return losses_sta_ii, times_ii # ,x,

        

    def cur_vary_sketch_size_nonava_all(self,c_col_size,r_row_size,c_sketch_size,r_sketch_size,  sketch, nnz, n_trials,  alpha_c,alpha_r):
        losses_all = []
        times_all = []

        # losses /= losses[0]

        sketch_size=c_sketch_size
        r_sketch_size_0=r_sketch_size
        for ii, sketch_size_value in enumerate(sketch_size):

            c_sketch_size=sketch_size_value
            r_sketch_size= r_sketch_size_0

            losses_ii, times_ii = self.cur_vary_sketch_size_nonava(c_col_size,r_row_size,c_sketch_size,r_sketch_size,  sketch, nnz, n_trials,  alpha_c,alpha_r)

            losses_ii = torch.tensor(losses_ii)
            times_ii = torch.tensor(times_ii)

            losses_all.append(losses_ii.cpu().numpy().tolist())
            # self.ols_(x, sketch_size_value, sketch, lambda_ridge, nnz,alpha)
            times_all.append(times_ii.cpu().numpy().tolist())

        losses_all = np.array(losses_all)

        return losses_all, times_all  # ,x,

    def debia_fast_cur(self,c_col_size,r_row_size,c_sketch_size,r_sketch_size,sketch, nnz,   alpha_c,alpha_r):

        start = time()
        # self.sa_c,self.sa_r= self._c_r_computation_random(c_col_size,r_row_size)

        sc, sr, sa_cr = SKETCH_FN_debia[sketch](self.A,self.sa_c, self.sa_r, c_sketch_size,r_sketch_size, nnz=nnz,  alpha_c=alpha_c,alpha_r=alpha_r)

        # check_beta =torch.linalg.pinv(sa)@ sb
        #  torch.linalg.lstsq(sa, sb).solution

        u_fast=torch.linalg.pinv(sc)@ sa_cr@ torch.linalg.pinv(sr)
        cur_fast=self.sa_c@u_fast@self.sa_r

        
        return cur_fast, time()-start

    
        
    def debia_cur_vary_sketch_size_nonava(self,c_col_size,r_row_size,c_sketch_size,r_sketch_size,  sketch, nnz, n_trials,  alpha_c,alpha_r):


        dem=( (self.A ) **
                        2).sum()

        div_mean_ii = 0
        div_sta_mean_ii = 0
        times_mean_ii = 0
       
        for ii in range(n_trials):

            cur_fast,time_=self.debia_fast_cur(c_col_size,r_row_size,c_sketch_size,r_sketch_size,sketch, nnz,   alpha_c,alpha_r)
            time_=torch.tensor(time_,dtype=torch.float64)

            div_sta_=self.A -cur_fast
             
            div_sta_mean_ii += div_sta_ / n_trials
            times_mean_ii += time_ / n_trials
        
         # dem=(torch.linalg.norm(self.A_H_inv-self.b, ord=2)) **2
        div_sta_norm_ii = ((div_sta_mean_ii) ** 2 ).sum()/ dem
      
        times_ii = times_mean_ii
        
        losses_sta_ii =div_sta_norm_ii
        return losses_sta_ii, times_ii # ,x,
            
       
     

    def debia_cur_vary_sketch_size_nonava_all(self,c_col_size,r_row_size,c_sketch_size,r_sketch_size,  sketch, nnz, n_trials,  alpha_c,alpha_r):
        losses_all = []
        times_all = []

        # losses /= losses[0]

        sketch_size=c_sketch_size
        r_sketch_size_0=r_sketch_size

        for ii, sketch_size_value in enumerate(sketch_size):
            c_sketch_size=sketch_size_value
            r_sketch_size=r_sketch_size_0 
            

                        
            losses_ii, times_ii = self.debia_cur_vary_sketch_size_nonava(c_col_size,r_row_size,c_sketch_size,r_sketch_size,  sketch, nnz, n_trials,  alpha_c,alpha_r)

            losses_ii = torch.tensor(losses_ii)
            times_ii = torch.tensor(times_ii)

            losses_all.append(losses_ii.cpu().numpy().tolist())
           
            times_all.append(times_ii.cpu().numpy().tolist())

        losses_all = np.array(losses_all)

        return losses_all, times_all  # ,x,

    