# pip install matplotlib
import csv
import pandas as pd
import os
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.linalg import qr

import matplotlib.pyplot as plt
import matplotlib as mpl

from time import time


from sketches_cur import  (hadamard, sjlt, lev_approx, rrs_uniform,rrs_uniform_debia,rrs_lev_scores,
                       rrs_lev_scores_debia,rrs_shrinkage,rrs_shrinkage_debia,srht_opt,srht_opt_debia )



from sklearn.kernel_approximation import RBFSampler


SKETCH_FN = { 'rrs_uniform': rrs_uniform,'rrs_lev_scores': rrs_lev_scores, 'rrs_shrinkage': rrs_shrinkage,
              'srht_opt': srht_opt}

SKETCH_FN_debia = {  'rrs_lev_scores': rrs_lev_scores_debia,'rrs_shrinkage': rrs_shrinkage_debia, 'srht_opt': srht_opt_debia}
             

torch.set_default_dtype(torch.float64)



class CURdecomposition_baseline:
    
    def __init__(self, A):
        self.A = A
        # self.sa_c=sa_c
        # self.sa_r=sa_r
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
        # weight_c=torch.sqrt(torch.tensor(c_col_size*s_prob_col.reshape((1, -1)),dtype=torch.float64))
        # print("size_sa_matrix_c:", sa_matrix_c.shape)
        # print("size_weight_c:", weight_c.shape)
        sa_c=sa_matrix_c 
        
       

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
        # weight_r =torch.sqrt(torch.tensor(r_row_size*s_prob_row.reshape((-1, 1)),dtype=torch.float64))
        sa_r=sa_matrix_r  
        # sy = sy_response / weight
        
        sa_matrix_u  =sa_c[s_index_row, ::]
        
        sa_u=sa_matrix_u 

        return sa_c, sa_r, sa_u

    

    
    def cpqr(self,matrix, k):

        Q, R, piv = qr(matrix, pivoting=True, mode='economic')

        # # A[:, piv] = Q @ R
        # err = np.linalg.norm(self. A[:, piv] - Q @ R) / np.linalg.norm(self.A)
        # print("rel err:", err)

        # 前 k 个 pivot 就是“最重要的列索引”
        J = piv[:k]
        # print("top-k pivots:", J)
        return J
    
    def _srht(self, indices, v):
        n = v.shape[0]
        if n == 1:
            return v
        i1 = indices[indices < n//2]
        i2 = indices[indices >= n//2]
        if len(i1) == 0:
            return self._srht(i2-n//2, v[:n//2,::]-v[n//2:,::])
        elif len(i2) == 0:
            return self._srht(i1, v[:n//2,::]+v[n//2:,::])
        else:
            return torch.cat([self._srht(i1, v[:n//2,::]+v[n//2:,::]), self._srht(i2-n//2, v[:n//2,::]-v[n//2:,::])], axis=0)

    #
    def srht_opt(self,  sketch_size):
        #device = matrix.device
        #matrix = matrix.cpu().numpy()
        matrix=self.A
        if matrix.ndim == 1:
            matrix = matrix.reshape((-1,1))
        # pad matrix with 0 if first dimension is not a power of 2
        n = matrix.shape[0]
        if n & (n-1) != 0:
            new_dim = 2**(int(np.log(n) / np.log(2))+1)
            matrix = torch.cat([matrix, torch.zeros(new_dim - n, matrix.shape[1]).to(matrix.device)], axis=0)
        n = matrix.shape[0]
        # indices = np.sort(np.random.choice(np.arange(n), sketch_size, replace=False))
        samples = np.random.choice(np.arange(n), sketch_size, replace=True)
        indices, counts = np.unique(samples, return_counts=True)
        v = torch.tensor(np.random.choice([-1,1], n, replace=True)).reshape((-1,1)).to(matrix.device)
        matrix = v * matrix
        # response=v*response
        sa =self._srht(indices, matrix)
        # sy= _srht(indices, response)
        counts_tensor = torch.tensor(counts, dtype=sa.dtype, device=sa.device).reshape(-1, 1)
        return np.sqrt(1/sketch_size)*np.sqrt(counts_tensor) *sa

     ########## sparse lev
    # define countsketch
    def sjlt(self, sketch_size, nnz=None):
        matrix=self.A
        n, d = matrix.shape 
        indices = np.vstack([np.random.choice(sketch_size, n).reshape((1,-1)), np.arange(n)])
        values = np.random.choice(np.array([-1,1], dtype=np.float64), size=n)
        S = torch.sparse_coo_tensor(indices, values, (sketch_size, n)).to(matrix.device)  #torch.sparse_coo_tensor is used to save the sparse tensor
        sa = S @ matrix
        return sa

    def lev_approx(self,alpha_k):
        matrix=self.A
        # alpha=50
        # n, d = matrix.shape
        m = int(alpha_k)
        sa=self.sjlt( m, nnz=None)
        return sa


    def _c_r_computation(self,J_c,I_r):
        
        sa_matrix_c  =self.A[ :,J_c]
    
        sa_c=sa_matrix_c 
        
        sa_matrix_r  =self.A[I_r, ::]
        sa_r=sa_matrix_r  
        # sy = sy_response / weight
        
        sa_matrix_u  =sa_c[I_r, ::]
        
        sa_u=sa_matrix_u 

        return sa_c, sa_r, sa_u



    def rand_pivot(self,para,baseline,k):
        """
        Algorithm 4.1: Pivoting on a random sketch.
        Input: A (m x n), target rank k
        Output: row indices I (k,), col indices J (k,)
        """
        if self.A.ndim != 2:
            raise ValueError("A must be 2D")
        # n,d = self.A.shape
        # device, dtype = A.device, A.dtype

    #    # 1) Draw random embedding Omega in R^{k x m}
    #     if omega == "gaussian":
    #         Omega = torch.randn(k, self, device=device, dtype=dtype)
    #     elif omega == "rademacher":
    #         Omega = torch.randint(0, 2, (k, m), device=device).to(dtype) * 2 - 1
    #     else:
    #         raise ValueError("ome ga must be 'gaussian' or 'rademacher'")

        # 2) X = Omega A (row sketch)
        
        if  baseline in ['OSP_SRHT']:
             X =self.srht_opt(para)
        else:

             X =self.lev_approx(para)
            



        # para=20
        # X =self.srht_opt(para* k)

        # # alpha=1
        # X =self.lev_approx(alpha*k)
        # print("X:",X)

        # 3) CPQR on X -> k column pivots J
        J = self.cpqr(X, k)  # (k,)
        # print("colun_pivot:",J)

        # 4) CPQR on A(:,J)^T -> k row pivots I
        AJT = self.A[:, J].T               # (k x m)  columns correspond to rows of A
        I =self.cpqr(AJT, k)  # pivots among m columns -> row indices
        J_c=J
        I_r=I

        return  J_c,I_r


    def oversampling_OS(self,sa_c, J_c,I_r,k,  p):
        """
        Algorithm 4.2: Oversampling indices.

        Require:
            B in R^{n x k} full column rank, n >= k
            I index set with |I| = k  (row indices)
            p <= k oversampling parameter
        Ensure:
            Extra indices I0 with |I0| = p

        Steps:
            1) [Q_B, ~] = qr(B, 0)
            2) [~, ~, V] = svd(Q_B(I,:))
            3) V_-p = V(:, k-p+1 : k)  (trailing p right singular vectors)
            4) CPQR on ( Q_B([n]-I,:) V_-p )^T ; take p pivots as I0 among complement.
        """
      
        # sa_c,sa_r=self._c_r_computation(J_c,I_r)
        # 1) Thin QR: B = Q_B R
        Q_c, _ = torch.linalg.qr(sa_c, mode="reduced")  # (n x k)

        # 2) SVD of Q_B(I,:)
        QI = Q_c[I_r, :]                               # (k x k)
        # Vh: (k x k), with V = Vh^T
        _, _, Vh = torch.linalg.svd(QI, full_matrices=False)
        V = Vh.T                                     # (k x k)

        # 3) trailing p right singular vectors
        V_minus_p = V[:, (k - p):k]                  # (k x p)

        # 4) Build complement [n] - I
        mask = torch.ones(self.n, dtype=torch.bool)
        mask[I_r] = False
        I_r_remin = torch.arange(self.n)[mask]    # (n-k,)

        # Matrix for CPQR: (Q_B(Ic,:) V_-p)^T  shape (p x (n-k))
        M = (Q_c[I_r_remin, :] @ V_minus_p).T               # (p x (n-k))

        # CPQR on M -> pivots among columns -> indices in Ic
        piv_local = self.cpqr(M, p)         # (p,) in [0, n-k)
        I0 =I_r_remin[piv_local]                            # map back to original row indices

        return I0
    
    def oversampling_cur(self,para,baseline,k,p ):

        J_c,I_r=self.rand_pivot(para,baseline,k)

        sa_c,sa_r,sa_u=self._c_r_computation(J_c,I_r)

        I_0=self.oversampling_OS(sa_c,J_c,I_r,k,  p)

        sa_r_over  =self.A[I_0, ::]
        # sa_r  =self.A[I_r, ::]

        sa_r_all = torch.cat([sa_r, sa_r_over], dim=0)  
        # sy = sy_response / weight
        
        sa_u_over  =sa_c[I_0, ::]
        sa_u_all = torch.cat([sa_u, sa_u_over], dim=0)  
        return sa_c,sa_r_all,sa_u_all


        
        


        











    def fast_cur(self,para,baseline,c_col_size_base,r_row_over):

        start = time()
        k=c_col_size_base
        p=r_row_over
        self.sa_c,self.sa_r,self.sa_u= self.oversampling_cur(para, baseline,k,p )

        # sc, sr, sa_cr = SKETCH_FN[sketch](self.A,self.sa_c, self.sa_r, c_sketch_size,r_sketch_size, nnz=nnz,  alpha_c=alpha_c,alpha_r=alpha_r)

        # check_beta =torch.linalg.pinv(sa)@ sb
        #  torch.linalg.lstsq(sa, sb).solution
        # print("size_c:",self.sa_c.shape)
        # print("size_r:",self.sa_r.shape)
        # print("size_u:",self.sa_u.shape)

        u_fast=torch.linalg.pinv(self.sa_u)
        cur_approx=self.sa_c@u_fast@self.sa_r

        
        return cur_approx, time()-start
    

    
    def cur_vary_sketch_size_nonava(self,para,baseline,c_col_size_base,r_row_over,n_trials):

        losses = []
        times = []

        # time_sketch = profile_times(A, sketch, sketch_size, nnz)

        # x = 1. / np.sqrt(self.d) * torch.randn(self.d, self.c).to(self.device)
        for ii in range(n_trials):
            cur_approx,time_=self.fast_cur(para,baseline,c_col_size_base,r_row_over)
            time_=torch.tensor(time_,dtype=torch.float64)
            num=((self.A -cur_approx) ** 2).sum()
            den=( (self.A ) **
                        2).sum()
            loss_check = num/den

            losses.append( loss_check.cpu().numpy().item())
            # self.ols_(x, sketch_size_value, sketch, lambda_ridge, nnz,alpha)
            times.append(time_.cpu().numpy().item())

        losses = np.array(losses)
        # losses /= losses[0]

        return losses, times  # ,x,

    def cur_vary_sketch_size_nonava_all(self,para,baseline,c_col_size_base,r_row_over,n_trials):
        losses_all = []
        times_all = []

        # losses /= losses[0]

        sketch_size=c_col_size_base
        para_all=para

        for ii, sketch_size_value in enumerate(sketch_size):

            para=para_all[ii]

            c_col_size_base=sketch_size_value
            r_sketch_size=  int(0.5*sketch_size_value)   #0.5*sketch_size_value


            losses_ii, times_ii = self.cur_vary_sketch_size_nonava(para,baseline,c_col_size_base,r_row_over,n_trials)

            losses_ii = torch.tensor(losses_ii)
            times_ii = torch.tensor(times_ii)

            losses_all.append(losses_ii.cpu().numpy().tolist())
            # self.ols_(x, sketch_size_value, sketch, lambda_ridge, nnz,alpha)
            times_all.append(times_ii.cpu().numpy().tolist())

        losses_all = np.array(losses_all)

        return losses_all, times_all  # ,x,

    