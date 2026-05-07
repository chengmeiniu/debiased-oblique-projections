# pip install matplotlib
import csv
import pandas as pd
import os
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl


import matplotlib.pyplot as plt
from pathlib import Path




from time import time

from sketches_cur import  (hadamard, sjlt, lev_approx, rrs_uniform,rrs_uniform_debia,rrs_lev_scores,
                       rrs_lev_scores_debia,rrs_shrinkage,rrs_shrinkage_debia,srht_opt,srht_opt_debia )
    # gaussian, less, sparse_rademacher, srht, rrs, rrs_lev_scores,rrs_lev_scores)

from sklearn.kernel_approximation import RBFSampler


from solvers_cur import  CURdecomposition
from solvers_cur_corss_u import CURdecomposition_baseline



SKETCH_FN = { 'rrs_uniform': rrs_uniform,'rrs_lev_scores': rrs_lev_scores, 'rrs_shrinkage': rrs_shrinkage,
              'srht_opt': srht_opt}

SKETCH_FN_debia = {  'rrs_uniform':  rrs_uniform_debia, 'rrs_lev_scores': rrs_lev_scores_debia,'rrs_shrinkage': rrs_shrinkage_debia, 'srht_opt': srht_opt_debia}


OUT = Path(__file__).with_suffix("") 
OUT_DIR = Path(__file__).resolve().parent / "figs" / OUT.name
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name):
    plt.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {OUT_DIR/name}.png")



def _c_r_computation_random(A, c_col_size,r_row_size):
    
    n, d = A.shape
     # local random generator with fixed seed
    seed=2
    rng = np.random.default_rng(seed)
    # compute the A.^2

    A_ele_squ= A.pow(2)

    """
    Return C= A@S^T
    Column sampling probabilities proportional to ||A[:,j]||_2^2.
    """

    # shape: (p,)  sum(dim=0)the sum of column
    probs_col = (A_ele_squ.sum(dim=0)) #.clamp_min(eps)
    prob_col=probs_col / probs_col.sum()

    s_index_col=rng.choice(d, c_col_size, replace=True, p=prob_col)
    sa_matrix_c  =A[ :,s_index_col]
    # sy_response = response[s_index, ::]
    s_prob_col=prob_col[s_index_col]
    weight_c=torch.sqrt(torch.tensor(c_col_size*s_prob_col.reshape((1, -1)),dtype=torch.float64))
    sa_c=sa_matrix_c /weight_c 
    # sa_c=sa_matrix_c
    """
    Return R=S @ A
    Row sampling probabilities proportional to ||A[i,:]||_2^2.
    """
    
    probs_row = (A_ele_squ.sum(dim=1))   #clamp_min(eps)
    prob_row= probs_row  / probs_row.sum()


    s_index_row=rng.choice(n, r_row_size, replace=True, p=prob_row)
    sa_matrix_r  =A[s_index_row, ::]
    # sy_response = response[s_index, ::]
    s_prob_row=prob_row[s_index_row]
    weight_r =torch.sqrt(torch.tensor(r_row_size*s_prob_row.reshape((-1, 1)),dtype=torch.float64))
    sa_r=sa_matrix_r  /weight_r 
    # sa_r=sa_matrix_r 
    # sy = sy_response / weight

    return sa_c, sa_r
    
def load_cifar(n=2 ** 14, d=2 ** 6):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    seed = 8
    torch.manual_seed(seed)
    

    g = torch.Generator()
    g.manual_seed(seed)

    trainloader = DataLoader(trainset, batch_size=n, shuffle=True, num_workers=2, generator=g)


    A_, b_ =  next(iter(trainloader))
    print(b_)
    A_ = A_.reshape((A_.shape[0], -1)).clone().detach().to(torch.float64)[:, :d]

   

    b = torch.tensor([-1 if b_[ii] % 2 == 0 else 1 for ii in range(len(b_))], dtype=torch.float64).reshape((-1, 1))

    return A_, b






def main():
    torch.set_default_dtype(torch.float64)
   




    # Load CIFAR-10 data
    A_, b = load_cifar(n=2 ** 13, d=2 ** 11)
    A = A_



    n_trials =200

    m = 400
    sketch_size = [500,1000,2000,3000]
    nnz = 0.005
    

    A = A
    b=b
    n, d = A.shape
    print(n, d)

    alpha_c =2

    alpha_r =3

    c_col_size= 30
    r_row_size=60
    c_sketch_size= [500,1000,2000,3000] 
    r_sketch_size=  500 
    sketch_size= c_sketch_size

    
    
    start_cr=time()
    sa_c, sa_r =_c_r_computation_random(A, c_col_size,r_row_size)
    time_cr=time()-start_cr

    lcur = CURdecomposition( A,sa_c, sa_r)  # define the class   OLSRegression
    lcur_baseline=CURdecomposition_baseline(A)
   

    losses_cur = {}
    times_cur = {}
    losses_cur_debia = {}
    times_cur_debia = {}

    
    sketches = ['rrs_uniform','rrs_lev_scores', 'srht_opt']  # 'rrs_lev_scores','rrs_uniform','rrs_uniform_debia',
    # 'rs_norm', 'rrs_uniform',
    for sketch in sketches:
        print('cur: ', sketch)

        losses_, times_ =lcur.cur_vary_sketch_size_nonava_all(c_col_size=c_col_size,r_row_size=r_row_size, c_sketch_size=c_sketch_size,r_sketch_size=r_sketch_size,  sketch=sketch,  nnz=nnz,
                                                               n_trials=n_trials,  alpha_c=alpha_c,alpha_r=alpha_r) 

        losses_cur[sketch] = losses_
        times_cur[sketch] = times_   #+ time_cr 

        print('cur_debia: ', sketch)

        losses_debia_, times_debia_ =lcur.debia_cur_vary_sketch_size_nonava_all(c_col_size=c_col_size,r_row_size=r_row_size, c_sketch_size=c_sketch_size,r_sketch_size=r_sketch_size,  sketch=sketch,  nnz=nnz,
                                                               n_trials=n_trials,  alpha_c=alpha_c,alpha_r=alpha_r)
   
        losses_cur_debia[sketch] = losses_debia_
        times_cur_debia[sketch] = times_debia_    #+ time_cr 


    

    
    sketches = ['rrs_uniform','rrs_lev_scores', 'srht_opt']
    
    label_loss =  ['rrs_uniform','rrs_lev_scores', 'srht_opt']
    label_loss_debia=  [ 'rrs_uniform_debia','rrs_lev_scores_debia', 'srht_opt_debia'] 
   
    marker_loss = ['o',  'v', 'X']
    marker_loss_debia = [ 's', '*','_']
    
    plt.figure()

    for ii, sketch in enumerate(sketches):
        
        # losses_cur_j = torch.tensor(losses_cur[sketch], dtype=torch.float64)
        torch.set_printoptions(precision=6)
        losses_cur_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in losses_cur[sketch]]) 
        losses_cur_sketch_mean = losses_cur_sketch.mean(dim=1)  

        print('cur: ', sketch)
        print('losses_mean:', losses_cur_sketch_mean)
        
        losses_cur_sketch_debia = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in losses_cur_debia[sketch]])  
        losses_cur_sketch_mean_debia = losses_cur_sketch_debia.mean(dim=1)  

        print('cur_debia: ', sketch)
        print('losses_mean_debia:', losses_cur_sketch_mean_debia)



        plt.plot(sketch_size, losses_cur_sketch_mean, label=label_loss[ii], marker=marker_loss[ii])
        plt.plot(sketch_size, losses_cur_sketch_mean_debia, label=label_loss_debia[ii], marker=marker_loss_debia[ii])


    
    

    # plt.yscale('log')
    
    plt.title('Error by Sketching Methods')
    plt.xlabel('Sketch size')
    plt.ylabel('Error')

    
    plt.legend()
    plt.yscale('log')
   
    plt.show()

    savefig("loss_skesize_all_cifar_10")

    plt.close()



    plt.figure()

    for ii, sketch in enumerate(sketches):
        

        times_cur_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in times_cur[sketch]]) +time_cr   
        times_cur_sketch_mean = times_cur_sketch.mean(dim=1) 

        # losses_cur_sketch = losses_cur_j
        print('cur: ', sketch)
        # print(losses_cur_sketch)
        # print('times_item:', times_cur_sketch_item)
        print('times_mean:', times_cur_sketch_mean)

        times_cur_sketch_debia = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in times_cur_debia[sketch]]) +time_cr  
        times_cur_sketch_mean_debia = times_cur_sketch_debia.mean(dim=1) 

        # losses_cur_sketch = losses_cur_j
        print('cur_debia: ', sketch)
        # print(losses_cur_sketch)
        # print('times_item:', times_cur_sketch_item)
        print('times_mean_debia:', times_cur_sketch_mean_debia)


        plt.plot(sketch_size, times_cur_sketch_mean, label=label_loss[ii], marker=marker_loss[ii])
        plt.plot(sketch_size, times_cur_sketch_mean_debia, label=label_loss_debia[ii], marker=marker_loss_debia[ii])

  

    plt.title('Error by Sketching Methods')
    plt.xlabel('Sketch size')
    plt.ylabel('Walk clock time')

   
    plt.legend()

    # plt.yscale('log')
    plt.show()
    savefig("time_skesize_all_cifar_10")

    plt.close()

       
    




if __name__ == '__main__':
    main()



