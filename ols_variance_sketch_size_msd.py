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

from sketches import  (hadamard, sjlt, lev_approx, rrs_uniform,rrs_uniform_debia,rrs_lev_scores,
                       rrs_lev_scores_debia,rrs_shrinkage,rrs_shrinkage_debia,srht_opt,srht_opt_debia,less )
    

from sklearn.kernel_approximation import RBFSampler


from solvers_lr import  OLSRegression



SKETCH_FN = { 'rrs_uniform': rrs_uniform,'rrs_uniform_debia': rrs_uniform_debia,'rrs_lev_scores': rrs_lev_scores,
                       'rrs_lev_scores_debia': rrs_lev_scores_debia,'rrs_shrinkage': rrs_shrinkage,
              'rrs_shrinkage_debia': rrs_shrinkage_debia, 'srht_opt': srht_opt,'srht_opt_debia': srht_opt_debia,'less': less}


OUT = Path(__file__).with_suffix("")  
OUT_DIR = Path(__file__).resolve().parent / "figs" / OUT.name
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name):
    plt.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {OUT_DIR/name}.png")




def main():
    torch.set_default_dtype(torch.float64)


    csv_path = r"YearPredictionMSD.csv"
    df = pd.read_csv(csv_path, header=None)

   
    arr = df.to_numpy(dtype=np.float64)

   
    year = arr[:, 0]
    X_np = arr[:, 1:]
    y_np = np.log(year)  # y = log(year)


   
    mask = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np) & (year > 0)
    X_np = X_np[mask]
    y_np = y_np[mask]



    n_keep = 2**14
    rng = np.random.default_rng(11010074)
    idx = rng.choice(X_np.shape[0], size=n_keep, replace=False)

    X_np = X_np[idx]
    y_np = y_np[idx]


   
    eps=1e-6
    mu_X  = X_np.mean(axis=0, keepdims=True)
    std_X = X_np.std(axis=0, keepdims=True)

    std_X = np.maximum(std_X, eps)   
    X_np = (X_np - mu_X) / std_X

    
    X = torch.from_numpy(X_np)  # (n, p)
    y = torch.from_numpy(y_np).view(-1, 1)  # (n, 1)

    print(X.shape, y.shape)


   
    n_trials =500

    m = 400
    sketch_size =[5000,6000,7000,8000]
    nnz = 0.005
    

    A = X
    b=y
    n, d = A.shape
    print(n, d)

    alpha =2

    # Possible further processing...
    lreg = OLSRegression(A, b)  # define the class   OLSRegression

    

    losses_ols = {}
    times_ols = {}
    sketches = ['rrs_uniform_debia','rrs_uniform','rrs_lev_scores', 'rrs_lev_scores_debia',
               'srht_opt','srht_opt_debia','less']  # 'rrs_lev_scores','rrs_uniform','rrs_uniform_debia',
    for sketch in sketches:
        print('ols: ', sketch)
        losses_, times_ = lreg.ols_vary_sketch_size_nonava_all(sketch_size=sketch_size, sketch=sketch,  nnz=nnz,
                                                               n_trials=n_trials, alpha=alpha)
        losses_ols[sketch] = losses_
        times_ols[sketch] = times_




    

        
    sketches = [ 'rrs_uniform', 'rrs_uniform_debia','rrs_lev_scores', 'rrs_lev_scores_debia',  'srht_opt', 'srht_opt_debia', 'less']
    label_loss = [ 'rrs_uniform','rrs_uniform_debia', 'rrs_lev_scores', 'rrs_lev_scores_debia', 'srht_opt', 'srht_opt_debia', 'less']
   
    marker_loss = ['v', 'X', '_', '+', '*', '^', 'P','o']
   
    plt.figure()

    for ii, sketch in enumerate(sketches):
        
        torch.set_printoptions(precision=6)
        losses_ols_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in losses_ols[sketch]])  
        losses_ols_sketch_mean = losses_ols_sketch.mean(dim=1) 

        print('ols: ', sketch)
        print('losses_mean:', losses_ols_sketch_mean)

        plt.plot(sketch_size, losses_ols_sketch_mean, label=label_loss[ii], marker=marker_loss[ii])
    # plt.plot(times_newton, losses_newton, label='Newton', marker='*')

    # plt.yscale('log')
    
    plt.title('Error by Sketching Methods')
    plt.xlabel('Sketch size')
    plt.ylabel('Error')

    
    plt.legend()
    plt.yscale('log')
    
    plt.show()

    savefig("loss_skesize_uni_lev_srht_less_year")

    plt.close()




    plt.figure()

    for ii, sketch in enumerate(sketches):
        # loss_sketch=torch.tensor(losses_ols[sketch])
        # loss_log=torch.log10( loss_sketch)
        times_ols_j = torch.tensor(times_ols[sketch], dtype=torch.float64)

        # losses_ols_j = torch.tensor(losses_ols[sketch], dtype=torch.float64)

        times_ols_sketch_item = times_ols_j

        times_ols_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in times_ols[sketch]]) 
        times_ols_sketch_mean = times_ols_sketch.mean(dim=1)  

        # losses_ols_sketch = losses_ols_j
        print('ols: ', sketch)
        # print(losses_ols_sketch)
        # print('times_item:', times_ols_sketch_item)
        print('times_mean:', times_ols_sketch_mean)

        plt.plot(sketch_size, times_ols_sketch_mean, label=label_loss[ii], marker=marker_loss[ii])
    # plt.plot(times_newton, losses_newton, label='Newton', marker='*')

    # plt.yscale('log')
    

    plt.title('Error by Sketching Methods')
    plt.xlabel('Sketch size')
    plt.ylabel('Walk clock time')

    
    plt.legend()

    # plt.yscale('log')
    plt.show()
    savefig("time_skesize_uni_lev_srht_less_year")

    plt.close()

    

        
   
   
if __name__ == '__main__':
    main()



