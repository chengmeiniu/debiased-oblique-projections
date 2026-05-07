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


from solvers_lr_bias import  OLSRegression




SKETCH_FN = { 'rrs_uniform': rrs_uniform,'rrs_lev_scores': rrs_lev_scores, 'rrs_shrinkage': rrs_shrinkage,
              'srht_opt': srht_opt, 'less': less}

SKETCH_FN_debia = { 'rrs_uniform':  rrs_uniform_debia,  'rrs_lev_scores': rrs_lev_scores_debia,'rrs_shrinkage': rrs_shrinkage_debia, 'srht_opt': srht_opt_debia}


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

    std_X = np.maximum(std_X, eps)   # 防止除0
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
    losses_ols_debia = {}
    times_ols_debia = {}
    sketches = ['rrs_uniform', 'rrs_lev_scores', 'srht_opt']  # 'rrs_lev_scores','rrs_uniform','rrs_uniform_debia',
    # 'rs_norm', 'rrs_uniform',
    for sketch in sketches:

        print('ols_debia: ', sketch)

        losses_debia_, times_debia_ = lreg. debia_ols_vary_sketch_size_nonava_all(sketch_size=sketch_size, sketch=sketch,  nnz=nnz,
                                                               n_trials=n_trials, alpha=alpha)
                                               

        losses_ols_debia[sketch] = losses_debia_
        times_ols_debia[sketch] = times_debia_   #+ time_cr

        print('ols: ', sketch)

        losses_, times_ = lreg.ols_vary_sketch_size_nonava_all(sketch_size=sketch_size, sketch=sketch, nnz=nnz,
                                                               n_trials=n_trials, alpha=alpha)

        losses_ols[sketch] = losses_
        times_ols[sketch] = times_  # + time_cr

    sketches_basline = ['less']

    for sketch in sketches_basline:
        print('ols: ', sketch)

        losses_, times_ =lreg.ols_vary_sketch_size_nonava_all(sketch_size=sketch_size, sketch=sketch,  nnz=nnz,
                                                               n_trials=n_trials, alpha=alpha)                                                    
        losses_ols[sketch] = losses_
        times_ols[sketch] = times_   #+ time_cr



    

    sketches = ['rrs_uniform', 'rrs_lev_scores',  'srht_opt']
    # print(losses_ols, times_ols)

    
    label_loss = ['rrs_uniform','rrs_lev_scores', 'srht_opt']
    label_loss_debia=  ['rrs_uniform_debia', 'rrs_lev_scores_debia','srht_opt_debia'] 
    
    
    marker_loss = ['o',  'v', 'X']
    marker_loss_debia = [ 's', '*','_']
   
    plt.figure()

    for ii, sketch in enumerate(sketches):
        # loss_sketch=torch.tensor(losses_ols[sketch])
        # loss_log=torch.log10( loss_sketch)

        # losses_ols_j = torch.tensor(losses_ols[sketch], dtype=torch.float64)
        torch.set_printoptions(precision=6)

        losses_ols_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in losses_ols[sketch]]) 
        losses_ols_sketch_mean = losses_ols_sketch 

        print('ols: ', sketch)
        print('losses_mean:', losses_ols_sketch_mean)
        
        losses_ols_sketch_debia = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in losses_ols_debia[sketch]])  
        losses_ols_sketch_mean_debia = losses_ols_sketch_debia 

        print('ols_debia: ', sketch)
        print('losses_mean_debia:', losses_ols_sketch_mean_debia)



        plt.plot(sketch_size, losses_ols_sketch_mean, label=label_loss[ii], marker=marker_loss[ii])
        plt.plot(sketch_size, losses_ols_sketch_mean_debia, label=label_loss_debia[ii], marker=marker_loss_debia[ii])
    # plt.plot(times_newton, losses_newton, label='Newton', marker='*')

    sketches_basline = ['less']
    label_loss_basline = ['less']
   
    marker_loss_basline = ['P'] 
    for ii, sketch in enumerate(sketches_basline):
            # loss_sketch=torch.tensor(losses_ols[sketch])
            # loss_log=torch.log10( loss_sketch)

            # losses_ols_j = torch.tensor(losses_ols[sketch], dtype=torch.float64)
            torch.set_printoptions(precision=6)

            losses_ols_sketch = torch.stack(
                [torch.tensor(x, dtype=torch.float64) for x in losses_ols[sketch]])  
            losses_ols_sketch_mean = losses_ols_sketch 

            print('ols: ', sketch)
            print('losses_mean:', losses_ols_sketch_mean)

            plt.plot(sketch_size, losses_ols_sketch_mean, label=label_loss_basline[ii], marker=marker_loss_basline[ii])
    

    # plt.yscale('log')
    
    plt.title('Error by Sketching Methods')
    plt.xlabel('Sketch size')
    plt.ylabel('Error')

   
    plt.legend()
    plt.yscale('log')
    # 显示图形
    plt.show()

    savefig("loss_skesize_all_year")

    plt.close()



    plt.figure()

    for ii, sketch in enumerate(sketches):
        

        times_ols_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in times_ols[sketch]])  
        times_ols_sketch_mean = times_ols_sketch  

        # losses_ols_sketch = losses_ols_j
        print('ols_debia: ', sketch)
        # print(losses_ols_sketch)
        # print('times_item:', times_ols_sketch_item)
        print('times_mean_debia:', times_ols_sketch_mean)

        times_ols_sketch_debia = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in times_ols_debia[sketch]]) 
        times_ols_sketch_mean_debia = times_ols_sketch_debia 

        # losses_ols_sketch = losses_ols_j
        print('ols_debia: ', sketch)
        # print(losses_ols_sketch)
        # print('times_item:', times_ols_sketch_item)
        print('times_mean_debia:', times_ols_sketch_mean_debia)


        plt.plot(sketch_size, times_ols_sketch_mean, label=label_loss[ii], marker=marker_loss[ii])
        plt.plot(sketch_size, times_ols_sketch_mean_debia, label=label_loss_debia[ii], marker=marker_loss_debia[ii])

    for ii, sketch in enumerate(sketches_basline):
        

        times_ols_sketch = torch.stack(
            [torch.tensor(x, dtype=torch.float64) for x in times_ols[sketch]])  
        times_ols_sketch_mean = times_ols_sketch 

        # losses_ols_sketch = losses_ols_j
        print('ols: ', sketch)
        # print(losses_ols_sketch)
        # print('times_item:', times_ols_sketch_item)
        print('times_mean:', times_ols_sketch_mean)

        plt.plot(sketch_size, times_ols_sketch_mean, label=label_loss_basline[ii], marker=marker_loss_basline[ii])
    # plt.plot(times_newton, losses_newton, label='Newton', marker='*')

    # plt.yscale('log')
   

    plt.title('Error by Sketching Methods')
    plt.xlabel('Sketch size')
    plt.ylabel('Walk clock time')

    
    plt.legend()

    # plt.yscale('log')
    plt.show()
    savefig("time_skesize_all_year")

    plt.close()


        



    
    


if __name__ == '__main__':
    main()



