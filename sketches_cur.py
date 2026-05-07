import torch
import numpy as np 

from scipy.linalg import hadamard as hadam_scipy

from time import time


torch.set_default_dtype(torch.float64)


def _hadamard(matrix):
    n = matrix.shape[0]
    if n == 1:
        return matrix
    _t1 = _hadamard(matrix[:n//2,::]+matrix[n//2:,::])
    _t2 = _hadamard(matrix[:n//2,::]-matrix[n//2:,::])
    return torch.cat((_t1, _t2), 0)


def hadamard(matrix,response):
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
    n = matrix.shape[0]
    if n & (n-1) != 0:
        new_dim = 2**(int(np.ceil(np.log(n)/np.log(2))))
        pad_matrix = torch.zeros(new_dim-n, matrix.shape[1]).to(matrix.device)
        matrix = torch.cat((matrix, pad_matrix))
        pad_response = torch.zeros(new_dim-n,response.shape[1]).to(response.device)
        response = torch.cat((response , pad_response ))
    n = matrix.shape[0]
    diag = np.random.choice([-1,1], n, replace=True).reshape((-1,1))
    matrix = torch.Tensor(diag).to(matrix.device) * matrix
    response=torch.Tensor(diag).to(response.device) * response
    return 1./np.sqrt(n) * _hadamard(matrix), 1./np.sqrt(n) * _hadamard(response)



def rrs_uniform(matrix, sa_c, sa_r, c_sketch_size,r_sketch_size,  nnz=None,alpha_c=5,alpha_r=4):
    n,d= matrix.shape
    s_index_c=np.random.choice(np.arange(n), c_sketch_size, replace=True)
    sc=np.sqrt(n/(c_sketch_size))*sa_c[s_index_c,::]
    sa=np.sqrt(n/(c_sketch_size))*matrix[s_index_c,::]
     
    s_index_r=np.random.choice(np.arange(d), r_sketch_size, replace=True)
    sr=np.sqrt(d/(r_sketch_size))*sa_r[::,s_index_r]
    sa_cr=np.sqrt(d/(r_sketch_size))*sa[::, s_index_r]

    
    return sc, sr, sa_cr

def rrs_uniform_debia(matrix, sa_c, sa_r, c_sketch_size,r_sketch_size,  nnz=None,alpha_c=5,alpha_r=4):
    n,d= matrix.shape
    s_index_c=np.random.choice(np.arange(n), c_sketch_size, replace=True)

    ## exact lev
    y_mat_c, _, _ = torch.linalg.svd(sa_c, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_c = torch.sum(y_mat_c ** 2, axis=1)
    slev_c = lev_scores_c[s_index_c]

    weight_c = torch.sqrt(torch.tensor((c_sketch_size / n) - slev_c.reshape(-1, 1), dtype=torch.float64))
        
    sc=sa_c[s_index_c,::]/weight_c
    sa=matrix[s_index_c,::]/weight_c

    s_index_r=np.random.choice(np.arange(d), r_sketch_size, replace=True)

    ## exact lev
    y_mat_r, _, _ = torch.linalg.svd(sa_r.T, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_r = torch.sum(y_mat_r** 2, axis=1)
    slev_r = lev_scores_r[s_index_r]

    weight_r = torch.sqrt(torch.tensor((r_sketch_size / d) - slev_r.reshape(1, -1), dtype=torch.float64))
    
    sr=sa_r[::,s_index_r]/ weight_r
    sa_cr=sa[::, s_index_r]/ weight_r

    return sc, sr, sa_cr
  

def sjlt(matrix, sketch_size, nnz=None):
    n, d = matrix.shape 
    indices = np.vstack([np.random.choice(sketch_size, n).reshape((1,-1)), np.arange(n)])
    values = np.random.choice(np.array([-1,1], dtype=np.float64), size=n)
    S = torch.sparse_coo_tensor(indices, values, (sketch_size, n)).to(matrix.device)  #torch.sparse_coo_tensor is used to save the sparse tensor
    sa = S @ matrix
    return sa


def sparse_rademacher(matrix,response, sketch_size, nnz=None):
    n, d = matrix.shape 
    if nnz is None:
        nnz = d/n
    d_tilde = int(nnz*n)
    indices = np.vstack([np.repeat(np.arange(sketch_size), d_tilde).reshape((1,-1)), 
                         np.random.choice(n,size=sketch_size*d_tilde).reshape((1,-1))])
    values = np.random.choice(np.array([-1,1], dtype=np.float64), size=sketch_size*d_tilde)
    S = torch.sparse_coo_tensor(indices, values, (sketch_size, n)).to(matrix.device)
    sa=np.sqrt(n/(sketch_size*nnz*n)) * S @ matrix
    sy=np.sqrt(n/(sketch_size*nnz*n)) * S @ response
    return sa,sy



def rrs_lev_scores(matrix,sa_c, sa_r, c_sketch_size,r_sketch_size,nnz=None,alpha_c=5,alpha_r=4):
    n, d = matrix.shape
    lev_scores_c =lev_approx(sa_c, alpha=alpha_c)
    sum_lev_c=lev_scores_c.sum()

    prob_c = lev_scores_c/sum_lev_c
    s_index_c=np.random.choice(n, c_sketch_size, replace=True, p=prob_c)
    s_prob_c=prob_c[s_index_c]
    weight_c=torch.sqrt(torch.tensor(c_sketch_size*s_prob_c.reshape((-1, 1)),dtype=torch.float64))
    sc=sa_c[s_index_c, ::] /weight_c
    sa = matrix[s_index_c, ::]/weight_c

    lev_scores_r=lev_approx(sa_r.T, alpha=alpha_r)
    sum_lev_r=lev_scores_r.sum()

    prob_r= lev_scores_r/sum_lev_r
    s_index_r=np.random.choice(d, r_sketch_size, replace=True, p=prob_r)
    s_prob_r=prob_r[s_index_r]
    weight_r=torch.sqrt(torch.tensor(r_sketch_size*s_prob_r.reshape((1,-1)),dtype=torch.float64))

    sa_cr = sa[::, s_index_r] /weight_r
    sr=sa_r[::, s_index_r] /weight_r

    # print("alpha:",alpha)
    return  sc, sr, sa_cr

def rrs_lev_scores_debia(matrix,sa_c, sa_r, c_sketch_size,r_sketch_size, nnz=None,alpha_c=5,alpha_r=4):

    n, d = matrix.shape
    lev_scores_c =lev_approx(sa_c, alpha=alpha_c)
    sum_lev_c=lev_scores_c.sum()

    prob_c = lev_scores_c/sum_lev_c
    s_index_c=np.random.choice(n, c_sketch_size, replace=True, p=prob_c)
    # print(" c_sketch_size:", c_sketch_size)
    # print("  s_index_c:",  s_index_c)
    s_prob_c=prob_c[s_index_c]

    ## exact lev
    y_mat_c, _, _ = torch.linalg.svd(sa_c, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_c = torch.sum(y_mat_c ** 2, axis=1)
    slev_c = lev_scores_c[s_index_c]

    weight_c = torch.sqrt(torch.tensor(c_sketch_size*s_prob_c.reshape((-1, 1)) ,dtype=torch.float64)- slev_c.reshape(-1,1))
   
    sc=sa_c[s_index_c,::]/weight_c
    sa=matrix[s_index_c,::]/weight_c

    lev_scores_r=lev_approx(sa_r.T, alpha=alpha_r)
    sum_lev_r=lev_scores_r.sum()

    prob_r= lev_scores_r/sum_lev_r
    s_index_r=np.random.choice(d, r_sketch_size, replace=True, p=prob_r)
    s_prob_r=prob_r[s_index_r]
    
     ## exact lev
    y_mat_r, _, _ = torch.linalg.svd(sa_r.T, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_r = torch.sum(y_mat_r** 2, axis=1)
    slev_r = lev_scores_r[s_index_r]

    weight_r=torch.sqrt(torch.tensor(r_sketch_size*s_prob_r.reshape((1,-1)),dtype=torch.float64)- slev_r.reshape(1, -1))


    sa_cr = sa[::, s_index_r] /weight_r
    sr=sa_r[::, s_index_r] /weight_r

    return  sc, sr, sa_cr





def rrs_shrinkage(matrix,sa_c, sa_r, c_sketch_size,r_sketch_size, nnz=None,alpha_c=5,alpha_r=4):
    n, d = matrix.shape
    wei=0
    if wei==0:
       prob_c =np.ones(n) / n
    else:
        lev_scores_c =lev_approx(sa_c, alpha=alpha_c)
        sum_lev_c=lev_scores_c.sum()
        prob_c =wei* lev_scores_c/sum_lev_c+(1-wei)*1/n

    
   
    s_index_c=np.random.choice(n, c_sketch_size, replace=True, p=prob_c)
    s_prob_c=prob_c[s_index_c]
    weight_c=torch.sqrt(torch.tensor(c_sketch_size*s_prob_c.reshape((-1, 1)),dtype=torch.float64))
    sc=sa_c[s_index_c, ::] /weight_c
    sa = matrix[s_index_c, ::]/weight_c

    wei=0
    if wei==0:
       prob_r =np.ones(d) /d
    else:
        lev_scores_r=lev_approx(sa_r.T, alpha=alpha_r)
        sum_lev_r=lev_scores_r.sum()
        prob_r=wei* lev_scores_r/sum_lev_r +(1-wei)*1/d

    
    s_index_r=np.random.choice(d, r_sketch_size, replace=True, p=prob_r)
    s_prob_r=prob_r[s_index_r]
    weight_r=torch.sqrt(torch.tensor(r_sketch_size*s_prob_r.reshape((1,-1)),dtype=torch.float64))

    sa_cr = sa[::, s_index_r] /weight_r
    sr=sa_r[::, s_index_r] /weight_r

    return  sc, sr, sa_cr



def rrs_shrinkage_debia(matrix,sa_c, sa_r, c_sketch_size,r_sketch_size, nnz=None,alpha_c=5,alpha_r=4):

    n, d = matrix.shape

    wei=0
    if wei==0:
       prob_c =np.ones(n) / n
    else:
        lev_scores_c =lev_approx(sa_c, alpha=alpha_c)
        sum_lev_c=lev_scores_c.sum()
        prob_c =wei* lev_scores_c/sum_lev_c+(1-wei)*1/n

    s_index_c=np.random.choice(n, c_sketch_size, replace=True, p=prob_c)
    s_prob_c=prob_c[s_index_c]

    ## exact lev
    y_mat_c, _, _ = torch.linalg.svd(sa_c, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_c = torch.sum(y_mat_c ** 2, axis=1)
    slev_c = lev_scores_c[s_index_c]

    weight_c = torch.sqrt(torch.tensor(c_sketch_size*s_prob_c.reshape((-1, 1)) ,dtype=torch.float64)- slev_c.reshape(-1,1))

        
    sc=sa_c[s_index_c,::]/weight_c
    sa=matrix[s_index_c,::]/weight_c

    wei=0
    if wei==0:
       prob_r =np.ones(d) / d
    else:
        lev_scores_r=lev_approx(sa_r.T, alpha=alpha_r)
        sum_lev_r=lev_scores_r.sum()
        prob_r=wei* lev_scores_r/sum_lev_r +(1-wei)*1/d


    s_index_r=np.random.choice(d, r_sketch_size, replace=True, p=prob_r)
    s_prob_r=prob_r[s_index_r]
    
     ## exact lev
    y_mat_r, _, _ = torch.linalg.svd(sa_r.T, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_r = torch.sum(y_mat_r** 2, axis=1)
    slev_r = lev_scores_r[s_index_r]

    weight_r=torch.sqrt(torch.tensor(r_sketch_size*s_prob_r.reshape((1,-1)),dtype=torch.float64)- slev_r.reshape(1, -1))


    sa_cr = sa[::, s_index_r] /weight_r
    sr=sa_r[::, s_index_r] /weight_r

    return  sc, sr, sa_cr


def _srht(indices, v):
    n = v.shape[0]
    if n == 1:
        return v
    i1 = indices[indices < n//2]
    i2 = indices[indices >= n//2]
    if len(i1) == 0:
        return _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])
    elif len(i2) == 0:
        return _srht(i1, v[:n//2,::]+v[n//2:,::])
    else:
        return torch.cat([_srht(i1, v[:n//2,::]+v[n//2:,::]), _srht(i2-n//2, v[:n//2,::]-v[n//2:,::])], axis=0)

#
def srht_opt(matrix,sa_c, sa_r, c_sketch_size,r_sketch_size,  nnz=None,alpha_c=5,alpha_r=4):
    
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
 
    n,d = matrix.shape
    if n & (n-1) != 0:
        new_dim = 2**(int(np.log(n) / np.log(2))+1)
        matrix = torch.cat([matrix, torch.zeros(new_dim - n, matrix.shape[1]).to(matrix.device)], axis=0)
    n = matrix.shape[0]
    # indices = np.sort(np.random.choice(np.arange(n), sketch_size, replace=False))
    samples_c = np.random.choice(np.arange(n), c_sketch_size, replace=True)
    indices_c, counts_c = np.unique(samples_c, return_counts=True)
    v_c = torch.tensor(np.random.choice([-1,1], n, replace=True)).reshape((-1,1)).to(matrix.device)
    sa_c = v_c * sa_c
    matrix=v_c*matrix
    
    counts_tensor_c = torch.tensor(counts_c, dtype=matrix.dtype, device=matrix.device).reshape(-1, 1)
    weight_c=np.sqrt(1/c_sketch_size)*np.sqrt(counts_tensor_c) 
    sa = weight_c* _srht(indices_c, matrix)
    sc= weight_c* _srht(indices_c, sa_c)
    
    
    if d & (d-1) != 0:
        sa_r_trans=sa_r.T
        new_dim = 2**(int(np.log(d) / np.log(2))+1)
        sa_r_trans= torch.cat([ sa_r_trans, torch.zeros(new_dim - d,  sa_r_trans.shape[1]).to( sa_r_trans.device)], axis=0)
        sa_r=sa_r_trans.T
        sa_trans=sa.T
        # new_dim = 2**(int(np.log(d) / np.log(2))+1)
        sa_trans= torch.cat([ sa_trans, torch.zeros(new_dim - d,  sa_trans.shape[1]).to( sa_trans.device)], axis=0)
        sa=sa_trans.T
        

    # print("d_size", d)
    d = sa_r.shape[1]
    # indices = np.sort(np.random.choice(np.arange(n), sketch_size, replace=False))
    samples_r= np.random.choice(np.arange(d), r_sketch_size, replace=True)
    indices_r, counts_r= np.unique(samples_r, return_counts=True)
    v_r= torch.tensor(np.random.choice([-1,1], d, replace=True)).reshape((-1,1)).to(matrix.device)



    sa_r= v_r* sa_r.T
    sa=v_r*sa.T
    
    counts_tensor_r= torch.tensor(counts_r, dtype=sa.dtype, device=sa.device).reshape(-1, 1)
    weight_r=np.sqrt(1/r_sketch_size)*np.sqrt(counts_tensor_r) 
    sa_cr = weight_r* _srht(indices_r, sa)
    sr= weight_r* _srht(indices_r, sa_r)
    sa_cr= sa_cr .T
    sr=sr.T

    return sc, sr, sa_cr



def srht_opt_debia(matrix,sa_c, sa_r, c_sketch_size,r_sketch_size,  nnz=None,alpha_c=5,alpha_r=4):

    n, d = matrix.shape

    hada_sa_c,hada_matrix=hadamard(sa_c,matrix)
    s_index_c=np.random.choice(np.arange(n), c_sketch_size, replace=True)

    ## exact lev
    y_mat_c, _, _ = torch.linalg.svd(sa_c, full_matrices=False)
    lev_scores_c = torch.sum(y_mat_c ** 2, axis=1)
    slev_c = lev_scores_c[s_index_c]

    weight_c = torch.sqrt(torch.tensor((c_sketch_size / n) - slev_c.reshape(-1, 1), dtype=torch.float64))
   
        
    sc=hada_sa_c[s_index_c,::]/weight_c
    sa=hada_matrix[s_index_c,::]/weight_c


    hada_sa_r,hada_sa=hadamard(sa_r.T,sa.T)

    s_index_r=np.random.choice(np.arange(d), r_sketch_size, replace=True)

    ## exact lev
    y_mat_r, _, _ = torch.linalg.svd(sa_r.T, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores_r = torch.sum(y_mat_r** 2, axis=1)
    slev_r = lev_scores_r[s_index_r]

    weight_r = torch.sqrt(torch.tensor((r_sketch_size / d) - slev_r.reshape(-1, 1), dtype=torch.float64))
   
    
    
    sr=hada_sa_r[s_index_r,::]/ weight_r
    sa_cr=hada_sa[ s_index_r, ::]/ weight_r
    
    sa_cr= sa_cr .T
    sr=sr.T

    return sc, sr, sa_cr


    # sy = np.sqrt(n / sketch_size) * response[s_index, ::]
    #
    # return np.sqrt(1/sketch_size)*np.sqrt(counts_tensor) *sa, np.sqrt(1/sketch_size)*np.sqrt(counts_tensor) *sy



def lev_approx(matrix, alpha=50):
    n, d = matrix.shape
    m = int(alpha * d)
    sa=sjlt(matrix, m, nnz=None)

    _, sig_vec, v_mat = torch.linalg.svd(sa, full_matrices=False)
    y_mat = matrix @ v_mat.T / ((sig_vec.reshape((1, -1))  ) )
    lev_vec = torch.sum(y_mat ** 2, axis=1)
    # end = time()
    # print(alpha)
    return lev_vec.cpu().numpy()




