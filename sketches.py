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
    n = matrix.shape[0]
    diag = np.random.choice([-1,1], n, replace=True).reshape((-1,1))
    matrix = torch.Tensor(diag).to(matrix.device) * matrix
    response=torch.Tensor(diag).to(response.device) * response
    return 1./np.sqrt(n) * _hadamard(matrix), 1./np.sqrt(n) * _hadamard(response)



def rrs_uniform(matrix, response, sketch_size,  nnz=None,alpha=5):
    n = matrix.shape[0]
    s_index=np.random.choice(np.arange(n), sketch_size, replace=True)
    sa=np.sqrt(n/(sketch_size))*matrix[s_index,::]
    sy=np.sqrt(n/(sketch_size))*response[s_index,::]
    return sa,sy

def rrs_uniform_debia(matrix, response, sketch_size,  nnz=None,alpha=5):
    n = matrix.shape[0]
    s_index=np.random.choice(np.arange(n), sketch_size, replace=True)

    ## exact lev
    y_mat, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores = torch.sum(y_mat ** 2, axis=1)
    slev = lev_scores[s_index]

    weight = torch.sqrt(torch.tensor((sketch_size / n) - slev.reshape(-1, 1), dtype=torch.float64))

   
    sa=matrix[s_index,::]/weight

 
    sy=response[s_index,::]/ weight
    return sa,sy


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



def less(matrix,response,sketch_size,llev_scores=False,  nnz=None,alpha=5):
    n, d = matrix.shape
    if not llev_scores:

        hada_matrix,hada_response=hadamard(matrix,response)

        sa,sy = sparse_rademacher(hada_matrix,hada_response, sketch_size, nnz)
        return  sa,sy

    else:

        rlev_scores = lev_approx(matrix, alpha=5)
        sum_rlev = rlev_scores.sum()
        prob = rlev_scores / sum_rlev
        samples = torch.tensor(np.random.multinomial(d, pvals=prob, size=sketch_size)).to(matrix.device)
        samples = samples / (d*prob.reshape((1,-1)))
        debias_para = torch.tensor(sketch_size/(sketch_size - sum_rlev), dtype=torch.float64)
        print(rlev_scores.sum(), torch.tensor(sketch_size - rlev_scores.sum(), dtype=torch.float64))
        S = torch.sqrt((samples/sketch_size)*debias_para) * torch.tensor(np.random.choice([-1,1], size=(sketch_size,n))).to(matrix.device)
        return S @ matrix,S@response


def rrs_lev_scores(matrix,response, sketch_size, nnz=None,alpha=5):
    n, d = matrix.shape
    lev_scores =lev_approx(matrix, alpha=alpha)
    sum_lev=lev_scores.sum()
    prob = lev_scores/sum_lev
    s_index=np.random.choice(n, sketch_size, replace=True, p=prob)
    sa_matrix  = matrix[s_index, ::]
    sy_response=response[s_index, ::]
    s_prob=prob[s_index]
    weight=torch.sqrt(torch.tensor(sketch_size*s_prob.reshape((-1, 1)),dtype=torch.float64))
    sa=sa_matrix /weight
    sy = sy_response /weight
    return  sa,sy

def rrs_lev_scores_debia(matrix,response, sketch_size, nnz=None,alpha=5):
    n, d = matrix.shape
    lev_scores =lev_approx(matrix, alpha=alpha)
    sum_lev=lev_scores.sum()
    prob = lev_scores/sum_lev
    s_index=np.random.choice(n, sketch_size, replace=True, p=prob)
    sa_matrix  = matrix[s_index, ::]
    sy_response=response[s_index, ::]
    s_prob=prob[s_index]
    # slev=lev_scores[s_index]

    ## exact lev
    y_mat, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    lev_scores = torch.sum(y_mat ** 2, axis=1)
    slev = lev_scores[s_index]


    weight=torch.sqrt(torch.tensor(sketch_size*s_prob.reshape((-1, 1)) ,dtype=torch.float64)- slev.reshape(-1,1))


    sa=sa_matrix /weight
    sy = sy_response /weight
    return  sa,sy





def rrs_shrinkage(matrix,response,  sketch_size, nnz=None,alpha=5):
    n, d = matrix.shape
    wei=0
    if wei==0:
        prob =np.ones(n) / n
    else:
        lev_scores =lev_approx(matrix, alpha=alpha)
        sum_lev=lev_scores.sum()
        prob = wei*lev_scores/sum_lev+(1-wei)*1/n
    s_index=np.random.choice(n, sketch_size, replace=True, p=prob)
    sa_matrix  = matrix[s_index, ::]
    sy_response = response[s_index, ::]
    s_prob=prob[s_index]
    weight=torch.sqrt(torch.tensor(sketch_size*s_prob.reshape((-1, 1)),dtype=torch.float64))
    sa=sa_matrix /weight
    sy = sy_response / weight

    return  sa,sy

def rrs_shrinkage_debia(matrix,response,  sketch_size, nnz=None,alpha=5):
    n, d = matrix.shape
    wei=0
   
    if wei==0:
        prob =np.ones(n) / n
    else:
        lev_scores =lev_approx(matrix, alpha=alpha)
        sum_lev=lev_scores.sum()
        prob = wei*lev_scores/sum_lev+(1-wei)*1/n
    s_index=np.random.choice(n, sketch_size, replace=True, p=prob)
    sa_matrix  = matrix[s_index, ::]
    sy_response = response[s_index, ::]
    s_prob=prob[s_index]
    # slev = lev_scores[s_index]

    ## exact lev
    y_mat, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_scores = torch.sum(y_mat ** 2, axis=1)
    slev = lev_scores[s_index]

    weight=torch.sqrt(torch.tensor(sketch_size*s_prob.reshape((-1, 1)),dtype=torch.float64)-slev.reshape(-1,1))


    sa=sa_matrix /weight
    sy = sy_response / weight
    return  sa,sy


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
def srht_opt(matrix,response,sketch_size,  nnz=None,alpha=None):
   
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1,1))
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
    response=v*response
    sa = _srht(indices, matrix)
    sy= _srht(indices, response)
    counts_tensor = torch.tensor(counts, dtype=sa.dtype, device=sa.device).reshape(-1, 1)
    return np.sqrt(1/sketch_size)*np.sqrt(counts_tensor) *sa, np.sqrt(1/sketch_size)*np.sqrt(counts_tensor) *sy



def srht_opt_debia(matrix,response,sketch_size,  nnz=None,alpha=None):

    n, d = matrix.shape
    hada_matrix,hada_response = hadamard(matrix,response)  # hard_A
    s_index=np.random.choice(n, sketch_size, replace=True)


    y_mat, _, _ = torch.linalg.svd(hada_matrix, full_matrices=False)
    # y_mat =hada_matrix @ v_mat.T / ((sig_vec.reshape((1, -1))))
    lev_vec = torch.sum(y_mat ** 2, axis=1)

    slev=lev_vec[s_index]

    weight = torch.sqrt(torch.tensor(sketch_size / n - slev.reshape(-1, 1), dtype=torch.float64))

    sa = hada_matrix[s_index, ::] / weight
    sy = hada_response[s_index, ::] / weight
    return sa, sy

def lev_approx(matrix, alpha=50):
    n, d = matrix.shape
    m = int(alpha * d)
    sa=sjlt(matrix, m, nnz=None)

    _, sig_vec, v_mat = torch.linalg.svd(sa, full_matrices=False)
    y_mat = matrix @ v_mat.T / ((sig_vec.reshape((1, -1))  ) )
    lev_vec = torch.sum(y_mat ** 2, axis=1)
    return lev_vec.cpu().numpy()




    
    
    
    