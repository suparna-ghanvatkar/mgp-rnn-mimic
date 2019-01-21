#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:20:59 2017

Some useful functions.

@author: josephfutoma
"""
import tensorflow as tf
import numpy as np

def pad_rawdata(T,Y,ind_kf,ind_kt,X,meds_on_grid,covs):
    """ 
    Helper func. Pad raw data so it's in a padded array to be fed into the graph,
    since we can't pass in arrays of arrays directly.
    
    Inputs:
        arrays of data elements:
            T: array of arrays, with raw observation times
            Y,ind_kf,ind_kt: array of arrays;
                observed lab/vitals,
                indices into Y (same size)
            X: grid points
            meds_on_grid: list of arrays, each is grid_size x num_meds
            covs: matrix of baseline covariates for each patient. 
                to be tiled at each time and combined w meds
    Returns:
        Padded 2d np arrays of data, now of dim batchsize x batch_maxlen
    """
    N = np.shape(T)[0] #num in batch
    num_meds = np.shape(meds_on_grid[0])[1]
    num_covs = np.shape(covs)[1]
    
    T_lens = np.array([len(t) for t in T])
    T_maxlen = np.max(T_lens)
    T_pad = np.zeros((N,T_maxlen))
    
    Y_lens = np.array([len(y) for y in Y])
    Y_maxlen = np.max(Y_lens) 
    Y_pad = np.zeros((N,Y_maxlen))
    ind_kf_pad = np.zeros((N,Y_maxlen))
    ind_kt_pad = np.zeros((N,Y_maxlen))
    
    grid_lens = np.array([np.shape(m)[0] for m in meds_on_grid])
    grid_maxlen = np.max(grid_lens)
    meds_cov_pad = np.zeros((N,grid_maxlen,num_meds+num_covs))
    X_pad = np.zeros((N,grid_maxlen))
    
    for i in range(N):
        T_pad[i,:T_lens[i]] = T[i]
        Y_pad[i,:Y_lens[i]] = Y[i]
        ind_kf_pad[i,:Y_lens[i]] = ind_kf[i]
        ind_kt_pad[i,:Y_lens[i]] = ind_kt[i]
        X_pad[i,:grid_lens[i]] = X[i]
        meds_cov_pad[i,:grid_lens[i],:num_meds] = meds_on_grid[i]
        meds_cov_pad[i,:grid_lens[i],num_meds:] = np.tile(covs[i],(grid_lens[i],1))
                    
    return T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad

#####
##### Some TensorFlow functions used in the modeling
#####

def SE_kernel(length,x1,x2):
    x1 = tf.reshape(x1,[-1,1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-tf.pow(x1-x2,2.0)/length)
    return K

def OU_kernel(length,x1,x2):
    x1 = tf.reshape(x1,[-1,1]) #colvec
    x2 = tf.reshape(x2,[1,-1]) #rowvec
    K = tf.exp(-tf.abs(x1-x2)/length)
    return K

def dot(x,y):
    """ dot product of two vectors """
    return tf.reduce_sum(tf.multiply(x,y))

def CG(A,b):
    """ Conjugate gradient, to get solution x = A^-1 * b,
    can be faster than using the Cholesky for large scale problems
    """
    b = tf.reshape(b,[-1])
    n = tf.shape(A)[0]
    x = tf.zeros([n]) 
    r_ = b 
    p = r_ 
    
    #These settings are somewhat arbitrary
    #You might want to test sensitivity to these
    CG_EPS = tf.cast(n/1000,"float")
    MAX_ITER = tf.div(n,250) + 3
    
    def cond(i,x,r,p):
        return tf.logical_and(i < MAX_ITER, tf.norm(r) > CG_EPS)
    
    def body(i,x,r_,p):        
        p_vec = tf.reshape(p,[-1,1])
        Ap = tf.reshape(tf.matmul(A,p_vec),[-1]) #make a vector
        
        alpha = dot(r_,r_)/dot(p,Ap)
        x = x + alpha*p
        r = r_ - alpha*Ap
        beta = dot(r,r)/dot(r_,r_)
        p = r + beta*p

        return i+1,x,r,p
    
    i = tf.constant(0)
    i,x,r,p = tf.while_loop(cond,body,loop_vars=[i,x,r_,p])
    
    return tf.reshape(x,[-1,1])

def Lanczos(Sigma_func,b):
    """ Lanczos method to approximate Sigma^1/2 * b, with b random vec
    
    Note: this only gives you a single draw though, which may not be ideal.
    
    Inputs:
        Sigma_func: function to premultiply a vector by Sigma, which you 
            might not want to explicitly construct if it's huge.
        b: random vector of N(0,1)'
    
    Returns:
        random vector approximately equal to Sigma^1/2 * b
    """
    n = tf.shape(b)[0]
    k = tf.div(n,500) + 3 #this many Lanczos iterations

    betas = tf.zeros(1)
    alphas = tf.zeros(0)
    D = tf.zeros((n,1))
    
    b_norm = tf.norm(b)
    D = tf.concat([D,tf.reshape(b/b_norm,[-1,1])],1)
    
    def cond(j,alphas,betas,D):
        return j < k+1
    
    def body(j,alphas,betas,D):     
        d_j = tf.slice(D,[0,j],[-1,1])
        d = Sigma_func(d_j) - tf.slice(betas,[j-1],[1])*tf.slice(D,[0,j-1],[-1,1]) 
        alphas = tf.concat([alphas,[dot(d_j,d)]],0)
        d = d - tf.slice(alphas,[j-1],[1])*d_j
        betas = tf.concat([betas,[tf.norm(d)]],0)
        D = tf.concat([D,d/tf.slice(betas,[j],[1])],1)
        return j+1,alphas,betas,D
    
    j = tf.constant(1)
    j,alphas,betas,D = tf.while_loop(cond,body,loop_vars=[j,alphas,betas,D],
        shape_invariants=[j.get_shape(),tf.TensorShape([None]),
                          tf.TensorShape([None]),tf.TensorShape([None,None])])
    
    betas_ = tf.diag(tf.slice(betas,[1],[k-1]))
    D_ = tf.slice(D,[0,1],[-1,k])
    
    #build out tridiagonal H: alphas_1:k on main, betas_2:k on off 
    H = tf.diag(alphas) + tf.pad(betas_,[[1,0],[0,1]]) + tf.pad(betas_,[[0,1],[1,0]])
    
    e,v = tf.self_adjoint_eig(H)
    e_pos = tf.maximum(0.0,e)+1e-6 #make sure positive definite 
    e_sqrt = tf.diag(tf.sqrt(e_pos))
    sq_H = tf.matmul(v,tf.matmul(e_sqrt,tf.transpose(v)))

    out = b_norm*tf.matmul(D_,sq_H) 
    return tf.slice(out,[0,0],[-1,1]) #grab last column = *e_1

def block_CG(A_,B_):
    """
    block version of CG. Get solution to matrix equation AX = B, ie
    X = A^-1 * B. Will be much faster than Cholesky for large-scale problems.
    """
    n = tf.shape(B_)[0]
    m = tf.shape(B_)[1]
    
    X = tf.zeros((n,m))
    V_ = tf.zeros((n,m))
    R = B_
    R_ = tf.matrix_set_diag(tf.zeros((n,m)),tf.ones([m]))
        
    #somewhat arbitrary again, may want to check sensitivity
    CG_EPS = tf.cast(n/1000,"float")
    MAX_ITER = tf.div(n,250) + 3
    
    def cond(i,X,R_,R,V_):
        return tf.logical_and(i < MAX_ITER, tf.norm(R) > CG_EPS)
    
    def body(i,X,R_,R,V_):   
        S = tf.matrix_solve(tf.matmul(tf.transpose(R_),R_),
                            tf.matmul(tf.transpose(R),R))
        V = R + tf.matmul(V_,S)
        T = tf.matrix_solve(tf.matmul(tf.transpose(V),tf.matmul(A_,V)),
                            tf.matmul(tf.transpose(R),R))
        X = X + tf.matmul(V,T)
        V_ = V
        R_ = R
        R = R - tf.matmul(A_,tf.matmul(V,T))
        return i+1,X,R_,R,V_
    
    i = tf.constant(0)
    i,X,_,_,_ = tf.while_loop(cond,body,[i,X,R_,R,V_])
    return X

def block_Lanczos(Sigma_func,B_,n_mc_smps):
    """
    block Lanczos method to approx Sigma^1/2 * B, with B matrix of N(0,1)'s.
    Used to generate multiple approximate large normal draws.
    
    """
    n = tf.shape(B_)[0] #dim of the multivariate normal
    s = n_mc_smps #number of samples to draw
    k = tf.div(n,500) + 3 #number of Lanczos iterations
    
    betas = tf.zeros([1,s])
    alphas = tf.zeros([0,s])
    D = tf.zeros([s,n,1])
    B_norms = tf.norm(B_,axis=0)
    D = tf.concat([D,tf.expand_dims(tf.transpose(B_/B_norms),2)],2)
    
    def cond(j,alphas,betas,D):
        return j < k+1
    
    #TODO: use block-CG in place of Sigma
    def body(j,alphas,betas,D):  
        d_j = tf.squeeze(tf.slice(D,[0,0,j],[-1,-1,1]))
        d = Sigma_func(tf.transpose(d_j)) - (tf.slice(betas,[j-1,0],[1,-1])*
                tf.transpose(tf.squeeze(tf.slice(D,[0,0,j-1],[-1,-1,1]))))
        alphas = tf.concat([alphas,[tf.diag_part(tf.matmul(d_j,d))]],0)
        d = d - tf.slice(alphas,[j-1,0],[1,-1])*tf.transpose(d_j)
        betas = tf.concat([betas,[tf.norm(d,axis=0)]],0)
        D = tf.concat([D,tf.expand_dims(tf.transpose(d/tf.slice(betas,[j,0],[1,-1])),2)],2)
        return j+1,alphas,betas,D
    
    j = tf.constant(1)
    j,alphas,betas,D = tf.while_loop(cond,body,loop_vars=[j,alphas,betas,D],
        shape_invariants=[j.get_shape(),tf.TensorShape([None,None]),
                          tf.TensorShape([None,None]),tf.TensorShape([None,None,None])])
    
    D_ = tf.slice(D,[0,0,1],[-1,-1,k])
    
    ##TODO: replace loop
    H = tf.zeros([0,k,k])
    
    for ss in range(s):
        this_beta = tf.diag(tf.squeeze(tf.slice(betas,[1,ss],[k-1,1])))
        #build out tridiagonal H: alphas_1:k on main, betas_2:k on off 
        this_H = (tf.diag(tf.squeeze(tf.slice(alphas,[0,ss],[-1,1]))) +
                  tf.pad(this_beta,[[1,0],[0,1]]) +
                   tf.pad(this_beta,[[0,1],[1,0]]))
        H = tf.concat([H,tf.expand_dims(this_H,0)],0)    
    
    E,V = tf.self_adjoint_eig(H)
    E_sqrt = tf.zeros([0,k,k])
    #TODO: replace loop
    for ss in range(s): 
        #ensure positive definite
        E_sqrt = tf.concat([E_sqrt,tf.expand_dims(tf.diag(tf.squeeze(tf.sqrt(tf.maximum(tf.slice(E,[ss,0],[1,-1]),1e-6)))),0)],0)
    sq_H = tf.matmul(V,tf.matmul(E_sqrt,tf.transpose(V,perm=[0,2,1])))
        
    e1 = tf.expand_dims(tf.transpose(tf.tile(tf.slice(tf.eye(k),[0,0],[-1,1]),[1,s])),2)
    out = B_norms*tf.transpose(tf.squeeze(tf.matmul(D_,tf.matmul(sq_H,e1))))
    return out
