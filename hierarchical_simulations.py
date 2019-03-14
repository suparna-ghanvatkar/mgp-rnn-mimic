import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os
from data_prep import dataset_prep
import pickle
from math import ceil
import argparse
from util import pad_rawdata,SE_kernel,OU_kernel,dot,CG,Lanczos,block_CG,block_Lanczos
from simulations import *

#os.environ["CUDA_VISIBLE_DEVICES"]="1"#####
##### Numpy functions used to simulate data and pad raw data to feed into TF
#####
seed = 8675309
rs = np.random.RandomState(seed) #fixed seed in np
def gen_MGP_params(M):
    """
    Generate some MGP params for each class.
    Assume MGP is stationary and zero mean, so hyperparams are just:
        Kf: covariance across time series
        length: length scale for shared kernel across time series
        noise: noise level for each time series
    """
    true_Kfs = []
    true_noises = []
    true_lengths = []

    #Class 0
    tmp = rs.normal(0,.2,(M,M))
    true_Kfs.append(np.dot(tmp,tmp.T))
    true_lengths.append(1.0)
    true_noises.append(np.linspace(.02,.08,M))

    #Class 1
    tmp = rs.normal(0,.4,(M,M))
    true_Kfs.append(np.dot(tmp,tmp.T))
    true_lengths.append(2.0)
    true_noises.append(np.linspace(.1,.2,M))

    return true_Kfs,true_noises,true_lengths

def add_high_freq(end_time):
    """
    Returns a high frequency data along with observation times
    The freq is 125 hz which implies the obs times progress by 0.008 second.
    However, the hours granularity is too large..so change to minutes which is 0.000133 minutes

    """
    num_obs = int(ceil(end_time*60*60*125))
    y_i = np.random.normal(0,1,num_obs)
    #ind_kf = np.array([index]*num_obs)
    #ind_kt = np.arange(num_obs)
    #obs_times = (np.arange(num_obs)*0.000133).tolist()
    return y_i#, ind_kf, ind_kt

def kron(a,b):
    res = np.zeros((a.shape[0]*b.shape[0],a.shape[1]*b.shape[1]))
    for index,i in np.ndenumerate(a):
        start_row = index[0]*3
        start_col = index[1]*3
        res[start_row:start_row+3, start_col:start_col+3] = i
    return res*np.tile(b,a.shape)


def retrieve_hierarchical_sim():
    num_obs_times = pickle.load(open('num_obs_times_hierarchical.pickle','r'))
    num_obs_values = pickle.load(open('num_obs_values_hierarchical.pickle','r'))
    num_rnn_grid_times = pickle.load(open('num_rnn_grid_times_hierarchical.pickle','r'))
    rnn_grid_times = pickle.load(open('rnn_grid_times_hierarchical.pickle','r'))
    labels = pickle.load(open('labels_hierarchical.pickle','r'))
    T = pickle.load(open('T_hierarchical.pickle','r'))
    Y = pickle.load(open('Y_hierarchical.pickle','r'))
    ind_kf = pickle.load(open('ind_kf_hierarchical.pickle','r'))
    ind_kt = pickle.load(open('ind_kt_hierarchical.pickle','r'))
    meds_on_grid = pickle.load(open('meds_on_grid_hierarchical.pickle','r'))
    baseline_covs = pickle.load(open('baseline_covs_hierarchical.pickle','r'))
    H = pickle.load(open('H_hierarchical.pickle','r'))
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs,H)

def sim_dataset(num_encs,M,n_covs,n_meds,pos_class_rate = 0.5,trainfrac=0.2):
    """
    Returns everything we need to run the model.

    Each simulated patient encounter consists of:
        Multivariate time series (labs/vitals)
        Static baseline covariates
        Medication administration times (no values; just a point process)
    """
    true_Kfs,true_noises,true_lengths = gen_MGP_params(M-1)

    #end_times = np.random.uniform(10,120,num_encs) #last observation time of the encounter
    #HOURS granularity again!
    hour_conv = 0.008/3600
    end_times = np.random.uniform(10,50,num_encs) #last observation time of the encounter
    num_obs_times = np.random.poisson(end_times,num_encs)+3 #number of observation time points per encounter, increase with  longer series
    num_obs_values = np.array(num_obs_times*M*trainfrac,dtype="int")
    #print num_obs_times
    #number of inputs to RNN. will be a grid on integers, starting at 0 and ending at next integer after end_time
    num_rnn_grid_times = np.array(np.floor(end_times)+1,dtype="int")
    rnn_grid_times = []
    labels = rs.binomial(1,pos_class_rate,num_encs)

    T = [];  #actual observation times
    Y = []; ind_kf = []; ind_kt = [] #actual data; indices pointing to which lab, which time
    H = []
    baseline_covs = np.zeros((num_encs,n_covs))
    #each contains an array of size num_rnn_grid_times x n_meds
    #   simulates a matrix of indicators, where each tells which meds have been given between the
    #   previous grid time and the current. (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,times,
    #   times and will need to convert to this form, for feeding into the RNN
    meds_on_grid = []

    print('Simming data...')
    for i in range(num_encs):
        sys.stdout.flush()
        if i%500==0:
            print('%d/%d' %(i,num_encs))
        obs_times = np.insert(np.sort(np.random.uniform(0,end_times[i],num_obs_times[i]-1)),0,0)
        #obs_times = np.arange(num_obs_times[i])*hour_conv
        #print(num_obs_times[i])
        #obs_ind = np.random.choice(num_obs_times[i], num_obs_times_[i], replace=False).tolist()
        #print obs_times
        #obs_times_ = obs_times[obs_ind]
        T.append(obs_times)
        l = labels[i]
        y_i,ind_kf_i,ind_kt_i = sim_multitask_GP(obs_times,true_lengths[l],true_noises[l],true_Kfs[l],trainfrac)
        #y_i,ind_kf_i,ind_kt_i = sim_multitask_GP(end_times[i], obs_times_,true_lengths[l],true_noises[l],true_Kfs[l],trainfrac)
        h_i = add_high_freq(end_times[i])
        #print "in sim data"
        #print y_i, ind_kf_i, ind_kt_i, obs_times
        H.append(h_i)
        Y.append(y_i); ind_kf.append(ind_kf_i); ind_kt.append(ind_kt_i)
        rnn_grid_times.append(np.arange(num_rnn_grid_times[i]))
        if l==0: #sim some different baseline covs; meds for 2 classes
            baseline_covs[i,:int(n_covs/2)] = rs.normal(0.1,1.0,int(n_covs/2))
            baseline_covs[i,int(n_covs/2):] = rs.binomial(1,0.2,int(n_covs/2))
            meds = rs.binomial(1,.02,(num_rnn_grid_times[i],n_meds))
        else:
            baseline_covs[i,:int(n_covs/2)] = rs.normal(0.2,1.0,int(n_covs/2))
            baseline_covs[i,int(n_covs/2):] = rs.binomial(1,0.1,int(n_covs/2))
            meds = rs.binomial(1,.04,(num_rnn_grid_times[i],n_meds))
        #print "Meds:",meds
        meds_on_grid.append(meds)
    T = np.array(T)
    Y = np.array(Y); ind_kf = np.array(ind_kf); ind_kt = np.array(ind_kt)
    meds_on_grid = np.array(meds_on_grid)
    rnn_grid_times = np.array(rnn_grid_times)
    '''
    pickle.dump(num_obs_times, open('num_obs_times_hierarchical.pickle','w'))
    pickle.dump(num_obs_values, open('num_obs_values_hierarchical.pickle','w'))
    pickle.dump(num_rnn_grid_times, open('num_rnn_grid_times_hierarchical.pickle','w'))
    pickle.dump(rnn_grid_times, open('rnn_grid_times_hierarchical.pickle','w'))
    pickle.dump(labels, open('labels_hierarchical.pickle','w'))
    pickle.dump(T, open('T_hierarchical.pickle','w'))
    pickle.dump(Y, open('Y_hierarchical.pickle','w'))
    pickle.dump(ind_kf, open('ind_kf_hierarchical.pickle','w'))
    pickle.dump(ind_kt, open('ind_kt_hierarchical.pickle','w'))
    pickle.dump(meds_on_grid, open('meds_on_grid_hierarchical.pickle','w'))
    pickle.dump(baseline_covs, open('baseline_covs_hierarchical.pickle','w'))
    pickle.dump(H, open('H_hierarchical.pickle', 'w'))
    '''
    return (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,
            labels,T,Y,ind_kf,ind_kt,meds_on_grid,baseline_covs,H)


def sim_multitask_GP(times, length,noise_vars,K_f,trainfrac):
    """
    draw from a multitask GP.

    we continue to assume for now that the dim of the input space is 1, ie just time

    M: number of tasks (labs/vitals/time series)

    train_frac: proportion of full M x N data matrix Y to include

    """
    M = np.shape(K_f)[0]
    N = len(times)
    #print M
    #print N
    n = N*M
    K_t = OU_kernel_np(length,times) #just a correlation function
    Sigma = np.diag(noise_vars)

    #print K_f.shape, K_t.shape, Sigma.shape
    K = kron(K_f,K_t) + kron(Sigma,np.eye(N)) + 1e-6*np.eye(n)
    L_K = np.linalg.cholesky(K)
    y = np.dot(L_K,np.random.normal(0,1,n)) #Draw normal
    #y = np.random.normal(0,1,M*N)
    #get indices of which time series and which time point, for each element in y
    ind_kf = np.tile(np.arange(M),(N,1)).flatten('F') #vec by column
    ind_kx = np.tile(np.arange(N),(M,1)).flatten()
    #print ind_kf, ind_kx
    #randomly dropout some fraction of fully observed time series
    perm = np.random.permutation(n)
    n_train = int(trainfrac*n)
    train_inds = perm[:n_train]
    #high_freq_ind = np.where(ind_kx==0)
    #train_inds = np.unique(np.concatenate((high_freq_ind,train_inds)))
    y = y[train_inds]
    ind_kf = ind_kf[train_inds]
    ind_kx = ind_kx[train_inds]
    #print "after dropout"
    #print ind_kf_, ind_kx_
    return y,ind_kf,ind_kx

def OU_kernel_np(length,x):
    """ just a correlation function, for identifiability
    """
    x1 = np.reshape(x,[-1,1]) #colvec
    x2 = np.reshape(x,[1,-1]) #rowvec
    #print x1.shape, x2.shape, length
    K_xx = np.exp(-np.abs(x1-x2)/length)
    #print K_xx.shape
    return K_xx

if __name__=="__main__":
    sim_dataset(4, 4, 3, 4)
