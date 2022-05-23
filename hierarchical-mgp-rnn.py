#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 5 Jun 2017.

Fit MGP-RNN model on full data, with Lanczos and CG to speed things up.

@author: josephfutoma
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os
import pickle
from math import ceil
import argparse
from hierarchical_util import *
#from hierarchical_util import pad_rawdata,SE_kernel,OU_kernel,dot,CG,Lanczos,block_CG,block_Lanczos
#from simulations import *
#from tf.keras.layers import *
#from patient_events import *
from patient_traintest_final import *
#from patient_hierar_prep import *
from tensorflow.python import debug as tf_debug
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#####
##### Tensorflow functions
#####

def draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti):
    """
    given GP hyperparams and data values at observation times, draw from
    conditional GP

    inputs:
        length,noises,Lf,Kf: GP params
        Yi: observation values
        Ti: observation times
        Xi: grid points (new times for rnn)
        ind_kfi,ind_kti: indices into Y
    returns:
        draws from the GP at the evenly spaced grid times Xi, given hyperparams and data
    """
    ny = tf.shape(Yi)[0]
    K_tt = OU_kernel(length,Ti,Ti)
    D = tf.diag(noises)

    grid_f = tf.meshgrid(ind_kfi,ind_kfi) #same as np.meshgrid
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_f[0],grid_f[1]),-1))

    grid_t = tf.meshgrid(ind_kti,ind_kti)
    Kt_big = tf.gather_nd(K_tt,tf.stack((grid_t[0],grid_t[1]),-1))

    Kf_Ktt = tf.multiply(Kf_big,Kt_big)

    DI_big = tf.gather_nd(D,tf.stack((grid_f[0],grid_f[1]),-1))
    DI = tf.diag(tf.diag_part(DI_big)) #D kron I

    #data covariance.
    #Either need to take Cholesky of this or use CG / block CG for matrix-vector products
    Ky = Kf_Ktt + DI + 1e-6*tf.eye(ny)

    ### build out cross-covariances and covariance at grid

    nx = tf.shape(Xi)[0]

    K_xx = OU_kernel(length,Xi,Xi)
    K_xt = OU_kernel(length,Xi,Ti)

    ind = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)
    grid = tf.meshgrid(ind,ind)
    Kf_big = tf.gather_nd(Kf,tf.stack((grid[0],grid[1]),-1))
    ind2 = tf.tile(tf.range(nx),[M])
    grid2 = tf.meshgrid(ind2,ind2)
    Kxx_big =  tf.gather_nd(K_xx,tf.stack((grid2[0],grid2[1]),-1))

    K_ff = tf.multiply(Kf_big,Kxx_big) #cov at grid points

    full_f = tf.concat([tf.tile([i],[nx]) for i in range(M)],0)
    grid_1 = tf.meshgrid(full_f,ind_kfi,indexing='ij')
    Kf_big = tf.gather_nd(Kf,tf.stack((grid_1[0],grid_1[1]),-1))
    full_x = tf.tile(tf.range(nx),[M])
    grid_2 = tf.meshgrid(full_x,ind_kti,indexing='ij')
    Kxt_big = tf.gather_nd(K_xt,tf.stack((grid_2[0],grid_2[1]),-1))

    K_fy = tf.multiply(Kf_big,Kxt_big)

    #now get draws!
    y_ = tf.reshape(Yi,[-1,1])
    Ly = tf.cholesky(Ky)
    #Mu = tf.matmul(K_fy,CG(Ky,y_)) #May be faster with CG for large problems
    #printop = tf.Print(y_,[y_,Ly],"The y is :",-1,1000)
    #with tf.control_dependencies([printop]):
    xi = tf.random_normal((nx*M,n_mc_smps))
    Mu = tf.matmul(K_fy,tf.cholesky_solve(Ly,y_))

    #TODO: it's worth testing to see at what point computation speedup of Lanczos algorithm is useful & needed.
    # For smaller examples, using Cholesky will probably be faster than this unoptimized Lanczos implementation.
    # Likewise for CG and BCG vs just taking the Cholesky of Ky once
    """
    #Never need to explicitly compute Sigma! Just need matrix products with Sigma in Lanczos algorithm
    def Sigma_mul(vec):
        # vec must be a 2d tensor, shape (?,?)
        return tf.matmul(K_ff,vec) - tf.matmul(K_fy,block_CG(Ky,tf.matmul(tf.transpose(K_fy),vec)))

    def small_draw():
        return Mu + tf.matmul(tf.cholesky(Sigma),xi)
    def large_draw():
        return Mu + block_Lanczos(Sigma_mul,xi,n_mc_smps) #no need to explicitly reshape Mu

    BLOCK_LANC_THRESH = 1000
    draw = tf.cond(tf.less(nx*M,BLOCK_LANC_THRESH),small_draw,large_draw)
    """

    Sigma = K_ff - tf.matmul(K_fy,tf.cholesky_solve(Ly,tf.transpose(K_fy))) + 1e-6*tf.eye(tf.shape(K_ff)[0])
    draw = Mu + tf.matmul(tf.cholesky(Sigma),xi)
    draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw),[n_mc_smps,M,nx]),perm=[0,2,1])
    return draw_reshape

def get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                   num_rnn_grid_times,med_cov_grid):
    """
    returns samples from GP at evenly-spaced gridpoints
    """
    grid_max = tf.shape(X)[1]
    Z = tf.zeros([0,grid_max,input_dim])
    #printop = tf.Print(ind_kt, [ind_kt, num_obs_values],"The ind kt is:",-1, 15)
    #with tf.control_dependencies([printop]):
    N = tf.shape(T)[0] #number of observations

    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,Z):
        return i<N

    def body(i,Z):
        Yi = tf.reshape(tf.slice(Y,[i,0],[1,num_obs_values[i]]),[-1])
        Ti = tf.reshape(tf.slice(T,[i,0],[1,num_obs_times[i]]),[-1])
        #printop = tf.Print(i, [i,Yi,tf.shape(waveform_outputs)], "current i is:",-1,1000)
        #with tf.control_dependencies([printop]):
        ind_kfi = tf.reshape(tf.slice(ind_kf,[i,0],[1,num_obs_values[i]]),[-1])
        ind_kti = tf.reshape(tf.slice(ind_kt,[i,0],[1,num_obs_values[i]]),[-1])
        Xi = tf.reshape(tf.slice(X,[i,0],[1,num_rnn_grid_times[i]]),[-1])
        X_len = num_rnn_grid_times[i]

        GP_draws = draw_GP(Yi,Ti,Xi,ind_kfi,ind_kti)
        pad_len = grid_max-X_len #pad by this much
        with tf.name_scope('GP_samp_concats'):
            padded_GP_draws = tf.concat([GP_draws,tf.zeros((n_mc_smps,pad_len,M))],1)

            medcovs = tf.slice(med_cov_grid,[i,0,0],[1,-1,-1])
            tiled_medcovs = tf.tile(medcovs,[n_mc_smps,1,1])
            padded_GPdraws_medcovs = tf.concat([padded_GP_draws,tiled_medcovs],2)
            waves = tf.slice(waveform_outputs,[i,0,0],[1,-1,-1])
            tiled_waves = tf.tile(waves,[n_mc_smps,1,1])
            #printop = tf.Print(tiled_waves,[tf.shape(padded_GPdraws_medcovs), tf.shape(tiled_waves)],"medcovs and waveform shape:",-1,100)
            #with tf.control_dependencies([printop]):
            padded_GPdraws_medcovs_waves = tf.concat([padded_GPdraws_medcovs,tiled_waves],2)
            Z = tf.concat([Z,padded_GPdraws_medcovs_waves],0)

        return i+1,Z

    i = tf.constant(0)
    i,Z = tf.while_loop(cond,body,loop_vars=[i,Z],
                shape_invariants=[i.get_shape(),tf.TensorShape([None,None,None])])

    return Z

def get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
              num_rnn_grid_times,med_cov_grid):
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation,
    then feeds it all through the LSTM to get predictions.

    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_rnn_grid_times: length of even spaced RNN grid per encounter

    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """

    Z = get_GP_samples(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,
                       num_rnn_grid_times,med_cov_grid) #batchsize*num_MC x batch_maxseqlen x num_inputs
    Z.set_shape([None,None,input_dim]) #somehow lost shape info, but need this
    #wshape = Z.get_shape()
    #wshape.set_shape([None,None,1])
    #x = tf.concat([Z, waveform_outputs],2)
    N = tf.shape(T)[0] #number of observations
    #duplicate each entry of seqlens, to account for multiple MC samples per observation
    seqlen_dupe = tf.reshape(tf.tile(tf.expand_dims(num_rnn_grid_times,1),[1,n_mc_smps]),[N*n_mc_smps])

    #with tf.variable_scope("",reuse=True):
    outputs, states = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=Z,
                                            dtype=tf.float32,
                                            sequence_length=seqlen_dupe)

    final_outputs = states[n_layers-1][1]
    preds =  tf.matmul(final_outputs, out_weights) + out_biases
    return preds

def get_probs_and_accuracy(preds,O):
    """
    helper function. we have a prediction for each MC sample of each observation
    in this batch.  need to distill the multiple preds from each MC into a single
    pred for this observation.  also get accuracy. use true probs to get ROC, PR curves in sklearn
    """
    all_probs = tf.exp(preds[:,1] - tf.reduce_logsumexp(preds, axis = 1)) #normalize; and drop a dim so only prob of positive case
    N = tf.cast(tf.shape(preds)[0]/n_mc_smps,tf.int32) #actual number of observations in preds, collapsing MC samples

    #predicted probability per observation; collapse the MC samples
    probs = tf.zeros([0]) #store all samples in a list, then concat into tensor at end
    #setup tf while loop (have to use this bc loop size is variable)
    def cond(i,probs):
        return i < N
    def body(i,probs):
        probs = tf.concat([probs,[tf.reduce_mean(tf.slice(all_probs,[i*n_mc_smps],[n_mc_smps]))]],0)
        return i+1,probs
    i = tf.constant(0)
    i,probs = tf.while_loop(cond,body,loop_vars=[i,probs],shape_invariants=[i.get_shape(),tf.TensorShape([None])])

    #compare to truth; just use cutoff of 0.5 for right now to get accuracy
    correct_pred = tf.equal(tf.cast(tf.greater(probs,0.5),tf.int32), O)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return probs,accuracy

def vectorize(l,ind):
    return [l[i] for i in ind]

flags = tf.app.flags
#params obtained after hyperparameter optimization
flags.DEFINE_float("lr",0.00057,"")
flags.DEFINE_float("l2_penalty",1e-3,"")
flags.DEFINE_float("n_layers",2.0,"")
flags.DEFINE_float("epochs",140.0,"")
flags.DEFINE_float("batch",30.0,"")
flags.DEFINE_float("beta1",0.9,"")
flags.DEFINE_float("beta2",0.999,"")
flags.DEFINE_float("epsilon",1e-8,"")
'''
flags.DEFINE_float("lr",0.001,"")
flags.DEFINE_float("l2_penalty",1e-3,"")
flags.DEFINE_float("epochs",55.0,"")
flags.DEFINE_float("batch",30.0,"")
flags.DEFINE_float("n_layers",3.0,"")
#flags.DEFINE_float("wave_layers",2.0,"")
'''
FLAGS=flags.FLAGS

if __name__ == "__main__":
    seed = 8675309
    rs = np.random.RandomState(seed) #fixed seed in np

    parser = argparse.ArgumentParser()
    #parser.add_argument('high', type=str, help='high frequency or low only. type high/low')
    parser.add_argument('sim', type=str, help='prevsim/sim/data/prevdata')
    parser.add_argument('mode', type=str, help='trainonly/trainval/test')
    parser.add_argument('fold',type=int,help='fold number')
    parser.add_argument('-lr',type=float, help='lr value')
    parser.add_argument("-l2_penalty",type=float)
    parser.add_argument("-epochs",type=int)
    parser.add_argument("-batch",type=float)
    parser.add_argument("-n_layers",type=float)
    parser.add_argument("-beta1",type=float)
    parser.add_argument("-beta2",type=float)
    parser.add_argument("-epsilon",type=float)
    parser.add_argument("-tensorboard",type=str,default="tensorboard_icis2019/")
    #parser.add_argument("-wave_layers",type=float)
    parser.add_argument('--debug', dest='debug')
    args = parser.parse_args()
    mode = args.mode
    #####
    ##### Setup ground truth and sim some data from a GP
    #####
    #if args.sim=="sim":
    #    num_encs=50#100#10000
    #    M=10#17#10
    #    n_covs=3#10
    #    n_meds=10#2938#5
    #    (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,times,
    #    values,ind_lvs,ind_times,meds_on_grid,covs,high_freq) = sim_dataset(num_encs,M,n_covs,n_meds)#retrieve_sim_dataset
    if args.sim=="data":
        #'''
        if mode=="trainonly":
            (num_obs_times_tr,num_obs_values_tr,num_rnn_grid_times_tr,rnn_grid_times_tr,labels_tr,times_tr,
            values_tr,ind_lvs_tr,ind_times_tr,meds_on_grid_tr,covs_tr, H_tr) = prep_mimic('train',args.fold)
            num_enc = len(num_obs_times_tr)
        elif mode=="trainval":
            (num_obs_times_tr,num_obs_values_tr,num_rnn_grid_times_tr,rnn_grid_times_tr,labels_tr,times_tr,
            values_tr,ind_lvs_tr,ind_times_tr,meds_on_grid_tr,covs_tr, H_tr) = prep_mimic('train',args.fold)
            (num_obs_times_te,num_obs_values_te,num_rnn_grid_times_te,rnn_grid_times_te,labels_te,times_te,
            values_te,ind_lvs_te,ind_times_te,meds_on_grid_te,covs_te, H_te) = prep_mimic('test',args.fold)
            num_enc = len(num_obs_times_tr)
        else:
            (num_obs_times_te,num_obs_values_te,num_rnn_grid_times_te,rnn_grid_times_te,labels_te,times_te,
            values_te,ind_lvs_te,ind_times_te,meds_on_grid_te,covs_te, H_te) = prep_mimic('test',args.fold)
        '''
        (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,times,
        values,ind_lvs,ind_times,meds_on_grid,covs, high_freq) = prep_mimic('train',args.fold)
        '''
        M = 21
        n_meds = 5
        n_covs = 9
    else:
        (num_obs_times,num_obs_values,num_rnn_grid_times,rnn_grid_times,labels,times,
        values,ind_lvs,ind_times,meds_on_grid,covs, high_freq) = prep_baseline_mgp()
        num_enc = len(num_obs_times)
        M = 25
        n_meds = 5
        n_covs = 9

    #N_tot = len(labels) #total encounters
    if args.lr:
        FLAGS.lr = args.lr
    if args.l2_penalty:
        FLAGS.l2_penalty = args.l2_penalty
    if args.epochs:
        FLAGS.epochs = args.epochs
    if args.batch:
        FLAGS.batch = args.batch
    if args.n_layers:
        FLAGS.n_layers = args.n_layers
    if args.beta1:
        FLAGS.beta1 = args.beta1
    if args.beta2:
        FLAGS.beta2 = args.beta2
    if args.epsilon:
        FLAGS.epsilon = args.epsilon
    #if args.wave_layers:
    #    FLAGS.wave_layers = args.wave_layers
    '''
    train_test_perm = rs.permutation(N_tot)
    #train_test_perm = range(N_tot)
    #val_frac = 0.1 #fraction of full data to set aside for testing
    val_size = 150
    te_ind = train_test_perm[:val_size]
    labels_te = [labels[i] for i in te_ind]
    while (0 not in labels_te) or (1 not in labels_te):
        train_test_perm = rs.permutation(N_tot)
        te_ind = train_test_perm[:val_size]
        labels_te = [labels[i] for i in te_ind]
    tr_ind = train_test_perm[val_size:]
    labels_tr = [labels[i] for i in tr_ind]
    Nte = len(te_ind); Ntr = len(tr_ind)

    print("Total samples:%s, Train/tes:%s/%s"%(N_tot,Ntr,Nte))
    #print tr_ind
    #print te_ind
    #Break everything out into train/test
    covs_tr = [covs[i] for i in tr_ind]; covs_te = [covs[i] for i in te_ind]
    labels_tr = [labels[i] for i in tr_ind]; labels_te = [labels[i] for i in te_ind]
    times_tr = [times[i] for i in tr_ind]; times_te = [times[i] for i in te_ind]
    values_tr = [values[i] for i in tr_ind]; values_te = [values[i] for i in te_ind]
    ind_lvs_tr = [ind_lvs[i] for i in tr_ind]; ind_lvs_te = [ind_lvs[i] for i in te_ind]
    ind_times_tr = [ind_times[i] for i in tr_ind]; ind_times_te = [ind_times[i] for i in te_ind]
    meds_on_grid_tr = [meds_on_grid[i] for i in tr_ind]; meds_on_grid_te = [meds_on_grid[i] for i in te_ind]
    num_obs_times_tr = [num_obs_times[i] for i in tr_ind]; num_obs_times_te = [num_obs_times[i] for i in te_ind]
    num_obs_values_tr = [num_obs_values[i] for i in tr_ind]; num_obs_values_te = [num_obs_values[i] for i in te_ind]
    rnn_grid_times_tr = [rnn_grid_times[i] for i in tr_ind]; rnn_grid_times_te = [rnn_grid_times[i] for i in te_ind]
    num_rnn_grid_times_tr = [num_rnn_grid_times[i] for i in tr_ind]; num_rnn_grid_times_te = [num_rnn_grid_times[i] for i in te_ind]
    H_tr = [high_freq[i] for i in tr_ind]; H_te = [high_freq[i] for i in te_ind]
    #'''
    if mode=="trainonly":
        Ntr = len(covs_tr)
    elif mode=="trainval":
        Ntr = len(covs_tr)
        Nte = len(covs_te)
    else:
        Nte = len(covs_te)
    #print("Train/test split : %s-%s"%(Ntr,Nte))
    print("data fully setup!")
    #print("test labels are:%s"%labels_te)
    sys.stdout.flush()

    #####
    ##### Setup model and graph
    #####

    # Learning Parameters
    learning_rate = FLAGS.lr #0.0001#NOTE may need to play around with this or decay it
    L2_penalty = FLAGS.l2_penalty #NOTE may need to play around with this some or try additional regularization
    #TODO: add dropout regularization
    training_iters = int(FLAGS.epochs) #num epochs
    batch_size = int(FLAGS.batch) #NOTE may want to play around with this
    if mode=="trainval" or mode=="trainonly":
        test_freq = Ntr/batch_size #eval on test set after this many batches

    # Network Parameters
    waveform_dim = 10
    n_hidden = 20 # hidden layer num of features; assumed same
    #wave_layers = int(FLAGS.wave_layers)
    n_layers = int(FLAGS.n_layers) # number of layers of stacked LSTMs
    n_classes = 2 #binary outcome
    input_dim = M+n_meds+n_covs+waveform_dim #dimensionality of input sequence.
    n_mc_smps = 25

    # Create graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    ops.reset_default_graph()
    sess = tf.Session(config=config)
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    if mode=="test" or mode=="trainval":
        T_pad_te,Y_pad_te,ind_kf_pad_te,ind_kt_pad_te,X_pad_te,meds_cov_pad_te, H_pad_te = pad_data(
            times_te,values_te,ind_lvs_te,ind_times_te,rnn_grid_times_te,meds_on_grid_te,covs_te,H_te)

    ##### tf Graph - inputs
    #observed values, times, inducing times; padded to longest in the batch
    Y = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
    T = tf.placeholder("float", [None,None]) #batchsize x batch_maxdata_length
    ind_kf = tf.placeholder(tf.int32, [None,None]) #index tasks in Y vector
    ind_kt = tf.placeholder(tf.int32, [None,None]) #index inputs in Y vector
    X = tf.placeholder("float", [None,None]) #grid points. batchsize x batch_maxgridlen
    med_cov_grid = tf.placeholder("float", [None,None,n_meds+n_covs]) #combine w GP smps to feed into RNN

    O = tf.placeholder(tf.int32, [None]) #labels. input is NOT as one-hot encoding; convert at each iter
    num_obs_times = tf.placeholder(tf.int32, [None]) #number of observation times per encounter
    num_obs_values = tf.placeholder(tf.int32, [None]) #number of observation values per encounter
    num_rnn_grid_times = tf.placeholder(tf.int32, [None]) #length of each grid to be fed into RNN in batch

    waveform = tf.placeholder("float",[None,None,3600*2])

    N = tf.shape(Y)[0]

    #also make O one-hot encoding, for the loss function
    O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(O,1),[1,n_mc_smps]),[N*n_mc_smps]),n_classes)

    ##### tf Graph - variables to learn

    ### GP parameters (unconstrained)

    #in fully separable case all labs share same time-covariance
    log_length = tf.Variable(tf.random_normal([1],mean=1,stddev=0.1),name="GP-log-length")
    length = tf.exp(log_length)

    #different noise level of each lab
    log_noises = tf.Variable(tf.random_normal([M],mean=-2,stddev=0.1),name="GP-log-noises")
    noises = tf.exp(log_noises)

    #init cov between labs
    L_f_init = tf.Variable(tf.eye(M),name="GP-Lf")
    Lf = tf.matrix_band_part(L_f_init,-1,0)
    Kf = tf.matmul(Lf,tf.transpose(Lf))

    ### RNN params

    # Create network
    #x = tf.keras.layers.Embedding(output_dim=512, input_dim=10000, input_length=100)(waveform)
    #auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

    with tf.name_scope('waveform_rnn'):
        with tf.variable_scope('waveform_rnn'):
            waveform_cell = tf.contrib.rnn.LSTMCell(waveform_dim)
            #stacked_waveform_cell = tf.contrib.rnn.MultiRNNCell([waveform_cell for _ in range(wave_layers)])
            #initial_state = waveform_cell.zero_state(batch_size, dtype=tf.float32)
            #waveform_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(waveform_dim, input_shape=(None,3600,2), output_shape=(None,waveform_dim)))(waveform)
            waveform_outputs, s = tf.nn.dynamic_rnn(cell=waveform_cell, inputs=waveform, dtype=tf.float32)
            waveform_outputs = tf.reshape(waveform_outputs,[batch_size,-1,waveform_dim])
    #wshape = waveform_outputs.get_shape().as_list()
    #waveform_outputs = tf.reshape(waveform_outputs, [-1,wshape[1],1])
    #print waveform_outputs
    #auxiliary_out = tf.layers.dense(waveform_outputs, 1, name='aux_output')
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])

    # Weights at the last layer given deep LSTM output
    out_weights = tf.Variable(tf.random_normal([n_hidden, n_classes],stddev=0.1),name="Softmax/W")
    out_biases = tf.Variable(tf.random_normal([n_classes],stddev=0.1),name="Softmax/b")

    ##### Get predictions and feed into optimization
    preds = get_preds(Y,T,X,ind_kf,ind_kt,num_obs_times,num_obs_values,num_rnn_grid_times,med_cov_grid)
    probs,accuracy = get_probs_and_accuracy(preds,O)
    tf.summary.scalar('accuracy',accuracy)
    # Define optimization problem
    loss_fit = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=O_dupe_onehot))
    #loss_wave = tf.reduce_sum(tf.square(tf.get_variable('aux_output')))
    with tf.variable_scope("",reuse=True):
        loss_reg = L2_penalty*tf.reduce_sum(tf.square(out_weights))
        for i in range(n_layers):
            loss_reg = L2_penalty+tf.reduce_sum(tf.square(tf.get_variable('rnn/multi_rnn_cell/cell_'+str(i)+'/basic_lstm_cell/kernel')))
    loss = loss_fit + loss_reg #+ loss_wave
    train_op = tf.train.AdamOptimizer(learning_rate,FLAGS.beta1,FLAGS.beta2,FLAGS.epsilon).minimize(loss)
    tf.summary.scalar('loss',loss)

    #Create a visualizer object
    merged = tf.summary.merge_all()
    ##### Initialize globals and get ready to start!
    sess.run(tf.global_variables_initializer())
    print("Graph setup!")
    saver = tf.train.Saver(max_to_keep = 30)
    #Initializing the saver
    tb_dir = args.tensorboard
    train_writer = tf.summary.FileWriter(tb_dir+'mts_kalman/train/'+str(args.fold),sess.graph)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    test_writer = tf.summary.FileWriter(tb_dir+'mts_kalman/test/'+str(args.fold))


    if mode!="test":
        #setup minibatch indices
        #print Ntr
        #print batch_size
        starts = np.arange(0,Ntr,batch_size)
        ends = np.arange(batch_size,Ntr+1,batch_size)
        #print ends
        if ends[-1]<Ntr:
            ends = np.append(ends,int(ceil((Ntr*1.0)/batch_size))*batch_size)
        num_batches = len(ends)


        ##### Main training loop
        checkpoint_freq = 5*num_batches
        epoch_loss = 0.0
        total_batches = 0
        i = 0
        thresh = 500
        #for i in range(training_iters):
        while epoch_loss>=30.0 or total_batches==0 or i<=thresh:
            #train
            metric_opt = 0.0 #the metric which is given to hyperparam opt. Here the te_auc will be passed
            epoch_start = time()
            print("Starting epoch "+"{:d}".format(i))
            epoch_loss = 0.0
            perm = rs.permutation(Ntr)
            if Ntr%batch_size!=0:
                rem = perm[:batch_size-(Ntr%batch_size)]
                perm = np.append(perm,rem)
                print(len(perm))
            batch = 0
            for s,e in zip(starts,ends):
                batch_start = time()
                inds = perm[s:e]
                #print(inds)
                T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad, H_pad = pad_data(
                        vectorize(times_tr,inds),vectorize(values_tr,inds),vectorize(ind_lvs_tr,inds),vectorize(ind_times_tr,inds),
                        vectorize(rnn_grid_times_tr,inds),vectorize(meds_on_grid_tr,inds),vectorize(covs_tr,inds),vectorize(H_tr,inds))
                #print H_pad.shape
                #print T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad
                feed_dict={waveform: H_pad, Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad,
                med_cov_grid:meds_cov_pad,num_obs_times:vectorize(num_obs_times_tr,inds),
                num_obs_values:vectorize(num_obs_values_tr,inds),
                num_rnn_grid_times:vectorize(num_rnn_grid_times_tr,inds),O:vectorize(labels_tr,inds)}
                #summary,loss_,_ = sess.run([merged,loss,train_op],feed_dict)
                #'''
                try:
                    summary,loss_,_ = sess.run([merged,loss,train_op],feed_dict)
                    epoch_loss += loss_
                    train_writer.add_summary(summary,i)
                except:
                    print("problem in "+str(batch))
                    batch+=1; total_batches+=1
                    continue
                #'''
                #print("Batch "+"{:d}".format(batch)+"/"+"{:d}".format(num_batches)+\
                #    ", took: "+"{:.3f}".format(time()-batch_start)+", loss: "+"{:.5f}".format(loss_))
                sys.stdout.flush()
                batch += 1; total_batches += 1

                correct_proc = True
                if mode=="trainval":
                    if total_batches % num_batches == 0: #Check val set every so often for early stopping
                        #TODO: may also want to check validation performance at additional X hours back
                        #from the event time, as well as just checking performance at terminal time
                        #on the val set, so you know if it generalizes well further back in time as well
                        #print("Testing")
                        epoch_loss = epoch_loss/num_batches
                        print("The average loss in the epoch is:%s"%epoch_loss)
                        #epoch_loss = 0.0
                        test_t = time()
                        acc = 0.0
                        auc = 0.0
                        prc = 0.0
                        te_starts = np.arange(0,Nte,batch_size)
                        te_ends = np.arange(batch_size,Nte+1,batch_size)
                        te_perm = np.arange(Nte)
                        no_to_pad = batch_size-(Nte%batch_size)
                        #print ends
                        if te_ends[-1]<Nte:
                            te_ends = np.append(te_ends,int(ceil((Nte*1.0)/batch_size))*batch_size)
                            te_rem = te_perm[:no_to_pad]
                            te_perm = np.append(te_perm,te_rem)
                        #print(ends)
                        #num_batches = len(ends)
                        no_iters = int(ceil((Nte*1.0)/batch_size))
                        start_i = 0
                        pred_probs = []
                        #print(perm)
                        batch = 0
                        for ts,te in zip(te_starts,te_ends):
                            batch_start = time()
                            inds = te_perm[ts:te]
                            #print(inds)
                            T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad, H_pad = pad_data(
                                    vectorize(times_te,inds),vectorize(values_te,inds),vectorize(ind_lvs_te,inds),vectorize(ind_times_te,inds),
                                    vectorize(rnn_grid_times_te,inds),vectorize(meds_on_grid_te,inds),vectorize(covs_te,inds),vectorize(H_te,inds))
                            #print H_pad.shape
                            feed_dict={waveform: H_pad, Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad,
                            med_cov_grid:meds_cov_pad,num_obs_times:vectorize(num_obs_times_te,inds),
                            num_obs_values:vectorize(num_obs_values_te,inds),
                            num_rnn_grid_times:vectorize(num_rnn_grid_times_te,inds),O:vectorize(labels_te,inds)}
                            #summary,loss_,_ = sess.run([merged,loss,train_op],feed_dict)
                            try:
                                summary,te_probs,te_acc,te_loss = sess.run([merged,probs,accuracy,loss],feed_dict)
                                test_writer.add_summary(summary,i)
                                #print "Te probs:"+str(te_probs)
                                pred_probs.extend(te_probs)
                                acc += te_acc
                                #auc += te_auc
                                #prc += te_prc
                            except:
                                correct_proc = False
                        if correct_proc:
                            acc = acc/no_iters
                            te_auc = roc_auc_score(labels_te, pred_probs[:len(labels_te)])
                            te_prc = average_precision_score(labels_te, pred_probs[:len(labels_te)])
                            print("Epoch "+str(i)+", seen "+str(total_batches)+" total batches. Testing Took "+\
                                "{:.2f}".format(time()-test_t)+\
                                ". OOS, "+str(0)+" hours back: Loss: "+"{:.5f}".format(te_loss)+ \
                                " Acc: "+"{:.5f}".format(acc)+", AUC: "+ \
                                "{:.5f}".format(te_auc)+", AUPR: "+"{:.5f}".format(te_prc))
                            sys.stdout.flush()
                            #print(te_auc)
                            metric_opt = te_auc
                        else:
                            #print("0")
                            metric_opt = 0
                            epoch_loss = 0.0

                    #create a folder and put model checkpoints there
            if mode=="trainonly":
                epoch_loss= epoch_loss/num_batches
            #if total_batches%checkpoint_freq==0:
            saver.save(sess, "/data/suparna/icis2019/HIERARCHICAL_MGP_noglascow_kalman/"+str(args.fold)+"/", global_step=total_batches)
            print("Finishing epoch "+"{:d}".format(i)+", took "+\
                  "{:.3f}".format(time()-epoch_start)+" with loss:"+\
                  "{:.3f}".format(epoch_loss))
            print(metric_opt)
            i += 1

            ### Takes about ~1-2 secs per batch of 50 at these settings, so a few minutes each epoch
            ### Should converge reasonably quickly on this toy example with these settings in a few epochs

    if mode=="test":
        ckpt_dir ="/data/suparna/icis2019/HIERARCHICAL_MGP_noglascow_kalman/"+str(args.fold)+"/"
        ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
        saver.restore(sess,ckpt_state.model_checkpoint_path)
        print("Model restored")
        test_t = time()
        acc = 0.0
        auc = 0.0
        prc = 0.0
        starts = np.arange(0,Nte,batch_size)
        ends = np.arange(batch_size,Nte+1,batch_size)
        perm = np.arange(Nte)
        no_to_pad = batch_size-(Nte%batch_size)
        #print(no_to_pad)
        #print ends
        if ends[-1]<Nte:
            ends = np.append(ends,int(ceil((Nte*1.0)/batch_size))*batch_size)
            rem = perm[:no_to_pad]
            perm = np.append(perm,rem)
        #print(ends)
        num_batches = len(ends)
        no_iters = int(ceil((Nte*1.0)/batch_size))
        start_i = 0
        pred_probs = []
        #print(perm)
        batch = 0
        for s,e in zip(starts,ends):
            batch_start = time()
            inds = perm[s:e]
            #print(inds)
            T_pad,Y_pad,ind_kf_pad,ind_kt_pad,X_pad,meds_cov_pad, H_pad = pad_data(
                    vectorize(times_te,inds),vectorize(values_te,inds),vectorize(ind_lvs_te,inds),vectorize(ind_times_te,inds),
                    vectorize(rnn_grid_times_te,inds),vectorize(meds_on_grid_te,inds),vectorize(covs_te,inds),vectorize(H_te,inds))
            #print H_pad.shape
            """
        for j in range(no_iters):
            end_i = start_i+batch_size
            H_pad_bte = H_pad_te[start_i:end_i]
            Y_pad_bte = Y_pad_te[start_i:end_i]
            T_pad_bte = T_pad_te[start_i:end_i]
            ind_kf_pad_bte = ind_kf_pad_te[start_i:end_i]
            ind_kt_pad_bte = ind_kt_pad_te[start_i:end_i]
            X_pad_bte = X_pad_te[start_i:end_i]
            meds_cov_pad_bte = meds_cov_pad_te[start_i:end_i]
            num_obs_times_bte = num_obs_times_te[start_i:end_i]
            num_obs_values_bte = num_obs_values_te[start_i:end_i]
            num_rnn_grid_times_bte = num_rnn_grid_times_te[start_i:end_i]
            labels_bte = labels_te[start_i:end_i]
            if end_i>Nte:
                H_pad_bte += H_pad_bte[:no_to_pad]
            feed_dict={waveform:H_pad, Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad,
            med_cov_grid:meds_cov_pad,num_obs_times:num_obs_times,
            num_obs_values:num_obs_values,num_rnn_grid_times:num_rnn_grid_times,O:labels_te}
            """
            feed_dict={waveform: H_pad, Y:Y_pad,T:T_pad,ind_kf:ind_kf_pad,ind_kt:ind_kt_pad,X:X_pad,
            med_cov_grid:meds_cov_pad,num_obs_times:vectorize(num_obs_times_te,inds),
            num_obs_values:vectorize(num_obs_values_te,inds),
            num_rnn_grid_times:vectorize(num_rnn_grid_times_te,inds),O:vectorize(labels_te,inds)}
            #summary,loss_,_ = sess.run([merged,loss,train_op],feed_dict)
            summary,te_probs,te_acc,te_loss = sess.run([merged,probs,accuracy,loss],feed_dict)
            test_writer.add_summary(summary,i)
            #print "Te probs:"+str(te_probs)
            pred_probs.extend(te_probs)
            acc += te_acc
            #start_i = end_i
        print("AUROC:%s"%(roc_auc_score(labels_te, pred_probs[:Nte])))
        print("AUPR:%s"%(average_precision_score(labels_te, pred_probs[:Nte])))
        pickle.dump(labels_te,open('icis_revision/hierarchical_noglascow_kalman_targ_fold'+str(args.fold)+'.pickle','wb'))
        pickle.dump(pred_probs, open('icis_revision/hierarchical_noglascow_kalman_predprobs_fold'+str(args.fold)+'.pickle','wb'))
