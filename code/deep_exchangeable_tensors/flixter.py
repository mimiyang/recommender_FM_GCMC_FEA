from __future__ import print_function
# Standard lib imports
from builtins import range
import sys
import gc
import glob
import math
import time
import logging
from collections import OrderedDict
# Helper lib imports
from tqdm import tqdm
# Math imports
import tensorflow as tf
from scipy.sparse import csr_matrix
# Model imports
from base import Model
from util import get_data
from sparse_util import *
from dataset_transfer import load_matlab_file, eval_dataset

def sample_submatrix(mask_,#mask, used for getting concentrations
                     maxN, maxM,
                     sample_uniform=False):
    '''
    sampling mini-batches. Currently it is sampling rows and columns based on the number of non-zeros.
    In the sparse implementation, we could sample non-zero entries and directly.
    '''
    pN, pM = mask_.sum(axis=1)[:,0], mask_.sum(axis=0)[:,0]
    pN /= pN.sum()#non-zero dist per row
    pM /= pM.sum()#non-zero dist per column
    
    N, M, _ = mask_.shape
    for n in range(N // maxN):
        for m in range(M // maxM):
            if N == maxN:
                ind_n = np.arange(N)
            else:
                ind_n = np.random.choice(N, size=maxN, replace=False, p = pN)#select a row based on density of nonzeros
            if M == maxM:
                ind_m = np.arange(M)
            else:
                ind_m = np.random.choice(M, size=maxM, replace=False, p = pM)
            yield ind_n, ind_m 


def sample_dense_values_uniform(mask_indices, minibatch_size, iters_per_epoch):
    num_vals = mask_indices.shape[0]
    minibatch_size = np.minimum(minibatch_size, num_vals)
    for n in range(iters_per_epoch):
        sample = np.random.choice(num_vals, size=minibatch_size, replace=False)
        yield sample

def sample_dense_values_uniform_val(mask_indices, mask_tr_val_split, minibatch_size, iters_per_epoch):
    num_vals_tr = mask_indices[mask_tr_val_split == 0].shape[0]
    num_vals_val = mask_indices[mask_tr_val_split == 1].shape[0]
    minibatch_size_tr = np.minimum(int(.9 * minibatch_size), num_vals_tr)
    minibatch_size_val = np.minimum(int(.1 * minibatch_size), num_vals_val)
    for n in range(iters_per_epoch):
        sample_tr = np.random.choice(num_vals_tr, size=minibatch_size_tr, replace=False)
        sample_val = np.random.choice(num_vals_val, size=minibatch_size_val, replace=False)
        yield sample_tr, sample_val

def reindex_mask(mask):
    index = np.zeros_like(mask)
    for c in [0, 1]:
        _, index[:, c] = np.unique(mask[:,c], return_inverse=True)
    return index

def re_mask(*args):
    lengths = [m.shape[0] for m in args]
    merged = np.concatenate(args, axis=0)
    ids = reindex_mask(merged)
    mask_ids = []
    k = 0
    for l in lengths:
        mask_ids.append((k, k+l))
        k += l
    return [ids[i:j, :] for i, j in mask_ids]

def conditional_sample_sparse(mask_indices, tr_val_split, shape, maxN, maxM): # AKA Kevin sampling 
    N,M,_ = shape
    num_vals = mask_indices.shape[0]

    for n in range(N // maxN):
        for m in range(M // maxM):

            pN = np.bincount(mask_indices[:,0], minlength=N).astype(np.float32)
            pN /= pN.sum()
            ind_n = np.arange(N)[pN!=0] # If there are 0s in p and replace is False, we cant select N=maxN unique values. Filter out 0s.
            pN = pN[pN!=0]
            maxN = min(maxN, ind_n.shape[0])
            ind_n = np.random.choice(ind_n, size=maxN, replace=False, p=pN)

            select_row = np.in1d(mask_indices[:,0], ind_n)
            rows = mask_indices[select_row==True]

            pM = np.bincount(rows[:,1], minlength=M).astype(np.float32)
            pM /= pM.sum()
            ind_m = np.arange(M)[pM!=0] # If there are 0s in p and replace is False, we cant select M=maxM unique values. Filter out 0s.
            pM = pM[pM!=0] 
            maxM = min(maxM, ind_m.shape[0])
            ind_m = np.random.choice(ind_m, size=maxM, replace=False, p=pM)

            select_col = np.in1d(mask_indices[:,1], ind_m)
            select_row_col = np.logical_and(select_row, select_col)

            inds_all = np.arange(num_vals)[select_row_col==True]
            
            split = tr_val_split[inds_all]
            
            inds_tr = inds_all[split==0]
            inds_val = inds_all[split==1]
            inds_tr_val = inds_all[split<=1]
            inds_ts = inds_all[split==2]

            yield inds_tr, inds_val, inds_tr_val, inds_ts, inds_all

def get_neighbours(seed_set, full_set, axis=None):
    if axis is None:
        axis = np.random.randint(2)
    in_neighbourhood = np.isin(full_set[:,axis], seed_set[:, axis])
    return full_set[in_neighbourhood, :]

def sample_neighbours(seed_set, full_set, n=None):
    out = np.zeros((0,2), dtype="int")
    for axis in [0, 1]:
        neighbours = get_neighbours(seed_set, full_set, axis)
        if n is None:
            n = neighbours.shape[0]
        if n > neighbours.shape[0]:
            out = np.concatenate([out, neighbours], axis=0)
        else:
            idx = np.random.choice(neighbours.shape[0], n, replace=False)
            out = np.concatenate([out, neighbours[idx, :]], axis=0)
    return out

def sample_k_neighbours(seed_set, full_set, hops=4, n=None):
    if n is not None:
        n /= 2
    out = np.zeros((0,2), dtype="int")
    extra_nodes = seed_set
    for hop in range(hops):
        extra_nodes = sample_neighbours(extra_nodes, full_set, n)
        out = np.concatenate([out, extra_nodes], axis=0)
    return out

def neighbourhood_sampling(mask_indices, minibatch_size, iters_per_epoch, hops=4):
    num_vals = mask_indices.shape[0]
    minibatch_size = np.minimum(minibatch_size, num_vals) // (1+hops)
    
    for n in range(iters_per_epoch):
        sample = np.random.choice(num_vals, size=minibatch_size, replace=False)
        seed_set = mask_indices[sample]
        extra_nodes = seed_set
        for hop in range(hops):
            extra_nodes = sample_neighbours(extra_nodes, mask_indices, minibatch_size//2)
            seed_set = np.concatenate([seed_set, extra_nodes], axis=0)
        yield seed_set

def neighbourhood_validate(sparse_matrix, mat_values_val, mask_indices_val, mask_indices_tr, 
                           mask_indices_all, tf_dic,
                           hops=4, n_samp=100, lossfn="ce", minibatch_size=None):
    if minibatch_size is None:
        minibatch_size = mat_values_val.shape[0]
    loss_val_ = 0.
    n = 0
    sess = tf_dic["sess"]
    iters_per_val = max(1, mat_values_val.shape[0] / minibatch_size)
    idxes = np.random.permutation(np.arange(mat_values_val.shape[0]))
    for sample_ in tqdm(np.array_split(idxes, iters_per_val), 
                        total=iters_per_val):
        sample_tr_ = sample_k_neighbours(mask_indices_val[sample_, :], 
                                         mask_indices_tr, hops, n_samp)
        sample_val_ = mask_indices_val[sample_, :]
        mat_values_tr_ = np.array(sparse_matrix[sample_tr_[:,0], 
                                                sample_tr_[:,1]]).flatten()
        mat_values_val_ = np.array(sparse_matrix[sample_val_[:,0], 
                                                 sample_val_[:,1]]).flatten()
        
        mask_indices_tr_ = sample_tr_
        mask_indices_val_ = sample_val_

        mask_split_ = np.ones(sample_.shape[0])
        
        val_dict = {tf_dic["mat_values_tr"]:one_hot(mat_values_tr_),
                    tf_dic["mask_indices_tr"]:mask_indices_tr_,
                    tf_dic["mat_values_val"]:one_hot(mat_values_val_),
                    tf_dic["mask_indices_val"]:mask_indices_val_,
                    tf_dic["mask_indices_tr_val"]:mask_indices_val_,
                    tf_dic["mask_split"]:mask_split_
                    }

        bloss_val, = sess.run([tf_dic["rec_loss_val"]], feed_dict=val_dict)
        loss_val_ += bloss_val * sample_.shape[0]
        n += sample_.shape[0]
    return np.sqrt(loss_val_ / float(n))

def rec_loss_fn_sp(mat_values, mask_indices, rec_values, mask_split=None):
    if mask_split is None:
        mask_split = tf.ones_like(mat_values)
    return tf.reduce_sum((mat_values - rec_values)**2 * mask_split) / tf.cast(tf.reduce_sum(mask_split), tf.float32)

def get_losses(lossfn, reg_loss, mat_values_tr, mat_values_val, mask_indices_tr, 
               mask_indices_val, out_tr, out_val, mask_split, expected_value, num_outputs=5):
    if lossfn == "mse":
        rec_loss = rec_loss_fn_sp(mat_values_tr, mask_indices_tr, out_tr)
        rec_loss_val = rec_loss_fn_sp(mat_values_val, mask_indices_val, out_val, mask_split)
        total_loss = rec_loss + reg_loss
    elif lossfn == "ce":
        # Train cross entropy for optimization
        out = tf.reshape(out_tr, shape=[-1, num_outputs])
        rec_loss = - tf.reduce_mean(mask_split * tf.reduce_sum(tf.reshape(mat_values_tr, shape=[-1,num_outputs]) * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)), axis=1))
        rec_loss = rec_loss
        total_loss = rec_loss + reg_loss

        # Train RMSE for reporting
        evalues_tr = expected_value(tf.nn.softmax(tf.reshape(out_tr, shape=[-1,num_outputs])))
        avalues_tr = expected_value(tf.reshape(mat_values_tr, shape=[-1,num_outputs])) # actual values
        rec_loss = rec_loss_fn_sp(avalues_tr, mask_indices_tr, evalues_tr)

        # Validation
        evalues_val = expected_value(tf.nn.softmax(tf.reshape(out_val, shape=[-1,num_outputs])))
        avalues_val = expected_value(tf.reshape(mat_values_val, shape=[-1,num_outputs])) # actual values
        rec_loss_val = rec_loss_fn_sp(avalues_val, mask_indices_val, evalues_val, mask_split)
    else:
        raise KeyError("Unknown loss function: %s" % lossfn)
    return rec_loss, rec_loss_val, total_loss


def masked_inner_product(nvec, mvec, mask):
    ng = tf.gather(nvec, mask[:,0], axis=0)
    mg = tf.gather(mvec, mask[:,1], axis=1)
    return tf.reduce_sum(ng * tf.transpose(mg, (1, 0, 2)), axis=2, keep_dims=False)

def one_hot(x):
    values, x = np.unique(x, return_inverse=True)
    num_outputs = values.shape[0]
    n = x.shape[0]
    out = np.zeros((n,num_outputs))
    out[np.arange(n), x] = 1
    return out.flatten()

def prep_ev(ratings):
    values = tf.convert_to_tensor(np.unique(ratings), np.float32)
    def expected_value(output):
        return tf.reduce_sum(output * values[None,:], axis=-1)
    return expected_value

def get_optimizer(loss, opts):
    optimizer = opts.get("optimizer", "adam")
    opt_options = opts.get("opt_options", {})
    logging.info("Optimizing with %s with options: %s" % (optimizer, opt_options))
    if isinstance(optimizer, str):
        if optimizer == "adam":
            train_step = tf.train.AdamOptimizer(opts['lr'], **opt_options).minimize(loss)
        elif optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(opts['lr'], **opt_options).minimize(loss)
        elif optimizer == "sgd":
            train_step = tf.train.GradientDescentOptimizer(opts['lr']).minimize(loss)
        else:
            raise KeyError("Unknown optimizer: %s" % optimizer)
    else:
        train_step = optimizer.minimize(loss)
    return train_step

def build_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        '''
        Exponential moving average custom getter
        Source: http://ruishu.io/2017/11/22/ema/
        '''
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def setup_ema(scope, decay=0.998):
    '''
    Exponential moving average setup
    Source: http://ruishu.io/2017/11/22/ema/
    '''
    if decay < 1.:
        var_class = tf.get_collection('trainable_variables', scope) # Get list of the classifier's trainable variables
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_op = ema.apply(var_class)
        getter = build_getter(ema)
        return [ema_op], getter
    else:
        ema_op = []
        getter = None
        return ema_op, getter

def np_softmax(x):
    e = np.exp(x - np.max(x, axis=1)[:,None])
    return e / e.sum(axis=1)[:,None]

def load_data(p_validation = 0.07):
    path_dataset = "./data/flixster/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    index = np.array(np.where(M)).T
    ratings = M[np.where(M)]
    rng = np.random.RandomState(1234)
    validation_set = (Otraining[np.where(M)] * rng.binomial(1, p_validation, Otraining[np.where(M)].shape))
    mask_tr_val_split = np.array(Otest[np.where(M)] * 2 + validation_set, "int") # 


    flixter = {}
    flixter["mat_values_all"] = ratings
    flixter["mat_values_tr"] = ratings[mask_tr_val_split == 0]
    flixter["mat_values_val"] = ratings[mask_tr_val_split == 1]
    flixter["mat_values_test"] = ratings[mask_tr_val_split == 2]
    flixter["mat_values_tr_val"] = ratings[mask_tr_val_split <= 1]

    flixter["mask_indices_all"] = index
    flixter["mask_indices_tr"] = index[mask_tr_val_split == 0, :]
    flixter["mask_indices_val"] = index[mask_tr_val_split == 1, :]
    flixter["mask_indices_test"] = index[mask_tr_val_split == 2, :]
    flixter["mask_indices_tr_val"] = index[mask_tr_val_split <= 1, :]

    flixter["mask_tr_val_split"] = mask_tr_val_split

    n, m = index.max(axis=0) + 1
    flixter['mat_shape'] = (n, m, 10)
    return flixter

def main(opts, logfile=None, restore_point=None):
    if logfile is not None:
        logging.basicConfig(format='%(asctime)s %(message)s', filename=logfile, level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    try:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        path = opts['data_path']

        data = load_data()
        
        #build encoder and decoder and use VAE loss
        N, M, num_features = data['mat_shape']
        maxN, maxM = opts['maxN'], opts['maxM']

        if N < maxN: maxN = N
        if M < maxM: maxM = M
        lossfn = opts.get("loss", "mse")

        if opts['verbose'] > 0:
            logging.info('Factorized Autoencoder run settings:')
            logging.info('dataset: %s' % path)
            logging.info('Exchangable layer pool mode: %s' % opts['defaults']['matrix_sparse']['pool_mode'])
            logging.info('Pooling layer pool mode: %s' % opts['defaults']['matrix_pool_sparse']['pool_mode'])
            logging.info('learning rate: %s' % opts['lr'])
            logging.info('activation: %s' % opts['defaults']['matrix_sparse']['activation'])
            logging.info('number of latent features: %s' % opts['encoder'][-2]['units'])
            logging.info('maxN: %s' % opts['maxN'])
            logging.info('maxM: %s' % opts['maxM'])

        with tf.Graph().as_default():
            mat_values_tr = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr')
            mask_indices_tr = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_tr')

            mat_values_val = tf.placeholder(tf.float32, shape=[None], name='mat_values_val')
            mask_split = tf.placeholder(tf.float32, shape=[None], name='mat_values_val')
            mask_indices_val = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_val')
            mask_indices_tr_val = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_tr_val')
            expected_value = prep_ev(data["mat_values_all"])

            tr_dict = {'input':mat_values_tr,
                        'mask_indices':mask_indices_tr,
                        'units':1 if lossfn == "mse" else num_features, 
                        'shape':[N,M],
                        }

            
            val_dict = {'input':mat_values_tr,
                        'mask_indices':mask_indices_tr,
                        'units':1 if lossfn == "mse" else num_features,
                        'shape':[N,M],
                        }

            encoder = Model(layers=opts['encoder'], layer_defaults=opts['defaults'], scope="encoder", verbose=2) #define the encoder
            out_enc_tr = encoder.get_output(tr_dict) #build the encoder
            enc_ema_op, enc_getter = setup_ema("encoder", opts.get("ema_decay", 1.))
            out_enc_val = encoder.get_output(val_dict, reuse=True, verbose=0, is_training=False, getter=enc_getter)#get encoder output, reusing the neural net
            tr_dict = {'nvec':out_enc_tr['nvec'],
                        'mvec':out_enc_tr['mvec'],
                        'units':out_enc_tr['units'],
                        'mask_indices':mask_indices_tr,
                        'shape':out_enc_tr['shape'],
                        }

            val_dict = {'nvec':out_enc_val['nvec'],
                        'mvec':out_enc_val['mvec'],
                        'units':out_enc_val['units'],
                        'mask_indices':mask_indices_tr_val,
                        'shape':out_enc_val['shape'],
                        }

            decoder = Model(layers=opts['decoder'], layer_defaults=opts['defaults'], scope="decoder", verbose=2)#define the decoder
            out_dec_tr = decoder.get_output(tr_dict)#build it
            out_tr = out_dec_tr['input']
            dec_ema_op, dec_getter = setup_ema("decoder", opts.get("ema_decay", 1.))
            ema_op = enc_ema_op + dec_ema_op

            out_dec_val = decoder.get_output(val_dict, reuse=True, verbose=0, is_training=False, getter=dec_getter)#reuse it for validation
            out_val = out_dec_val['input']
            
            eout_val = expected_value(tf.nn.softmax(tf.reshape(out_val, shape=[-1,num_features])))

            #loss and training
            reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
            
            rec_loss, rec_loss_val, total_loss = get_losses(lossfn, reg_loss, 
                                                            mat_values_tr, 
                                                            mat_values_val, 
                                                            mask_indices_tr, 
                                                            mask_indices_val, 
                                                            out_tr, out_val,
                                                            mask_split,
                                                            expected_value,
                                                            num_outputs=num_features)
            train_step = get_optimizer(total_loss, opts)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess.run(tf.global_variables_initializer())

            if 'by_row_column_density' in opts['sample_mode'] or 'conditional_sample_sparse' in opts['sample_mode']:
                iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
            elif 'uniform_over_dense_values' in opts['sample_mode']:
                minibatch_size = np.minimum(opts['minibatch_size'], data['mask_indices_tr'].shape[0])
                iters_per_epoch = data['mask_indices_tr'].shape[0] // minibatch_size
            elif 'neighbourhood' in opts['sample_mode']:
                minibatch_size = np.minimum(opts['minibatch_size'], data['mask_indices_tr'].shape[0])
                weights = csr_matrix((np.ones_like(data['mat_values_tr']), 
                    (data['mask_indices_tr'][:,0], 
                    data['mask_indices_tr'][:,1])),
                    data["mat_shape"][0:2])

                sp_mat = csr_matrix((data['mat_values_all'], 
                    (data['mask_indices_all'][:,0], 
                    data['mask_indices_all'][:, 1])),
                    data["mat_shape"][0:2])
            
            min_loss = np.inf
            min_train = np.inf
            min_loss_epoch = 0
            losses = OrderedDict()
            losses["train"] = []
            losses["valid"] = []
            losses["test"] = []
            min_ts_loss = np.inf
            min_val_ts = np.inf
        
            saver = tf.train.Saver(max_to_keep=1000) # keep all checkpoints
            if restore_point is not None:
                saver.restore(sess, restore_point)

            best_log = "logs/best_" + opts.get("model_name", "TEST") + ".log"
            print("epoch,train,valid,test", file=open(best_log, "w"))

            tf_nodes = {"sess":sess,"mat_values_tr":mat_values_tr, "mask_indices_tr":mask_indices_tr,
                     "mat_values_val":mat_values_val,"mask_indices_val":mask_indices_val,
                     "mask_indices_tr_val":mask_indices_tr_val,"mask_split":mask_split,
                     "total_loss":total_loss, "rec_loss":rec_loss,"rec_loss_val":rec_loss_val,
                     "out_tr":out_tr,"out_val":out_val}

            for ep in range(opts.get('restore_point_epoch', 0), opts['epochs'] + opts.get('restore_point_epoch', 0)):
                begin = time.time()
                loss_tr_, rec_loss_tr_, loss_val_, loss_ts_ = 0.,0.,0.,0.

                if 'by_row_column_density' in opts['sample_mode']:
                    for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM, sample_uniform=False), 
                                            total=iters_per_epoch):#go over mini-batches
                        
                        inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies                    
                        mat_sp = data['mat_tr_val'][inds_] * data['mask_tr'][inds_]
                        mat_values = dense_array_to_sparse(mat_sp)['values']
                        mask_indices = dense_array_to_sparse(data['mask_tr'][inds_])['indices'][:,0:2]

                        tr_dict = {mat_values_tr:mat_values if lossfn == "mse" else one_hot(mat_values),
                                    mask_indices_tr:mask_indices,
                                    mask_split:np.ones_like(mat_values)
                                    }
                        
                        returns = sess.run([train_step, total_loss, rec_loss] + ema_op, feed_dict=tr_dict)
                        bloss_, brec_loss_ = [i for i in returns[1:3]]

                        loss_tr_ += np.sqrt(bloss_)
                        rec_loss_tr_ += np.sqrt(brec_loss_)

                elif 'uniform_over_dense_values' in opts['sample_mode']:
                    for sample_ in tqdm(sample_dense_values_uniform(data['mask_indices_tr'], minibatch_size, iters_per_epoch), 
                                        total=iters_per_epoch):
                        mat_values = data['mat_values_tr'][sample_]
                        mask_indices = data['mask_indices_tr'][sample_]

                        tr_dict = {mat_values_tr:mat_values if lossfn == "mse" else one_hot(mat_values),
                                    mask_indices_tr:mask_indices,
                                    mask_split:np.ones_like(mat_values)
                                    }
                        
                        returns = sess.run([train_step, total_loss, rec_loss] + ema_op, feed_dict=tr_dict)
                        bloss_, brec_loss_ = [i for i in returns[1:3]] # ema_op may be empty and we only need these two outputs

                        loss_tr_ += bloss_
                        rec_loss_tr_ += np.sqrt(brec_loss_)
                        gc.collect()


                elif 'conditional_sample_sparse' in opts['sample_mode']:
                    for _, _, _, _, sample_ in tqdm(conditional_sample_sparse(data['mask_indices_tr'], 
                                                                              data['mask_tr_val_split'], 
                                                                              [N,M,1], maxN, maxM), 
                                                    total=iters_per_epoch):
                        mat_values = data['mat_values_tr'][sample_]
                        mask_indices = data['mask_indices_tr'][sample_]

                        tr_dict = {mat_values_tr:mat_values if lossfn == "mse" else one_hot(mat_values),
                                    mask_indices_tr:reindex_mask(mask_indices),
                                    mask_split:np.ones_like(mat_values)
                                    }
                        
                        returns = sess.run([train_step, total_loss, rec_loss] + ema_op, feed_dict=tr_dict)
                        bloss_, brec_loss_ = [i for i in returns[1:3]] # ema_op may be empty and we only need these two outputs

                        loss_tr_ += bloss_
                        rec_loss_tr_ += np.sqrt(brec_loss_)
                        gc.collect()

                else:
                    raise ValueError('\nERROR - unknown <sample_mode> in main()\n')

                loss_tr_ /= iters_per_epoch
                rec_loss_tr_ /= iters_per_epoch
                losses['train'].append(loss_tr_)

                logging.info("Training: epoch {:d} took {:.1f} train loss {:.3f} (rec:{:.3f});".format(ep+1, time.time() - begin, 
                                                                                                    loss_tr_, rec_loss_tr_))

                if (ep+1) % opts['validate_interval'] == 0: # Validate and test every validate_interval epochs
                                ## Validation Loss 
                    val_ratings = data['mat_values_all'].copy()
                    val_ratings[data['mask_tr_val_split'] == 0] = 3.
                    val_dict = {mat_values_tr:data['mat_values_tr'] if lossfn =="mse" else one_hot(data['mat_values_tr']),
                                mask_indices_tr:data['mask_indices_tr'],
                                mat_values_val:val_ratings if lossfn =="mse" else one_hot(val_ratings),
                                mask_indices_val:data['mask_indices_all'],
                                mask_indices_tr_val:data['mask_indices_all'],
                                mask_split:(data['mask_tr_val_split'] == 1) * 1.
                                }

                    bloss_val, = sess.run([rec_loss_val], feed_dict=val_dict)
                    loss_val_ += np.sqrt(bloss_val)

                    ## Test Loss     
                    test_dict = {mat_values_tr:data['mat_values_tr'] if lossfn =="mse" else one_hot(data['mat_values_tr']),
                                mask_indices_tr:data['mask_indices_tr'],
                                mat_values_val:val_ratings if lossfn =="mse" else one_hot(val_ratings),
                                mask_indices_val:data['mask_indices_all'],
                                mask_indices_tr_val:data['mask_indices_all'],
                                mask_split:(data['mask_tr_val_split'] == 2) * 1.
                                }

                    bloss_test, = sess.run([rec_loss_val], feed_dict=test_dict)

                    loss_ts_ += np.sqrt(bloss_test)
                   
                    losses['valid'].append(loss_val_)
                    losses['test'].append(loss_ts_)
    
                    if loss_val_ < min_loss: # keep track of the best validation loss 
                        min_loss = loss_val_
                        min_loss_epoch = ep+1
                        min_train = rec_loss_tr_
                        min_test = loss_ts_
                        print("{:d},{:4},{:4},{:4}".format(ep, loss_tr_, loss_val_, loss_ts_), file=open(best_log, "a"))
                        if opts.get("save_best", False): 
                            save_path = saver.save(sess, opts['ckpt_folder'] + "/%s_best.ckpt" % opts.get('model_name', "test"))
                            logging.info("Model saved in file: %s" % save_path)
                    
                    if loss_ts_ < min_ts_loss: # keep track of the best test loss 
                        min_ts_loss = loss_ts_
                        min_val_ts = loss_val_                

                    logging.info("Validation: epoch {:d} took {:.1f} train loss {:.3f} (rec:{:.3f}); valid: {:.3f}; min valid loss: {:.3f} (train: {:.3}, test: {:.3}) at epoch: {:d}; test loss: {:.3f} (best test: {:.3f} with val {:.3f})."
                                        .format(ep+1, time.time() - begin, loss_tr_, 
                                                rec_loss_tr_, loss_val_, min_loss, 
                                                min_train, min_test, min_loss_epoch, loss_ts_,
                                                min_ts_loss, min_val_ts))
                    gc.collect()

                if (ep+1) % opts.get("checkpoint_interval", 10000000) == 0:
                    save_path = saver.save(sess, opts['ckpt_folder'] + "/%s_checkpt_ep_%05d.ckpt" % (opts.get('model_name', "test"), ep + 1))
                    logging.info("Model saved in file: %s" % save_path)                                

                #if loss_val_ > min_loss * 1.075:
                #    logging.info("Overfitting... exiting")
                    # overfitting: break if validation loss diverges
    #                break
        return losses
    except Exception as e:
        logging.exception("Training failed")


if __name__ == "__main__":
    auto_restore = False

    if len(sys.argv) > 1:
        LOG = open(sys.argv[1], "w", 0)
    # path = 'movielens-TEST'
    path = 'movielens-100k'
    # path = 'movielens-1M'
    # path = 'netflix/6m'
    #path = 'netflix/full'

    ap = False# use attention pooling
    lossfn = "ce"    
    skip_connections = False
    name = "flixter"
    ## 100k Configs
    if 'movielens-100k' in path:        
        maxN = 3000
        maxM = 3000
        minibatch_size = 5000000
        units = 220
        latent_features = 100
        learning_rate = 0.0005
        validate_interval = 1 # perform validation every validate_interval epochs 
        checkpoint_interval = 10 

    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 800
        maxM = 1300
        minibatch_size = 70000
        units = 120
        latent_features = 50
        learning_rate = 0.0005
        validate_interval = 10 # perform validation every validate_interval epochs
        checkpoint_interval = 10

    if 'netflix/6m' in path:
        maxN = 1100
        maxM = 1100
        minibatch_size = 5000000
        units = 110
        latent_features = 50
        learning_rate = 0.0005
        validate_interval = 10 # perform validation every validate_interval epochs
        checkpoint_interval = 1 

    if 'netflix/full' in path:
        maxN = 1100 
        maxM = 1100 
        minibatch_size = 80000
        units = 150
        latent_features = 100
        learning_rate = 0.0005
        validate_interval = 1 # perform validation every validate_interval epochs
        checkpoint_interval = 1
    #name = name + "-%d-%d" % (units, latent_features)
    path = 'flixter'
    log_name = "logs/%s.log" % name
    opts ={'epochs': 6000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'scratch/checkpoints/movielens',
           'model_name':name,
           'ema_decay':1.,
           'verbose':2,
           'loss':lossfn,
           'optimizer':"adam",
           'opt_options':{"epsilon":1e-6},
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
           'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':maxM,#num movies per submatrix
           'minibatch_size':minibatch_size,
           'visualize':False,
           'save':True,
           'data_path':path,
           'output_file':'output',
           'validate_interval':validate_interval,
           'checkpoint_interval':checkpoint_interval,
           'save_best':True,
           'encoder':[
               {'type':'matrix_sparse', 'units':units},
               {'type':'matrix_sparse', 'units':units},
               #{'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':latent_features, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool_sparse'},
               ],
            'decoder':[
               {'type':'matrix_sparse', 'units':units},
               {'type':'matrix_sparse', 'units':units},
               {'type':'matrix_sparse', 'units':units},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':10 if lossfn == "ce" else 1, 'activation':None},
            ],
            'defaults':{#default values for each layer type (see layer.py)
                 'bilinear_sparse':{
                    # 'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    'activation':tf.nn.relu,
                    # 'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'max',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                    'skip_connections':skip_connections,
                },
                'matrix_sparse':{
                    # 'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    #'activation':tf.nn.relu,
                    'activation':lambda x: tf.nn.relu(x) - 0.01*tf.nn.relu(-x), # Leaky Relu
                    # 'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'mean',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(1e-10),
                    'skip_connections':skip_connections,
                    'attention_sharpen':1.,
                    'attention_pooling':ap
                },
                'dense':{#not used
                    'activation':tf.nn.elu,
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'matrix_pool_sparse':{
                    'pool_mode':'mean',
                },
                'channel_dropout_sparse':{
                    'rate':.5,
                },
                'matrix_dropout_sparse':{
                    'rate':.05,
                }
            },
           'lr':learning_rate,
           'sample_mode':'conditional_sample_sparse'#neighbourhood'conditional_sample_sparse' # by_row_column_density, uniform_over_dense_values, conditional_sample_sparse
           
    }
    if auto_restore:        

        ## glob doesn't find any files since they are named *.ckpt.meta, etc. So do it with string split
        
        # checkpoints = sorted(glob.glob(opts['ckpt_folder'] + "/%s_checkpt_ep_*.ckpt" % (opts.get('model_name', "test"))))
        # if len(checkpoints) > 0:
        #     restore_point = checkpoints[-1]
        #     logging.info("Restoring from %s" % restore_point)
        # else:
        #     restore_point = None

        restore_point_epoch = sorted(glob.glob(opts['ckpt_folder'] + "/%s_checkpt_ep_*.ckpt*" % (opts.get('model_name', "test"))))[-1].split(".")[0].split("_")[-1]
        restore_point = opts['ckpt_folder'] + "/%s_checkpt_ep_" % (opts.get('model_name', "test")) + restore_point_epoch + ".ckpt"
        logging.info("Restoring from %s" % restore_point)

        opts["restore_point_epoch"] = int(restore_point_epoch) # Pass num_epochs so far to start counting from there. In case of another crash 


    else:
        restore_point = None
    main(opts, logfile=log_name, restore_point=restore_point)


