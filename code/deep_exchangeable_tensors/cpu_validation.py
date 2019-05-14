from __future__ import print_function
# Standard lib imports
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

def rec_loss_fn_sp(mat_values, rec_values, mask_split=None):
    if mask_split is None:
        mask_split = tf.ones_like(mat_values)
    return tf.reduce_sum((mat_values - rec_values)**2 * mask_split) / tf.cast(tf.reduce_sum(mask_split), tf.float32)

def get_losses(lossfn, reg_loss, mat_values_tr, mat_values_val, out_tr, out_val, mask_split):
    if lossfn == "mse":
        rec_loss = rec_loss_fn_sp(mat_values_tr, out_tr)
        rec_loss_val = rec_loss_fn_sp(mat_values_val, out_val, mask_split)
        total_loss = rec_loss + reg_loss
    elif lossfn == "ce":
        # Train cross entropy for optimization
        out = tf.reshape(out_tr, shape=[-1,5])
        rec_loss = - tf.reduce_mean(mask_split * tf.reduce_sum(tf.reshape(mat_values_tr, shape=[-1,5]) * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)), axis=1))
        total_loss = rec_loss + reg_loss

        # Train RMSE for reporting
        evalues_tr = expected_value(tf.nn.softmax(tf.reshape(out_tr, shape=[-1,5])))
        avalues_tr = expected_value(tf.reshape(mat_values_tr, shape=[-1,5])) # actual values
        rec_loss = rec_loss_fn_sp(avalues_tr, evalues_tr)

        # Validation
        evalues_val = expected_value(tf.nn.softmax(tf.reshape(out_val, shape=[-1,5])))
        avalues_val = expected_value(tf.reshape(mat_values_val, shape=[-1,5])) # actual values
        rec_loss_val = rec_loss_fn_sp(avalues_val, evalues_val, mask_split)
    else:
        raise KeyError("Unknown loss function: %s" % lossfn)
    return rec_loss, rec_loss_val, total_loss


def masked_inner_product(nvec, mvec, mask):
    ng = tf.gather(nvec, mask[:,0], axis=0)
    mg = tf.gather(mvec, mask[:,1], axis=1)
    return tf.reduce_sum(ng * tf.transpose(mg, (1, 0, 2)), axis=2, keep_dims=False)

def one_hot(x):
    n = x.shape[0]
    out = np.zeros((n,5))
    out[np.arange(n), x-1] = 1
    return out.flatten()

def expected_value(output):
    return tf.reduce_sum(output * tf.range(1,6, dtype="float32")[None,:], axis=-1)

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

def main(opts, logfile=None, restore_point=None):
    if logfile is not None:
        logging.basicConfig(format='%(asctime)s %(message)s', filename=logfile, level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    try:
        cpu_config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )

        path = opts['data_path']

        if 'movielens-100k' in path:
            data = get_data(path, train=.75, valid=.05, test=.2, mode='sparse', fold=1) # ml-100k uses official test set so only the valid paramter matters
        else: 
            data = get_data(path, train=.6, valid=.2, test=.2, mode='sparse', fold=1)
        
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
            mask_indices_tr_val = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_tr_val')

            tr_dict = {'input':mat_values_tr,
                        'mask_indices':mask_indices_tr,
                        'units':1 if lossfn == "mse" else 5, 
                        'shape':[N,M],
                        }

            
            val_dict = {'input':mat_values_tr,
                        'mask_indices':mask_indices_tr,
                        'units':1 if lossfn == "mse" else 5,
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

            eout_val = expected_value(tf.nn.softmax(tf.reshape(out_val, shape=[-1,5])))

            #loss and training
            reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
            
            rec_loss, rec_loss_val, total_loss = get_losses(lossfn, reg_loss, 
                                                            mat_values_tr, 
                                                            mat_values_val,
                                                            out_tr, out_val,
                                                            mask_split)
            train_step = get_optimizer(total_loss, opts)
            sess = tf.Session(config=cpu_config)
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
        
            saver = tf.train.Saver()
            if restore_point is not None:
                saver.restore(sess, restore_point)
            logging.info("Restored successfully, running validation")
            ## Validation Loss
            train_data = data['mat_values_all'][data['mask_tr_val_split'] == 0]
            train_mask = data['mask_indices_all'][data['mask_tr_val_split'] == 0,:]
            val_dict = {mat_values_tr:train_data if lossfn =="mse" else one_hot(train_data),
                        mat_values_val:data['mat_values_all'] if lossfn =="mse" else one_hot(data['mat_values_all']),
                        mask_indices_tr:train_mask,
                        mask_indices_tr_val:data['mask_indices_all'],
                        mask_split:(data['mask_tr_val_split'] == 1) * 1.
                        }

            bloss_val, = sess.run([rec_loss_val], feed_dict=val_dict)
            loss_val_ = np.sqrt(bloss_val)
            logging.info("Validation complete. Got {:4}".format(loss_val_))

            ## Test Loss     
            test_dict = {mat_values_tr:train_data if lossfn =="mse" else one_hot(train_data),
                        mat_values_val:data['mat_values_all'] if lossfn =="mse" else one_hot(data['mat_values_all']),
                        mask_indices_tr:train_mask,
                        mask_indices_tr_val:data['mask_indices_all'],
                        mask_split:(data['mask_tr_val_split'] == 2) * 1.
                        }

            bloss_test, = sess.run([rec_loss_val], feed_dict=test_dict)
            loss_test_ = np.sqrt(bloss_test)
            print("Valid: {:4}, Test: {:4}\n".format(loss_val_, loss_test_))
    except Exception as e:
        logging.exception("Training failed")

if __name__ == "__main__":
    auto_restore = False

    if len(sys.argv) > 1:
        LOG = open(sys.argv[1], "w", 0)
    # path = 'movielens-TEST'
    # path = 'movielens-100k'
    path = 'movielens-1M'
    # path = 'netflix/6m'
    # path = 'netflix/full'

    ap = False# use attention pooling
    lossfn = "ce"    
    skip_connections = False 
    name = "movielens-1M-conds"
    ## 100k Configs
    if 'movielens-100k' in path:        
        maxN = 300
        maxM = 300
        minibatch_size = 5000000        
        units = 220
        latent_features = 100
        learning_rate = 0.0005
        validate_interval = 10 # perform validation every validate_interval epochs 
        checkpoint_interval = 10 

    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 800
        maxM = 1300
        minibatch_size = 50000
        units = 220
        latent_features = 100
        learning_rate = 0.0005
        validate_interval = 5 # perform validation every validate_interval epochs
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
        minibatch_size = 5000000
        units = 110
        latent_features = 50
        learning_rate = 0.0005
        validate_interval = 10 # perform validation every validate_interval epochs
        checkpoint_interval = 1 

    name = name + "-%d-%d" % (units, latent_features)
    log_name = "logs/%s.log" % name
    opts ={'epochs': 10000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/netflix',
           'model_name':name,
           'ema_decay':0.9,
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
               {'type':'matrix_sparse', 'units':units, "attention_pooling":ap},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":ap},
               #{'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':latent_features, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool_sparse'},
               ],
            'decoder':[
               {'type':'matrix_sparse', 'units':units, "attention_pooling":ap},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":ap},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":ap},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":ap},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':5 if lossfn == "ce" else 1, 'activation':None},
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
    restore_point = "checkpoints/netflix/movielens-1M-conds-220-100_checkpt_ep_00500.ckpt"
    log_name=None
    main(opts, logfile=log_name, restore_point=restore_point)


