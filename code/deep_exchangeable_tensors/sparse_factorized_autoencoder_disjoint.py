from __future__ import print_function

import tensorflow as tf
import sys
from base import Model
from util import get_data
from sparse_util import *
import math
import time
from tqdm import tqdm
from collections import OrderedDict
from disjoint_data import load_ratings
import gc

LOG = sys.stdout

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

def neighbourhood_sampling(mask, n=1, batch_size=None, replace=False, unique_seed=False, hops=1):
    axis = np.random.randint(0,2)
    idx = np.arange(mask.shape[0])
    if unique_seed:
        seed_set = np.unique(mask[:, axis])
    else:
        seed_set = mask[:, axis]
    seed_nodes = np.random.choice(seed_set, n, replace)
    batch = np.array([], dtype="int")
    for _ in xrange(hops):
        neighbours_idx = idx[np.isin(mask[:, axis], np.unique(seed_nodes))]
        neighbours_idx = np.random.choice(neighbours_idx, size=[n], replace=False)
        neighbours = mask[neighbours_idx, :]
        axis = (axis + 1) % 2
        seed_nodes = neighbours[:, axis]
        batch = np.concatenate([batch, neighbours_idx], axis=0)
    return batch

def rec_loss_fn_sp(mat_values, mask_indices, rec_values, mask_split=None):
    if mask_split is None:
        mask_split = tf.ones_like(mat_values)
    return tf.reduce_sum((mat_values - rec_values)**2 * mask_split) / tf.cast(tf.reduce_sum(mask_split), tf.float32)

def get_losses(lossfn, reg_loss, mat_values_tr, mat_values_val, mask_indices_tr, mask_indices_val, out_tr, out_val, mask_split):
    if lossfn == "mse":
        rec_loss = rec_loss_fn_sp(mat_values_tr, mask_indices_tr, out_tr)
        rec_loss_val = rec_loss_fn_sp(mat_values_val, mask_indices_val, out_val, mask_split)
        total_loss = rec_loss + reg_loss
    elif lossfn == "ce":
        # Train cross entropy for optimization
        out = tf.reshape(out_tr, shape=[-1,5])
        rec_loss = -tf.reduce_mean(tf.reshape(mat_values_tr, shape=[-1,5]) * (out - tf.reduce_logsumexp(out, axis=1, keep_dims=True)))
        total_loss = rec_loss + reg_loss

        # Train RMSE for reporting
        evalues_tr = expected_value(tf.nn.softmax(tf.reshape(out_tr, shape=[-1,5])))
        avalues_tr = expected_value(tf.reshape(mat_values_tr, shape=[-1,5])) # actual values
        rec_loss = rec_loss_fn_sp(avalues_tr, mask_indices_tr, evalues_tr)

        # Validation
        evalues_val = expected_value(tf.nn.softmax(tf.reshape(out_val, shape=[-1,5])))
        avalues_val = expected_value(tf.reshape(mat_values_val, shape=[-1,5])) # actual values
        rec_loss_val = rec_loss_fn_sp(avalues_val, mask_indices_val, evalues_val, mask_split)
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
    print("Optimizing with %s with options: %s" % (optimizer, opt_options))
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
        LOG = open(logfile, "w", 0)
    else:
        LOG = sys.stdout
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)    
    path = opts['data_path']
    data, eval_data = load_ratings() 
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat_shape']
    maxN, maxM = opts['maxN'], opts['maxM']

    if N < maxN: maxN = N
    if M < maxM: maxM = M
    lossfn = opts.get("loss", "mse")

    if opts['verbose'] > 0:
        print('\nFactorized Autoencoder run settings:', file=LOG)
        print('dataset: ', path, file=LOG)
        print('Exchangable layer pool mode: ', opts['defaults']['matrix_sparse']['pool_mode'], file=LOG)
        print('Pooling layer pool mode: ', opts['defaults']['matrix_pool_sparse']['pool_mode'], file=LOG)
        print('learning rate: ', opts['lr'], file=LOG)
        print('activation: ', opts['defaults']['matrix_sparse']['activation'], file=LOG)
        print('number of latent features: ', opts['encoder'][-2]['units'], file=LOG)
        print('maxN: ', opts['maxN'], file=LOG)
        print('maxM: ', opts['maxM'], file=LOG)
        print('', file=LOG)

    with tf.Graph().as_default():
        mat_values_tr = tf.placeholder(tf.float32, shape=[None], name='mat_values_tr')
        mask_indices_tr = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_tr')

        mat_values_val = tf.placeholder(tf.float32, shape=[None], name='mat_values_val')
        mask_split = tf.placeholder(tf.float32, shape=[None], name='mat_values_val')
        mask_indices_val = tf.placeholder(tf.int32, shape=[None, 2], name='mask_indices_val')
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

        #loss and training
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
        
        rec_loss, rec_loss_val, total_loss = get_losses(lossfn, reg_loss, 
                                                        mat_values_tr, 
                                                        mat_values_val, 
                                                        mask_indices_tr, 
                                                        mask_indices_val, 
                                                        out_tr, out_val,
                                                        mask_split)
        train_step = get_optimizer(total_loss, opts)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        if 'by_row_column_density' in opts['sample_mode']:
            iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
        elif 'uniform_over_dense_values' in opts['sample_mode']:
            minibatch_size = np.minimum(opts['minibatch_size'], data['mask_indices_tr'].shape[0])
            iters_per_epoch = data['mask_indices_tr'].shape[0] // minibatch_size
        
        
        min_loss = 5
        min_train = 5
        min_loss_epoch = 0
        losses = OrderedDict()
        losses["train"] = []
        losses["valid"] = []
        losses["test"] = []
        min_ts_loss = 5 
        min_val_ts = 5 
       
        saver = tf.train.Saver()
        if restore_point is not None:
            saver.restore(sess, restore_point)

        best_log = "logs/" + opts.get("model_name", "TEST") + "_best.log"
        print("epoch,train,valid,test\n", file=open(best_log, "a"))
        for ep in range(opts['epochs']):
            begin = time.time()
            loss_tr_, rec_loss_tr_, loss_val_, loss_ts_ = 0,0,0,0

            if 'by_row_column_density' in opts['sample_mode']:
                for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM, sample_uniform=False), 
                                        total=iters_per_epoch):#go over mini-batches
                    
                    inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies                    
                    mat_sp = data['mat_tr_val'][inds_] * data['mask_tr'][inds_]
                    mat_values = dense_array_to_sparse(mat_sp)['values']
                    mask_indices = dense_array_to_sparse(data['mask_tr'][inds_])['indices'][:,0:2]

                    tr_dict = {mat_values_tr:mat_values if lossfn == "mse" else one_hot(mat_values),
                                mask_indices_tr:mask_indices,
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

            ## Validation Loss     
            val_dict = {mat_values_tr:data['mat_values_tr'] if lossfn =="mse" else one_hot(data['mat_values_tr']),
                        mask_indices_tr:data['mask_indices_tr'],
                        mat_values_val:data['mat_values_tr_val'] if lossfn =="mse" else one_hot(data['mat_values_tr_val']),
                        mask_indices_val:data['mask_indices_val'],
                        mask_indices_tr_val:data['mask_indices_tr_val'],
                        mask_split:(data['mask_tr_val_split'] == 1) * 1.
                        }

            bloss_val, = sess.run([rec_loss_val], feed_dict=val_dict)
            loss_val_ += np.sqrt(bloss_val)

            ## Test Loss     
            test_dict = {mat_values_tr:eval_data['mat_values_tr'] if lossfn =="mse" else one_hot(eval_data['mat_values_tr']),
                         mask_indices_tr:eval_data['mask_indices_tr'],
                         mat_values_val:eval_data['mat_values_tr_val'] if lossfn =="mse" else one_hot(eval_data['mat_values_tr_val']),
                         mask_indices_val:eval_data['mask_indices_test'],
                         mask_indices_tr_val:eval_data['mask_indices_tr_val'],
                         mask_split:(eval_data['mask_tr_val_split'] == 2) * 1.
                        }

            bloss_test, = sess.run([rec_loss_val], feed_dict=test_dict)

            loss_ts_ += np.sqrt(bloss_test)
            if loss_ts_ < min_ts_loss: # keep track of the best validation loss 
                min_ts_loss = loss_ts_
                min_val_ts = loss_val_
            if loss_val_ < min_loss: # keep track of the best validation loss 
                min_loss = loss_val_
                min_loss_epoch = ep
                min_train = rec_loss_tr_
                min_test = loss_ts_
                print("{:d},{:4},{:4},{:4}\n".format(ep, loss_tr_, loss_val_, loss_ts_), file=open(best_log, "a"))
                if ep > 1000 and (min_loss < 0.942):
                    save_path = saver.save(sess, opts['ckpt_folder'] + "/%s_best.ckpt" % opts.get('model_name', "test"))
                    print("Model saved in file: %s" % save_path, file=LOG)
            if (ep+1) % 500 == 0:
                save_path = saver.save(sess, opts['ckpt_folder'] + "/%s_checkpt_ep_%05d.ckpt" % (opts.get('model_name', "test"), ep + 1))
                print("Model saved in file: %s" % save_path, file=LOG)

            losses['train'].append(loss_tr_)
            losses['valid'].append(loss_val_)
            losses['test'].append(loss_ts_)

            print("epoch {:d} took {:.1f} train loss {:.3f} (rec:{:.3f}); valid: {:.3f}; min valid loss: {:.3f} \
(train: {:.3}, test: {:.3}) at epoch: {:d}; test loss: {:.3f} (best test: {:.3f} with val {:.3f})".format(ep, time.time() - begin, loss_tr_, 
                                                                              rec_loss_tr_, loss_val_, min_loss, 
                                                                              min_train, min_test, min_loss_epoch, loss_ts_,
                                                                              min_ts_loss, min_val_ts), file=LOG)
            gc.collect()
            if loss_val_ > min_loss * 1.075:
                # overfitting
                break
    
    saver.restore(sess, opts['ckpt_folder'] + "/%s_best.ckpt" % opts.get('model_name', "test"))
    return losses, {"sess":sess,"mat_values_tr":mat_values_tr, "mask_indices_tr":mask_indices_tr,
                     "mat_values_val":mat_values_val,"mask_indices_val":mask_indices_val,
                     "mask_indices_tr_val":mask_indices_tr_val,"mask_split":mask_split,
                     "total_loss":total_loss, "rec_loss":rec_loss,"rec_loss_val":rec_loss_val,
                     "out_tr":out_tr,"out_val":out_val}

def set_opts(epochs=100000, learning_rate=0.0005, units=220, latent_features=100, path="movielens-100k", 
             attention_pooling=False, lossfn="ce", maxN=100, maxM=100, skip_connections=False, model_name="disjoint_fac_ae",
             ema_decay=0.9, minibatch_size=1000000):
    return {'epochs': epochs,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/%s' % model_name,
           'model_name':model_name,
           'ema_decay':ema_decay,
           'verbose':2,
           'loss':lossfn,
           'optimizer':"adam",
           'opt_options':{"epsilon":1e-6},
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
           'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':maxM,#num movies per submatrix
           'minibatch_size':minibatch_size,
           'save':True,
           'data_path':path,
           'output_file':'output',
           'encoder':[
               {'type':'matrix_sparse', 'units':units, "attention_pooling":attention_pooling},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":attention_pooling},
               #{'type':'matrix_dropout_sparse'},
               {'type':'matrix_sparse', 'units':latent_features, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               {'type':'matrix_pool_sparse'},
               ],
            'decoder':[
               {'type':'matrix_sparse', 'units':units, "attention_pooling":attention_pooling},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":attention_pooling},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":attention_pooling},
               {'type':'channel_dropout_sparse'},
               {'type':'matrix_sparse', 'units':units, "attention_pooling":attention_pooling},
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
           'sample_mode':'uniform_over_dense_values' # by_row_column_density, uniform_over_dense_values
           
    }

if __name__ == "__main__":

    #restore_point = "checkpoints/factorized_ae/test_fac_ae_checkpt_ep_05500.ckpt"
    restore_point = None
    if len(sys.argv) > 1:
        LOG = open(sys.argv[1], "w", 0)
    # path = 'movielens-TEST'
    path = 'movielens-100k'
    #path = 'movielens-1M'
    # path = 'netflix/6m'

    ap = False# use attention pooling
    lossfn = "ce"
    ## 100k Configs
    if 'movielens-100k' in path:
        maxN = 100
        maxM = 100
        minibatch_size = 5000000
        skip_connections = False 
        units = 220
        latent_features = 100
        learning_rate = 0.0005

    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 250000
        maxM = 150000
        minibatch_size = 50000
        skip_connections = False
        units = 150
        latent_features = 50
        learning_rate = 0.001

    if 'netflix/6m' in path:
        maxN = 300
        maxM = 300
        minibatch_size = 5000000
        skip_connections = False
        units = 16
        latent_features = 5
        learning_rate = 0.005


    opts = set_opts(epochs=3000, learning_rate=learning_rate, units=units, latent_features=latent_features, path=path, 
             attention_pooling=ap, lossfn=lossfn, maxN=maxN, maxM=maxM, skip_connections=skip_connections)
    
    main(opts, restore_point=restore_point)


