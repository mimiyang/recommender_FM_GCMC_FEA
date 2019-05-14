import tensorflow as tf
from base import Model
from util import *
from sparse_util import *
import math
import time
from tqdm import tqdm 

def sample_submatrix(mask_,#mask, used for getting concentrations
                     maxN, maxM):
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


def rec_loss_fn(mat, mask, rec):
    return (tf.reduce_sum(((mat - rec)**2)*mask)) / tf.reduce_sum(mask)#average l2-error over non-zero entries


def dae_loss_fn(mat, mask, noise_mask, rec, alpha):    
    diff = (mat - rec)**2
    mask_c = mask * noise_mask
    mask_u = mask - mask_c
    loss_c = tf.reduce_sum( diff * mask_c )
    loss_u = tf.reduce_sum( diff * mask_u )
    eps = 1e-8
    return alpha * loss_c / (tf.reduce_sum(mask_c) + eps) + (1-alpha) * loss_u / (tf.reduce_sum(mask_u) + eps)


def main(opts):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    path = opts['data_path']
    data = get_data(path, train=.8, valid=.2, test=.001)
    
    #build encoder and decoder and use VAE loss
    N, M, num_features = data['mat_tr_val'].shape
    maxN, maxM = opts['maxN'], opts['maxM']

    if N < maxN: maxN = N
    if M < maxM: maxM = M

    # if opts['verbose'] > 0:
    #     print('\nRun Settings:')
    #     print('dataset: ', path)
    #     print('drop mask: ', opts['defaults']['matrix_dense']['drop_mask'])
    #     print('Exchangable layer pool mode: ', opts['defaults']['matrix_dense']['pool_mode'])
    #     print('Pooling layer pool mode: ', opts['defaults']['matrix_pool']['pool_mode'])
    #     print('learning rate: ', opts['lr'])
    #     print('activation: ', opts['defaults']['matrix_dense']['activation'])
    #     print('maxN: ', opts['maxN'])
    #     print('maxM: ', opts['maxM'])
    #     print('')
        

    with tf.Graph().as_default():

        mat = tf.placeholder(tf.float32, shape=(maxN, maxM, num_features), name='mat')#data matrix for training
        mask_tr = tf.placeholder(tf.float32, shape=(maxN, maxM, 1), name='mask_tr')
        #for validation, since we need less memory (forward pass only), we are feeding the whole matrix. This is only feasible for this smaller dataset. In the long term we could perform validation on CPU to avoid memory problems
        mat_val = tf.placeholder(tf.float32, shape=(N, M, num_features), name='mat')##data matrix for validation: 
        mask_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_val')#the entries not present during training
        mask_tr_val = tf.placeholder(tf.float32, shape=(N, M, 1), name='mask_tr_val')#both training and validation entries

        noise_mask = tf.placeholder(tf.float32, shape=(maxN, maxM, 1), name='noise_mask')
        mask_tr_noise = tf.placeholder(tf.float32, shape=(maxN, maxM, 1), name='mask_tr_noise')
        
        with tf.variable_scope("encoder"):
            tr_dict = {'input':mat,
                       'mask':mask_tr_noise}
            val_dict = {'input':mat_val,
                        'mask':mask_tr_val}

            encoder = Model(layers=opts['encoder'], layer_defaults=opts['defaults'], verbose=2) #define the encoder

            out_enc_tr = encoder.get_output(tr_dict) #build the encoder
            out_enc_val = encoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)#get encoder output, reusing the neural net
            

        with tf.variable_scope("decoder"):
            tr_dict = {'input':out_enc_tr['input'],
                       'mask':mask_tr}
            val_dict = {'input':out_enc_val['input'],
                        'mask':mask_val}

            decoder = Model(layers=opts['decoder'], layer_defaults=opts['defaults'], verbose=2)#define the decoder

            out_tr = decoder.get_output(tr_dict)['input']#build it
            out_val = decoder.get_output(val_dict, reuse=True, verbose=0, is_training=False)['input']#reuse it for validation

        #loss and training
        rec_loss = dae_loss_fn(mat, mask_tr, noise_mask, out_tr, opts['dae_loss_alpha'])
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # regularization
        rec_loss_val = rec_loss_fn(mat_val, mask_val, out_val)
        total_loss = rec_loss + reg_loss 

        train_step = tf.train.AdamOptimizer(opts['lr']).minimize(total_loss)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())

        iters_per_epoch = math.ceil(N//maxN) * math.ceil(M//maxM) # a bad heuristic: the whole matrix is in expectation covered in each epoch
        
        min_loss = 5
        min_loss_epoch = 0

        for ep in range(opts['epochs']):
            begin = time.time()
            loss_tr_, rec_loss_tr_, loss_val_ = 0,0,0
            for indn_, indm_ in tqdm(sample_submatrix(data['mask_tr'], maxN, maxM), total=iters_per_epoch):#go over mini-batches
                inds_ = np.ix_(indn_,indm_,[0])#select a sub-matrix given random indices for users/movies

                noise_rate = opts['dae_noise_rate']
                noise = np.random.choice([0,1], size=[maxN,maxM,1], p=[noise_rate, 1-noise_rate])

                tr_dict = {mat:data['mat_tr_val'][inds_],
                           mask_tr:data['mask_tr'][inds_],
                           mask_tr_noise:(data['mask_tr'][inds_] * noise),
                           noise_mask:noise}

                _, bloss_, brec_loss_ = sess.run([train_step, total_loss, rec_loss], feed_dict=tr_dict)

                loss_tr_ += np.sqrt(bloss_)
                rec_loss_tr_ += np.sqrt(brec_loss_)

            loss_tr_ /= iters_per_epoch
            rec_loss_tr_ /= iters_per_epoch

            val_dict = {mat_val:data['mat_tr_val'],
                        mask_val:data['mask_val'],
                        mask_tr_val:data['mask_tr']}
        
            bloss_, = sess.run([rec_loss_val], feed_dict=val_dict)
            loss_val_ += np.sqrt(bloss_)
            if loss_val_ < min_loss: # keep track of the best validation loss 
                min_loss = loss_val_
                min_loss_epoch = ep
            print("epoch {:d} took {:.1f} training loss {:.3f} (rec:{:.3f}) \t validation: {:.3f} \t minimum validation loss: {:.3f} at epoch: {:d}".format(ep, time.time() - begin, loss_tr_, rec_loss_tr_,  loss_val_, min_loss, min_loss_epoch), flush=True)


if __name__ == "__main__":

    # path = 'movielens-TEST'
    path = 'movielens-100k'
    # path = 'movielens-1M'


    ## TEST configurations 
    if 'movielens-TEST' in path:
        maxN = 100  
        maxM = 100
        skip_connections = True
        units = 32
        latent_features = 5
        learning_rate = 0.001

    ## 100k Configs
    if 'movielens-100k' in path:
        maxN = 500
        maxM = 500
        skip_connections = True
        units = 32
        latent_features = 5
        learning_rate = 0.001

    ## 1M Configs
    if 'movielens-1M' in path:
        maxN = 320
        maxM = 220
        skip_connections = True
        units = 54
        latent_features = 10
        learning_rate = 0.001

    dae_noise_rate = .5 # drop out this proportion of imput values 
    dae_loss_alpha = .5  # proportion of loss assigned to predicting droped out values 
    
    print('\n')
    print('dae_noise_rate: ', dae_noise_rate)
    print('dae_loss_alpha: ', dae_loss_alpha)
    print('')   

    opts ={'epochs':5000,#never-mind this. We have to implement look-ahead to report the best result.
           'ckpt_folder':'checkpoints/factorized_ae',
           'model_name':'test_fac_ae',
           'verbose':2,
           # 'maxN':943,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           # 'maxM':1682,#num movies per submatrix
           'maxN':maxN,#num of users per submatrix/mini-batch, if it is the total users, no subsampling will be performed
           'maxM':maxM,#num movies per submatrix
           'visualize':False,
           'save':False,
           'data_path':path,
           'encoder':[
               {'type':'matrix_dense', 'units':units},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':True},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':1, 'activation':None},#units before matrix-pool is the number of latent features for each movie and each user in the factorization
               # {'type':'matrix_pool'},
               ],
            'decoder':[
               {'type':'matrix_dense', 'units':units},
               # {'type':'matrix_dropout'},
               {'type':'matrix_dense', 'units':units, 'skip_connections':True},
               # {'type':'matrix_dropout'},
                {'type':'matrix_dense', 'units':1, 'activation':None},
            ],
            'defaults':{#default values for each layer type (see layer.py)s
                'matrix_dense':{
                    # 'activation':tf.nn.tanh,
                    # 'activation':tf.nn.sigmoid,
                    'activation':tf.nn.relu,
                    'drop_mask':False,#whether to go over the whole matrix, or emulate the sparse matrix in layers beyond the input. If the mask is droped the whole matrix is used.
                    'pool_mode':'mean',#mean vs max in the exchangeable layer. Currently, when the mask is present, only mean is supported
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                    'skip_connections':False,
                },
                'dense':{#not used
                    'activation':tf.nn.elu, 
                    'kernel_initializer': tf.random_normal_initializer(0, .01),
                    'regularizer': tf.contrib.keras.regularizers.l2(.00001),
                },
                'matrix_pool':{
                    'pool_mode':'max',
                },
                'matrix_dropout':{
                    'rate':.2,
                },                
            },
           'lr':learning_rate,
           'dae_noise_rate':dae_noise_rate,
           'dae_loss_alpha':dae_loss_alpha,
    }
    
    main(opts)


