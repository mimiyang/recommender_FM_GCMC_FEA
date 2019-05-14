from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
from sparse_factorized_autoencoder_disjoint import main as train_model_with_opts
from sparse_factorized_autoencoder_disjoint import set_opts, one_hot
from disjoint_data import train_test_valid

def eval_dataset(fns, data, v=2):
    sess = fns["sess"]
    val_dict = {fns["mat_values_tr"]:one_hot(data["mat_values_tr"]),
                fns["mask_indices_tr"]:data["mask_indices_tr"],
                fns["mat_values_val"]:one_hot(data["mat_values_tr_val"]),
                fns["mask_indices_val"]:data["mask_indices_test"],
                fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                fns["mask_split"]:(data["mask_tr_val_split"] == v) * 1.
                }
    return np.sqrt(sess.run([fns["rec_loss_val"]], val_dict)[0])

def load_matlab_file(path_file, name_field):
    """
    Source: https://github.com/riannevdberg/gc-mc/blob/master/gcmc/preprocessing.py

    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T
    db.close()
    return out

def douban():
    path_dataset = "./data/douban/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    users, items = np.where(M)
    ratings = M[np.where(M)]
    data = {}
    data["mask_indices_tr_val"] = np.array([users, items], dtype="int").T
    data["mat_values_tr_val"] = np.array(ratings, dtype="int")
    data["mat_values_tr"] = np.array(ratings[Otraining[np.where(M)] == 1], dtype="int")
    data["mask_indices_tr"] = data["mask_indices_tr_val"][Otraining[np.where(M)] == 1, :]
    data["mask_indices_test"] = data["mask_indices_tr_val"][Otest[np.where(M)] == 1, :]
    data["mask_tr_val_split"] = np.array(Otest[np.where(M)] * 2, "int")
    return data, eval_dataset

def yahoo():
    path_dataset = "./data/yahoo_music/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    users, items = np.where(M)
    ratings = np.round(M[np.where(M)]/25.) + 1

    yahoo = {}
    yahoo["mask_indices_tr_val"] = np.array([users, items], dtype="int").T
    yahoo["mat_values_tr_val"] = np.array(ratings, dtype="int")
    yahoo["mat_values_tr"] = np.array(ratings[Otraining[np.where(M)] == 1], dtype="int")
    yahoo["mask_indices_tr"] = yahoo["mask_indices_tr_val"][Otraining[np.where(M)] == 1, :]
    yahoo["mask_indices_test"] = yahoo["mask_indices_tr_val"][Otest[np.where(M)] == 1, :]
    yahoo["mask_tr_val_split"] = np.array(Otest[np.where(M)] * 2, "int")
    
    def eval_yahoo(fns, data, v=2):
        '''
        custom eval function to map the 1-5 scale to 1-100 for yahoo.
        '''
        sess = fns["sess"]
        val_dict = {fns["mat_values_tr"]:one_hot(data["mat_values_tr"]),
                    fns["mask_indices_tr"]:data["mask_indices_tr"],
                    fns["mat_values_val"]:one_hot(data["mat_values_tr_val"]),
                    fns["mask_indices_val"]:data["mask_indices_test"],
                    fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                    fns["mask_split"]:(data["mask_tr_val_split"] == v) * 1.
                    }
        out = sess.run([fns["out_val"]], val_dict)[0]
        expval = (((softmax(out.reshape(-1,5))) * np.arange(1,6)[None, :]).sum(axis=1) - 0.5) * 25
        return np.sqrt(np.square(Otest[np.where(M)] *(expval- M[np.where(M)])).sum() / (Otest[np.where(M)]).sum())
    return yahoo, eval_yahoo

def softmax(x):
    e = np.exp(x - np.max(x, axis=1)[:,None])
    return e / e.sum(axis=1)[:,None]

def flixter():
    path_dataset = "./data/flixster/training_test_dataset.mat" 
    M = load_matlab_file(path_dataset, 'M')
    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')
    num_users = M.shape[0]
    num_items = M.shape[1]
    users, items = np.where(M)
    ratings = np.round(M[np.where(M)])

    flixter = {}
    flixter["mask_indices_tr_val"] = np.array([users, items], dtype="int").T
    flixter["mat_values_tr_val"] = np.array(ratings, dtype="int")
    flixter["mat_values_tr"] = np.array(ratings[Otraining[np.where(M)] == 1], dtype="int")
    flixter["mask_indices_tr"] = flixter["mask_indices_tr_val"][Otraining[np.where(M)] == 1, :]
    flixter["mask_indices_test"] = flixter["mask_indices_tr_val"][Otest[np.where(M)] == 1, :]
    flixter["mask_tr_val_split"] = np.array(Otest[np.where(M)] * 2, "int")
    
    def eval_flixter(fns, data, v=2):
        '''
        custom eval function to map the 1-5 scale to 1-100 for flixter.
        '''
        sess = fns["sess"]
        val_dict = {fns["mat_values_tr"]:one_hot(data["mat_values_tr"]),
                    fns["mask_indices_tr"]:data["mask_indices_tr"],
                    fns["mat_values_val"]:one_hot(data["mat_values_tr_val"]),
                    fns["mask_indices_val"]:data["mask_indices_test"],
                    fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                    fns["mask_split"]:(data["mask_tr_val_split"] == v) * 1.
                    }
        out = sess.run([fns["out_val"]], val_dict)[0]
        expval = (((softmax(out.reshape(-1,5))) * np.arange(1,6)[None, :]).sum(axis=1))
        return np.sqrt(np.square(Otest[np.where(M)] *(expval- M[np.where(M)])).sum() / (Otest[np.where(M)]).sum())
    return flixter, eval_flixter

def split_dataset_with_ratings(ratings, seed=1234, p_known=1., p_new=0.5, max_id_u=None, max_id_m=None):
    rng = np.random.RandomState(seed)
    n_ratings = ratings.rating.shape[0]
    n_users = np.max(ratings.user_id)
    n_movies = np.max(ratings.movie_id)
    # get unique users / movies 
    movie_id = np.unique(ratings["movie_id"])
    user_id = np.unique(ratings["user_id"])

    # split users into known (observed) and new (hidden) users (make 30% known)
    observed_users, hidden_users = random_split(user_id, 0.3, rng=rng)
    observed_movies, hidden_movies = random_split(movie_id, 0.3, rng=rng)

    # get a subset of users and movies simmilar in size to ml-100k for training
    p = p_known
    training_users,_ = random_split(observed_users, p, rng=rng)
    training_movies,_ = random_split(observed_movies, p, rng=rng)
    # get subset of dataframe containing selected users / movies
    known_ratings = get_subset(ratings, training_users, training_movies)
    print("Known movies / users size: %d" % known_ratings.shape[0])
    # construct new ids for the new data frame
    _, users = np.unique(known_ratings.user_id, return_inverse=True)

    known_ratings.loc[:,"user_id"] = users
    _, movies = np.unique(known_ratings.movie_id, return_inverse=True)
    known_ratings.loc[:,"movie_id"] = movies

    # split into train / valid / test as usual
    train, valid, test = train_test_valid(known_ratings, train=0.75, valid=0.05, test=0.2, rng=rng)
    # build data dictionary
    known_dat = prep_data_dict(train, valid, test, n_users, n_movies) # users.max() + 1, movies.max() + 1)

    p = p_new
    # get another subset of users and movies (distinct from previous users / movies) for evaluation.
    # the steps are the same except the users and movies are different
    new_users,_ = random_split(hidden_users, p, rng=rng)
    new_movies,_ = random_split(hidden_movies, p, rng=rng)
    new_ratings = get_subset(ratings, new_users, new_movies)
    
    _, users = np.unique(new_ratings.user_id, return_inverse=True)
    new_ratings.loc[:,"user_id"] = users
    _, movies = np.unique(new_ratings.movie_id, return_inverse=True)
    new_ratings.loc[:,"movie_id"] = movies
    if max_id_u is not None:
        new_ratings = new_ratings.loc[new_ratings.user_id < max_id_u,:]
    if max_id_m is not None:
        new_ratings = new_ratings.loc[new_ratings.movie_id < max_id_m,:]
    print("New movies / users size: %d" % new_ratings.shape[0])

    new_obs, new_hidden = train_test_valid(new_ratings, train=0.8, valid=0., test=0.2, rng=rng)
    new_dat = prep_data_dict(new_obs, test=new_hidden, n_users=n_users, n_movies=n_movies)
    return known_dat, new_dat, new_ratings

def netflix():
    '''
    TODO: finish netflix with neighhood sampling for evaluation.
    '''
    #ratings_df = pd.read_csv("./data/Netflix/NF_TRAIN/nf.train.txt", sep='\t', 
    #                  names=["user_id", "movie_id", "rating"], encoding='latin-1')
    #known_dat, new_dat, new_ratings = split_dataset_with_ratings(ratings_df, p_known=0., 
    #                                    p_new=0.07, max_id_m=10000, max_id_u=17770)
    return {}, lambda x, y: 0.

def compbio(seed=12345, n=300000):
    rng = np.random.RandomState(seed)
    mat = pd.read_csv("./data/bio/nonRedun.mat", sep="\t", index_col=0)
    mask = np.zeros((0, 2), "int")
    values = np.zeros((0))
    for i in xrange(mat.shape[1]):
        col = mat.loc[:, mat.columns[i]]
        idx = np.logical_not(col.isnull())
        matrix = np.arange(mat.shape[0])
        rows = matrix[idx]
        mask = np.concatenate((mask, np.concatenate([rows[:, None], np.repeat([i], rows.shape[0])[:, None]], axis=1)), axis=0)
        values = np.concatenate((values, col[idx]))
    
    idx = values < np.percentile(values, 99.95)
    values = values[idx]
    mask = mask[idx, :]
    idx = rng.permutation(values.shape[0])[0:n]
    values = values[idx]
    mask = mask[idx, :]
    values_disc = np.zeros_like(values, dtype="int")
    centers = np.zeros(5)
    for i, p in enumerate([20, 40, 60, 80, 100]):
        idx = np.logical_and(values_disc == 0, values <= np.percentile(values, p))
        values_disc[idx] = i+1
        centers[i] = (values[idx]).mean()
    
    (train_mask, train_values), (test_mask, test_values), idx = train_test_valid(np.concatenate([mask, values_disc[:, None]], axis=1), 
                                                                            train=0.8, test=0.2, valid=0., return_idx=True)
    
    split_pts = np.zeros_like(values, "int")
    split_pts[idx[1]] = 2
    data = {}
    data["mask_indices_tr_val"] = mask
    data["mat_values_tr_val"] = values_disc
    data["mat_values_tr"] = train_values
    data["mask_indices_tr"] = train_mask
    data["mask_indices_test"] = test_mask
    data["mask_tr_val_split"] = split_pts

    def eval_compbio(fns, data, v=2):
        '''
        custom eval function to map the 1-5 scale to 1-100 for flixter.
        '''
        sess = fns["sess"]
        val_dict = {fns["mat_values_tr"]:one_hot(data["mat_values_tr"]),
                    fns["mask_indices_tr"]:data["mask_indices_tr"],
                    fns["mat_values_val"]:one_hot(data["mat_values_tr_val"]),
                    fns["mask_indices_val"]:data["mask_indices_test"],
                    fns["mask_indices_tr_val"]:data["mask_indices_tr_val"],
                    fns["mask_split"]:(data["mask_tr_val_split"] == v) * 1.
                    }
        out = sess.run([fns["out_val"]], val_dict)[0]
        expval = (((softmax(out.reshape(-1,5))) * centers[None, :]).sum(axis=1))
        mask = (split_pts == 2)
        return np.sqrt(np.square(mask * (expval - values)).sum() / (mask).sum())
    return data, eval_compbio

def main():
    print("Training")
    losses, fns = train_model_with_opts(set_opts(epochs=0))
    print("Training complete....")
    with open("results/tranfer_learning_results.log", "w") as save_file:
        print("model,rmse", file=save_file)
    for name, dataset in {"douban":douban, "flixter":flixter, "yahoo":yahoo, "netflix":netflix, "compbio":compbio}.iteritems():
        print(name)
        data, eval_fn = dataset()
        rmse = eval_fn(fns, data)
        print(name, rmse)
        with open("results/tranfer_learning_results.log", "a") as save_file:
            print(name, rmse, sep=",", file=save_file)

if __name__ == '__main__':
    main()
