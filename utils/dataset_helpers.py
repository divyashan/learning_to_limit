#import py_uci
import pdb
import numpy as np
from funk_svd import SVD

def set_to_array(idx_set):
    idx_arr = list(map(tuple, idx_set))
    idx_arr = np.array(idx_arr)
    return idx_arr

def get_dataset(name):
    uci_names = ['superconduct', 'abalone', 'iris', 'forest_fire', 
                 'wine_quality', 'wine_red', 'wine_white', 'boston', 'adult', 'parkinsons', 'concrete']
    if name in uci_names:
        return get_uci(name)
    elif name == 'three_blobs':
        return np.loadtxt('./datasets/three_blobs.txt')
    elif name == 'one_blob':
        return np.loadtxt('./datasets/one_blob')
    else:
        print("Dataset not supported!")

def get_uci(name):
    dataset = py_uci.get(name)
    return dataset.data

def get_movielens():
    # Helper function that returns movielens data
    # Pre-processed to 2000 users w the most ratings
    # and 1000 movies w the most ratings
    pass

def get_SVD_pred(dataset, rank, observed_idxs, test_idxs):
    train_ratings = dataset.get_funksvd_df(observed_idxs)
    test_ratings = dataset.get_funksvd_df(test_idxs)
    svd = SVD(learning_rate=0.001, regularization=.005, n_epochs=30,
              n_factors=rank, min_rating=1, max_rating=5)
    svd.fit(train_ratings, early_stopping=True, shuffle=False)
    try:
        pred = svd.predict(test_ratings)
    except:
        pdb.set_trace()
    return pred
