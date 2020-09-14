import copy
import pdb
from alipy.query_strategy.query_features import QueryFeatureStability
from alipy.query_strategy.query_features import QueryFeatureAFASMC
from alipy.query_strategy.query_features import IterativeSVD_mc
from eval_fs import quantity_cost_mask
import numpy as np
from scipy.spatial.distance import cdist
from scipy import linalg
from utils.dataset_helpers import set_to_array, get_SVD_pred

def perform_svd(a,rank):
    u, s, v = linalg.svd(a)
    ur = u[:, :rank]
    sr = np.matrix(linalg.diagsvd(s[:rank], rank,rank))
    vr = v[:rank, :]
    return np.asarray(ur*sr*vr)

def get_acquisition_model(name, config):
    if name == 'QBC':
        return QBC()
    elif name == 'Weighted':
        return Weighted(config['l'])
    else:
        return Rand()

def normalize(vals):
    min_val = np.min(vals) + 1e-10
    max_val = np.max(vals)
    norm_vals =(vals - min_val)/(max_val-min_val)
    return norm_vals

class Baseline(object):

    def __init__(self):
        pass


    def return_all_check(self, dataset, n_features):
        if n_features > len(dataset.pool_idxs):
            pool_idxs = set_to_array(dataset.pool_idxs)
            return True, [tuple(i) for i in pool_idxs]
        return False, []
    
    def acquire_features(self, dataset, n_features):
        acquired_idxs = []
        for i in range(n_features):
            # logic for choosing the next index
            acquired_idx.append(idx)
        return acquired_idxs


class Rand(Baseline):

    def acquire_features(self, dataset, n_features):
        cond, idxs = self.return_all_check(dataset, n_features)
        if cond:
            return idxs

        pool_idxs = list(dataset.pool_idxs)
        n_pool = len(pool_idxs)
        idx = np.random.choice(np.arange(n_pool), n_features, replace=False)
        return [pool_idxs[int(i)] for i in idx]

class QBC(Baseline):

    def get_neighbor_idxs(self, user_ratings, X_k, k):
        comp_idxs = np.where(user_ratings != 0)[0]
        X_k_comp = X_k[:,comp_idxs]
        user_comp = user_ratings[comp_idxs]
        dists = cdist([user_comp], X_k_comp).squeeze()
        return np.argsort(dists)[:k]
    
    def knn_impute(self, dataset, k=3):
        X = dataset.X.copy()
        observed_mask = dataset.observed_mask()
        unobserved_mask = np.logical_not(observed_mask)
        X[unobserved_mask] = 0
       
        svd_pred = get_SVD_pred(dataset, 30, dataset.available_idxs(), dataset.pool_idxs)
        pool_idx_list = set_to_array(dataset.pool_idxs)
        X_k = np.zeros(dataset.X.shape)
        for i, idx in enumerate(pool_idx_list):
            X_k[idx[0], idx[1]] = svd_pred[i]
        
        for i in range(dataset.n_us):
            user_ratings = X[i]
            neighbor_idxs = self.get_neighbor_idxs(user_ratings, X_k, k)
            imputed = np.mean(X_k[neighbor_idxs,:], axis=0)
            imputed_idxs = np.where(user_ratings == 0)[0]
            X[i,imputed_idxs] = imputed[imputed_idxs]
        print("Done with KNN impute")
        return X
    
    def em_impute(self, dataset, k=3):
        X = dataset.X.copy()
        observed_mask = dataset.observed_mask()
        unobserved_mask = np.logical_not(observed_mask)
        X[unobserved_mask] = 0
        
        known_idxs = set_to_array(dataset.available_idxs())
        known_row_idxs = known_idxs[:,0]
        known_col_idxs = known_idxs[:,1]
       
        # This doesn't include the validation idxs..?
        unknown_idxs = dataset.pool_idxs.union(dataset.test_idxs)
        unknown_idxs = set_to_array(unknown_idxs)
        unknown_row_idxs = unknown_idxs[:,0]
        unknown_col_idxs = unknown_idxs[:,1]

        init_val = np.mean(X[known_row_idxs,known_col_idxs])
        X[unknown_row_idxs, unknown_col_idxs] = init_val
        delta = 1
        threshold = 1e-2
        while delta > threshold:
            X_k = perform_svd(X, 3)
            X_prev = X.copy()
            X[unknown_row_idxs, unknown_col_idxs] = X_k[unknown_row_idxs, unknown_col_idxs]
            delta = np.linalg.norm(X-X_prev)/np.linalg.norm(X_prev)
            print(delta)
        return X

    def svd_impute(self, dataset, rank_opt=30):
        # TODO: Move this to FunkSVD
        X = dataset.X.copy()
        pred = get_SVD_pred(dataset, rank_opt, dataset.available_idxs(), 
                            dataset.pool_idxs)
        pool_idx_list = set_to_array(dataset.pool_idxs)
        X_k = np.zeros(dataset.X.shape)
        for i, idx in enumerate(pool_idx_list):
            X_k[idx[0], idx[1]] = pred[i]
        return X_k
    
    def acquire_features(self, dataset, n_features):
        cond, idxs = self.return_all_check(dataset, n_features)
        if cond:
            return idxs
        imputations = [self.knn_impute(dataset),
                       self.em_impute(dataset),
                       self.svd_impute(dataset)]
        uncertainty = np.var(np.array(imputations), axis=0)
        
        pool_idxs = set_to_array(dataset.pool_idxs)
        row_idxs = pool_idxs[:,0]
        col_idxs = pool_idxs[:,1]
        v = uncertainty[row_idxs,col_idxs]
        selected_idxs = np.argsort(v)[:n_features]
        return [tuple(pool_idxs[i]) for i in selected_idxs]


class Weighted(Baseline):

    def __init__(self, l, n_svd_ranks=3):
        self.l = l
        self.svd_ranks = [int(i+1) for i in range(n_svd_ranks)]
    
    def acquire_features(self, dataset, k):
        cond, idxs = self.return_all_check(dataset, k)
        if cond:
            return idxs
        
        pool_preds = []
        for i in self.svd_ranks:
            pool_pred = get_SVD_pred(dataset, i, dataset.available_idxs(), dataset.pool_idxs)
            pool_preds.append(pool_pred)
        svd_uncertainty = np.var(np.array(pool_preds), axis=0)
        # subselect to the dataset.pool_idxs
        mask = dataset.observed_mask()
        pool_idxs = set_to_array(dataset.pool_idxs)
        row_idxs = pool_idxs[:,0]
        col_idxs = pool_idxs[:,1]
        v = svd_uncertainty
        #p = [quantity_cost_mask(mask, tuple(idx))[1] for idx in pool_idxs]
        v = normalize(np.log(v)) 
        #p = normalize(p)
        p = np.zeros(v.shape)
        c = -1*(1-self.l)*v + self.l*np.array(p)
        selected_idxs = np.argsort(c)[:k]
        return [tuple(pool_idxs[i]) for i in selected_idxs]


