import numpy as np
import pdb

from collections import Counter
from alipy.query_strategy.query_features import IterativeSVD_mc
from utils.dataset_helpers import set_to_array, get_SVD_pred

def all_MSE(true, pred, idxs, worst_mse, best_mse):
    micro_mse, micro_pct = micro_pct_MSE(true, pred, idxs,
                                         worst_mse, best_mse)
    #macro_mse = macro_MSE(true, pred, idxs)
    # TODO: include pct of user performance, fix macro_mse
    macro_mse = -1
    return micro_mse, macro_mse, micro_pct

def micro_MSE(true, pred, idxs):
    # true: NxD matrix of observations for N users, D features
    # pred: N_tx1 matrix of predictions for N_t user-feature pairs
    # idxs: N_tx2 matrix of idxs to calculate MSE over
    diffs = []
    for i,idx in enumerate(idxs):
        diffs.append(pred[i] - true[idx[0], idx[1]])
    return np.sum(np.square(diffs))

def macro_MSE(true, pred, idxs):
    # true: NxD matrix of observations for N users, D features
    # pred: NxD matrix of predicted observations
    # idxs: idxs to calculate MSE over
    # TODO: change this to accept pred as a list of values 
    # corresponding to idxs
    n_users = true.shape[0]
    observed_mat = np.zeros(true.shape)
    for i in idxs:
        observed_mat[i[0], i[1]] = 1
    
    user_mses = []
    for i in range(n_users):
        test_feats = np.where(observed_mat[i] == 0)[0]

        if len(test_feats):
            diff = pred[i,test_feats] - true[i,test_feats]
            user_mses.append(np.mean(np.square(diff)))
    return np.mean(user_mses), user_mses

def micro_pct_MSE(true, pred, idxs, worst_mse, best_mse):
    best_mse_diff = worst_mse - best_mse 
    current_mse = micro_MSE(true, pred, idxs)
    mse_diff = worst_mse - current_mse 
    return current_mse, mse_diff/best_mse_diff

def best_possible_MSE(dataset, rank_opt):
    pred = get_SVD_pred(dataset, rank_opt, dataset.observable_idxs(), 
                        dataset.test_idxs)
    return micro_MSE(dataset.X, pred, dataset.test_idxs)

def worst_possible_MSE(dataset, rank_opt):
    pred = get_SVD_pred(dataset, rank_opt, dataset.init_idxs,
                        dataset.test_idxs)
    return micro_MSE(dataset.X, pred, dataset.test_idxs)

def macro_pct_MSE(true, pred, idxs):
    n_users = true.shape[0]
    n_feats = true.shape[1]
    svd_mc = IterativeSVD_mc(rank=n_feats-1)
    observed_mask = np.ones(true.shape)
    for i in idxs:
        observed_mask[i[0], i[1]] = 0
    pred_full = svd_mc.impute(true, observed_mask=observed_mask)
    
    user_pcts = []
    for i in range(n_users):
        test_feats = np.where(observed_mask[i] == 0)[0]
        if len(test_feats):
            diff_total = pred_full[i,test_feats] - true[i, test_feats]
            diff = pred[i,test_feats] - true[i, test_feats]
            mse_total = np.mean(np.square(diff_total))
            mse = np.mean(np.square(diff))
            user_pct_mse = mse/mse_total
            user_pcts.append(user_pct_mse)
    return np.mean(user_pcts), user_pcts

def classification_acc(true, pred, user=None):
    # true: Nx1 matrix of labels
    # pred: Nx1 matrix of predictions
    # user: idx of user to calculate metric for. Calculated
    #       for all users if None.
    pass

def quantity_cost(dataset, idx=None):
    # data: Instance of Dataset class
    # Outputs mean + variance in user data acquisition cost 
    # todo: incorporate diff. costs of features
    #idxs = dataset.acqu_idxs
    #if idx:
    #    idxs.append(idx)
    user_counts = np.sum(dataset.observed_mask(), axis=1)
    return np.mean(user_counts), np.var(user_counts)

def quantity_cost_mask(mask, idx=[]):
    try:
        if len(idx):
            mask[idx] = 1
    except:
        pdb.set_trace()
    user_counts = np.sum(mask, axis=1)
    return np.mean(user_counts), np.var(user_counts)

def quality_cost(dataset):
    pass
