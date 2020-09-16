import numpy as np
import pdb

from collections import Counter
from alipy.query_strategy.query_features import IterativeSVD_mc
from utils.dataset_helpers import set_to_array, get_SVD_pred

def all_MSE(true, pred, idxs, worst_mse, best_mse):
    micro_mse, micro_pct = micro_pct_MSE(true, pred, idxs,
                                         worst_mse, best_mse)
    macro_mse, macro_mses = macro_MSE(true, pred, idxs)
    return micro_mse, macro_mse, micro_pct, macro_mses

def micro_MSE(true, pred, idxs):
    # idxs: N_tx2 matrix of idxs to calculate MSE over
    diffs = []
    for i,idx in enumerate(idxs):
        diffs.append(pred[i] - true[idx[0], idx[1]])
    return np.sum(np.square(diffs))

def macro_MSE(true, pred, idxs):
    # idxs: idxs to calculate MSE over
    n_users = true.shape[0]
    observed_mat = np.zeros(true.shape)
    pred_mat = np.zeros(true.shape)
    idxs = set_to_array(idxs)
    for i, idx in enumerate(idxs):
        observed_mat[idx[0], idx[1]] = 1
        pred_mat[idx[0], idx[1]] = pred[i]

    user_mses = []
    for i in range(n_users):
        test_feats = np.where(observed_mat[i] == 1)[0]
        diff = pred_mat[i,test_feats] - true[i,test_feats]
        user_mses.append(np.mean(np.square(diff)))
    return np.mean(user_mses), user_mses

def micro_pct_MSE(true, pred, idxs, worst_mse, best_mse):
    best_mse_diff = worst_mse - best_mse 
    current_mse = micro_MSE(true, pred, idxs)
    mse_diff = worst_mse - current_mse 
    return current_mse, mse_diff/best_mse_diff

def best_possible_MSE(dataset, rank_opt, mode="test"):
    if mode == "test":
        pred = get_SVD_pred(dataset, rank_opt, dataset.observable_idxs(), 
                            dataset.test_idxs)
        return micro_MSE(dataset.X, pred, dataset.test_idxs)
    else:
        pred = get_SVD_pred(dataset, rank_opt, dataset.observable_idxs(), 
                            dataset.val_idxs)
        return micro_MSE(dataset.X, pred, dataset.val_idxs)

def worst_possible_MSE(dataset, rank_opt, mode="test"):
    if mode == "test":
        pred = get_SVD_pred(dataset, rank_opt, dataset.init_idxs,
                            dataset.test_idxs)
        return micro_MSE(dataset.X, pred, dataset.test_idxs)
    else:
        pred = get_SVD_pred(dataset, rank_opt, dataset.init_idxs, 
                            dataset.val_idxs)
        return micro_MSE(dataset.X, pred, dataset.val_idxs)

