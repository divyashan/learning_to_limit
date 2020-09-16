from scipy import linalg
import numpy as np

def perform_svd(a,rank):
    u, s, v = linalg.svd(a)
    ur = u[:, :rank]
    sr = np.matrix(linalg.diagsvd(s[:rank], rank,rank))
    vr = v[:rank, :]
    return np.asarray(ur*sr*vr)

def normalize(vals):
    min_val = np.min(vals) + 1e-10
    max_val = np.max(vals)
    norm_vals =(vals - min_val)/(max_val-min_val)
    return norm_vals

def calc_pct(pred_best, pred_worst, pred_curr):
    return (pred_worst - pred_curr)/(pred_worst - pred_best)
