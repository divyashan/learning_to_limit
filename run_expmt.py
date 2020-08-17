import os
import pdb
import pickle

import numpy as np
import pandas as pd

from dataset import Dataset
from acquisition import get_acquisition_model
from eval_fs import MSE, privacy_cost
from alipy.query_strategy.query_features import IterativeSVD_mc

config = {'n_acquisitions': 501,
        'init_pct': .10, 'test_pct': .20, 
        'init_mode': 'uniform', 'feat_pct': .5, 'user_pct': .5, 
        'feature_cost': 'uniform', 'l': 0.0,
        'checks': False, 'log_interval': 50, 'n_runs': 1000}

config_sig = '_'.join([str(x) for x in config.values()])
n_acquisitions = config['n_acquisitions']
log_interval = config['log_interval']
n_runs = config['n_runs']

datasets = ['three_blobs']
model_names = ['Weighted'] 
mode = 'batch'
for dataset_name in datasets:
    for model_name in model_names:
        results = []
        observed_mats = []
        init_mats = []
        results_path = "./results/" + dataset_name + "/" + model_name
        results_path += "/" + config_sig 
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        results_f_path = results_path + "/results_df"
        for j in range(n_runs):
            # save experiment config to the directory
            dataset = Dataset(dataset_name, config)
            n_feats = dataset.n_fs
            i = 0
            while i < n_acquisitions:
                if i % log_interval == 0:
                    print("logging")
                    # record metrics, save to results arrive
                    # imputation = impute_X(dataset)
                    svd_mc = IterativeSVD_mc(rank=n_feats-1)
                    pred = svd_mc.impute(dataset.X, 
                            observed_mask=dataset.observed_mask()) 
                    mse = MSE(dataset.X, pred, dataset.test_idxs)
                    cost_mean, cost_var = privacy_cost(dataset)
                    n_available = dataset.n_available_idxs() 
                    result = {'dataset': dataset, 'acq_model': model_name,
                              'mse': mse, 'cost_var': cost_var,
                              'cost_mean': cost_mean, 'run': j,
                              'n_available': n_available,
                              'mc_model': 'iterative_SVD'}
                    results.append(result)
                    pd.DataFrame(results).to_csv(results_f_path)
                
                model = get_acquisition_model(model_name, config)
                if model_name == 'Weighted' and mode == 'batch':
                    acquired_idxs = model.acquire_features(dataset, log_interval)
                    dataset.add(acquired_idxs)
                    i += log_interval
                else:
                    acquired_idxs = model.acquire_feature(dataset)
                    dataset.add(acquired_idxs)
                    i += 1
            
            observed_idxs = np.array(dataset.init_idxs + dataset.acqu_idxs)
            init_idxs = np.array(dataset.init_idxs)
            observed_mats.append(observed_idxs)
            init_mats.append(init_idxs)
            obs_mat_path = results_path + '/observed_mats.pkl'
            init_mat_path = results_path + '/init_mats.pkl'
            pickle.dump(np.array(observed_mats), open(obs_mat_path, 'wb'))
            pickle.dump(np.array(init_mats), open(init_mat_path, 'wb'))
