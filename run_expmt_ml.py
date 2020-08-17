import os
import pdb
import pickle

import numpy as np
import pandas as pd

from dataset import MovieLensDataset
from utils.dataset_helpers import get_SVD_pred 
from acquisition import get_acquisition_model
from eval_fs import all_MSE, quantity_cost, best_possible_MSE, worst_possible_MSE
import pdb

config = {'n_acquisitions': 10000,
        'init_pct': .10, 'split_num': 1, 'user_pct': .5, 
        'item_pct': .5, 'init_mode': 'uniform',
        'feature_cost': 'uniform', 'l': 0.0, 'rank_opt': 30,
        'checks': False, 'log_interval': 1000, 'n_runs': 5}

config_sig = '_'.join([str(x) for x in config.values()])
n_acquisitions = config['n_acquisitions']
log_interval = config['log_interval']
n_runs = config['n_runs']

datasets = ['ml-20m-tiny']
model_names = ['Weighted'] 
rank_opt = config['rank_opt']
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
            dataset = MovieLensDataset(dataset_name, config)
            best_mse = best_possible_MSE(dataset, rank_opt) 
            worst_mse = worst_possible_MSE(dataset, rank_opt)
            n_feats = dataset.n_fs
            i = 0
            while i < n_acquisitions:
                available_idxs = dataset.available_idxs()
                test_idxs = dataset.test_idxs
                pred = get_SVD_pred(dataset, rank_opt, available_idxs,
                                    test_idxs)
                mses = all_MSE(dataset.X, pred, dataset.test_idxs,
                              worst_mse, best_mse)
                micro_mse, macro_mse, micro_pct = mses
                print("Calculated MSE")
                cost_mean, cost_var = quantity_cost(dataset)
                n_available = dataset.n_available_idxs() 
                pct_available = dataset.pct_available_idxs()
                result = {'dataset': dataset, 'acq_model': model_name,
                        'micro_mse': micro_mse, 'micro_pct': micro_pct,
                        'macro_mse': macro_mse,
                        'quantity_cost': cost_var,
                        'run': j, 'n_available': n_available,
                        'pct_available': pct_available,
                        'mc_model': 'FunkSVD'}
                results.append(result)
                pd.DataFrame(results).to_csv(results_f_path)
                print("Acquiring features")    
                model = get_acquisition_model(model_name, config)
                acquired_idxs = model.acquire_features(dataset, log_interval)
                print("Acquired features")    
                dataset.add(acquired_idxs)
                i += log_interval
            
            observed_idxs = np.array(dataset.init_idxs.union(dataset.acqu_idxs))
            init_idxs = np.array(dataset.init_idxs)
            observed_mats.append(observed_idxs)
            init_mats.append(init_idxs)
            obs_mat_path = results_path + '/observed_mats.pkl'
            init_mat_path = results_path + '/init_mats.pkl'
            pickle.dump(np.array(observed_mats), open(obs_mat_path, 'wb'))
            pickle.dump(np.array(init_mats), open(init_mat_path, 'wb'))
