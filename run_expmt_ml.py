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

init_pct = .1
init_mode = 'item_subset'
ml_config = {'n_acquisitions': 1000001,
        'init_pct': init_pct, 'split_num': 1, 'user_pct': .5, 
        'item_pct': .5, 'init_mode': init_mode,
        'feature_cost': 'uniform', 'l': 0.0, 'rank_opt': 30,
        'checks': False, 'log_interval': 100000, 'n_runs': 5}

mltiny_config = {'n_runs': 5, 'checks': False, 'init_pct': init_pct,
        'test_pct': .4, 'init_mode': init_mode, 'log_interval': 21000,
        'step_size': 5250, 't': 0,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 213973,
          'item_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .85}

gltiny_config = {'n_runs': 5, 'checks': False, 'init_pct': init_pct,
        'test_pct': .4, 'init_mode': init_mode, 'log_interval': 9400,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 94001,
        'step_size': 1880, 't': 0,
          'item_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .8}

gl_config = {'n_runs': 5, 'checks': False, 'init_pct': init_pct,
        'test_pct': .4, 'init_mode': init_mode, 'log_interval': 29000,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 290001,
        'step_size': 5800, 't': 0,
          'item_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .8}

config = ml_config
# nacq = 3800, 380 for rtr
# nacq = 10000, 1000 for mltiny - should it be more for ml-uniform?
# nacq = 200000, 20000 for gl 
config_sig = '_'.join([str(x) for x in config.values()])
n_acquisitions = config['n_acquisitions']
log_interval = config['log_interval']
n_runs = config['n_runs']

datasets = ['gl', 'gl-tiny', 'ml-20m-uniform', 'ml-20m-tiny']
datasets = ['ml-20m-uniform']
model_names = ['Random', 'Weighted', 'QBC'] 
rank_opt = config['rank_opt']
for dataset_name in datasets:
    for model_name in model_names:
        results = []
        all_observed_mats = []
        init_mats = []
        all_user_mses = []
        results_path = "./results/" + dataset_name + "/" + model_name
        results_path += "/" + config_sig 
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        results_f_path = results_path + "/results_df"
        for j in range(n_runs):
            dataset = MovieLensDataset(dataset_name, config)
            best_mse = best_possible_MSE(dataset, rank_opt) 
            worst_mse = worst_possible_MSE(dataset, rank_opt)
            print("Done calculating best/wrost")
            n_feats = dataset.n_fs
            i = 0
            user_mses = []
            observed_mats = []
            while i < n_acquisitions:
                available_idxs = dataset.available_idxs()
                test_idxs = dataset.test_idxs
                pred = get_SVD_pred(dataset, rank_opt, available_idxs,
                                    test_idxs)
                mses = all_MSE(dataset.X, pred, test_idxs,
                              worst_mse, best_mse)
                micro_mse, macro_mse, micro_pct, macro_mses = mses
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
                user_mses.append(macro_mses)
                observed_idxs = np.array(dataset.init_idxs.union(dataset.acqu_idxs))
                observed_mats.append(observed_idxs)

            init_idxs = np.array(dataset.init_idxs)
            all_observed_mats.append(observed_mats)
            init_mats.append(init_idxs)
            all_user_mses.append(user_mses)
            obs_mat_path = results_path + '/observed_mats.pkl'
            init_mat_path = results_path + '/init_mats.pkl'
            user_mses_path = results_path + '/user_mses.pkl'
            pickle.dump(np.array(all_observed_mats), open(obs_mat_path, 'wb'))
            pickle.dump(np.array(init_mats), open(init_mat_path, 'wb'))
            pickle.dump(np.array(all_user_mses), open(user_mses_path, 'wb'))
