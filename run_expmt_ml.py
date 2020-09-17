import os
import pdb
import pickle

import numpy as np
import pandas as pd

from dataset import MovieLensDataset
from utils.dataset_helpers import get_SVD_pred 
from acquisition import get_acquisition_model
from eval_fs import all_MSE, best_possible_MSE, worst_possible_MSE
import pdb

init_pct = .1
init_mode = 'item_subset'
base_config = {'init_pct': init_pct, 'split_num': 0, 'user_pct': .5,
               'item_pct': .5, 'init_mode': init_mode, 'rank_opt': 30,
               'checks': False, 'n_runs': 5}

ml_config = base_config.copy()
ml_config.update({'n_acquisitions': 1000001, 'log_interval': 100000})

mltiny_config = base_config.copy()
mltiny_config.update({'log_interval': 21000, 'n_acquisitions': 213973})

gltiny_config = base_config.copy()
gltiny_config.update({'log_interval': 9400,'n_acquisitions': 94001})

gl_config = base_config.copy()
gl_config.update({'log_interval': 29000, 'n_acquisitions': 290001})

config = ml_config
config_sig = '_'.join([str(x) for x in config.values()])

n_runs = config['n_runs']
rank_opt = config['rank_opt']
log_interval = config['log_interval']
n_acquisitions = config['n_acquisitions']

datasets = ['ml-20m-tiny', 'ml-20m-uniform', 'gl-tiny', 'gl']]
model_names = ['Random', 'Weighted', 'QBC'] 
for dataset_name in datasets:
    
    for model_name in model_names:
        results = []
        init_mats, all_observed_mats, all_user_mses = [], [], []
        
        results_path = "./results/" + dataset_name + "/" + model_name
        results_path += "/" + config_sig 
        obs_mat_path = results_path + '/observed_mats.pkl'
        init_mat_path = results_path + '/init_mats.pkl'
        user_mses_path = results_path + '/user_mses.pkl'
        results_f_path = results_path + "/results_df"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        for j in range(n_runs):
            dataset = MovieLensDataset(dataset_name, config)
            best_mse = best_possible_MSE(dataset, rank_opt) 
            worst_mse = worst_possible_MSE(dataset, rank_opt)
            n_feats = dataset.n_fs
    
            i = 0
            user_mses, observed_mats = [], []
            while i < n_acquisitions:
                test_idxs = dataset.test_idxs
                available_idxs = dataset.available_idxs()
                pred = get_SVD_pred(dataset, rank_opt, available_idxs,
                                    test_idxs)
                
                mses = all_MSE(dataset.X, pred, test_idxs, worst_mse, best_mse)
                micro_mse, macro_mse, micro_pct, macro_mses = mses
                n_available = dataset.n_available_idxs() 
                pct_available = dataset.pct_available_idxs()
                
                result = {'dataset': dataset, 'acq_model': model_name,
                        'micro_mse': micro_mse, 'micro_pct': micro_pct,
                        'macro_mse': macro_mse,
                        'run': j, 'n_available': n_available,
                        'pct_available': pct_available,
                        'mc_model': 'FunkSVD'}
                results.append(result)
                pd.DataFrame(results).to_csv(results_f_path)
                print("Written results")
                 
                model = get_acquisition_model(model_name)
                acquired_idxs = model.acquire_features(dataset, log_interval)
                print("Acquired features")    
                
                i += log_interval
                dataset.add(acquired_idxs)
                user_mses.append(macro_mses)
                observed_idxs = np.array(dataset.init_idxs.union(dataset.acqu_idxs))
                observed_mats.append(observed_idxs)

            init_idxs = np.array(dataset.init_idxs)
            all_observed_mats.append(observed_mats)
            init_mats.append(init_idxs)
            all_user_mses.append(user_mses)
        pickle.dump(np.array(all_observed_mats), open(obs_mat_path, 'wb'))
        pickle.dump(np.array(init_mats), open(init_mat_path, 'wb'))
        pickle.dump(np.array(all_user_mses), open(user_mses_path, 'wb'))
