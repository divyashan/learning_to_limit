import os
import numpy as np
import pandas as pd
import pdb

from dataset import Dataset, MovieLensDataset
from curve_models import get_curve_model
from acquisition import get_acquisition_model
from utils.dataset_helpers import set_to_array, get_SVD_pred
from eval_fs import all_MSE, micro_MSE, quantity_cost, best_possible_MSE, worst_possible_MSE 

def subsample_idxs(dataset, config, sample_sizes=[], mses=[]):
    step_size = config['step_size']
    n_runs = 1 
    n_feats = dataset.n_fs
    available_idxs = set_to_array(dataset.available_idxs())
    n_available = dataset.n_available_idxs()
    
    max_existing_sample = 0
    if sample_sizes:
        max_existing_sample = np.max(sample_sizes)

    sample_opts = [int(x) for x in np.arange(0, n_available, step_size)]
    sample_opts = [opt for opt in sample_opts if (opt > max_existing_sample)] 
    new_mses = []
    new_sample_sizes = []
    test_mses = []
    for sample_size in sample_opts:
        for i in range(n_runs):
            sample_idxs = np.random.choice(range(n_available), 
                                           sample_size, replace=False)
            sample_idxs = [available_idxs[i] for i in sample_idxs]
            val_pred = get_SVD_pred(dataset, 30, sample_idxs, dataset.val_idxs)
            mse = micro_MSE(dataset.X, val_pred, dataset.val_idxs)

            test_pred = get_SVD_pred(dataset, 30, sample_idxs, dataset.test_idxs)
            test_mse = micro_MSE(dataset.X, test_pred, dataset.test_idxs)
            new_sample_sizes.append(sample_size)
            new_mses.append(mse)
            test_mses.append(mse)
    if len(sample_sizes) == 0 and 1 == 0:
        # Add point for no data aka a random guess
        new_sample_sizes.append(1e-6)
        fill_val = np.mean(dataset.get_funksvd_df(dataset.available_idxs())['rating'])
        val_pred = np.full((len(val_pred)), fill_val)
        mse = micro_MSE(dataset.X, val_pred, dataset.val_idxs)
        new_mses.append(mse)
    sample_sizes.extend(new_sample_sizes)
    mses.extend(new_mses)
    return sample_sizes, mses

def run_mse_curve_expmt(dataset_name, config):
    n_runs = config['n_runs']
    rank_opt = config['rank_opt']
    goal_pct = config['global_goal']
    batch_size = config['batch_size']
    n_acquisitions = config['n_acquisitions'] 
    acq_model_names = ['Random', 'Weighted']
    curve_model_names = ['NLS', 'NLS_w', 'NLS_rse']
    config_sig = '_'.join([str(x) for x in config.values()])

    sample_sizes = [] 
    mses = []
    for acq_model_name in acq_model_names:
        results = []
        acq_model = get_acquisition_model(acq_model_name, config)
        # TODO: write to results path
        for curve_model_name in curve_model_names:
            curve_model = get_curve_model(curve_model_name, config)
            results_path = "./results/forecasting/" + dataset_name + '_' + config_sig
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            for j in range(n_runs):
                dataset = MovieLensDataset(dataset_name, config) 
                best_mse = best_possible_MSE(dataset, rank_opt)
                worst_mse = worst_possible_MSE(dataset, rank_opt)
                
                i = 0
                stopping = False
                while i < n_acquisitions and stopping == False:
                    acquired_idxs = acq_model.acquire_features(dataset, batch_size) 
                    dataset.add(acquired_idxs)
                    sample_sizes, mses = subsample_idxs(dataset, config, sample_sizes, mses)
                    np.savetxt('ss', sample_sizes)
                    np.savetxt('mses', mses)
                    curve_model.fit(sample_sizes, mses)

                    n_init = len(dataset.init_idxs)
                    n_acquired = dataset.n_available_idxs()
                    n_observable = dataset.n_observable_idxs()
                    stopping = curve_model.stop_condition(goal_pct, n_init, n_acquired, n_observable)
                    i += batch_size
                
                pred = get_SVD_pred(dataset, rank_opt, 
                                    dataset.available_idxs(),
                                    dataset.test_idxs) 
                mse_vals = all_MSE(dataset.X, pred, dataset.test_idxs, 
                               worst_mse, best_mse)
                micro_mse, macro_mse, micro_pct = mse_vals
                _, q_cost = quantity_cost(dataset)
                n_available = dataset.n_available_idxs()
                pct_available = dataset.pct_available_idxs()
                result = {'curve_model': curve_model_name, 
                        'acq_model': acq_model_name,
                        'n_acquired': i, 'pct_available': pct_available, 
                        'micro_mse': micro_mse, 'micro_pct': micro_pct, 
                        'goal_pct': goal_pct, 'n_available': n_available,
                        'macro_mse': macro_mse, 'quantity_cost': q_cost}
                results.append(result)
                pd.DataFrame(results).to_csv(results_path + '/results_df')

mltiny_config = {'n_runs': 5, 'checks': False, 'init_pct': .1, 
        'test_pct': .4, 'init_mode': 'user_subset', 'batch_size': 1000,
        'step_size': 250,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 20000,
          'feat_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .85} 

mluniform_config = {'n_runs': 5, 'checks': False, 'init_pct': .1, 
        'test_pct': .4, 'init_mode': 'user_subset', 'batch_size': 40000,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 1000000,
        'step_size': 10000,
          'feat_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .8} 

dataset_names = [('ml-20m-tiny', mltiny_config), ('ml-20m-uniform', mluniform_config)]
dataset_names = [('ml-20m-uniform', mluniform_config)]
dataset_names = [('ml-20m-tiny', mltiny_config)]


# to run multiple experiments, create multiple configs
init_modes = ['user_subset', 'item_subset']
global_goals = [.6, .8, .85, .9, .95]
for dataset_name, config in dataset_names:
    for init_mode in init_modes:
        for global_goal in global_goals:
            config['init_mode'] = init_mode
            config['global_goal'] = global_goal
            run_mse_curve_expmt(dataset_name, config)














