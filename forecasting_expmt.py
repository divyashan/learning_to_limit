import os
import numpy as np
import pandas as pd
import pdb

from dataset import Dataset, MovieLensDataset
from curve_models import get_curve_model
from acquisition import get_acquisition_model
from utils.dataset_helpers import set_to_array, get_SVD_pred
from eval_fs import all_MSE, micro_MSE, quantity_cost, best_possible_MSE, worst_possible_MSE, macro_MSE 

def subsample_idxs(dataset, config, sample_sizes=[], mses=[], test_mses=[], macro_mses=[],
                   test_macro_mses=[]):
    """ Subsamples dataset to produce (sample size, validation performance) pairs

    dataset : Dataset to subsample
    config : Dictionary including keys 'step_size' and 'rank_opt'
    sample_sizes : Includes previously collected sample sizes
    mses, test_mses, macro_mses, test_macro_mses  : "" 
    Returns arrays corresponding the the sample size, validation mse, test mse, validation macro mses, 
    and test macro mses
    """
    
    # step_size dictates the query size
    # rank_opt dictates the assumed latent rank in the dataset
    step_size = config['step_size']
    rank_opt = config['rank_opt']
    
    available_idxs = set_to_array(dataset.available_idxs())
    n_available = dataset.n_available_idxs()
    n_feats = dataset.n_fs
    
    # Dictates the number of samples drawn for a given sample size
    n_samples = 1 
    
    # Skipping previous sample measurements
    max_existing_sample = 0
    if sample_sizes:
        max_existing_sample = np.max(sample_sizes)
    sample_opts = [int(x) for x in np.arange(0, n_available, step_size)]
    sample_opts = [opt for opt in sample_opts if (opt > max_existing_sample)] 
    
    new_mses, new_sample_sizes  = [], []
    new_test_mses, new_macro_mses, new_test_macro_mses = [], [], []
    for sample_size in sample_opts:
        for i in range(n_samples):
            sample_idxs = np.random.choice(range(n_available), sample_size, replace=False)
            sample_idxs = [available_idxs[i] for i in sample_idxs]

            val_pred = get_SVD_pred(dataset, rank_opt, sample_idxs, dataset.val_idxs)
            mse = micro_MSE(dataset.X, val_pred, dataset.val_idxs)
            macro_mse, m_mses = macro_MSE(dataset.X, val_pred, dataset.val_idxs)
            
            test_pred = get_SVD_pred(dataset, rank_opt, sample_idxs, dataset.test_idxs)
            test_mse = micro_MSE(dataset.X, test_pred, dataset.test_idxs)
            test_macro_mse, test_m_mses = macro_MSE(dataset.X, test_pred, dataset.test_idxs)
            
            new_sample_sizes.append(sample_size)
            new_mses.append(mse)
            new_test_mses.append(test_mse)
            new_macro_mses.append(m_mses)
            new_test_macro_mses.append(test_m_mses)
    sample_sizes.extend(new_sample_sizes)
    mses.extend(new_mses)
    test_mses.extend(new_test_mses)
    macro_mses.extend(np.array(new_macro_mses))
    test_macro_mses.extend(new_test_macro_mses)
    return sample_sizes, mses, test_mses, macro_mses, test_macro_mses

def run_mse_curve_expmt(dataset_name, config):
    """ Produces (sample size, mse) pairs for a given dataset.

    Saves results to directory for processing by curve_fitting_expmt.py.

    """
    n_runs = config['n_runs']
    rank_opt = config['rank_opt']
    batch_size = config['batch_size']
    n_acquisitions = config['n_acquisitions'] 
    
    acq_model_names = ['Weighted'] 
    curve_model_names = ['NLS_w', 'NLS_rse', 'linear', 'initial']
    config_sig = '_'.join([str(x) for x in config.values()])
    
    for acq_model_name in acq_model_names:
        acq_model = get_acquisition_model(acq_model_name)
        
        results = []
        results_path = "./results/forecasting/" + dataset_name + '/' + acq_model_name + '/' + config_sig
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        all_sample_sizes, all_mses = [], []
        all_test_mses, all_macro_mses, all_test_macro_mses = [], [], []           
        for j in range(n_runs):
            config['split_num'] = j
            dataset = MovieLensDataset(dataset_name, config) 
            best_mse = best_possible_MSE(dataset, rank_opt)
            worst_mse = worst_possible_MSE(dataset, rank_opt)
            best_val_mse = best_possible_MSE(dataset, rank_opt, "val")
            worst_val_mse = worst_possible_MSE(dataset, rank_opt, "val")

            i = 0
            mses, sample_sizes = [], []
            test_mses, macro_mses, test_macro_mses = [], [], []
            curve_models = [get_curve_model(cm_name) for cm_name in curve_model_names]
            while i < n_acquisitions: 
                acquired_idxs = acq_model.acquire_features(dataset, batch_size)
                dataset.add(acquired_idxs)
                mses = subsample_idxs(dataset, config, sample_sizes, mses,
                                      test_mses, macro_mses, test_macro_mses)
                sample_sizes, mses, test_mses, macro_mses, test_macro_mses = mses
                
                n_init = len(dataset.init_idxs)
                n_available = dataset.n_available_idxs()
                n_observable = dataset.n_observable_idxs()
                pred = get_SVD_pred(dataset, rank_opt, 
                                    dataset.available_idxs(),
                                    dataset.test_idxs) 
                mse_scores = all_MSE(dataset.X, pred, dataset.test_idxs, 
                               worst_mse, best_mse)
                micro_mse, macro_mse, micro_pct, _ = mse_scores
                
                val_pred = get_SVD_pred(dataset, rank_opt, 
                                    dataset.available_idxs(),
                                    dataset.val_idxs) 
                mse_scores_val = all_MSE(dataset.X, val_pred, dataset.val_idxs, 
                               worst_val_mse, best_val_mse)
                micro_mse_val, macro_mse_val, micro_pct_val, _ = mse_scores_val
                
                # TODO: This portion might not need to be run, since we operate on the other files
                """
                for cm in curve_models:
                    if i == 0:
                        cm.fit(sample_sizes, mses)
                    elif 'NLS' in cm.name:
                        cm.fit(sample_sizes, mses)
                    stop, pred_pct = cm.stop_condition(goal_pct, n_init, n_available, n_observable)
                    pred_curr = cm.f(n_available, **cm.p)
                    pred_best = cm.f(n_observable, **cm.p)
                    pred_worst = cm.f(n_init, **cm.p)
                    pct_available = dataset.pct_available_idxs()
                    result = {'curve_model': cm.name, 
                            'acq_model': acq_model_name,
                            'n_acquired': i, 'pct_available': pct_available,
                            'n_observable': n_observable,
                            'micro_mse': micro_mse, 'micro_pct': micro_pct, 
                            'goal_pct': goal_pct, 'n_available': n_available, 'n_init': n_init,
                            'n_observable': n_observable,
                            'macro_mse': macro_mse, 'run': j,
                            'micro_mse_val': micro_mse_val, 
                            'macro_mse_val': macro_mse_val,
                            'micro_pct_val': micro_pct_val,
                            'pred_pct': pred_pct, 'pred_curr': pred_curr,
                            'pred_best': pred_best, 'pred_worst': pred_worst,
                            'best_mse': best_mse, 'worst_mse': worst_mse}
                    results.append(result)
                    pd.DataFrame(results).to_csv(results_path + '/results_df')
                """
                i += batch_size
            all_sample_sizes.append(sample_sizes)
            all_mses.append(mses)
            all_test_mses.append(test_mses)
            all_macro_mses.append(macro_mses)
            all_test_macro_mses.append(test_macro_mses)
        np.savetxt(results_path + '/sample_sizes', all_sample_sizes)
        np.savetxt(results_path + '/mses', all_mses)
        np.savetxt(results_path + '/test_mses', all_test_mses)
        np.save(results_path + '/macro_mses', all_macro_mses)
        np.save(results_path + '/test_macro_mses', all_test_macro_mses)


# Configs for earlier part of data collection curve
init_pct = .0001
base_config = {'init_pct': init_pct, 'n_runs': 5, 'checks': False, 
               'init_mode': 'uniform', 'rank_opt': 30, 'item_pct': .5,
               'user_pct': .5}

mltiny_config = base_config.copy()
mltiny_config.update({'batch_size': 200, 'n_acquisitions': 2000, 'step_size': 50})

mluniform_config = base_config.copy()
mluniform_config.update({'batch_size': 600, 'n_acquisitions': 1200, 'step_size': 30})

gltiny_config = base_config.copy()
gltiny_config.update({'batch_size': 200, 'step_size': 25, 'n_acquisitions': 1000})

gl_config = base_config.copy()
gl_config.update({'step_size': 50, 'batch_size': 580, 'n_acquisitions': 5800})

# Configs for later part of data collection curve
init_pct = .10
base_config = {'init_pct': init_pct, 'n_runs': 5, 'checks': False, 
               'init_mode': 'uniform', 'rank_opt': 30, 'item_pct': .5,
               'user_pct': .5}

mltiny_config = base_config.copy()
mltiny_config.update({'step_size': 5250, 'batch_size': 21000, 'n_acquisitions': 213973})

mluniform_config = base_config.copy()
mluniform_config.update({'step_size': 20000, 'batch_size': 100000, 'n_acquisitions': 1000001})

gltiny_config = base_config.copy()
gltiny_config.update({'step_size': 1880, 'batch_size': 9400, 'n_acquisitions': 94001})

gl_config = base_config.copy()
gl_config.update({'step_size': 5800, 'batch_size': 29000, 'n_acquisitions': 290001})

dataset_names = [('ml-20m-tiny', mltiny_config), ('gl-tiny', gltiny_config),
                 ('ml-20m-uniform', mluniform_config),
                 ('gl', gl_config)]

dataset_names = [('ml-20m-tiny', mltiny_config)]
init_modes = ['uniform']
for dataset_name, config in dataset_names:
    for init_mode in init_modes:
        config['init_mode'] = init_mode
        run_mse_curve_expmt(dataset_name, config)














