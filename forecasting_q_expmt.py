import os
import numpy as np
import pandas as pd
import pdb

from dataset import Dataset, MovieLensDataset
from curve_models import get_curve_model
from acquisition import get_acquisition_model
from utils.dataset_helpers import set_to_array, get_SVD_pred
from eval_fs import all_MSE, micro_MSE, quantity_cost, best_possible_MSE, worst_possible_MSE 

def subsample_idxs(dataset, config, sample_sizes=[], mses=[], test_mses=[]):
    step_size = config['step_size']
    rank_opt = config['rank_opt']
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
    new_test_mses = []
    for sample_size in sample_opts:
        for i in range(n_runs):
            sample_idxs = np.random.choice(range(n_available), 
                                           sample_size, replace=False)
            sample_idxs = [available_idxs[i] for i in sample_idxs]
            val_pred = get_SVD_pred(dataset, rank_opt, sample_idxs, dataset.val_idxs)
            mse = micro_MSE(dataset.X, val_pred, dataset.val_idxs)

            test_pred = get_SVD_pred(dataset, rank_opt, sample_idxs, dataset.test_idxs)
            test_mse = micro_MSE(dataset.X, test_pred, dataset.test_idxs)
            new_sample_sizes.append(sample_size)
            new_mses.append(mse)
            new_test_mses.append(test_mse)
    sample_sizes.extend(new_sample_sizes)
    mses.extend(new_mses)
    test_mses.extend(new_test_mses)
    return sample_sizes, mses, test_mses

def run_mse_curve_expmt(dataset_name, config):
    n_runs = config['n_runs']
    rank_opt = config['rank_opt']
    goal_pct = config['global_goal']
    batch_size = config['batch_size']
    n_acquisitions = config['n_acquisitions'] 
    
    acq_model_names = ['Random'] 
    curve_model_names = ['NLS_w', 'NLS_rse', 'linear', 'initial']
    q_strategy = ['Const', 'Mult', 'Adapt']
    config_sig = '_'.join([str(x) for x in config.values()])
    
    for acq_model_name in acq_model_names:
        results = []
        results_path = "./results/q_strategy/" + dataset_name + '/' + acq_model_name + '/' + config_sig
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        acq_model = get_acquisition_model(acq_model_name, config)
        all_sample_sizes = []
        all_mses = []
        for j in range(n_runs):
            config['split_num'] = j
            dataset = MovieLensDataset(dataset_name, config) 
            best_mse = best_possible_MSE(dataset, rank_opt)
            worst_mse = worst_possible_MSE(dataset, rank_opt)
            best_val_mse = best_possible_MSE(dataset, rank_opt, "val")
            worst_val_mse = worst_possible_MSE(dataset, rank_opt, "val")

            mses = []
            sample_sizes = [] 
            all_test_mses = []            
            curve_models = [get_curve_model(cm_name, config) for cm_name in curve_model_names]
            i = 0
            while i < n_acquisitions: 
                acquired_idxs = acq_model.acquire_features(dataset, batch_size)
                dataset.add(acquired_idxs)
                sample_sizes, mses, test_mses = subsample_idxs(dataset, config, sample_sizes, mses)


                n_init = len(dataset.init_idxs)
                n_available = dataset.n_available_idxs()
                n_observable = dataset.n_observable_idxs()
                pred = get_SVD_pred(dataset, rank_opt, 
                                    dataset.available_idxs(),
                                    dataset.test_idxs) 
                mse_scores = all_MSE(dataset.X, pred, dataset.test_idxs, 
                               worst_mse, best_mse)
                micro_mse, macro_mse, micro_pct, macro_mses = mse_scores
                
                val_pred = get_SVD_pred(dataset, rank_opt, 
                                    dataset.available_idxs(),
                                    dataset.val_idxs) 
                mse_scores_val = all_MSE(dataset.X, val_pred, dataset.val_idxs, 
                               worst_val_mse, best_val_mse)
                micro_mse_val, macro_mse_val, micro_pct_val, _ = mse_scores_val
                for cm in curve_models:
                    if cm.name in finished_cms:
                        continue

                    if i == 0:
                        cm.fit(sample_sizes, mses)
                    elif 'NLS' in cm.name:
                        cm.fit(sample_sizes, mses)
                    stop, pred_pct = cm.stop_condition(goal_pct, n_init, n_available, n_observable)
                    if stop:
                        finished_cms.append(cm.name)
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

                i += batch_size
            all_sample_sizes.append(sample_sizes)
            all_mses.append(mses)
            all_test_mses.append(test_mses)
        np.savetxt(results_path + '/sample_sizes', all_sample_sizes)
        np.savetxt(results_path + '/mses', all_mses)
        np.savetxt(results_path + '/test_mses', all_test_mses)



mltiny_config = {'n_runs': 5, 'checks': False, 'init_pct': .1, 
        'test_pct': .4, 'init_mode': 'uniform', 'batch_size': 20000,
        'step_size': 4000, 't': 0,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 200000,
          'feat_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .85} 

mluniform_config = {'n_runs': 5, 'checks': False, 'init_pct': .1, 
        'test_pct': .4, 'init_mode': 'uniform', 'batch_size': 100000,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 1000000,
        'step_size': 20000, 't': 0,
          'feat_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .8} 

gltiny_config = {'n_runs': 5, 'checks': False, 'init_pct': .1, 
        'test_pct': .4, 'init_mode': 'uniform', 'batch_size': 9400,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 94000,
        'step_size': 1880, 't': 0,
          'feat_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .8} 

gl_config = {'n_runs': 5, 'checks': False, 'init_pct': .1, 
        'test_pct': .4, 'init_mode': 'uniform', 'batch_size': 19000,
        'rank_opt': 30, 'split_num': 1, 'n_acquisitions': 190000,
        'step_size': 3800, 't': 0, 
          'feat_pct': .5, 'user_pct': .5, 'l': 0, 'global_goal': .8} 


dataset_names = [('ml-20m-tiny', mltiny_config), ('ml-20m-uniform', mluniform_config)]
dataset_names = [('ml-20m-tiny', mltiny_config), ('gl-tiny', gltiny_config),
                 ('ml-20m-uniform', mluniform_config),
                 ('gl', gl_config)]
#dataset_names = [('gl-tiny', gltiny_config)]
#dataset_names = [('gl', gl_config)]
#dataset_names = [('ml-20m-uniform', mluniform_config)]
#dataset_names = [('ml-20m-tiny', mltiny_config)]
# to run multiple experiments, create multiple configs
init_modes = ['user_subset', 'item_subset']
global_goals = [.8, .85, .9, .95]
global_goals = [.8]
for dataset_name, config in dataset_names:
    for init_mode in init_modes:
        for global_goal in global_goals:
            config['init_mode'] = init_mode
            config['global_goal'] = global_goal
            run_mse_curve_expmt(dataset_name, config)














