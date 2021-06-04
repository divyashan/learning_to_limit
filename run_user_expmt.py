import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append("../")
from tqdm import tqdm
from curve_models import power_law, BrokenCurve
import pdb
def calc_pct(pred_best, pred_worst, pred_curr):
    return (pred_worst - pred_curr)/(pred_worst - pred_best)

dataset_titles = {'gl': 'GoogleLocal-L', 'gl-tiny': 'GoogleLocal-S', 'ml-20m-tiny': 'MovieLens-S',
                 'ml-20m-uniform': 'MovieLens-L'}
gl_expmt = '0.1_5_False_uniform_30_0.5_0.5_5800_29000_290001'
ml_expmt = '0.1_5_False_uniform_30_0.5_0.5_20000_100000_1000001'
mltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_5250_21000_213973'
gltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_1880_9400_94001'

#expmts = [('gl', gl_expmt), ('gl-tiny', gltiny_expmt), ('ml-20m-uniform', ml_expmt)] 
expmts = [('ml-20m-tiny', mltiny_expmt)]
for dataset_name, expmt in expmts:
    results_path = './results/forecasting/' + dataset_name + '/Random/' + expmt +'/'
    print(dataset_name)
    all_ss = np.loadtxt(results_path + 'sample_sizes')
    all_mses = np.loadtxt(results_path + 'mses')
    all_test_mses = np.loadtxt(results_path + 'test_mses')
    all_macro_mses = np.load(results_path + 'macro_mses.npy')

    data = pd.read_csv(results_path + 'results_df')
    n_init = data['n_init'].iloc[0]
    n_observable = data['n_observable'].iloc[0]
    results = []
    user_results = []
    n_runs = 1 
    pcts_available = []

    for j in tqdm(range(n_runs)):
        worst_mse = data[data['run'] == j]['worst_mse'].iloc[0]
        best_mse = data[data['run'] == j]['best_mse'].iloc[0]
        ss = all_ss[j]
        mses = all_mses[j]
        test_mses = all_test_mses[j]
        macro_mses = all_macro_mses[j]
        n_pts = len(np.where(ss) > n_init)
        for i, (size, mse) in tqdm(enumerate(zip(ss, mses))):
            if size < n_init:
                continue
            # this method fits a beginning power law curve and an end one
            pct_available = (size-n_init)/(n_observable-n_init)
            pcts_available.append(pct_available)
            true_pct = (worst_mse - mse)/(worst_mse - best_mse)
            
            broken = BrokenCurve(power_law, 'broken')
            broken.fit(ss[:i+1], mses[:i+1])

            pred_worst = broken.pred(n_init)
            pred_best = broken.pred(n_observable)
            pred_curr = broken.pred(size)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'broken', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})
            
            n_users = macro_mses.shape[1]
            for k in range(n_users):
                user_macro_mses = macro_mses[:,k]
                broken = BrokenCurve(power_law, 'broken')
                broken.fit(ss[:i+1], user_macro_mses[:i+1])
                pred_worst = broken.pred(n_init)
                pred_best = broken.pred(n_observable)
                pred_curr = broken.pred(size)
                pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
                true_pct = calc_pct(user_macro_mses[i], user_macro_mses[5], 
                                    user_macro_mses[-1])
                user_results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 
                                     'pred_curr': pred_curr, 'cm': 'broken',
                                     'pred_pct': pred_pct, 'pct_available': pct_available,
                                     'run': j, 'user': k, 'true_best': user_macro_mses[-1],
                                     'true_worst': user_macro_mses[5],
                                     'true_pct': true_pct})
                user_results.append({'pred_best': user_macro_mses[-1], 
                                     'pred_worst': user_macro_mses[5], 
                                     'pred_curr': user_macro_mses[i], 'cm': 'True',
                                     'pred_pct': true_pct, 'true_best': user_macro_mses[-1],
                                     'true_worst': user_macro_mses[5],
                                     'true_pct': true_pct, 'pct_available': pct_available,
                                     'run': j, 'user': k})
        pdb.set_trace()
    pd.DataFrame(results).to_csv('./results/forecasting/' + dataset_name + '/pred_performance')
    pd.DataFrame(user_results).to_csv('./results/forecasting/' + dataset_name + '/user_pred_performance')
