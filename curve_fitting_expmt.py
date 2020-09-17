import numpy as np
import pandas as pd
import seaborn as sns
import sys

import matplotlib.pyplot as plt
from curve_models import NLLS, NLLS_w, NLLS_rse, power_law, CurveModel, linearized_power_law
from utils.util import calc_pcti

gl_expmt = '0.1_5_False_uniform_30_0.5_0.5_5800_29000_290001'
ml_expmt = '0.1_5_False_uniform_30_0.5_0.5_20000_100000_1000001'
mltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_5250_21000_213973'
gltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_1880_9400_94001'

expmts = [('gl', gl_expmt), ('gl-tiny', gltiny_expmt), ('ml-20m-uniform', ml_expmt), ('ml-20m-tiny', mltiny_expmt)]
for dataset_name, expmt in expmts:
    results_path = '../results/forecasting/' + dataset_name + '/Random/' + expmt +'/'
    all_ss = np.loadtxt(results_path + 'sample_sizes')
    all_mses = np.loadtxt(results_path + 'mses')
    all_test_mses = np.loadtxt(results_path + 'test_mses')

    data = pd.read_csv(results_path + 'results_df')
    n_init = data['n_init'].iloc[0]
    n_observable = data['n_observable'].iloc[0]
    results = []
    n_runs = 5
    for j in range(n_runs):
        worst_mse = data[data['run'] == j]['worst_mse'].iloc[0]
        best_mse = data[data['run'] == j]['best_mse'].iloc[0]
        ss = all_ss[j]
        mses = all_mses[j]
        test_mses = all_test_mses[j]
        n_pts = len(np.where(ss) > n_init)
        for i, (size, mse) in enumerate(zip(ss, mses)):
            if size < n_init:
                continue
            
            pct_available = (size-n_init)/(n_observable-n_init)
            true_pct = (worst_mse - mse)/(worst_mse - best_mse)
            results.append({'pred_best': best_mse, 'pred_worst': worst_mse, 'pred_curr': mse,
                            'pred_pct': true_pct, 'true_curr': mse, 'cm': 'True', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})

            # Linear
			results.append({'pred_best': best_mse, 'pred_worst': worst_mse, 'pred_curr': mse,
                            'pred_pct': (size-n_init)/(n_observable-n_init), 'true_curr': mse, 
                            'cm': 'Linear', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})

            # initial
            nlls = NLLS(power_law)
            stop_pt = min(np.where(ss > n_init)[0])
            nlls.fit(ss[:stop_pt], mses[:stop_pt])
            pred_worst = nlls.f(n_init, **nlls.p)
            pred_best = nlls.f(n_observable, **nlls.p)
            pred_curr = nlls.f(size, **nlls.p)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_initial', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})

            # this baseline fits the power law curve to all points
            nlls = NLLS(power_law)
            nlls.fit(ss[:i+1], mses[:i+1])
            pred_worst = nlls.f(n_init, **nlls.p)
            pred_best = nlls.f(n_observable, **nlls.p)
            pred_curr = nlls.f(size, **nlls.p)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})

            # this method fits the weighted power law curve to all pts
            nlls_w = NLLS_w(power_law)
            nlls_w.fit(ss[:i+1], mses[:i+1])
            pred_worst = nlls_w.f(n_init, **nlls_w.p)
            pred_best = nlls_w.f(n_observable, **nlls_w.p)
            pred_curr = nlls_w.f(size, **nlls_w.p)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_w', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})
        
		    # this method fits a beginning power law curve and an end one
		    start_cm = NLLS_rse(linearized_power_law)
            start = 0
            end = int(i/2)
            start_cm.fit(ss[start:end], mses[start:end])

            end_cm = NLLS_rse(linearized_power_law)
            start = int(i/2)
            end_cm.fit(ss[start:i+1], mses[start:i+1])

            pred_worst = start_cm.f(n_init, **start_cm.p)
            pred_best = end_cm.f(n_observable, **end_cm.p)
            pred_curr = end_cm.f(size, **end_cm.p)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_mix', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j})

        pd.DataFrame(results).to_csv('../results/forecasting/' + dataset_name + '/pred_performance')
