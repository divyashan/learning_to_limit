import numpy as np
import pandas as pd
import seaborn as sns
import sys
import pdb
from tqdm import tqdm 

import matplotlib.pyplot as plt
from curve_models import NLLS, NLLS_w, NLLS_rse, power_law, CurveModel, linearized_power_law
from curve_models import NLLS_three_param, power_law_three_param, power_law_exp_three_param
from curve_models import BrokenCurve
from utils.util import calc_pct

mode = sys.argv[1]
if mode == 'random':
    gl_expmt = '0.1_5_False_uniform_30_0.5_0.5_5800_29000_290001'
    ml_expmt = '0.1_5_False_uniform_30_0.5_0.5_20000_100000_1000001'
    mltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_5250_21000_213973'
    gltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_1880_9400_94001'

    expmts = [('gl', gl_expmt), ('gl-tiny', gltiny_expmt), ('ml-20m-uniform', ml_expmt), ('ml-20m-tiny', mltiny_expmt)]
    expmts = [('ml-20m-tiny', mltiny_expmt)]
elif mode == 'weighted':
    mltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_5250_21000_213973'
    gltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_1880_9400_94001'
    gl_expmt = '0.1_5_False_uniform_30_0.5_0.5_5800_29000_290001'
    ml_expmt = '0.1_5_False_uniform_30_0.5_0.5_20000_100000_1000001'
    expmts = [('ml-20m-tiny', mltiny_expmt), ('gl-tiny', gltiny_expmt)]
    expmts = [('ml-20m-tiny', mltiny_expmt), ('gl-tiny', gltiny_expmt), ('gl', gl_expmt),
              ('ml-20m-uniform', ml_expmt)]
    #expmts = [('gl', gl_expmt)]
    #expmts = [('ml-20m-uniform', ml_expmt)]
    #expmts = [('ml-20m-tiny', mltiny_expmt)]
    #expmts = [('gl-tiny', gltiny_expmt)]
elif mode == 'qbc':
    mltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_5250_21000_213973'
    gltiny_expmt = '0.1_5_False_uniform_30_0.5_0.5_1880_9400_94001'
    ml_expmt = '0.1_5_False_uniform_30_0.5_0.5_20000_100000_1000001'
    #gl_expmt = '0.1_5_False_uniform_30_0.5_0.5_1880_9400_94001'
    expmts = [('ml-20m-tiny', mltiny_expmt), ('gl-tiny', gltiny_expmt)]
    
    #expmts = [('ml-20m-uniform', ml_expmt)]
              
    #expmts = [('gl-tiny', gltiny_expmt)]    
elif mode == 'step_sizes':
    step_size_expmts = ['0.1_5_False_uniform_30_0.5_0.5_1050_5250_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_2100_10500_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_4200_21000_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_6300_31500_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_8400_42000_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_10500_52500_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_12600_63000_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_14700_73500_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_16800_84000_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_18900_94500_210000', 
                        '0.1_5_False_uniform_30_0.5_0.5_21000_105000_210000'] 
    expmts = [('ml-20m-tiny', expmt) for expmt in step_size_expmts]

    step_size_expmts = ['0.1_5_False_uniform_30_0.5_0.5_1450_7250_290000', 
                         '0.1_5_False_uniform_30_0.5_0.5_2900_14500_290000', 
                         '0.1_5_False_uniform_30_0.5_0.5_5800_29000_290000', 
                         '0.1_5_False_uniform_30_0.5_0.5_8700_43500_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_11600_58000_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_14500_72500_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_17400_87000_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_20300_101500_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_23200_116000_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_26100_130500_290000', 
                        '0.1_5_False_uniform_30_0.5_0.5_29000_145000_290000']
    expmts = [('gl', expmt) for expmt in step_size_expmts] 
    
    step_size_expmts = ['0.1_5_False_uniform_30_0.5_0.5_5000_10000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_10000_20000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_20000_40000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_30000_60000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_40000_80000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_50000_100000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_60000_120000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_70000_140000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_80000_160000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_90000_180000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_100000_200000_1000000']
    step_size_expmts = ['0.1_5_False_uniform_30_0.5_0.5_10000_20000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_20000_40000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_30000_60000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_40000_80000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_50000_100000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_60000_120000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_70000_140000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_80000_160000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_90000_180000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_100000_200000_1000000']
    step_size_expmts = ['0.1_5_False_uniform_30_0.5_0.5_10000_20000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_20000_40000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_40000_80000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_60000_120000_1000000', 
                        '0.1_5_False_uniform_30_0.5_0.5_100000_200000_1000000']
    expmts = [('ml-20m-uniform', expmt) for expmt in step_size_expmts]

elif mode == 'ds_sizes':
    ds_size_expmts = ['0.1_5_False_uniform_30_0.5_0.5_855_4279_42794', 
                             '0.1_5_False_uniform_30_0.5_0.5_1283_6419_64191', 
                             '0.1_5_False_uniform_30_0.5_0.5_1711_8558_85589', 
                             '0.1_5_False_uniform_30_0.5_0.5_2139_10698_106986', 
                             '0.1_5_False_uniform_30_0.5_0.5_2567_12838_128383', 
                             '0.1_5_False_uniform_30_0.5_0.5_2995_14978_149781', 
                             '0.1_5_False_uniform_30_0.5_0.5_3423_17117_171178', 
                             '0.1_5_False_uniform_30_0.5_0.5_3851_19257_192575', 
                             '0.1_5_False_uniform_30_0.5_0.5_4279_21397_213973']
    expmts = [('ml-20m-tiny', expmt) for expmt in ds_size_expmts]


scale_dict = {'ml-20m-tiny': 30513, 'ml-20m-uniform': 158408, 
              'gl': 41870, 'gl-tiny':13713}

for dataset_name, expmt in tqdm(expmts):
    print(expmt)
    scale = scale_dict[dataset_name]
    afa_name = 'Random'
    scale = 1
    prefix = mode
    if mode == 'weighted':
        afa_name = 'Weighted'
    elif mode == 'qbc':
        afa_name = 'QBC'
    elif  mode == 'step_sizes':
        prefix = 'step_size_' + expmt.split('_')[-3] + '_' + expmt.split('_')[-1] 
    results_path = './results/forecasting/' + dataset_name + '/' + afa_name + '/' + expmt +'/'
    all_ss = np.loadtxt(results_path + 'sample_sizes')
    all_mses = np.loadtxt(results_path + 'mses')/scale
    all_test_mses = np.loadtxt(results_path + 'test_mses')/scale
    data = pd.read_csv(results_path + 'results_df')
    n_init = data['n_init'].iloc[0]
    n_observable = data['n_observable'].iloc[0]
    results = []
    n_runs =  5 
    n_unique_runs = 5 
    for j in range(n_runs):
        print("RUN: ", j)
        worst_mse = data[data['run'] == j % n_unique_runs]['worst_mse'].iloc[0]/scale
        best_mse = data[data['run'] == j %n_unique_runs]['best_mse'].iloc[0]/scale
        ss = all_ss[j % n_unique_runs]
        mses = all_mses[j % n_unique_runs]
        test_mses = all_test_mses[j %n_unique_runs]
        n_pts = len(np.where(ss) > n_init)
        for i, (size, mse) in enumerate(zip(ss, mses)):
            if size < n_init:
                continue
            if len(ss[:i+1]) < 3:
                # Depending on the step size, perhaps not enough data  
                continue 
                            
            pct_available = (size-n_init)/(n_observable-n_init)
            pct_observable = size/n_observable
            if run == 0:
                print(pct_observable, size, n_observable)
            true_pct = (worst_mse - mse)/(worst_mse - best_mse)
            results.append({'pred_best': best_mse, 'pred_worst': worst_mse, 'pred_curr': mse,
                            'pred_pct': true_pct, 'true_curr': mse, 'cm': 'True', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j, 'feature_acq': afa_name,
                            'pct_observable': pct_observable,
                            'true_best': best_mse})

            # Linear
            results.append({'pred_best': best_mse, 'pred_worst': worst_mse, 'pred_curr': mse,
                            'pred_pct': (size-n_init)/(n_observable-n_init), 'true_curr': mse, 
                            'cm': 'Linear', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j, 'feature_acq': afa_name,
                            'pct_observable': pct_observable,
                           'true_best': best_mse})

            # initial
            if mode != 'step_sizes':
                nlls = NLLS_w(power_law)
                stop_pt = min(np.where(ss > n_init)[0])
                nlls.fit(ss[:stop_pt], mses[:stop_pt])
                pred_worst = nlls.pred(n_init)
                pred_best = nlls.pred(n_observable)
                pred_curr = nlls.pred(size)
                pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
                results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                                'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_initial', 'true_pct': true_pct, 
                                'pct_available': pct_available, 'run': j, 'feature_acq': afa_name,
                                'pct_observable': pct_observable,
                               'true_best': best_mse})

            # this baseline fits the power law curve to all points
            nlls = NLLS(power_law)
            nlls.fit(ss[:i+1], mses[:i+1])
            pred_worst = nlls.pred(n_init)
            pred_best = nlls.pred(n_observable)
            pred_curr = nlls.pred(size)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j, 'feature_acq': afa_name,
                            'pct_observable': pct_observable,
                           'true_best': best_mse})
            
            # this method fits the weighted power law curve to all pts
            nlls_w = NLLS_w(power_law)
            nlls_w.fit(ss[:i+1], mses[:i+1])
            pred_worst = nlls_w.pred(n_init)
            pred_best = nlls_w.pred(n_observable)
            pred_curr = nlls_w.pred(size)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_w', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j, 'feature_acq': afa_name, 
                            'pct_observable': pct_observable,
                           'true_best': best_mse})
            
            # this method fits the three parameter power law curve
            nlls_3p = NLLS_three_param(power_law_three_param, "power_law_3p")
            try:
                nlls_3p.fit(ss[:i+1], mses[:i+1]) 
            except:
                print("\n Expmt: ", expmt , ", Run: ", j, ", Index: ", i)
                pdb.set_trace()
            
            pred_worst = nlls_3p.pred(n_init)
            pred_best = nlls_3p.pred(n_observable)
            pred_curr = nlls_3p.pred(size)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_power_law_3P', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j, 'feature_acq': afa_name, 
                           'true_best': best_mse})
            
            nlls_exp_3p = NLLS_three_param(power_law_exp_three_param, "power_law_exp_3p")
            nlls_exp_3p.fit(ss[:i+1], mses[:i+1])
            pred_worst = nlls_exp_3p.pred(n_init)
            pred_best = nlls_exp_3p.pred(n_observable)
            pred_curr = nlls_exp_3p.pred(size)
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'NLS_power_law_exp_3P', 
                            'true_pct': true_pct, 'feature_acq': afa_name, 
                            'pct_available': pct_available, 'run': j, 
                            'pct_observable': pct_observable,
                           'true_best': best_mse})
            
            broken = BrokenCurve(power_law, "broken_power_law")
            broken.fit(ss[:i+1], mses[:i+1])
            pred_worst = broken.pred(n_init)
            pred_best = broken.pred(n_observable)
            pred_curr = broken.pred(size) 
            pred_pct = calc_pct(pred_best, pred_worst, pred_curr)
            results.append({'pred_best': pred_best, 'pred_worst': pred_worst, 'pred_curr': pred_curr,
                            'pred_pct': pred_pct, 'true_curr': mse, 'cm': 'broken', 'true_pct': true_pct, 
                            'pct_available': pct_available, 'run': j, 'feature_acq': afa_name,
                            'pct_observable': pct_observable,
                           'true_best': best_mse})
            
        pd.DataFrame(results).to_csv('./results/forecasting/' + dataset_name + '/' + prefix + '_pred_performance')
