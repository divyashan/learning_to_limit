import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
import pdb

def get_curve_model(name):
    """ Returns curve model given name """ 
    if name == 'NLS':
        return NLLS(power_law, name='NLS')
    elif name == 'NLS_w':
        return NLLS_w(power_law, name='NLS_w')
    elif name == 'NLS_rse':
        return NLLS_rse(linearized_power_law, name='NLS_rse')
    elif name == 'initial':
        return NLLS(power_law, name='Initial')
    elif name == 'linear':
        return CurveModel(lambda x: x, name='Linear')

# Specific curve models i
def power_law(x, a, b):
    return (b*x**(-a))

def linearized_power_law(x, a, b):
    return np.exp(b + a*np.log(x))

def exponential_law(x, a, b):
    return (a*b**(1/x))

class CurveModel(object):

    def __init__(self, fit_f, init_params=None, name="cm"):
        self.f = fit_f
        self.p = {}
        self.name = name
        pass

    def fit(self, sample_sizes, sample_mses):
        pass

    def pred(self, n_samples):
        return self.f(n_samples, **self.p)
        
    def stop_condition(self, pct, n_init, n_samples, n_total):
        pred_progress = self.pred(n_init)-self.pred(n_samples)
        pred_total = self.pred(n_init) - self.pred(n_total)
        pred_pct = pred_progress/pred_total
        print("NAME: ", self.name)
        print("# init: ", n_init, "# samples", n_samples, "# total", n_total)
        print("pred pct", pred_pct, "goal ", pct)
        return (pred_pct > pct), pred_pct

class NLLS(CurveModel):
    def fit(self, sample_sizes, sample_mses):
        popt, pcov = curve_fit(self.f, xdata=sample_sizes, ydata=sample_mses, p0=[0,0], absolute_sigma=True)
        self.p['a'] = popt[0]
        self.p['b'] = popt[1]
        self.pcov = pcov

class NLLS_w(CurveModel):
    def weight_list(self, sample_sizes):
        return [1/(x**.5) for x in sample_sizes]

    def fit(self, sample_sizes, sample_mses):
        sigma = self.weight_list(sample_sizes)
        popt, pcov = curve_fit(self.f, xdata=sample_sizes, ydata=sample_mses, p0=[0,0], sigma=sigma,
                               absolute_sigma=True)
        self.p['a'] = popt[0]
        self.p['b'] = popt[1]
        self.pcov = pcov

class NLLS_el(CurveModel):

    def fit(self, sample_sizes, sample_mses):
        popt, pcov = curve_fit(self.f, xdata.sample_sizes, ydata=sample_mses, p0=[0,0], sigma=sigma, absolute_sigma=True)
        self.p['a'] = popt[0]
        self.p['b'] = popt[1]
        self.pcov = pcov

class NLLS_rse(CurveModel):
    def __init__(self, fit_f, init_params=None, name='cm'):
        self.f = fit_f
        self.p = {}
        self.name = name

    def fit(self, sample_sizes, sample_mses):
        ss, mses = [], []
        log_x = [np.log(i) for i in ss]
        log_y = [np.log(i) for i in mses]
        log_x = sm.add_constant(log_x)
        result = sm.OLS(log_y, log_x).fit()
        result = result.get_robustcov_results('HC3')
        self.p['a'] = result.params[1]
        self.p['b'] = result.params[0]
        self.pcov = result.bse
        self.rsq = result.rsquared
        self.rsq_adj = result.rsquared_adj
        self.pvalues = result.pvalues
        self.ll = result.llf
        self.aic = result.aic
        self.bic = result.bic
        self.f_pval = result.f_pvalue[0][0]
        self.f_test = result.f_test
