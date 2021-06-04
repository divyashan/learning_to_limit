import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm
import pdb
from scipy.signal import savgol_filter

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

def power_law_three_param(x, a, b, c):
    return (b*x**(-a) + c)

def power_law_exp_three_param(x, a, b, c):
    return (x**(-a))*np.exp(-b*x) + c

def linearized_power_law(x, a, b):
    return np.exp(b + a*np.log(x))

def exponential_law(x, a, b):
    return (a*b**(1/x))

class CurveModel(object):

    def __init__(self, fit_f, init_params=None, name="cm"):
        self.f = fit_f
        self.p = {}
        self.name = name
        self.xscale = 1
        self.yscale = 1

    def fit(self, sample_sizes, sample_mses):
        pass

    def pred(self, n_samples):
        return self.yscale*self.f(n_samples/self.xscale, **self.p)

    def slope(self, n_samples, delta=.1):
        return (self.pred(n_samples + delta) - self.pred(n_samples - delta))/(2*delta)

    def weight_list(self, sample_sizes):
        return [1/(x**.5) for x in sample_sizes]  
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

class NLLS_three_param(CurveModel):
    def fit(self, sample_sizes, sample_mses, p0=[0, 0, 0]):
        sigma = self.weight_list(sample_sizes)
        self.yscale = np.max(sample_mses)
        self.xscale = np.max(sample_sizes)
        #  p0[2] = np.min(sample_mses)/self.scale
        scaled_sample_sizes = [i/self.xscale for i in sample_sizes]
        scaled_sample_mses = [i/self.yscale for i in sample_mses]
        popt, pcov = curve_fit(self.f, xdata=scaled_sample_sizes, ydata=scaled_sample_mses, absolute_sigma=True, 
                               maxfev=12000, sigma=sigma)

        self.p['a'] = popt[0]
        self.p['b'] = popt[1]
        self.p['c'] = popt[2]
        

class NLLS_w(CurveModel):
    def fit(self, sample_sizes, sample_mses):
        sigma = self.weight_list(sample_sizes)
        self.yscale = np.max(sample_mses)
        self.xscale = np.max(sample_sizes)
        scaled_sample_sizes = [i/self.xscale for i in sample_sizes]
        scaled_sample_mses = [i/self.yscale for i in sample_mses]
        popt, pcov = curve_fit(self.f, xdata=scaled_sample_sizes, ydata=scaled_sample_mses, p0=[0,0], sigma=sigma,
                               absolute_sigma=True)
        self.p['a'] = popt[0]
        self.p['b'] = popt[1]
        self.pcov = pcov

class BrokenCurve(CurveModel):

    def __init__(self, fit_f, name):
        self.f = fit_f
        self.p1 = {}
        self.p2 = {}
        self.name = name
        self.yscale = 1
        self.xscale = 1
        self.thresh = 0
        
    def pred(self, n_samples):
        if n_samples < self.thresh:
            return self.f(n_samples/self.xscale, **self.p1)*self.yscale
        else:
            return self.f(n_samples/self.xscale, **self.p2)*self.yscale
        
    def fit(self, sample_sizes, sample_mses):
        # if number of observations < 5, just fit one curve
        #smoothed_sample_mses = savgol_filter(sample_mses, 5, 2)
        smoothed_sample_mses = sample_mses
        sigma = self.weight_list(sample_sizes)
        self.yscale = np.max(smoothed_sample_mses)
        self.xscale = np.max(sample_sizes)
        scaled_sample_sizes = [i/self.xscale for i in sample_sizes]
        scaled_sample_mses = [i/self.yscale for i in smoothed_sample_mses]
        if len(sample_sizes) < 5:
            popt, pcov = curve_fit(self.f, xdata=scaled_sample_sizes, sigma=sigma,
                                   ydata=scaled_sample_mses, p0=[0,0], absolute_sigma=True)
            self.p1['a'] = popt[0]
            self.p1['b'] = popt[1]
            self.p2['a'] = popt[0]
            self.p2['b'] = popt[1]
            self.p1cov, self.p2cov = pcov, pcov
        else:
            thresh_idxs = [2+i for i in list(range(len(sample_sizes)-4))]
            effect_sizes = []
            for i in thresh_idxs:
                first_ss, first_mses = scaled_sample_sizes[:i], scaled_sample_mses[:i]
                second_ss, second_mses = scaled_sample_sizes[i:], scaled_sample_mses[i:]
                first_sigma = self.weight_list(first_ss)
                second_sigma = self.weight_list(second_ss)

                try:
                    popt, pcov = curve_fit(self.f, xdata=first_ss, sigma = first_sigma,
                                       ydata=first_mses, p0=[0,0], absolute_sigma=True)
                    p1_a = popt[0]
                    popt, pcov = curve_fit(self.f, xdata=second_ss, 
                                           ydata=second_mses, sigma=second_sigma,
                                           p0=[0,0], absolute_sigma=True, maxfev=2000)
                    p2_a = popt[0]
                    effect_sizes.append(abs(p1_a - p2_a))
                except:
                    effect_sizes.append(0)
            thresh_idx = thresh_idxs[np.argmax(effect_sizes)]
            self.thresh = sample_sizes[thresh_idx]
            first_ss, first_mses = scaled_sample_sizes[:thresh_idx], scaled_sample_mses[:thresh_idx]
            second_ss, second_mses = scaled_sample_sizes[thresh_idx:], scaled_sample_mses[thresh_idx:]
            first_sigma = self.weight_list(first_ss)
            second_sigma = self.weight_list(second_ss)
            popt, pcov = curve_fit(self.f, xdata=first_ss, sigma=first_sigma,
                               ydata=first_mses, p0=[0,0], absolute_sigma=True, maxfev=2000)
            self.p1['a'] = popt[0]
            self.p1['b'] = popt[1]
            self.p1cov = pcov
            
            popt, pcov = curve_fit(self.f, xdata=second_ss, sigma=second_sigma,
                               ydata=second_mses, p0=[0,0], absolute_sigma=True, maxfev=2000)
            self.p2['a'] = popt[0]
            self.p2['b'] = popt[1]
            self.p2cov = pcov
            
        
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
        self.yscale = 1
        self.xscale = 1
    def fit(self, sample_sizes, sample_mses):
        log_x = [np.log(i) for i in sample_sizes]
        log_y = [np.log(i) for i in sample_mses]
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
        self.f_test = result.f_test
