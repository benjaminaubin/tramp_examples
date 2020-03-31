
from utils.plot.import_library_plot import *
from utils.functions import save_object, load_object, mean_squared_error, sort_lists
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from tramp.priors import GaussBernouilliPrior, GaussianPrior, MAP_LaplacePrior
from tramp.likelihoods import GaussianLikelihood
from tramp.channels import GaussianChannel, LinearChannel, AnalyticalLinearChannel
from tramp.ensembles import GaussianEnsemble, MarchenkoPasturEnsemble
from tramp.algos import ConstantInit, NoisyInit, CustomInit
from tramp.variables import SISOVariable as V, SILeafVariable as O, MILeafVariable, SIMOVariable
from tramp.algos import ExpectationPropagation, EarlyStoppingEP, TrackEvolution, TrackErrors, StateEvolution

from sklearn.linear_model import Lasso, LassoCV
import pymc3 as pm

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def plot_sparse_linear_benchmark(dics_Bayes, save_fig=False, block=True):
    n = 2
    fig, axs = plt.subplots(1, 2, figsize=(6*n, 6))
    tab_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams['lines.linewidth'] = 1.75
    i = 0
    for key in dics_Bayes.keys():
        tab_marker = {'Pymc3Bayes': 'o', 'TrampBayes': 'x--',
                      'LassoCV': '^--', 'TrampSE': '-'}
        tab_color = {
            'Pymc3Bayes': tab_col[1], 'TrampBayes': tab_col[0], 'LassoCV': tab_col[2], 'TrampSE': 'k'}
        tab_label = {'Pymc3Bayes': r'PyMC3',
                     'TrampBayes': r'Tramp', 'LassoCV': 'Lasso', 'TrampSE': 'Bayes opt.'}

        tab_x = np.array(dics_Bayes[key]['tab_alpha'])
        tab_y = np.array(dics_Bayes[key]['tab_mse'])

        if key == 'Pymc3Bayes':
            tab_y += 1e-9
        tab_x, tab_y = sort_lists(tab_x, tab_y)
        axs[0].plot(tab_x, tab_y / 0.05, tab_marker[key],
                    c=tab_color[key], label=tab_label[key])

        tab_x = np.array(dics_Bayes[key]['tab_alpha'])
        tab_y = np.array(dics_Bayes[key]['tab_time'])

        tab_x, tab_y = sort_lists(tab_x, tab_y)
        if key in ['TrampBayes', 'Pymc3Bayes']:
            tab_y = savgol_filter(tab_y.ravel(), 9, 1)
        if key not in ['TrampSE']:
            axs[1].plot(tab_x, tab_y, tab_marker[key],
                        c=tab_color[key])

        i += 1
    i = 0

    axs[0].legend()
    axs[0].set_xlim([0, 1])
    axs[1].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[1].set_yscale('log')

    """ Ticks Label   """
    axs[0].set_xlabel(r'$\alpha$')
    axs[0].set_ylabel(r'MSE / $\rho$')
    axs[1].set_ylabel(r'Time $(s)$')
    axs[1].set_xlabel(r'$\alpha$')

    fig.tight_layout()

    """ Save   """
    if save_fig:
        dir_fig = 'Figures/'
        os.makedirs(dir_fig) if not os.path.exists(dir_fig) else 0
        file_name = f'{dir_fig}/benchmark_sparse_linear.pdf'

        plt.tight_layout()
        plt.savefig(file_name, format='pdf', dpi=1000,
                    bbox_inches="tight", pad_inches=0.1)

    if block:
        plt.show(block=False)
        input('Press enter to continue')
        plt.close()
    else:
        plt.close()


class SparseTeacher():
    def __init__(self, N, alpha, rho, Delta):
        self.N = N
        self.alpha = alpha
        self.M = int(self.alpha * self.N)
        self.rho = rho
        self.Delta = Delta
        self.model = self.build_model()

    def build_model(self):
        self.prior = GaussBernouilliPrior(size=(self.N,), rho=self.rho)
        ensemble = GaussianEnsemble(self.M, self.N)
        self.A = ensemble.generate()
        model = self.prior @ V(id="x") @ LinearChannel(W=self.A) @ V(
            id='z') @ GaussianChannel(var=self.Delta) @ O(id="y")
        model = model.to_model()
        return model


class compare_tramp_lasso_pymc3(object):
    def __init__(self, alpha=0.5, rho=0.1, Delta=1e-2, N=1000):
        self.name = 'sparse_linear'
        self.alpha = alpha
        self.rho = rho
        self.Delta = Delta
        self.N = N

    def build_teacher(self, N, alpha, rho, Delta, seed=None):
        if seed is not None:
            np.random.seed(seed)
        teacher = SparseTeacher(
            N=N, alpha=alpha, rho=rho, Delta=Delta)
        return teacher

    def generate_sample(self, teacher):
        sample = (teacher.model).sample()
        data = {'x_true': sample['x'], 'y': sample['y'], 'A': teacher.A}
        return data

    def initialization(self, N, alpha, rho, Delta, seed=None):
        self.teacher = self.build_teacher(N, alpha, rho, Delta, seed)
        self.data = self.generate_sample(self.teacher)

    def average_reconstruction(self, algos, n_samples=5, plot=False, verbose=False, seed=None, coef_avg=1):
        df_mse = pd.DataFrame({algo: [] for algo in algos.keys()})
        df_time = pd.DataFrame({algo: [] for algo in algos.keys()})
        i = 0
        while i < n_samples:
            if int(i/n_samples * 100) % 10 == 0:
                print(f'{int(i/n_samples * 100)}%')
            res_mse, res_time = self.single_reconstruction(
                algos, plot=plot, seed=i, verbose=verbose)
            if res_mse is not None:
                i = i + 1
                df_mse = df_mse.append(pd.Series([res_mse[key]
                                                  for key in res_mse.keys()], index=res_mse.keys()), ignore_index=True)
                df_time = df_time.append(pd.Series([res_time[key]
                                                    for key in res_time.keys()], index=res_time.keys()), ignore_index=True)
        df_mse_sorted = (df_mse.sort_values(
            by=list(algos.keys())[-1]))[: max(1, int(n_samples*coef_avg))]
        mse_mean = df_mse_sorted.mean(axis=0)

        df_time_sorted = (df_time.sort_values(
            by=list(algos.keys())[-1]))[: max(1, int(n_samples*coef_avg))]
        time_mean = df_time_sorted.mean(axis=0)

        if verbose:
            print(mse_mean)
            print(time_mean)
        return {'mse': mse_mean, 'time': time_mean}

    def single_reconstruction(self, algos, plot=False, seed=None, verbose=False):
        dic = {}

        self.initialization(N=self.N, alpha=self.alpha,
                            rho=self.rho, Delta=self.Delta, seed=seed)

        if 'TrampBayes' in algos.keys():
            print('## Tramp Bayes ##')
            start = time.time()
            mse, dic_ = self.tramp_Bayes()
            end = time.time()
            dic['TrampBayes'] = {'mse': mse,
                                 'data': dic_, 'time': end-start}

        if 'TrampSE' in algos.keys():
            print('## Tramp SE ##')
            start = time.time()
            mse = self.tramp_Bayes_SE()
            end = time.time()
            dic['TrampSE'] = {'mse': mse,
                              'data': [], 'time': end-start}

        if 'LassoCV' in algos.keys():
            print('## Lasso Cross Validation ##')
            start = time.time()
            mse, dic_ = self.scikit_lassoCV()
            end = time.time()
            dic['LassoCV'] = {'mse': mse, 'data': dic_, 'time': end-start}

        if 'Pymc3Bayes' in algos.keys():
            print('## PyMC3 ##')
            start = time.time()
            mse, dic_ = self.pymc3_sparse()
            end = time.time()
            dic['Pymc3Bayes'] = {'mse': mse,
                                 'data': dic_, 'time': end-start}

        if plot:
            plot_compare(dic)

        if verbose:
            s = 'mse '
            for key in dic.keys():
                s += f"{key}:{dic[key]['mse']} ({dic[key]['time']:.2f}s) "
            print(s)

        return {key: dic[key]['mse'] for key in dic.keys()}, {key: dic[key]['time'] for key in dic.keys()}

    def tramp_Bayes(self, rho=None, Delta=None, max_iter=1000, damping=0.1):
        if rho is None:
            rho = self.rho
        if Delta is None:
            Delta = self.Delta
        prior = GaussBernouilliPrior(size=(self.N,), rho=rho)
        student = prior @ V(id="x") @ LinearChannel(W=self.data['A']) @ V(
            id='z') @ GaussianLikelihood(y=self.data['y'], var=Delta)
        student = student.to_model_dag()
        student = student.to_model()

        ep = ExpectationPropagation(student)
        ep.iterate(
            max_iter=max_iter, damping=damping, callback=EarlyStoppingEP(tol=1e-8))
        data_ep = ep.get_variables_data(['x'])
        mse = mean_squared_error(
            data_ep['x']['r'], self.data['x_true'])
        print(f'mse:{mse}')
        return mse, {'x_pred': data_ep['x']['r'], 'x_true': self.data['x_true']}

    def tramp_Bayes_SE(self, max_iter=1000, damping=0.1):
        prior = GaussBernouilliPrior(size=(self.N,), rho=self.rho)
        student = prior @ V(id="x") @ AnalyticalLinearChannel(ensemble=MarchenkoPasturEnsemble(
            alpha=self.alpha)) @ V(id='z') @ GaussianLikelihood(y=self.data['y'], var=self.Delta)
        student = student.to_model_dag()
        student = student.to_model()
        student.x_ids = ['x']

        # Informative
        a_init = [('x', '->', '0')]
        initializer = CustomInit(a_init=a_init, a=10**(4))
        se = StateEvolution(student)
        se.iterate(
            max_iter=max_iter, damping=damping, initializer=initializer)
        data_se = se.get_variables_data(['x'])
        mse_inf = data_se['x']['v']

        # Non-informative
        se = StateEvolution(student)
        se.iterate(
            max_iter=max_iter, damping=damping)
        data_se = se.get_variables_data(['x'])
        mse_uni = data_se['x']['v']

        print(f'mse inf:{mse_inf} mse uni:{mse_uni}')
        return mse_inf

    def scikit_lassoCV(self):
        lasso = LassoCV(cv=5)
        lasso.fit(self.data['A'], self.data['y'])
        mse = mean_squared_error(
            lasso.coef_, self.data['x_true'])
        print(f'mse:{mse}')
        return mse, {'x_pred': lasso.coef_, 'x_true': self.data['x_true']}

    def pymc3_sparse(self, max_iter=1000, rho=None, Delta=None):
        if rho is None:
            rho = self.rho
        if Delta is None:
            Delta = self.Delta
        basic_model = pm.Model()
        with basic_model:
            ber = pm.Bernoulli("ber", p=rho, shape=self.N)
            nor = pm.Normal("nor", mu=0, sd=1, shape=self.N)
            x = pm.Deterministic("x", ber * nor)
            likelihood = pm.Normal('y', mu=pm.math.dot(self.data['A'], x),
                                   sigma=np.sqrt(Delta), observed=self.data['y'])

            n_chains = 1
            start = pm.find_MAP()
            step = pm.NUTS()
            trace = pm.sample(int(max_iter/n_chains), step=step, start=start,
                              progressbar=True, chains=n_chains)

        x_samples = trace.get_values('x')
        x_pred = np.mean(x_samples, axis=0)
        mse = mean_squared_error(
            x_pred, self.data['x_true'])
        print(f'mse:{mse}')
        return mse, {'x_pred': x_pred, 'x_true': self.data['x_true']}


def run_experiment_mse_alpha(algos, rho=0.05, Delta=1e-2, N=1000,
                             n_samples=100, n_points=10,
                             save_data=True, verbose=False, seed=None):
    ## Tab alpha ##
    tab_alpha = np.linspace(0.1, 1, n_points)

    ## Records ##
    dics = {key: {
        'tab_alpha': [], 'tab_mse': [], 'tab_time': [],
        'rho': rho, 'algos': algos
    } for key in algos.keys()}

    if save_data:
        s = '_'
        str_name = s.join([key for key in algos.keys()])
        dir_data = 'Data'
        os.makedirs(dir_data) if not os.path.exists(dir_data) else 0
        file_name = f'{dir_data}/Benchmark_algos={str_name}_rho={rho:.2f}.pkl'
    seed_count = 0

    for alpha in tab_alpha:
        if seed is not None:
            seed = seed_count
        print(f'\n alpha:{alpha:.3f}')
        # Create benchmark #
        obj = compare_tramp_lasso_pymc3(
            alpha=alpha, rho=rho, Delta=Delta, N=N)
        # Run average benchmark #
        res = obj.average_reconstruction(
            algos=algos, n_samples=n_samples, verbose=verbose, seed=seed)
        # Store record #
        for key in dics.keys():
            dic = dics[key]
            dic['tab_alpha'].append(alpha)
            dic['tab_mse'].append(res['mse'][key])
            dic['tab_time'].append(res['time'][key])
        seed_count += 1

        if save_data:
            save_object(dics, file_name)

    return dics


if __name__ == "__main__":
    ## Parameters ##
    N, alpha, Delta, rho = 1000, 0.5, 1e-2, 0.05
    seed = True

    ## List algos to compare ##
    algos_tab = {
        'TrampSE': {'Delta': Delta, 'rho': rho},
        'TrampBayes': {'Delta': Delta, 'rho': rho},
        'LassoCV': {'lambd': 0},
        # 'Pymc3Bayes': {'lambd': 0}
    }
    list_algos = list(algos_tab.keys())
    algos = dict((name, algos_tab[name])
                 for name in algos_tab.keys())

    ## Run benchmark ##
    # Average experiment
    n_samples = 1
    # Number of points between [0:1]
    n_points = 50

    dic = run_experiment_mse_alpha(
        algos, rho=rho, Delta=Delta, N=N,
        n_samples=n_samples, n_points=n_points,
        save_data=False, verbose=False, seed=seed
    )

    plot_sparse_linear_benchmark(dic, save_fig=False)
