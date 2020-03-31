from utils.plot.import_library_plot import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.functions import save_object, load_object, mean_squared_error

from tramp.models import glm_generative, glm_state_evolution
from tramp.algos import ConstantInit, NoisyInit, CustomInit
from tramp.experiments import BayesOptimalScenario, TeacherStudentScenario
from tramp.priors import GaussBernouilliPrior, GaussianPrior, MAP_LaplacePrior
from tramp.likelihoods import GaussianLikelihood, AsymmetricAbsLikelihood
from tramp.channels import GaussianChannel, LinearChannel, AbsChannel, SgnChannel, AsymmetricAbsChannel
from tramp.ensembles import GaussianEnsemble
from tramp.algos import ConstantInit, NoisyInit
from tramp.variables import SISOVariable as V, SILeafVariable as O, MILeafVariable, SIMOVariable
from tramp.algos import ExpectationPropagation, EarlyStoppingEP, TrackEvolution, TrackErrors


def plot_sparse_(dic, save_fig=False, block=False):
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = cm.get_cmap('plasma_r')
    tab_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    tab_l1, tab_l2,  tab_l3 = [], [], []

    ind = np.where(np.array(dic['tab_mse_se_uni']) > 1e-3)[0][-1]
    ax.plot(dic['tab_alpha'][:ind+1], dic['tab_mse_se_uni'][:ind+1],
            color=tab_col[0], lw=1.75, label=r'SE')
    ax.plot([dic['tab_alpha'][ind], dic['tab_alpha'][ind]], [dic['tab_mse_se_uni'][ind], dic['tab_mse_se_uni'][ind+1]], ':',
            color=tab_col[0], lw=1.75)
    ax.plot(dic['tab_alpha'][ind+1:], dic['tab_mse_se_uni'][ind+1:],
            color=tab_col[0], lw=1.75)

    delta = 3
    ax.plot(dic['tab_alpha'][::delta], dic['tab_mse_ep'][::delta],
            'o', color=tab_col[1], label=r'EP')

    ax.plot(dic['tab_alpha'], dic['tab_mse_se_inf'], '--',
            color=tab_col[2], lw=1.75, label=r'Bayes opt.')

    """ Ticks   """
    ax.set_xlim([0, max(dic['tab_alpha'])])
    ax.set_ylim([-1e-3, max(dic['tab_mse_se_inf'])])

    """ Labels """
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'MSE')

    ax.legend(loc='upper right', fancybox=True,
              shadow=False, ncol=1)

    # Labels

    """ Save   """
    if save_fig:
        dir_fig = 'Figures/'
        os.makedirs(dir_fig) if not os.path.exists(dir_fig) else 0
        file_name = f'{dir_fig}/CS_rho={params["rho"]}_N={params["N"]}.pdf'
        plt.tight_layout()
        plt.savefig(file_name, format='pdf', dpi=1000,
                    bbox_inches="tight", pad_inches=0.1)

    """ Show   """
    if block:
        plt.show(block=False)
        input('Press enter to continue')
        plt.close()
    else:
        plt.close()


class Compressed_Sensing():
    def __init__(self, params, seed=False):
        self.N = params['N']
        self.alpha = params['alpha']
        self.M = int(self.alpha * self.N)
        self.rho = params['rho']
        self.model = self.build_model()
        self.scenario = self.build_scenario(seed)

    def build_model(self):
        self.prior = GaussBernouilliPrior(size=(self.N,), rho=self.rho)
        ensemble = GaussianEnsemble(self.M, self.N)
        self.A = ensemble.generate()
        model = self.prior @ V(id="x") @ LinearChannel(W=self.A) @ V(
            id='z') @ AsymmetricAbsChannel() @ V(
            id='z_') GaussianChannel(var=1e-10) @ O(id="y")
        model = model.to_model()
        return model

    def build_scenario(self, seed):
        scenario = BayesOptimalScenario(self.model)
        scenario.setup(seed=seed)
        return scenario


def run_ep(scenario, settings, n_samples=10):
    # Initialization with constant beliefs #
    initializer = ConstantInit(a=0.01, b=0.01)
    callback = None
    tab_mse = {'mse_ep': [], 'mse': []}

    # Average EP over n_samples #
    for i in range(n_samples):
        scenario.setup(seed=i)
        ep_x_data = scenario.run_ep(
            max_iter=settings['max_iter'], callback=callback,
            initializer=initializer, damping=settings['damping'])
        mse = mean_squared_error(scenario.x_pred['x'], scenario.x_true['x'])
        tab_mse['mse'].append(mse)
        tab_mse['mse_ep'].append(ep_x_data['x']['v'])
    mse, mse_ep = np.mean(tab_mse['mse']), np.mean(tab_mse['mse_ep'])
    print(f'mse ep:{mse_ep:.3e}')
    return mse


def run_se(scenario, settings):
    ## Setup ##
    # Use seed or not #
    seed = False
    scenario.setup(seed=seed)

    # UNIformative Initialization #
    initializer = ConstantInit(a=0.01, b=0.01)
    data_se = scenario.run_se(
        max_iter=settings['max_iter'], damping=settings['damping'], initializer=initializer)
    mse_se_uni = data_se['x']['v']
    print(f'mse se uni:{mse_se_uni:.3e}')

    # INFormative Initialization #
    # Adapt the informative initialization with alpha to:
    # - avoid issues at low alpha if init too large
    # - obtain the true IT transition
    assert params['alpha'] <= 1
    power = 3 * np.exp(params['alpha'])
    initializer = CustomInit(a_init=[('x', '->', '0')], a=10**(power))
    data_se = scenario.run_se(
        max_iter=settings['max_iter'], damping=settings['damping'], initializer=initializer)
    mse_se_inf = data_se['x']['v']
    print(f'mse se inf:{mse_se_inf:.3e}')

    return mse_se_uni, mse_se_inf


def compute_mse_curve(params, settings, n_points=10, n_samples=1):
    tab_alpha_ = np.linspace(0.0025, 1, n_points)
    dic = {key: [] for key in ['tab_alpha',
                               'tab_mse_se_inf', 'tab_mse_se_uni', 'tab_mse_ep']}
    for alpha in tab_alpha_:
        # Create TRAMP instance #
        print(f'\n alpha:{alpha}')
        params['alpha'] = alpha
        cs = Compressed_Sensing(params)
        scenario = cs.scenario
        # Run TRAMP EP ##
        mse_ep = run_ep(scenario, settings, n_samples=n_samples)
        # Run TRAMP SE ##
        mse_uni, mse_inf = run_se(scenario, settings)

        # Append data #
        dic['tab_alpha'].append(alpha)
        dic['tab_mse_se_inf'].append(mse_inf)
        dic['tab_mse_se_uni'].append(mse_uni)
        dic['tab_mse_ep'].append(mse_ep)

    return dic


if __name__ == "__main__":
    ## Define parameters and settings ##
    params = {'N': 1000, 'rho': 0.5}
    settings_ep = {'damping': 0.1, 'max_iter': 200}
    settings_exp = {'n_points': 50, 'n_samples': 1}

    ## Compute and plot MSE curve as a function of alpha ##
    dic = compute_mse_curve(
        params, settings_ep, n_points=settings_exp['n_points'], n_samples=settings_exp['n_samples'])
    # Plot #
    plot_sparse_CS(dic, block=True)
