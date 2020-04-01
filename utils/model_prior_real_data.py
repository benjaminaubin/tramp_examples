import numpy as np
import os
import random
import h5py
import time
import copy
import matplotlib.pyplot as plt

# Tramp package
import tramp
from tramp.algos import EarlyStopping, ExpectationPropagation, TrackEstimate, NoisyInit
from tramp.algos.metrics import mean_squared_error
from tramp.ensembles import GaussianEnsemble
from tramp.variables import SISOVariable as V, SILeafVariable as O, MISOVariable as M
from tramp.likelihoods import GaussianLikelihood
from tramp.priors import GaussianPrior
from tramp.channels import LinearChannel, ReshapeChannel, LeakyReluChannel, HardTanhChannel, BiasChannel

# Specific to keras
from keras.datasets import mnist, fashion_mnist
from keras.utils import normalize


class Model_Prior():
    """ Implements EP algorithm for model: 
        y = model(x_star) + sqrt(Delta) * xi

        - with models:
            * denoising : model(x) = x
            * inpainting: model(x) = x with n_rem erased elements

        - with x_star drawn from:
            * mnist test set
            * fashion mnist test set

        - trained with a VAE prior:
            VAE: [(20, 400) -> relu + bias -> (400, 784) -> sigmoid + bias]
    """

    def __init__(self, model_params={'name': 'denoising', 'N': 784, 'alpha': 0},
                 data_params={'name': 'mnist', 'category': 0},
                 prior_params={'name': 'VAE', 'type': 'mnist',
                               'id': '20_relu_400_sigmoid_784_bias'},
                 seed=False, plot_prior_sample=False):
        # Directory
        self.dir = 'utils/'

        # Model params
        self.model_params = model_params
        self.N = model_params['N']

        # Data params
        self.data_params = data_params

        # Prior params
        self.prior_params = prior_params

        # Seed
        self.seed = seed

        # Plot
        self.plot_prior_sample = plot_prior_sample

    def setup(self):
        # Build prior module
        prior_x = self.build_prior()

        # Sample from the prior
        self.sample_from_prior(prior_x)

        # Init the model
        model = self.init_model(prior_x)

        # Generate sample from dataset
        y = self.generate_sample()

        # Build the model with likelihood on y and prior_x on x
        model_ = self.build_model(model, y)
        self.model = model_.to_model()

    def build_prior(self):
        self.shape = (self.N)
        # VAE prior
        if self.prior_params['name'] == 'VAE':
            prior_x = self.build_VAE_prior(self.prior_params)

        else:
            raise NotImplementedError

        return prior_x

    def build_VAE_prior(self, params):
        shape = self.shape
        assert self.N == 784
        biases, weights = self.load_VAE_prior(params)
        # print(params['id'])

        if params['id'] == '20_relu_400_sigmoid_784_bias':
            D, N1, N = 20, 400, 28*28
            W1, W2 = weights
            b1, b2 = biases
            prior_x = (GaussianPrior(size=D) @ V(id="z_0") @
                       LinearChannel(W1, name="W_1") @ V(id="Wz_1") @ BiasChannel(b1) @ V(id="b_1") @ LeakyReluChannel(0) @ V(id="z_1") @
                       LinearChannel(W2, name="W_2") @ V(id="Wz_2") @ BiasChannel(b2) @ V(id="b_2") @ HardTanhChannel() @ V(id="z_2") @
                       ReshapeChannel(prev_shape=self.N, next_shape=self.shape))
        else:
            raise NotImplementedError

        return prior_x

    def load_VAE_prior(self, params):
        # load VAE weights
        file = h5py.File(
            f"{self.dir}/weights_vae/{params['type']}/vae_{params['type']}_{params['id']}.h5", "r")
        decoder = file['decoder']

        layers = [decoder[key] for key in list(decoder.keys())]
        weights = [layer["kernel:0"][()].T for layer in layers]
        try:
            biases = [layer["bias:0"][()] for layer in layers]
        except:
            #print('no biases')
            biases = []

        shapes = [weight.shape for weight in weights]
        #print(f'VAE weights loaded: {shapes}')
        return biases, weights

    def sample_from_prior(self, prior_x):
        if self.plot_prior_sample:
            model = prior_x @ O(id="y")
            model = model.to_model_dag()
            prior_sample = model.to_model()
            fig, axs = plt.subplots(4, 4, figsize=(4, 4))
            for ax in axs.ravel():
                sample = prior_sample.sample()['y']
                ax.set_axis_off()
                ax.imshow(sample.reshape(28, 28), cmap="gray")

            dir_fig = 'Figures/'
            os.makedirs(dir_fig) if not os.path.exists(dir_fig) else 0
            file_name = f"/{dir_fig}Prior_{self.prior_params['name']}_{self.prior_params['type']}_{self.prior_params['id']}.pdf"
            plt.savefig(file_name, format='pdf', dpi=1000,
                        bbox_inches="tight", pad_inches=0.1)
            plt.show(block=True)
            input("...")
            plt.close()

    def init_model(self, prior_x):
        self.y_ids = ['y']

        if self.model_params['name'] == 'denoising':
            # Model
            model = prior_x @ V(id="x")
            # Variables
            self.x_ids = ['x']

        elif self.model_params['name'] == 'inpainting':
            # Create sensing matrix
            N = self.model_params['N']
            F = np.identity(N)
            p_rem = self.model_params['p_rem']

            ## Remove a Band ##
            if self.model_params['type'] == 'band':
                N_rem = int(p_rem * N / 100)
                id_0 = int(N/2) - int(N_rem/2)
                for rem in range(id_0, id_0+N_rem):
                    F[rem, rem] = 0

            ## Remove randomly ##
            if self.model_params['type'] == 'uniform':
                tab = np.arange(1, N)
                np.random.shuffle(tab)
                for i in range(int(p_rem * N / 100)):
                    rem = tab[i]
                    F[rem, rem] = 0

            ## Diagonal ##
            if self.model_params['type'] == 'diagonal':
                l = int(p_rem * 28 / 100)
                for j in range(-int(l/2), int(l/2), 1):
                    for i in range(1, 27, 1):
                        ind = i * 28 + i + j
                        F[ind, ind] = 0
                        ind = i * 28 - i - j
                        F[ind, ind] = 0

            F_tot = F
            F_obs = np.delete(F, np.where(~F.any(axis=0))[0], axis=0)

            self.F = F_obs
            self.F_tot = F_tot
            # Model
            model = prior_x @ V(id="x") @ LinearChannel(
                F_obs, name="F") @ V(id="z")

            # Variables
            self.x_ids = ['x']

        else:
            raise NotImplementedError

        return model

    def generate_sample(self):
        self.x_true, self.y_true = {}, {}

        if self.data_params['name'] in ['mnist', 'fashion_mnist']:
            assert self.N == 784
            if self.data_params['name'] == 'mnist':
                (_, _), (X_test, Y_test) = mnist.load_data()
            else:
                (_, _), (X_test, Y_test) = fashion_mnist.load_data()

            # Transform data
            X_test_spec = 2 * (X_test / 255) - 1.
            X_test_spec = X_test_spec.reshape(
                10000, 784)-np.sum(X_test_spec.reshape(10000, 784), 1).reshape(10000, 1)/784
            X_test_spec = normalize(
                X_test_spec, axis=-1, order=2) * np.sqrt(784)

            X_test_ep = 2 * (X_test / 255) - 1.

            # Draw random sample from category
            indices = np.array(
                [i for i in range(len(Y_test)) if Y_test[i] == self.data_params['category']])
            if self.seed != 0:
                np.random.seed(self.seed)
            id = indices[np.random.randint(0, len(indices), 1)]

            # Choose x_star
            x_star = X_test_ep[id].reshape(self.N)
            x_star_spec = X_test_spec[id].reshape(self.N)

        else:
            raise NotImplementedError

        y = self.channel(x_star)
        self.x_true['x'] = x_star
        self.y_true['y'] = y
        y_spec = self.channel(x_star_spec)
        self.x_true['x_spec'] = x_star_spec
        self.y_true['y_spec'] = y_spec

        return y

    def channel(self, x):
        if self.model_params['name'] == 'denoising':
            noise = np.sqrt(
                self.model_params['Delta']) * np.random.randn(self.N)
            y = x + noise
        elif self.model_params['name'] == 'inpainting':
            y = self.F @ x
            self.y_inp = self.F_tot @ x
        else:
            raise NotImplementedError
        return y

    def build_model(self, model, y):
        if self.model_params['name'] == 'denoising':
            Delta = self.model_params['Delta']
        elif self.model_params['name'] == 'inpainting':
            Delta = 1e-2
        model = model @ GaussianLikelihood(y=y, var=Delta)
        model = model.to_model_dag()
        return model

    def run_ep(self, max_iter=250, damping=0.5):
        # Initialization
        initializer = NoisyInit()

        # Callback
        x_tracker = TrackEstimate(ids="x", every=100)
        callback = x_tracker

        # EP iterations
        ep = ExpectationPropagation(self.model)
        ep.iterate(
            max_iter=max_iter,
            callback=callback,
            initializer=initializer,
            damping=damping
        )

        df_track = x_tracker.get_dataframe()
        ep_x_data = ep.get_variables_data(self.x_ids)
        #ep_x_data_evo = df_track.loc[df_track['id'] == 'x']

        mse_ep, mse = self.compute_mse(ep_x_data)
        return mse_ep, mse

    ## Annex functions ##
    def compute_mse(self, ep_x_data):
        self.x_pred = {x_id: data["r"] for x_id, data in ep_x_data.items()}

        # MSE computed by ep
        self.mse_ep = {x_id: data["v"] for x_id, data in ep_x_data.items()}

        # Real MSE
        self.mse = min(mean_squared_error(self.x_true['x'], self.x_pred['x']), mean_squared_error(
            self.x_true['x'], -self.x_pred['x']))

        print(f"mse_ep: {self.mse_ep['x']:.3f} mse: {self.mse: .3f}")
        return self.mse_ep['x'], self.mse
