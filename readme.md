## TRAMP: Compositional Inference with TRee ApproximateMessage Passing
#### Antoine Baker, Benjamin Aubin, Florent Krzakala, Lenka Zdeborova

### Abstract:
We introduce tramp, standing for TRee Approximate Message Passing, a python package for compositional inference in Python that runs Expectation Propagation on high-dimensional tree-structured models. The package provides an unifying framework to study several approximate message passing algorithms previously derived for a variety of machine learning tasks such as generalized linear models, inference in multi-layer networks, matrix factorization, and reconstruction using non-separable penalties. For some models, the performance of the algorithm can be theoretically predicted by the state evolution, and the measurements entropy estimated by the free entropy formalism. The implementation is modular by design: each module, which implements a factor, can be composed at will with other modules to solve complex inference tasks. The user only needs to declare the factor graph of the model: the inference algorithm, state evolution and entropy estimation are fully automated. The source code is publicly available at [https://github.com/sphinxteam/tramp](https://github.com/sphinxteam/tramp).


This repo contains the few different examples presented in the above paper available on [ArXiv]():

### Notebooks (Python 3.6)
- [Sparse FFT](sparse_fft.ipynb) teacher-student scenario
- [Sparse gradient](sparse_gradient.ipynb) teacher-student scenario
- [Inpainting and denoising](inpainting_denoising_VAE.ipynb) on MNIST/FashionMNIST using a VAE prior trained on the same dataset

### Scripts (Python 3.6)
- [Sparse linear](sparse_linear_benchmark.py) benchmark: TRAMP EP vs TRAMP SE vs Lasso Scikit-Learn vs PyMC3
- [Sparse compressed sensing](sparse_compressed_sensing.py): produce MSE curves as a function of alpha at fixed rho
