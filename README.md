# baseline-estimate
This repo is for baseline correction.
Both parameter estimation and non-parameter estimation of the baseline are available.

## Prerequisites
- `Python 3`
- `numpy`, `scipy`, `math` (only required for non-parameter estimation's SNIP methods), `nlopt` (only required for parameter estimation)

## Usage
Class `NONPARAMS_EST` in `nonparams_est.py` provides two series of methods.
- `snip`: Sensitive Nonlinear Iterative Peak (SNIP) algorithms.
- `pls`: Reweighted Penalized Least Squares (PLS) algorithms. the AsLS, airPLS, arPLS, BrPLS methods are available. 

Class `PARAMES_EST` in `params_est.py` provides parameter baseline method based on Bayesian theorem.
- `ESTFUNC`: function of the baseline to be estimated. default: linear, quadratic, cubic, Lorentzian, Landau-Gaussian (`pylandau` required). Also user can define custom functions.
- `BAYESIAN`: main function for baseline estimation.

## Example
Examples of all the methods to estimate Gaussian peaks on linear and Lorentzian baseline are shown in `test.py`.

## License
This repository is licensed under the **GNU GPLv3**.
