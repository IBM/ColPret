from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy.optimize import curve_fit

from util.torch_fit import chinchilla_torch_fit


@dataclass
class FitInfo:
    name: str
    func: callable
    guess: Tuple
    bounds: Tuple
    fit_func: Union[callable, None] = None

    def __post_init__(self):
        if self.fit_func is None:
            self.fit_func = curve_fit


def test_fit(metadata, a, b, e, alpha, beta, rd_star, rn_star):
    """
    Make a simple function here to test that optimization is correct
    Args:
        metadata:
        a:
        b:
        e:
        alpha:
        beta:
        rd_star:
        rn_star:

    Returns:

    """
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    return A * metadata.iloc[:, 0] + B * metadata.iloc[:, 1] + E * metadata.iloc[:, 2]


TestFit = FitInfo(func=test_fit, guess=[6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743],
                  bounds=([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], [np.inf] * 7), name="test",
                  fit_func=None)


def bound_params(fit_info: FitInfo, bounded_vars: Tuple):
    """
    Bound some of the variables of a fitting function by order, skipping None (E.g., one can make a 5 parameter function 3 parameters by passing [None,1,None,3] (which will not change the 0,2,4 places of the original function)
    Args:
        fit_info:
        bounded_vars:

    Returns:

    """

    def bounded_func(metadata, *args, **kwargs):
        # assert len(bounded_vars) <= len(
        #     args), f"cannot bound more parameters then the original function had, trying to bind: {bounded_vars}"
        assert bounded_vars.count(None) <= len(args)
        new_args = []
        last_arg = 0
        last_bounded = 0
        while last_arg < len(args) or last_bounded < len(bounded_vars):
            if last_bounded < len(bounded_vars):
                if bounded_vars[last_bounded] is not None:
                    new_args.append(bounded_vars[last_bounded])
                else:
                    new_args.append(args[last_arg])
                    last_arg += 1
                last_bounded += 1
            else:
                new_args.append(args[last_arg])
                last_arg += 1
        return fit_info.func(metadata, *new_args, **kwargs)

    bounds = tuple(zip(*(bound for i, bound in enumerate(zip(*fit_info.bounds)) if
                         i >= len(bounded_vars) or bounded_vars[i] is None)))
    guess = tuple(g for i, g in enumerate(fit_info.guess) if i >= len(bounded_vars) or bounded_vars[i] is None)
    return FitInfo(func=bounded_func, bounds=bounds, guess=guess, name=fit_info.name + f"b:{bounded_vars}")


def chinchilla_one_model_fit(metadata, b, e, beta):
    return chinchilla_power_law_fit(metadata, 6.255414, b, e, 0.3526596, beta)


Chinchilla1ModelFit = FitInfo(func=chinchilla_one_model_fit, guess=(7.3049974, 0.6254804, 0.3526596),
                              bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf] * 5),
                              name="chinchilla:bounded_scaling")


def chinchilla_power_law_fit(metadata, a, b, e, alpha, beta):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["tokens_seen"].array, metadata[
        "tokens_per_epoch"].array
    training_tokens = tokens_seen
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    L = E + A / num_params ** alpha + B / training_tokens ** beta
    return L


def manual_chinchilla_power_law_fit(metadata, a, b, e):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["tokens_seen"].array, metadata[
        "tokens_per_epoch"].array
    training_tokens = tokens_seen
    alpha = 0.0116280 + a * 0.05004588
    beta = 0.23710262 + b * 0.03844353
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    L = E + A / num_params ** alpha + B / training_tokens ** beta
    return L


def PCA_chinchilla_power_law_fit(metadata, a, b, c):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["tokens_seen"].array, metadata[
        "tokens_per_epoch"].array
    training_tokens = tokens_seen
    pca_findings = np.array([[-0.10173792, 0.64745976, 0.75459505, -0.00767472, 0.03118762],
                             [0.07297588, 0.76113056, -0.64410298, 0.00285629, 0.02186262],
                             [0.99063847, 0.0113744, 0.12517598, 0.04886348, -0.02119861]]).T
    # full pca [[-0.10173792,  0.64745976,  0.75459505, -0.00767472,  0.03118762],
    #        [ 0.07297588,  0.76113056, -0.64410298,  0.00285629,  0.02186262],
    #        [ 0.99063847,  0.0113744 ,  0.12517598,  0.04886348, -0.02119861],
    #        [ 0.02367606, -0.03666797, -0.00683687, -0.02063154,  0.99881054],
    #        [ 0.04897741, -0.00148489, -0.00137544, -0.99855878, -0.02185124]]
    a, b, e, alpha, beta = pca_findings @ [a, b, c]
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    L = E + A / num_params ** alpha + B / training_tokens ** beta
    return L


def mult_power_law_fit(metadata, a, b, e, alpha, beta):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["tokens_seen"].array, metadata[
        "tokens_per_epoch"].array
    training_tokens = tokens_seen
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    L = E * A / num_params ** alpha * B / training_tokens ** beta
    return L


ChinchillaTorchFit = FitInfo(func=chinchilla_power_law_fit, guess=None,
                             bounds=None, name="chinchillaTorch", fit_func=chinchilla_torch_fit)
ChinchillaTorchGuessFit = FitInfo(func=chinchilla_power_law_fit,
                                  guess=(6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596),
                                  bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf] * 5), name="chinchillaGuessTorch",
                                  fit_func=chinchilla_torch_fit)
ChinchillaFit = FitInfo(func=chinchilla_power_law_fit, guess=(6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596),
                        bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf] * 5), name="chinchilla", fit_func=curve_fit)
MultFit = FitInfo(func=mult_power_law_fit, guess=(6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596),
                  bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf] * 5), name="mult", fit_func=curve_fit)
PCAFit = FitInfo(func=PCA_chinchilla_power_law_fit, guess=(6.255414, 7.3049974, 0.6254804),  # bad guess...
                 bounds=([-np.inf, -np.inf, -np.inf], [np.inf] * 3), name="pca", fit_func=curve_fit)
Manual2Fit = FitInfo(func=manual_chinchilla_power_law_fit, guess=(6.255414, 7.3049974, 0.6254804),  # bad guess...
                     bounds=([-np.inf, -np.inf, -np.inf], [np.inf] * 3), name="manual", fit_func=curve_fit)


def datablations_fit(metadata, a, b, e, alpha, beta, rd_star, rn_star):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["tokens_seen"].array, metadata[
        "tokens_per_epoch"].array
    unique_tokens = np.min([tokens_per_epoch, tokens_seen], axis=0)
    training_tokens = tokens_seen
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    return datablation_scaling_law(num_params, training_tokens, unique_tokens, A, B, E, G, alpha, beta, rd_star,
                                   rn_star)


def D_to_N(training_tokens, G, alpha, beta):
    return (training_tokens * G) ** (beta / alpha) * G


DATA_FIT_COLS = ("num_params", "tokens_seen", "tokens_per_epoch")


def datablation_scaling_law(num_params, training_tokens, unique_tokens, A, B, E, G, alpha, beta, rd_star, rn_star):
    """
    based on https://github.com/huggingface/datablations/tree/main#parametric-fit

    num_params(N): number of parameters
    training_tokens(D): number of total training tokens
    unique_tokens(U): number of unique training tokens
    """

    assert all(unique_tokens <= training_tokens), "Cannot have more unique tokens than total tokens"
    RD = np.maximum((training_tokens / unique_tokens) - 1, 0).astype(float)
    UN = np.minimum(num_params, D_to_N(training_tokens, G, alpha, beta)).astype(float)
    RN = np.maximum((num_params / UN) - 1, 0).astype(float)
    L = E + A / (UN + UN * rn_star * (1 - np.exp(-1 * RN / rn_star))) ** alpha \
        + B / (unique_tokens + unique_tokens * rd_star * (1 - np.exp(-1 * RD / rd_star))) ** beta

    return L


DatablationsFit = FitInfo(func=datablations_fit,
                          guess=[6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743],
                          bounds=([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], [np.inf] * 7),
                          name="datablations")
