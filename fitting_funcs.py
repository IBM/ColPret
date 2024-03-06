from typing import Tuple

from dataclasses import dataclass
import numpy as np


@dataclass
class FitInfo:
    func: callable
    guess: Tuple
    bounds: Tuple


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
                  bounds=([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], [np.inf] * 7))


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
    return FitInfo(func=bounded_func, bounds=bounds, guess=guess)


def chinchilla_one_model_fit(metadata, b, e, beta):
    return chinchilla_power_law_fit(metadata, 6.255414, b, e, 0.3526596, beta)


Chinchilla1ModelFit = FitInfo(func=chinchilla_one_model_fit, guess=(7.3049974, 0.6254804, 0.3526596),
                              bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf] * 5))


def chinchilla_power_law_fit(metadata, a, b, e, alpha, beta):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["tokens_seen"].array, metadata[
        "tokens_per_epoch"].array
    training_tokens = tokens_seen
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    L = E + A / num_params ** alpha + B / training_tokens ** beta
    return L


ChinchillaFit = FitInfo(func=chinchilla_power_law_fit, guess=(6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596),
                        bounds=([-np.inf, -np.inf, -np.inf, 0, 0], [np.inf] * 5))


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
                          bounds=([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], [np.inf] * 7))
