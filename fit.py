import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import seaborn as sns

sns.init()


# func = r"$L(N,D,R_N,R_D)=E + \frac{A}{(U_N + U_N * R_N^* * (1 - e^{(-1*R_N/(R_N^*))}))^\alpha} + \frac{B}{(U_D + U_D * R_D^* * (1 - e^{(-1*R_D/(R_D^*))}))^\beta}$"
# a, b, e, alpha, beta, rd_star, rn_star = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]


def datablations_fit(x, a, b, e, alpha, beta, rd_star, rn_star):
    num_params, training_tokens, unique_tokens = x
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))


def D_to_N(training_tokens, G, alpha, beta):
    return (training_tokens * G) ** (beta / alpha) * G


def datablation_scaling_law(num_params, training_tokens, unique_tokens, A,B,G, alpha, beta, rd_star, rn_star):
    """
    from https://github.com/huggingface/datablations/tree/main#parametric-fit

    num_params(N): number of parameters
    training_tokens(D): number of total training tokens
    unique_tokens(U): number of unique training tokens
    """
    assert unique_tokens <= training_tokens, "Cannot have more unique tokens than total tokens"

    RD = np.maximum((training_tokens / unique_tokens) - 1, 0)
    UN = np.minimum(num_params, D_to_N(unique_tokens,training_tokens, G, alpha, beta))
    RN = np.maximum((num_params / UN) - 1, 0)

    L = E + A / (UN + UN * rn_star * (1 - np.exp(-1 * RN / rn_star))) ** alpha + B / (
                unique_tokens + unique_tokens * rd_star * (1 - np.exp(-1 * RD / (rd_star)))) ** beta
    return L


popt, pcov = curve_fit(target_func, X, y)
