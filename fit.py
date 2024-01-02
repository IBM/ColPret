import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error
import matplotlib

from util.cache import get_cache, save_cache
from util.naming import to_int, to_str
from util.plots import plot_pred_actual_compare

# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

from util.read_data import get_data

plt.interactive(False)
import seaborn as sns

# func = r"$L(N,D,R_N,R_D)=E + \frac{A}{(U_N + U_N * R_N^* * (1 - e^{(-1*R_N/(R_N^*))}))^\alpha} + \frac{B}{(U_D + U_D * R_D^* * (1 - e^{(-1*R_D/(R_D^*))}))^\beta}$"
# a, b, e, alpha, beta, rd_star, rn_star = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]
epsilon = 1e-6


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


def prepare_for_fit(df, columns):
    # preprocess
    df["data"] = pd.get_dummies(df["data"])

    # remove non numerals
    assert not set(columns).difference(df.select_dtypes([np.number]).columns)
    # get necessary columns
    metadata = df.iloc[:, [col in columns for col in df.columns]]
    # metadata = list(df.itertuples())
    return metadata


def test_function_fit():
    df = get_data()
    subdf = df[df['data'] == "c4"]
    # models = num_params.apply(to_int), tokens_seen, tokens_per_epoch
    perf = subdf["loss"]
    target_func = test_fit
    metadata = prepare_for_fit(subdf, DATA_FIT_COLS)
    # params: a, b, e, alpha, beta, rd_star, rn_star
    guess = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]
    bounds = ([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], [np.inf] * len(guess))

    # Invent performance:
    perf = np.sum(metadata * [1e-11, 1e-9, 1e-11], axis=1)

    popt, pcov = curve_fit(target_func, metadata, perf, p0=guess, bounds=bounds)
    print("predicted", datablations_fit(metadata[:5], *popt))
    print("actual", perf[:5])


def fit(target_func, metadata, perf, guess=None, bounds=None):
    try:
        try:
            popt, pcov = curve_fit(target_func, metadata, perf, p0=guess, bounds=bounds)
        except RuntimeError as e:
            popt, pcov = curve_fit(target_func, metadata, perf, p0=guess,
                                   method='dogbox',
                                   bounds=bounds)  # not sure who of the three is best (lm can only be used without bounds)
            print(
                f"Was hard to fit, are some of the diagonal variances very high? This might indicate redundant parameters:{np.diag(pcov)}")
    except Exception as e:
        if metadata is None or metadata.empty:
            print("Fit failed, not metadata")
        elif not guess:
            print(f"Fit failed, guess is {guess}")
            return None
        else:
            print(
                f"Fit failed {metadata['model_name'].unique()[0]}, data size:{len(metadata)}, number of params to fit {len(guess)}")
        popt = np.zeros_like(guess) + epsilon
        pcov = None
    return popt


def aggregate_row_loss(row):
    return np.mean([row[x] for x in row["loss_cols"]])


def fit_per_model(df, predict_with=0.3):
    target_func = datablations_fit
    data = []
    cache_name = f"fit_per_model_{predict_with}"
    cache = get_cache(cache_name)
    for model_name in df["model_name"].unique():
        # if "2.8B" not in model_name:
        #     continue

        # params: a, b, e, alpha, beta, rd_star, rn_star
        guess = (6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743)
        bounds = ([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf], [np.inf] * len(guess))
        subdf = df[df["model_name"] == model_name]
        subdf = subdf.sort_values("tokens_seen")
        subdf["perf"] = subdf.apply(aggregate_row_loss, axis=1)
        metadata = subdf.iloc[:int(len(subdf["perf"]) * predict_with), :]
        perf = metadata["perf"]
        if model_name in cache:
            popt = cache[model_name]
        else:
            popt = fit(target_func, metadata, perf, guess=guess, bounds=bounds)
            if popt is None:
                popt = np.zeros_like(guess) + epsilon
            cache[model_name] = [x for x in popt]

        predicted = target_func(subdf, *popt)
        subdf = pd.merge(metadata, subdf, how='right', indicator="in_fit")
        subdf["in_fit"] = subdf["in_fit"].apply(lambda x: True if x == "both" else False)
        subdf["pred"] = predicted
        data.append(subdf)
    save_cache(cache, cache_name)
    data = pd.concat(data)
    plot_pred_actual_compare(data, show=True)


def data_aware_fit(show=False):
    df = get_data()
    subdf = df[df['data'] == "c4"]
    # models = num_params.apply(to_int), tokens_seen, tokens_per_epoch
    perf = subdf["loss"]
    target_func = datablations_fit
    metadata = prepare_for_fit(subdf, DATA_FIT_COLS)
    guess = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]
    # bounds = []
    popt, pcov = curve_fit(target_func, metadata, perf, p0=guess)
    predicted = datablations_fit(metadata, *popt)
    fig, ax = plt.subplots()

    # predictions only
    for num_params in metadata["num_params"].unique():
        for tokens_per_epoch in metadata["tokens_per_epoch"].unique():
            indx = (tokens_per_epoch == metadata["tokens_per_epoch"]) & (num_params == metadata["num_params"])
            x = metadata[indx]["tokens_seen"]
            # sns.lineplot(x=x, y=perf[indx], label=f"actual {num_params} {tokens_per_epoch}")
            # color = gca.lines[-1].get_color()
            sns.lineplot(x=x, y=predicted[indx], label=f"predicted {num_params} {tokens_per_epoch}",
                         linestyle='dashed')
            print(predicted)
    plt.xscale('log')
    plt.xlabel("tokens_seen")
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))
    if show:
        plt.show()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/predictions_only.pdf")
    plt.clf()

    # compare predictions to actual
    for num_params in metadata["num_params"].unique():
        for tokens_per_epoch in metadata["tokens_per_epoch"].unique():
            indx = (tokens_per_epoch == metadata["tokens_per_epoch"]) & (num_params == metadata["num_params"])
            x = metadata[indx]["tokens_seen"]
            ax = sns.lineplot(x=x, y=perf[indx], label=f"actual {num_params} {tokens_per_epoch}")
            color = ax.lines[-1].get_color()
            sns.lineplot(x=x, y=predicted[indx], color=color, label=f"predicted {num_params} {tokens_per_epoch}",
                         linestyle='dashed')
            print(predicted)
    plt.xscale('log')
    plt.xlabel("tokens_seen")
    plt.legend().remove()
    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    # ax.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))
    if show:
        plt.show()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/compare_predictions.pdf")
    plt.clf()

    # compare per trait
    for trait in metadata.columns:
        if len(metadata[trait].unique()) < 100:
            x = metadata[trait]
            x = np.concatenate([x, x])
            pred_n_perf = np.array([predicted, perf]).flatten()
            is_predicted = np.concatenate([np.ones_like(predicted), np.zeros_like(perf)])
            df = pd.DataFrame({"perf": pred_n_perf, "is_predicted": is_predicted, trait: x})
            sns.boxplot(data=df, x=trait, y="perf", hue="is_predicted", gap=0.1)
            # plt.xscale('log')
            plt.xlabel(trait)
            if show:
                plt.show()
            os.makedirs("graphs", exist_ok=True)
            plt.savefig(f"graphs/compare_{trait}.pdf")
            plt.clf()
        else:
            x = metadata[trait]
            sns.scatterplot(x=x, y=predicted, label="predicted")
            sns.scatterplot(x=x, y=perf, label="actual")
            plt.xscale('log')
            plt.xlabel(trait)
            if show:
                plt.show()
            os.makedirs("graphs", exist_ok=True)
            plt.savefig(f"graphs/compare_{trait}.pdf")
            plt.clf()
    print(popt, pcov)
    print("MSE", mean_squared_error(perf, predicted))
    print("predicted performance", predicted[:5])
    print("actual performance", perf[:5])


if __name__ == '__main__':
    df = get_data()
    # df = df[df["model_type"].isin(["OPT"])]
    # df = df[(df["model_type"].isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    fit_per_model(df)
    # data_aware_fit()
