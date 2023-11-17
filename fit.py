import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

plt.interactive(False)
import seaborn as sns


# func = r"$L(N,D,R_N,R_D)=E + \frac{A}{(U_N + U_N * R_N^* * (1 - e^{(-1*R_N/(R_N^*))}))^\alpha} + \frac{B}{(U_D + U_D * R_D^* * (1 - e^{(-1*R_D/(R_D^*))}))^\beta}$"
# a, b, e, alpha, beta, rd_star, rn_star = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]


def datablations_fit(metadata, a, b, e, alpha, beta, rd_star, rn_star):
    num_params, tokens_seen, tokens_per_epoch = metadata["num_params"].array, metadata["token_num"].array, metadata[
        "tokens_per_epoch"].array
    unique_tokens = np.min([tokens_per_epoch, tokens_seen], axis=0)
    training_tokens = tokens_seen
    A = np.exp(a)
    B = np.exp(b)
    E = np.exp(e)
    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    return datablation_scaling_law(num_params, training_tokens, unique_tokens, A, B, E, G, alpha, beta, rd_star,
                                   rn_star)


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


def D_to_N(training_tokens, G, alpha, beta):
    return (training_tokens * G) ** (beta / alpha) * G


DATA_COLS = ("num_params", "token_num", "tokens_per_epoch")


def datablation_scaling_law(num_params, training_tokens, unique_tokens, A, B, E, G, alpha, beta, rd_star, rn_star):
    """
    based on https://github.com/huggingface/datablations/tree/main#parametric-fit

    num_params(N): number of parameters
    training_tokens(D): number of total training tokens
    unique_tokens(U): number of unique training tokens
    """

    assert all(unique_tokens <= training_tokens), "Cannot have more unique tokens than total tokens"

    RD = np.maximum((training_tokens / unique_tokens) - 1, 0)
    UN = np.minimum(num_params, D_to_N(training_tokens, G, alpha, beta))
    RN = np.maximum((num_params / UN) - 1, 0)

    L = E + A / (UN + UN * rn_star * (1 - np.exp(-1 * RN / rn_star))) ** alpha + B / (
            unique_tokens + unique_tokens * rd_star * (1 - np.exp(-1 * RD / (rd_star)))) ** beta
    return L


def get_data():
    df = pd.read_csv("aggregated_eval/datablations_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    return df


def to_int(string):
    try:
        return int(string)
    except:
        string = string.strip()
        letter = string[-1]
        if letter.lower() == "b":
            scale = 1e9
        elif letter.lower() == "m":
            scale = 1e6
        elif letter.lower() == "t":
            scale = 1e12
        else:
            raise ValueError(f"Unexpected form for an int:{string}")
        num = float(string[:-1]) * scale
        return num


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
    metadata = prepare_for_fit(subdf, DATA_COLS)
    guess = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]

    # Invent performance:
    perf = np.sum(metadata * [1e-11, 1e-9, 1e-11], axis=1)

    # bounds = []
    popt, pcov = curve_fit(target_func, metadata, perf, p0=guess)
    print("predicted", datablations_fit(metadata[:5], *popt))
    print("actual", perf[:5])


def fit_per_model():
    df = get_data()


def data_aware_fit(show=False):
    df = get_data()
    subdf = df[df['data'] == "c4"]
    # models = num_params.apply(to_int), tokens_seen, tokens_per_epoch
    perf = subdf["loss"]
    target_func = datablations_fit
    metadata = prepare_for_fit(subdf, DATA_COLS)
    guess = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]
    # bounds = []
    popt, pcov = curve_fit(target_func, metadata, perf, p0=guess)
    predicted = datablations_fit(metadata, *popt)
    fig, ax = plt.subplots()
    gca = plt.gca()

    # predictions only
    for num_params in metadata["num_params"].unique():
        for tokens_per_epoch in metadata["tokens_per_epoch"].unique():
            indx = (tokens_per_epoch == metadata["tokens_per_epoch"]) & (num_params == metadata["num_params"])
            x = metadata[indx][
                "token_num"]
            # sns.lineplot(x=x, y=perf[indx], label=f"actual {num_params} {tokens_per_epoch}")
            # color = gca.lines[-1].get_color()
            sns.lineplot(x=x, y=predicted[indx], label=f"predicted {num_params} {tokens_per_epoch}",
                         linestyle='dashed')
            print(predicted)
    plt.xscale('log')
    plt.xlabel("token_num")
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
            x = metadata[indx][
                "token_num"]
            sns.lineplot(x=x, y=perf[indx], label=f"actual {num_params} {tokens_per_epoch}")
            color = gca.lines[-1].get_color()
            sns.lineplot(x=x, y=predicted[indx], color=color, label=f"predicted {num_params} {tokens_per_epoch}",
                         linestyle='dashed')
            print(predicted)
    plt.xscale('log')
    plt.xlabel("token_num")
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
    data_aware_fit()
