import json
import math
import os
import time

import numpy as np
import pandas as pd
import seaborn
from scipy.optimize import curve_fit

import sklearn.metrics
import matplotlib

from fitting_funcs import DATA_FIT_COLS, TestFit, FitInfo, ChinchillaFit, DatablationsFit
from util.cache import get_cache, save_cache
from util.naming import to_int, to_str
from util.plots import plot_pred_actual_compare

import matplotlib.pyplot as plt

matplotlib.use('QtAgg')

from util.read_data import get_data

plt.interactive(False)
import seaborn as sns

# func = r"$L(N,D,R_N,R_D)=E + \frac{A}{(U_N + U_N * R_N^* * (1 - e^{(-1*R_N/(R_N^*))}))^\alpha} + \frac{B}{(U_D + U_D * R_D^* * (1 - e^{(-1*R_D/(R_D^*))}))^\beta}$"
# a, b, e, alpha, beta, rd_star, rn_star = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]
epsilon = 1e-6


def mean_squared_error(pred, gold):
    try:
        return sklearn.metrics.mean_squared_error(pred, gold)
    except ValueError:
        if np.isinf(pred).any():
            return
        else:
            raise


def test_function_fit():
    df = get_data()
    subdf = df[df['data'] == "c4"]
    # models = num_params.apply(to_int), tokens_seen, tokens_per_epoch
    fit_info = TestFit
    metadata = prepare_for_fit(subdf, DATA_FIT_COLS)

    # Invent performance:
    perf = np.sum(metadata * [1e-11, 1e-9, 1e-11], axis=1)

    popt, pcov = curve_fit(fit_info.func, metadata, perf, p0=fit_info.guess, bounds=fit_info.bounds)
    print("predicted", fit_info.func(metadata[:5], *popt))
    print("actual", perf[:5])


def prepare_for_fit(df, columns):
    # preprocess
    df["data"] = pd.get_dummies(df["data"])

    # remove non numerals
    assert not set(columns).difference(df.select_dtypes([np.number]).columns)
    # get necessary columns
    metadata = df.iloc[:, [col in columns for col in df.columns]]
    # metadata = list(df.itertuples())
    return metadata


def fit(fit_info: FitInfo, metadata, perf, default=None):
    try:
        try:
            popt, pcov = curve_fit(fit_info.func, metadata, perf, p0=fit_info.guess, bounds=fit_info.bounds)
        except RuntimeError as e:
            popt, pcov = curve_fit(fit_info.func, metadata, perf, p0=fit_info.guess,
                                   method='dogbox',
                                   bounds=fit_info.bounds)  # not sure who of the three is best (lm can only be used without bounds)
            print(
                f"Was hard to fit, are some of the diagonal variances very high? This might indicate redundant parameters:{np.diag(pcov)}")
    except Exception as e:
        if metadata is None or metadata.empty:
            print("Fit failed, no metadata")
        elif not fit_info.guess:
            print(f"Fit failed, guess is {fit_info.guess}")
            return None
        else:
            print(
                f"Fit failed {metadata['model_name'].unique()[0]}, data size:{len(metadata)}, number of params to fit {len(fit_info.guess)}")
        if default is None:
            return default
        else:
            popt = np.zeros_like(fit_info.guess) + default
            pcov = None
    return popt


def aggregate_row_loss(row, graceful=False):
    return np.mean([row[x] for x in row["loss_cols"] if not graceful or x in row])


def normalize_metric(metric, score):
    if metric == "perp":
        return score
    elif "acc" in metric:
        if "100" not in metric:
            score *= 100
        return math.log(score)
    elif "likelihood" in metric:
        return score
    elif "unk" in metric:
        return score
    else:
        raise ValueError((metric, score))


def aggregate_loss(subdf, graceful=False):
    # col_to_metric = metric_per_column(subdf)
    # for col, metric in col_to_metric.items(): # assumes already normalized
    #     subdf[col] = subdf[col].apply(lambda x: normalize_metric(metric, x))
    return subdf.apply(lambda row: aggregate_row_loss(row, graceful=graceful), axis=1)


def fit_per_model(df, predict_with=0.3, force=False):
    fit_info = ChinchillaFit

    data = []
    cache_name = f"fit_per_model_{predict_with}"
    cache = get_cache(cache_name, force)
    for model_name in df["model_name"].unique():
        train_df = get_model_data(df=df, models=[model_name], max_percentage=predict_with)
        cash_id = model_name + str(predict_with)
        popt = cached_fit(cache, cash_id, fit_info, train_df, train_df["perf"])

        if popt is None:
            continue
        test_df = get_model_data(df=df, models=[model_name], min_percentage=predict_with)
        predicted = fit_info.func(test_df, *popt)
        subdf = pd.merge(train_df, test_df, how='right', indicator="in_fit")
        subdf["in_fit"] = subdf["in_fit"].apply(lambda x: True if x == "both" else False)
        subdf["pred"] = predicted
        data.append(subdf)
    save_cache(cache, cache_name)
    data = pd.concat(data)
    plot_pred_actual_compare(data, show=True)


def get_model_data(df, models, min_percentage=None, min_tokens=None, max_percentage=None, max_tokens=None):
    df = df.loc[df["model_name"].isin(models)]
    tokens = df["tokens_seen"].max()
    if min_tokens:
        df = df[df["tokens_seen"] > min_tokens]
    if max_tokens:
        df = df[df["tokens_seen"] <= max_tokens]
    for model in df["model_name"].unique():
        if min_percentage:
            min_model_tokens = tokens * min_percentage
            df = df.query("model_name != @model or tokens_seen >= @min_model_tokens")
        if max_percentage:
            max_model_tokens = tokens * max_percentage
            df = df.query("model_name != @model or tokens_seen < @max_model_tokens")
    return df


def cached_fit(cache, cache_id, fit_info, metadata, perf, default=None):
    if cache_id in cache:
        popt = cache[cache_id]
    else:
        popt = fit(fit_info, metadata, perf, default)
        cache[cache_id] = [x for x in popt]
    return popt


def normalize_loss(df, loss_types=None):
    metric_map = metric_per_column(df)
    if loss_types:
        drop_cols = [col for col, loss_type in metric_map.items() if loss_type not in loss_types and loss_type != "unk"]
        df = df.drop(columns=drop_cols)
    for col, loss_type in metric_map.items():
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda x: normalize_metric(loss_type, x))
    return df


def scale_fit_per_model(df, force=False, fig_dir=None, loss_types=["perp"], at_least_loss=float("inf")):
    fit_info = ChinchillaFit
    test_percentage = 0.3

    # data = []
    cache_name = f"scale_fit_per_model" + "_".join(loss_types)
    cache = get_cache(cache_name, force)
    df = normalize_loss(df, loss_types=loss_types)
    df = df.assign(perf=aggregate_loss(df, graceful=True))
    df = df.dropna(subset=["perf"])
    evals = []
    for model_name in df["model_name"].unique():
        if df.query("model_name==@model_name")["perf"].min() > at_least_loss:
            continue
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            cache_id = model_name + str(percentage)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                train_df = get_model_data(df=df, models=[model_name], max_percentage=percentage)
                test_df = get_model_data(df=df, models=[model_name], min_percentage=test_percentage)
                if train_df.empty or test_df.empty:
                    mse = None
                    mnd = None
                    predicted = None
                else:
                    popt = fit(fit_info, train_df, train_df["perf"])
                    if popt is None:
                        mse = None
                        mnd = None
                        predicted = None
                    else:
                        predicted = fit_info.func(test_df, *popt)
                        mse = mean_squared_error(predicted, test_df["perf"])
                        mnd = (np.abs(test_df["perf"] - predicted) / test_df["perf"]).mean()
                last_pred = predicted[-1] if predicted is not None else None
                res = (model_name, percentage, mse, mnd, last_pred)
                cache[cache_id] = res
                print(f"{model_name} {percentage}: {mse}, {mnd}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=["model_name", "percentage", "mse", "mnd", "last_pred"])

    plt.clf()
    # eval = "last_pred"
    # for i, model in enumerate(evals["model_name"].unique()):
    #     if i % 40 != 0:
    #         continue
    #     test_df = get_model_data(df=df, models=[model], min_percentage=test_percentage)
    #     subdf = evals.query("model_name==@model").dropna(subset=[eval])
    #     if subdf.empty:
    #         continue
    #     sns.lineplot(x=test_df["tokens_seen"], y=test_df["perf"])
    #     y = subdf[eval]
    #     x = [test_df["tokens_seen"].iloc[-1]] * len(y)
    #     sns.scatterplot(x=x, y=y.tolist())
    #
    # plt.ylim(bottom=0)
    # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # if fig_dir:
    #     plt.savefig(os.path.join(fig_dir, f"per_model_{eval}.png"), bbox_inches="tight")
    # plt.show()
    for eval in ["mse", "mnd"]:
        plt.clf()
        for model in evals["model_name"].unique():
            subdf = evals.query("model_name==@model").dropna(subset=[eval])
            if subdf.empty:
                continue
            x = subdf["percentage"]
            y = subdf[eval]
            sns.lineplot(x=x.tolist(), y=y.tolist(), label=model)

        plt.ylim(bottom=0)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        if fig_dir:
            plt.savefig(os.path.join(fig_dir, f"per_model_{eval}.png"), bbox_inches="tight")
        plt.show()



    print("done")
    # data = pd.concat(data)
    # plot_pred_actual_compare(data, show=True)


def scaling_scaling_law(df, force=False):
    fit_info = ChinchillaFit

    data = []
    cache_name = f"scaling_scaling_law"
    cache = get_cache(cache_name, force)
    for model_name in df["model_name"].unique():

        subdf = df[df["model_name"] == model_name]
        subdf = subdf.sort_values("tokens_seen")
        subdf["perf"] = aggregate_loss(subdf, graceful=True)

        metadata = subdf.iloc[:int(len(subdf["perf"]) * predict_with), :]
        perf = metadata["perf"]
        if model_name in cache:
            popt = cache[model_name]
        else:
            popt = fit(fit_info, metadata, perf, default=epsilon)
            cache[model_name] = [x for x in popt]

        predicted = fit_info.func(subdf, *popt)
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
    fit_info = DatablationsFit
    metadata = prepare_for_fit(subdf, DATA_FIT_COLS)
    popt, pcov = curve_fit(fit_info.func, metadata, perf, p0=fit_info.guess)
    predicted = fit_info.func(metadata, *popt)
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


def contains(str, lst):
    try:
        if lst in str:
            return True
        else:
            return False
    except TypeError:
        return any((x in str for x in lst))


_metric_map = {}


def metric_per_column(df):
    if _metric_map:
        return _metric_map
    for column in df.select_dtypes(include='number'):
        if column in _metric_map:
            continue
        elif contains(column, ("norm", "quasi")) and "acc" in column:
            _metric_map[column] = "norm_acc"
        elif contains(column, ("acc", "pct_ster", "exact_match", " em", "_em", "downstream_eval", "bpb")):
            _metric_map[column] = "1acc" if df[column].max() <= 1 else "100acc"
        elif contains(column, ("loss", "perp", "ppl")):
            _metric_map[column] = "perp"
        elif contains(column, "likelihood_difference"):
            _metric_map[column] = "likelihood_difference"
        elif contains(column, "bits_per_byte"):
            _metric_map[column] = "bpb"
        else:
            min_score, max_score = (df[column].min(), df[column].max())
            if (min_score >= 0 and max_score <= 1):
                _metric_map[column] = "1acc_like"
            elif (min_score >= 0 and 10 < max_score < 100):
                _metric_map[column] = "100acc_like"
            else:
                _metric_map[column] = "unk"
    return _metric_map


if __name__ == '__main__':
    force = True
    force = False
    data_path = '/Users/lc/PycharmProjects/CLPR/cache/data.csv'
    df = get_data(force=force)
    force = False
    force = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    # df = df[df["model_type"].isin(["OPT"])]
    # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    # fit_per_model(df, force=force)
    scale_fit_per_model(df, force=force, fig_dir=fig_dir, at_least_loss=10)
    # scale_fit_per_family(df, force=force, save_fig=fig_dir)
    # fit_on_smaller(df,force)
    # scaling_scaling_law(df, force)
    # cross_validate(df,force)

    # data_aware_fit()
