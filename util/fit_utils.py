import ast
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

from fitting_funcs import DATA_FIT_COLS, TestFit, FitInfo, ChinchillaFit, DatablationsFit, bound_params, \
    Chinchilla1ModelFit
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

def single_scaling(train_df, test_df, fit_info, abs_mnd):
    if train_df.empty or test_df.empty:
        popt = None
    else:
        popt = fit(fit_info, train_df, train_df["perf"])
    if popt is None:
        mse = None
        mnd = None
        train_mnd = None
        predicted = None
    else:
        predicted = fit_info.func(test_df, *popt)
        mse = mean_squared_error(predicted, test_df["perf"])
        mnd = mean_normalized_distance(predicted, test_df["perf"], abs_mnd)
        predicted_train = fit_info.func(train_df, *popt)
        train_mnd = mean_normalized_distance(predicted_train, train_df["perf"], abs_mnd)
    return mse,mnd,train_mnd,predicted, popt


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
    except AssertionError as e:
        raise
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
    return np.mean([row[x] for x in ast.literal_eval(row["loss_cols"]) if not graceful or x in row])


def normalize_metric(metric, score):
    if metric == "perp":
        return np.log2(score)
    if metric == "loss":
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
    cut_beginning = 10 ** 8

    data = []
    cache_name = f"fit_per_model_{predict_with}"
    cache = get_cache(cache_name, force)
    for model_name in df["model_name"].unique():
        train_df = get_model_data(df=df, models=[model_name], max_percentage=predict_with, min_tokens=cut_beginning)
        cash_id = model_name + str(predict_with)
        popt = cached_fit(cache, cash_id, fit_info, train_df, train_df["perf"])

        if popt is None:
            continue
        test_df = get_model_data(df=df, models=[model_name], min_percentage=predict_with, min_tokens=cut_beginning)
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


def get_perf_df(df, loss_types, graceful=True, force=False, save_in=None):
    if save_in and not force:
        if os.path.isfile(save_in):
            return pd.read_csv(save_in)
    df = normalize_loss(df, loss_types=loss_types)
    perf = aggregate_loss(df, graceful=graceful)
    df = df.assign(perf=perf)
    df = df.dropna(subset=["perf"])
    if save_in:
        df.to_csv(save_in, index=False)
    return df


def scale_fit_per_model(df, force=False, fig_dir=None, show=False, loss_types=["perp"], at_least_loss=float("inf"),
                        abs_mnd=True):
    """
    Predict each model with the begginning of its own training
    Args:
        df:
        force:
        fig_dir:
        show:
        loss_types:
        at_least_loss:
        abs_mnd:

    Returns:

    """
    cut_beginning = 10 ** 8
    fit_info = ChinchillaFit
    test_percentage = 0.7
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"scale_fit_per_model_" + str(abs_mnd) + "_".join(loss_types)
    cache = get_cache(cache_name, force)
    evals = []
    for model_name in df["model_name"].unique():
        if df.query("model_name==@model_name")["perf"].min() > at_least_loss:
            continue
        for percentage in train_percentages:
            cache_id = model_name + str(percentage)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                train_df = get_model_data(df=df, models=[model_name], max_percentage=percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[model_name], min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
                # replaced by single_scaling
                # if train_df.empty or test_df.empty:
                #     mse = None
                #     mnd = None
                #     predicted = None
                # else:
                #     popt = fit(fit_info, train_df, train_df["perf"])
                #     if popt is None:
                #         mse = None
                #         mnd = None
                #         predicted = None
                #     else:
                #         predicted = fit_info.func(test_df, *popt)
                #         mse = mean_squared_error(predicted, test_df["perf"])
                #         mnd = (test_df["perf"] - predicted) / test_df["perf"]
                #         if abs_mnd:
                #             mnd = np.abs(mnd)
                #         mnd = mnd.mean()
                mse, mnd, train_mnd, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_mnd=abs_mnd)

                last_pred = predicted[-1] if predicted is not None else None
                res = (model_name, percentage, mse, mnd, last_pred)
                cache[cache_id] = res
                print(f"{model_name} {percentage}: {mse}, {mnd}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=["model_name", "percentage", "mse", "mnd", "last_pred"])
    print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['model_name']}")

    # # plot on all models
    # for eval in ["mse", "mnd"]:
    #     plt.clf()
    #     for model in evals["model_name"].unique():
    #         subdf = evals.query("model_name==@model").dropna(subset=[eval])
    #         if subdf.empty:
    #             continue
    #         x = subdf["percentage"]
    #         y = subdf[eval]
    #         sns.lineplot(x=x.tolist(), y=y.tolist(), label=model)
    #
    #     plt.ylim(bottom=0)
    #     plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    #     if fig_dir:
    #         plt.savefig(os.path.join(fig_dir, f"per_model_{eval}.png"), bbox_inches="tight")
    #     plt.show()

    # Group by characteristics
    def add_info(row):
        to_add = df[row["model_name"] == df["model_name"]].iloc[0, :]
        metric_map = metric_per_column(df)
        to_add = to_add[
            (col for col in to_add.index if
             col not in row.index and (col not in metric_map or metric_map[col] == "unk"))]
        return pd.concat([row, to_add])

    evals = evals.apply(add_info, axis=1)
    # TODO model size, model family, final loss
    bins = np.percentile(evals["num_params"].unique(), np.linspace(20, 100, 5))
    # bins = [to_int(num) for num in ["100m", "500m", "1B", "5B", "10B"]]
    evals["model_size"] = np.digitize(evals["num_params"], bins)
    evals["best_loss"] = evals["model_name"].apply(lambda model: df.query("@model==model_name")["perf"].min())
    bins = np.percentile(evals["best_loss"].unique(), np.linspace(20, 100, 5))
    evals["best_loss_binned"] = np.digitize(evals["best_loss"], bins)

    evals["max_tokens"] = evals["model_name"].apply(lambda model: df.query("@model==model_name")["tokens_seen"].max())
    bins = np.percentile(evals["max_tokens"].unique(), np.linspace(20, 100, 5))
    evals["max_tokens_binned"] = np.digitize(evals["max_tokens"], bins)

    eval = "mnd"
    for col_group in ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type",
                      "best_loss_binned"]:
        plt.clf()
        # mnds = evals.groupby([col_group, "percentage"])["mnd"].mean().reset_index([1])
        for group in evals[col_group].unique():
            subdf = evals.query(f"{col_group}==@group").dropna(subset=[eval])
            subdf = subdf.groupby(["percentage"])[eval].mean().reset_index()
            if subdf.empty:
                continue
            x = subdf["percentage"]
            y = subdf[eval]
            sns.lineplot(x=x.tolist(), y=y.tolist(), label=group)

        if abs_mnd:
            plt.ylim(bottom=0)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title=col_group)
        plt.xlabel("Percentage of Training Trajectory Available")
        plt.ylabel("Mean Normalized Distance")
        if fig_dir:
            plt.savefig(os.path.join(fig_dir, f"per_model_{eval}_{col_group}.png"), bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()

    # compare predictions to actual
    eval = "mnd"
    for col_group in [
        "original_paper"]:  # ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type", "best_loss_binned"]
        for group in evals[col_group].unique():
            for model in evals[evals[col_group] == group]["model_name"].unique():
                # if "pythi" in group and "pythia-1.4b" not in model:
                #     continue
                subdf = df.query("model_name==@model & tokens_seen>=@cut_beginning")
                ax = sns.lineplot(x=subdf["tokens_seen"], y=subdf["perf"], label=f"{group} {model}")
                color = ax.lines[-1].get_color()
                subdf = evals.query("model_name==@model")
                x = subdf["max_tokens"]
                y = subdf["last_pred"]
                if y.dropna().empty:
                    continue
                sns.scatterplot(x=x, y=y, color=color)
                for i, txt in enumerate(train_percentages):
                    ax.annotate(txt, (x.iloc[i], y.iloc[i]))
            plt.xscale('log')
            plt.xlabel("tokens_seen")
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title=f"{col_group}:{group}")

            # plt.legend().remove()
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            # ax.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))
            if fig_dir:
                os.makedirs(os.path.join(fig_dir, "compare_predictions"), exist_ok=True)
                plt.savefig(os.path.join(fig_dir, "compare_predictions", f"{group}_{col_group}_{eval}.png"),
                            bbox_inches="tight")
            if show:
                plt.show()

            plt.clf()
    print("done")


def mean_normalized_distance(predicted, target, abs):
    mnd = (target - predicted) / target
    if abs:
        mnd = np.abs(mnd)
    mnd = mnd.mean()
    return mnd


def plot_models_percentage_hist(evals, eval, fig_dir, iterate_over="scaled_set", index="num_train_models",
                                columns="percentage", vmin=0, vmax=1, min_rows=2,
                                show=False):
    for scaled_set in evals[iterate_over].unique():
        pivot = evals.query(f"@{iterate_over}=={iterate_over}").pivot(index=index, columns=columns,
                                                                      values=eval)
        if len(pivot.dropna(axis=0, how='all')) < min_rows:
            print(f"Skipping {scaled_set}, no different scales of smaller models {pivot}")
            continue
        plt.title(scaled_set)
        sns.heatmap(100 * pivot, annot=True, vmin=100 * vmin, vmax=100 * vmax)
        if fig_dir:
            plt.savefig(os.path.join(fig_dir, f"hist-perc-{eval}_{scaled_set}.png"), bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()


def hist_one_model_fit(df, force=False, fig_dir=None, show=False, loss_types=["perp"], at_least_loss=float("inf"),
                       train_percentages=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
                       abs_mnd=True):
    cut_beginning = 10 ** 10
    fit_info = bound_params(ChinchillaFit, [6.255414, None, None, 0.3526596])
    # fit_info = Chinchilla1ModelFit
    test_percentage = 0.7

    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"hist_one_model_fit_{abs_mnd}_" + "_".join(loss_types)
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []
    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    for scaled_set in df["scaled_set"].unique():
        model_by_size = df.query("scaled_set==@scaled_set")[["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        largest_model = model_by_size["model_name"].iloc[-1]
        smaller_models = model_by_size["model_name"]  # to exclude the last one: .iloc[:-1]
        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue
        for percentage in train_percentages:
            for num_model in range(0, len(smaller_models)):
                cache_id = scaled_set + str(num_model) + str(percentage)
                current_model = smaller_models.to_list()[num_model]
                train_model_size = df.query("model_name==@current_model")["num_params"]
                assert train_model_size.nunique() == 1
                train_model_size = train_model_size.iloc[0] / 1e9
                if not force and cache_id in cache:
                    res = cache[cache_id]
                else:
                    train_df = get_model_data(df=df, models=[current_model],
                                              max_percentage=percentage,
                                              min_tokens=cut_beginning)
                    test_df = get_model_data(df=df, models=[largest_model], min_percentage=test_percentage,
                                             min_tokens=cut_beginning)
                    # replaced by single_scaling
                    # if train_df.empty or test_df.empty:
                    #     popt = None
                    # else:
                    #     popt = fit(fit_info, train_df, train_df["perf"])
                    # if popt is None:
                    #     mse = None
                    #     mnd = None
                    #     train_mnd = None
                    #     predicted = None
                    # else:
                    #     predicted = fit_info.func(test_df, *popt)
                    #     mse = mean_squared_error(predicted, test_df["perf"])
                    #     mnd = mean_normalized_distance(predicted, test_df["perf"], abs_mnd)
                    #     predicted_train = fit_info.func(train_df, *popt)
                    #     train_mnd = mean_normalized_distance(predicted_train, train_df["perf"], abs_mnd)
                    mse, mnd, train_mnd, predicted, popt = single_scaling(train_df, test_df, fit_info,abs_mnd=abs_mnd)
                    last_pred = predicted[-1] if predicted is not None else None
                    res = (scaled_set, percentage, mse, mnd, last_pred, largest_model, num_model + 1, current_model,
                           train_model_size,
                           tuple(popt) if popt is not None else None)
                    cache[cache_id] = res
                    print(f"{scaled_set} {percentage} {num_model}: {mse}, {mnd}, {train_mnd}")
                evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=["scaled_set", "percentage", "mse", "mnd", "last_pred", "largest_model",
                                         "num_train_models", "smaller_model", "train_model_size", "popt"])
    evals["popt"] = evals["popt"].apply(np.asarray)
    # print(f"Mean guess: {evals.groupby('scaled_set')['popt'].mean()}")
    # print(f"Mean guess: {evals['popt'].mean()}")
    print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['scaled_set']}")
    eval = "mnd"
    plot_models_percentage_hist(evals, eval=eval, index="train_model_size", columns="percentage", fig_dir=fig_dir,
                                show=show)


def hist_fit(df, force=False, fig_dir=None, show=False, loss_types=["perp"], at_least_loss=float("inf"),
             train_percentages=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
             abs_mnd=True):
    """
    Predict with each M models given x percentage of training the end of the last model's loss
    Args:
        df:
        force:
        fig_dir:
        show:
        loss_types:
        at_least_loss:
        train_percentages:
        abs_mnd:

    Returns:

    """
    cut_beginning = 10 ** 10
    fit_info = ChinchillaFit
    test_percentage = 0.7

    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"hist_fit_{abs_mnd}_" + "_".join(loss_types)
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []
    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    for scaled_set in df["scaled_set"].unique():
        model_by_size = df.query("scaled_set==@scaled_set")[["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        largest_model = model_by_size["model_name"].iloc[-1]
        smaller_models = model_by_size["model_name"].iloc[:-1]
        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue
        for percentage in train_percentages:
            for num_train_models in range(2, len(smaller_models) + 1):
                cache_id = scaled_set + str(num_train_models) + str(percentage)
                if not force and cache_id in cache:
                    res = cache[cache_id]
                else:
                    train_df = get_model_data(df=df, models=smaller_models.to_list()[:num_train_models],
                                              max_percentage=percentage,
                                              min_tokens=cut_beginning)
                    test_df = get_model_data(df=df, models=[largest_model], min_percentage=test_percentage,
                                             min_tokens=cut_beginning)
                    if train_df.empty or test_df.empty:
                        popt = None
                    else:
                        popt = fit(fit_info, train_df, train_df["perf"])
                    if popt is None:
                        mse = None
                        mnd = None
                        train_mnd = None
                        predicted = None
                    else:
                        predicted = fit_info.func(test_df, *popt)
                        mse = mean_squared_error(predicted, test_df["perf"])
                        mnd = mean_normalized_distance(predicted, test_df["perf"], abs_mnd)
                        predicted_train = fit_info.func(train_df, *popt)
                        train_mnd = mean_normalized_distance(predicted_train, train_df["perf"], abs_mnd)
                        if percentage > test_percentage:
                            end_train_df = get_model_data(df=df, models=smaller_models.to_list()[:num_train_models],
                                                          max_percentage=percentage,
                                                          min_percentage=percentage - test_percentage)
                            end_predicted_train = fit_info.func(end_train_df, *popt)
                            end_train_mnd = mean_normalized_distance(end_predicted_train, end_train_df["perf"], abs_mnd)
                            if np.abs(train_mnd - end_train_mnd) > 0.5:
                                print(
                                    f"{scaled_set} {percentage} {num_train_models + 1}: mnd "
                                    f"on all train:{train_mnd} and only on the end:{end_train_mnd}")
                    last_pred = predicted[-1] if predicted is not None else None
                    res = (scaled_set, percentage, mse, mnd, last_pred, largest_model, num_train_models + 1,
                           tuple(popt) if popt is not None else None)
                    cache[cache_id] = res
                    print(f"{scaled_set} {percentage} {num_train_models + 1}: {mse}, {mnd}, {train_mnd}")
                evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=["scaled_set", "percentage", "mse", "mnd", "last_pred", "largest_model",
                                         "num_train_models", "popt"])
    evals["popt"] = evals["popt"].apply(np.asarray)
    # print(f"Mean guess: {evals.groupby('scaled_set')['popt'].mean()}")
    # print(f"Mean guess: {evals['popt'].mean()}")
    print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['scaled_set']}")
    eval = "mnd"
    plot_models_percentage_hist(evals, eval=eval, fig_dir=fig_dir, show=show)

    # # plot on all models
    # for eval in ["mse", "mnd"]:
    #     plt.clf()
    #     for model in evals["model_name"].unique():
    #         subdf = evals.query("model_name==@model").dropna(subset=[eval])
    #         if subdf.empty:
    #             continue
    #         x = subdf["percentage"]
    #         y = subdf[eval]
    #         sns.lineplot(x=x.tolist(), y=y.tolist(), label=model)
    #
    #     plt.ylim(bottom=0)
    #     plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    #     if fig_dir:
    #         plt.savefig(os.path.join(fig_dir, f"per_model_{eval}.png"), bbox_inches="tight")
    #     plt.show()

    # Group by characteristics
    def add_info(row):
        to_add = df[row["largest_model"] == df["model_name"]].iloc[0, :]
        metric_map = metric_per_column(df)
        to_add = to_add[
            (col for col in to_add.index if
             col not in row.index and (col not in metric_map or metric_map[col] == "unk"))]
        return pd.concat([row, to_add])

    evals = evals.apply(add_info, axis=1)
    # TODO model size, model family, final loss
    bins = np.percentile(evals["num_params"].unique(), np.linspace(20, 100, 5))
    # bins = [to_int(num) for num in ["100m", "500m", "1B", "5B", "10B"]]
    evals["model_size"] = np.digitize(evals["num_params"], bins)
    evals["best_loss"] = evals["largest_model"].apply(lambda model: df.query("@model==model_name")["perf"].min())
    bins = np.percentile(evals["best_loss"].unique(), np.linspace(20, 100, 5))
    evals["best_loss_binned"] = np.digitize(evals["best_loss"], bins)

    evals["max_tokens"] = evals["largest_model"].apply(
        lambda model: df.query("@model==model_name")["tokens_seen"].max())
    bins = np.percentile(evals["max_tokens"].unique(), np.linspace(20, 100, 5))
    evals["max_tokens_binned"] = np.digitize(evals["max_tokens"], bins)

    eval = "mnd"
    for col_group in ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type",
                      "best_loss_binned"]:
        plt.clf()
        # mnds = evals.groupby([col_group, "percentage"])["mnd"].mean().reset_index([1])
        for group in evals[col_group].unique():
            subdf = evals.query(f"{col_group}==@group").dropna(subset=[eval])
            subdf = subdf.groupby(["percentage"])[eval].mean().reset_index()
            if df.dropna(axis=1, how='all').empty:
                continue
            x = subdf["percentage"]
            y = subdf[eval]
            sns.lineplot(x=x.tolist(), y=y.tolist(), label=group)

        if abs_mnd:
            plt.ylim(bottom=0)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title=col_group)
        plt.xlabel("Percentage of Training Trajectory Available")
        plt.ylabel("Mean Normalized Distance")
        if fig_dir:
            plt.savefig(os.path.join(fig_dir, f"perc-mnd_{eval}_{col_group}.png"), bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()

    # compare predictions to actual
    eval = "mnd"
    for col_group in [
        "original_paper"]:  # ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type", "best_loss_binned"]
        for group in evals[col_group].unique():
            for model in evals[evals[col_group] == group]["model_name"].unique():
                # if "pythi" in group and "pythia-1.4b" not in model:
                #     continue
                subdf = df.query("model_name==@model & tokens_seen>=@cut_beginning")
                ax = sns.lineplot(x=subdf["tokens_seen"], y=subdf["perf"], label=f"{group} {model}")
                color = ax.lines[-1].get_color()
                subdf = evals.query("model_name==@model")
                x = subdf["max_tokens"]
                y = subdf["last_pred"]
                if y.dropna().empty:
                    continue
                sns.scatterplot(x=x, y=y, color=color)
                for i, txt in enumerate(train_percentages):
                    ax.annotate(txt, (x.iloc[i], y.iloc[i]))
            plt.xscale('log')
            plt.xlabel("tokens_seen")
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title=f"{col_group}:{group}")

            # plt.legend().remove()
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            # ax.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))
            if fig_dir:
                os.makedirs(os.path.join(fig_dir, "compare_predictions"), exist_ok=True)
                plt.savefig(os.path.join(fig_dir, "compare_predictions", f"{group}_{col_group}_{eval}.png"),
                            bbox_inches="tight")
            if show:
                plt.show()

            plt.clf()
    print("done")

#
# def scaling_scaling_law(df, force=False):
#     fit_info = ChinchillaFit
#
#     data = []
#     cache_name = f"scaling_scaling_law"
#     cache = get_cache(cache_name, force)
#     for model_name in df["model_name"].unique():
#
#         subdf = df[df["model_name"] == model_name]
#         subdf = subdf.sort_values("tokens_seen")
#         subdf["perf"] = aggregate_loss(subdf, graceful=True)
#
#         metadata = subdf.iloc[:int(len(subdf["perf"]) * predict_with), :]
#         perf = metadata["perf"]
#         if model_name in cache:
#             popt = cache[model_name]
#         else:
#             popt = fit(fit_info, metadata, perf, default=epsilon)
#             cache[model_name] = [x for x in popt]
#
#         predicted = fit_info.func(subdf, *popt)
#         subdf = pd.merge(metadata, subdf, how='right', indicator="in_fit")
#         subdf["in_fit"] = subdf["in_fit"].apply(lambda x: True if x == "both" else False)
#         subdf["pred"] = predicted
#         data.append(subdf)
#     save_cache(cache, cache_name)
#     data = pd.concat(data)
#     plot_pred_actual_compare(data, show=True)


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
        elif contains(column, "bits_per_byte"):
            _metric_map[column] = "bpb"
        elif contains(column, ("loss")):
            _metric_map[column] = "loss"
        elif contains(column, ("perp", "ppl")):
            _metric_map[column] = "perp"
            if contains(column, "byte"):
                _metric_map[column] = "byte_perp"
        elif contains(column, "likelihood_difference"):
            _metric_map[column] = "likelihood_difference"
        else:
            min_score, max_score = (df[column].min(), df[column].max())
            if (min_score >= 0 and max_score <= 1):
                _metric_map[column] = "1acc_like"
            elif (min_score >= 0 and 10 < max_score < 100):
                _metric_map[column] = "100acc_like"
            else:
                _metric_map[column] = "unk"
    return _metric_map


def get_data_path(cache_dir):
    return os.path.join(cache_dir, "data.csv")


def get_perf_path(cache_dir, loss_types):
    return os.path.join(cache_dir, f"perfDF_{'_'.join(loss_types)}.csv")
