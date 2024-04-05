import os

import numpy as np
import pandas as pd

from util.fit_utils import plot_models_percentage_hist, mean_normalized_distance, fit, get_model_data, get_perf_df, \
    metric_per_column, mean_squared_error, get_perf_path, get_data_path, single_scaling
from fitting_funcs import ChinchillaFit, bound_params
from util.cache import save_cache, get_cache
from util.read_data import get_data


def predict_smallest(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"),
                     num_train_models=4, abs_mnd=True):
    cut_beginning = 10 ** 10
    fit_info = ChinchillaFit
    test_percentage = 0.7
    experiment_name = "predict_smallest"
    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = experiment_name + "_" + str(abs_mnd) + "_".join(loss_types) + "_" + str(num_train_models)
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []

    for scaled_set in df["scaled_set"].unique():
        model_by_size = df.query("scaled_set==@scaled_set")[["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        smallest_model = model_by_size["model_name"].iloc[0]
        larger_models = model_by_size["model_name"].iloc[1:]
        # skip bad models
        largest_model = model_by_size["model_name"].iloc[-1]
        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue
        for num_train_models in range(2, len(larger_models) + 1):
            cache_id = scaled_set + str(num_train_models) + str(test_percentage)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                train_df = get_model_data(df=df, models=larger_models.to_list()[:num_train_models],
                                          max_percentage=test_percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[smallest_model], min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
                mse, mnd, train_mnd, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_mnd)
                last_pred = predicted[-1] if predicted is not None else None
                res = (scaled_set, mse, mnd, last_pred, smallest_model, num_train_models + 1,
                       tuple(popt) if popt is not None else None)
                cache[cache_id] = res
                print(f"{scaled_set} {num_train_models + 1}: {mse}, {mnd}, {train_mnd}")
                evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)
    evals = pd.DataFrame(evals, columns=["scaled_set", "mse", "mnd", "last_pred", "smallest_model",
                                         "num_models", "params"])
    # print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['largest_model']}")
    evals = evals.loc[:, ["scaled_set", "mnd", "num_models"]]
    plot_models_percentage_hist(evals, eval="mnd", index="num_models", columns="scaled_set", fig_dir=fig_dir,
                                min_rows=1, show=show)

def closer_in_scale_is_predictive(df, force=False, fig_dir=None, show=False, loss_types=("perp"),
                                  at_least_loss=float("inf"),
                                  num_train_models=4, abs_mnd=True):
    cut_beginning = 10 ** 10
    fit_info = ChinchillaFit
    test_percentage = 0.7
    experiment_name = "closer_in_scale_is_predictive"
    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = experiment_name + "_" + str(abs_mnd) + "_".join(loss_types) + "_" + str(num_train_models)
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []
    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    for scaled_set in df["scaled_set"].unique():
        model_by_size = df.query("scaled_set==@scaled_set")[
            ["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        model_names = model_by_size["model_name"]
        largest_model = model_names.iloc[-1]
        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue
        for i in range(len(model_names) - num_train_models):
            train_models_ids = np.arange(i, i + num_train_models)
            cache_id = scaled_set + str(i)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                largest_train_model = model_names.iloc[train_models_ids].iloc[-1]
                num_params = df.query("model_name==@largest_train_model")["num_params"].iloc[0]
                train_df = get_model_data(df=df, models=model_names.iloc[train_models_ids],
                                          max_percentage=test_percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[largest_model],
                                         min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
                mse, mnd, train_mnd, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_mnd)

                last_pred = predicted[-1] if predicted is not None else None
                res = (scaled_set, mse, mnd, last_pred, largest_model, num_train_models + 1, num_params,
                       tuple(popt) if popt is not None else None)
                cache[cache_id] = res
                print(f"{scaled_set} {largest_model} {num_train_models + 1}: {mnd}, {train_mnd}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    evals = pd.DataFrame(evals, columns=["scaled_set", "mse", "mnd", "last_pred", "largest_model",
                                         "num_models", "num_params", "params"])
    print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['largest_model']}")
    evals = evals.loc[:, ["scaled_set", "mnd", "num_models", "num_params"]]
    plot_models_percentage_hist(evals, eval="mnd", index="num_models", columns="num_params", fig_dir=fig_dir,
                                min_rows=1, show=show)


def larger_is_predictable(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"),
                          num_train_models=4, abs_mnd=True):
    cut_beginning = 10 ** 10
    fit_info = ChinchillaFit
    test_percentage = 0.7
    experiment_name = "larger_is_predictable"
    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = experiment_name + "_" + str(abs_mnd) + "_".join(loss_types) + "_" + str(num_train_models)
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []
    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    for scaled_set in df["scaled_set"].unique():
        model_by_size = df.query("scaled_set==@scaled_set")[
            ["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        model_names = model_by_size["model_name"]
        best_model = model_names.iloc[-1]
        if df.query("model_name==@best_model")["perf"].min() > at_least_loss:
            continue
        for i in range(len(model_names) - num_train_models):
            train_models_ids = np.arange(i, i + num_train_models)
            cache_id = scaled_set + str(i)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                largest_model = model_names.iloc[i + num_train_models]
                num_params = df.query("model_name==@largest_model")["num_params"].iloc[0]
                train_df = get_model_data(df=df, models=model_names.iloc[train_models_ids],
                                          max_percentage=test_percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[largest_model],
                                         min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
                mse, mnd, train_mnd, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_mnd)

                last_pred = predicted[-1] if predicted is not None else None
                res = (scaled_set, mse, mnd, last_pred, largest_model, num_train_models + 1, num_params,
                       tuple(popt) if popt is not None else None)
                cache[cache_id] = res
                print(f"{scaled_set} {largest_model} {num_train_models + 1}: {mnd}, {train_mnd}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    evals = pd.DataFrame(evals, columns=["scaled_set", "mse", "mnd", "last_pred", "largest_model",
                                         "num_models", "num_params", "params"])
    print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['largest_model']}")
    evals = evals.loc[:, ["scaled_set", "mnd", "num_models", "num_params"]]
    plot_models_percentage_hist(evals, eval="mnd", index="num_models", columns="num_params", fig_dir=fig_dir,
                                min_rows=1, show=show)


if __name__ == '__main__':
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = ("perp", "loss")
    perf_path = get_perf_path(cache_dir, loss_types)
    df = get_perf_df(get_data(save_in=data_path, force=force), loss_types, save_in=perf_path, force=force)
    force = False
    force = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    # df = df[df["model_type"].isin(["OPT"])]
    # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    abs_mnd = True
    predict_smallest(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    closer_in_scale_is_predictive(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    larger_is_predictable(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
