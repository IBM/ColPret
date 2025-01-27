import os

import numpy as np
import pandas as pd
import tqdm

from fitting_funcs import ChinchillaFit, FitInfo
from util.cache import save_cache, get_cache
from util.fit_utils import aggregate_hist, get_model_nums, get_per_model_metadata, plot_models_percentage_hist, get_model_data, get_perf_df, \
    metric_per_column, get_perf_path, get_data_path, single_scaling, nunique_model_size
from util.read_data import get_data


def predict_smallest(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"),
                     num_train_models=4, abs_are=True, fit_info=ChinchillaFit):
    cut_beginning = 10 ** 10
    test_percentage = 0.7
    experiment_name = "predict_smallest"
    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = experiment_name + "_" + str(abs_are) + "_".join(loss_types) + "_" + str(
        num_train_models) + "_" + fit_info.name
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
            cache_id = scaled_set + \
                str(num_train_models) + str(test_percentage)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                train_df = get_model_data(df=df, models=larger_models.to_list()[:num_train_models],
                                          max_percentage=test_percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[smallest_model], min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
                unique_model_sizes = nunique_model_size(train_df)
                mse, are, huber, train_are, predicted, popt = single_scaling(
                    train_df, test_df, fit_info, abs_are)
                last_pred = predicted[-1] if predicted is not None else None
                res = (scaled_set, mse, are, huber, last_pred, smallest_model, unique_model_sizes,
                       tuple(popt) if popt is not None else None)
                cache[cache_id] = res
                print(
                    f"{scaled_set} {num_train_models + 1}: {mse}, {are}, {train_are}")
                evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)
    evals = pd.DataFrame(evals, columns=["scaled_set", "mse", "are", "huber", "last_pred", "smallest_model",
                                         "#Train models", "params"])
    # print(f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['largest_model']}")
    evals = evals.loc[:, ["scaled_set", "are", "#Train models"]]
    plot_models_percentage_hist(evals, eval="are", index="#Train models", columns="scaled_set", fig_dir=fig_dir,
                                min_rows=1, show=show)
    plot_models_percentage_hist(evals, eval="huber", index="#Train models", columns="scaled_set", fig_dir=fig_dir,
                                min_rows=1, show=show)


def closer_in_scale_is_predictive(df, force=False, fig_dir=None, show=False, loss_types=("perp"),
                                  at_least_loss=float("inf"),
                                  num_train_models=4, train_percentages=(0.1, 0.2, 0.3, 0.4, 0.5,
                                                                         0.6, 0.7, 0.8, 0.9, 1), include_test=False,
                                  abs_are=True, cut_beginning=10 ** 10, fit_info: FitInfo = ChinchillaFit,  verbose=False):

    test_percentage = 0.7
    if num_train_models is None:
        num_train_models = 4
    experiment_name = f"closer_in_scale_is_predictive_{num_train_models}models"
    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = "_".join([str(x) for x in [experiment_name, abs_are] + list(loss_types) + [
        num_train_models, include_test, fit_info.name]])
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    resulting_cols = ["scaled_set", "percentage", "mse", "are", "huber", "last_pred", "test_model",
                      "#Train models", "largest_train_model", "flops", "num_params", "popt"]
    evals = []
    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    for scaled_set in tqdm.tqdm(df["scaled_set"].unique(), desc=f"Scaling families ({num_train_models} models)"):
        model_by_size = df.query("scaled_set==@scaled_set")[
            ["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        if include_test:
            train_model_names = model_by_size["model_name"]
        else:
            train_model_names = model_by_size["model_name"].iloc[:-1]
        largest_model = model_by_size["model_name"].iloc[-1]

        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue
        for i in range(len(train_model_names) - num_train_models+1):
            train_models_ids = np.arange(i, i + num_train_models)
            for percentage in train_percentages:
                cache_id = scaled_set + str(i) + str(percentage)
                if not force and cache_id in cache:
                    res = cache[cache_id]
                else:
                    largest_train_model = train_model_names.iloc[train_models_ids].iloc[-1]
                    num_params = df.query(
                        "model_name==@largest_train_model")["num_params"].iloc[0]
                    train_models = train_model_names.iloc[train_models_ids].tolist(
                    )
                    train_df = get_model_data(df=df, models=train_models,
                                              max_percentage=percentage,
                                              min_tokens=cut_beginning)
                    test_df = get_model_data(df=df, models=[largest_model],
                                             min_percentage=test_percentage,
                                             min_tokens=cut_beginning)
                    mse, are, huber, train_are, predicted, popt = single_scaling(
                        train_df, test_df, fit_info, abs_are)
                    unique_model_sizes = nunique_model_size(train_df)
                    last_pred = predicted[-1] if predicted is not None else None
                    flops = train_df["flops"].sum()
                    res = (scaled_set, percentage, mse, are, huber, last_pred, largest_model, unique_model_sizes,
                           train_models[-1], flops, num_params,
                           tuple(popt) if popt is not None else None)
                    cache[cache_id] = res
                    if verbose:
                        print(
                            f"{scaled_set} {100 * percentage}% unique model sizes {unique_model_sizes}: mse {mse}, ARE {are}, train ARE{train_are}, popts {popt} predicted {np.mean(predicted) if predicted is not None else ''} actual {test_df['perf'].mean()}")
                    assert len(res) == len(
                        resulting_cols), "columns mismatch, ensure saved (res) and loaded (resulting_cols) values match"
                evals.append(res)
            save_cache(cache, cache_name)
        save_cache(cache, cache_name)

    eval = "are"
    evals = pd.DataFrame(evals, columns=resulting_cols)
    print(
        f"models with max relative error: {evals.sort_values(by='are').dropna()[-10:]['largest_train_model']}")
    # sub_evals = evals.loc[:, ["scaled_set",
    #                           "are", "#Train models", "num_params"]]
    metadata = get_per_model_metadata(df)
    plot_models_percentage_hist(evals, eval="are", iterate_over="scaled_set", index="num_params",
                                columns="percentage", fig_dir=fig_dir, iso="flops", eval_contours=[0.15, 0.10, 0.05], min_rows=1, show=show, metadata=metadata)

    # scaled_set_metadata = get_per_model_metadata(df, "scaled_set")
    # model_name_metadata = get_per_model_metadata(df, "model_name")
    # subfig_dir = os.path.join(fig_dir, "agg_hist_per_model_type")
    # for model_type in df["model_type"].unique():
    #     sub_evals = evals[evals["scaled_set"].apply(
    #         lambda x: scaled_set_metadata["model_type"][x] == model_type)]
    #     if sub_evals.empty:
    #         continue
    #     aggregate_hist(sub_evals, eval=eval, iso="flops", eval_contours=[0.15, 0.10, 0.05],
    #                    fig_dir=os.path.join(subfig_dir),
    #                    exp_name=f"{model_type}",
    #                    show=show,
    #                    metadata=model_name_metadata, vmin=0, vmax=0.35, single_scale=True)


def larger_is_predictable(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"),
                          num_train_models=4, abs_are=True, fit_info=ChinchillaFit):
    cut_beginning = 10 ** 10
    test_percentage = 0.7
    experiment_name = "larger_is_predictable"
    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = experiment_name + "_" + str(abs_are) + "_".join(loss_types) + "_" + str(
        num_train_models) + "_" + fit_info.name
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
                num_params = df.query(
                    "model_name==@largest_model")["num_params"].iloc[0]
                train_df = get_model_data(df=df, models=model_names.iloc[train_models_ids],
                                          max_percentage=test_percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[largest_model],
                                         min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
                mse, are, train_are, predicted, popt = single_scaling(
                    train_df, test_df, fit_info, abs_are)
                unique_model_size = nunique_model_size(train_df)
                last_pred = predicted[-1] if predicted is not None else None
                res = (scaled_set, mse, are, last_pred, largest_model, unique_model_size, num_params,
                       tuple(popt) if popt is not None else None)
                cache[cache_id] = res
                print(
                    f"{scaled_set} {largest_model} {num_train_models + 1}: {are}, {train_are}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    evals = pd.DataFrame(evals, columns=["scaled_set", "mse", "are", "huber", "last_pred", "largest_model",
                                         "#Train models", "num_params", "params"])
    print(
        f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['largest_model']}")
    evals = evals.loc[:, ["scaled_set", "are",
                          "huber", "#Train models", "num_params"]]
    plot_models_percentage_hist(evals, eval="are", index="#Train models", columns="num_params", fig_dir=fig_dir,
                                min_rows=1, show=show)
    plot_models_percentage_hist(evals, eval="huber", index="#Train models", columns="num_params", fig_dir=fig_dir,
                                min_rows=1, show=show)


if __name__ == '__main__':
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = ("perp", "loss")
    perf_path = get_perf_path(cache_dir, loss_types)
    df = get_perf_df(get_data(save_in=data_path, force=force),
                     loss_types, save_in=perf_path, force=force)
    force = False
    force = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    # df = df[df["model_type"].isin(["OPT"])]
    # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    abs_are = True
    predict_smallest(df, force=force, fig_dir=fig_dir,
                     at_least_loss=10, abs_are=abs_are)
    closer_in_scale_is_predictive(
        df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
    larger_is_predictable(df, force=force, fig_dir=fig_dir,
                          at_least_loss=10, abs_are=abs_are)
