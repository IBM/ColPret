import os

import numpy as np
import pandas as pd

from fit import plot_models_percentage_hist, mean_normalized_distance, fit, get_model_data, get_perf_df, \
    metric_per_column, mean_squared_error
from fitting_funcs import ChinchillaFit, bound_params
from util.cache import save_cache, get_cache
from util.read_data import get_data


def minimal_cut(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"),
                minimal_cuts=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.69), abs_mnd=True):
    cut_beginning = 10 ** 8
    fit_info = ChinchillaFit
    test_percentage = 0.7
    train_percentage = 0.7
    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"minimal_cut_" + str(abs_mnd) + "_".join(loss_types)
    cache = get_cache(cache_name, force)
    df = get_perf_df(df, loss_types)
    evals = []
    for model_name in df["model_name"].unique():
        if df.query("model_name==@model_name")["perf"].min() > at_least_loss:
            continue
        for min_percentage in minimal_cuts:
            cache_id = model_name + str(min_percentage)
            if not force and cache_id in cache:
                res = cache[cache_id]
            else:
                train_df = get_model_data(df=df, models=[model_name], max_percentage=train_percentage,
                                          min_percentage=min_percentage,
                                          min_tokens=cut_beginning)
                test_df = get_model_data(df=df, models=[model_name], min_percentage=test_percentage,
                                         min_tokens=cut_beginning)
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
                        mnd = (test_df["perf"] - predicted) / test_df["perf"]
                        if abs_mnd:
                            mnd = np.abs(mnd)
                        mnd = mnd.mean()
                last_pred = predicted[-1] if predicted is not None else None
                res = (model_name, min_percentage, mse, mnd, last_pred)
                cache[cache_id] = res
                print(f"{model_name} {min_percentage}: {mse}, {mnd}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=["model_name", "min_percentage", "mse", "mnd", "last_pred"])
    print(f"models with max normalized distance: {evals.sort_values(by='mnd').dropna()[-10:]['model_name']}")


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
    abs_mnd = True
    minimal_cut(df, force=force, fig_dir=os.path.join(fig_dir, "hist_1m"), at_least_loss=10, abs_mnd=abs_mnd)
