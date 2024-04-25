import ast
import os

import numpy as np

from experiments.minimal_cut import minimal_cut
from experiments.predict_with_size_or_number import predict_smallest, larger_is_predictable, \
    closer_in_scale_is_predictive
from fitting_funcs import MultFit, ChinchillaFit
from util.fit_utils import metric_per_column, get_perf_path, get_data_path, scale_fit_per_model, hist_fit, \
    hist_one_model_fit, get_perf_df, LossType, get_per_model_metadata
from util.read_data import get_data

if __name__ == '__main__':
    # acquire data
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = (LossType.PERP, LossType.LOSS)
    perf_path = get_perf_path(cache_dir, loss_types)
    all_df = get_data(save_in=data_path, force=force)
    df = get_perf_df(all_df, loss_types, save_in=perf_path, force=force)
    metadata = get_per_model_metadata(df)

    # fit
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
    fit_info = ChinchillaFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_mnd=abs_mnd, fit_info=fit_info)
    hist_one_model_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_1m"), at_least_loss=10, abs_mnd=abs_mnd)
    # scale_fit_per_model(df, force=force, fig_dir=os.path.join(fig_dir, "per_model"), at_least_loss=10, abs_mnd=abs_mnd)
    # minimal_cut(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd, fit_info=fit_info)
    # predict_smallest(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd, fit_info=fit_info)
    closer_in_scale_is_predictive(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd, fit_info=fit_info)
    # larger_is_predictable(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd, fit_info=fit_info)
    # # fit_on_smaller(df,force)
    # # scaling_scaling_law(df, force)
    # # cross_validate(df,force)
    #
    # # data_aware_fit()
    # Multiplicative fit
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/mult_fit/'
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/mult_fit'
    data_path = get_data_path(cache_dir)
    fit_func = MultFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_mnd=abs_mnd, fit_info=fit_func)

    # force = True
    # force = False
    # cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/single_loss/'
    # data_path = get_data_path(cache_dir)
    #
    # loss_types = (LossType.PERP, LossType.LOSS)
    # perf_path = get_perf_path(cache_dir, loss_types)
    # df = get_perf_df(get_data(save_in=data_path, force=force), loss_types, save_in=perf_path,
    #                  losses_aggregation_func=lambda losses: losses[0] if losses else np.nan, force=force)
    # force = True
    # force = False
    # fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/single_loss/'
    # os.makedirs(fig_dir, exist_ok=True)
    #
    # metric_per_column(df)
    # # df = df[df["model_type"].isin(["OPT"])]
    # # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # # df = df[df["original_paper"] == "pythia"]
    # # df = df[df["domain"] == "LM"]
    # abs_mnd = True
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_mnd=abs_mnd)
    # hist_one_model_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_1m"), at_least_loss=10, abs_mnd=abs_mnd)
    # scale_fit_per_model(df, force=force, fig_dir=os.path.join(fig_dir, "per_model"), at_least_loss=10, abs_mnd=abs_mnd)
    # minimal_cut(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    # predict_smallest(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    # closer_in_scale_is_predictive(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    # larger_is_predictable(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    # # fit_on_smaller(df,force)
    # # scaling_scaling_law(df, force)
    # # cross_validate(df,force)
    #
    # # data_aware_fit()
