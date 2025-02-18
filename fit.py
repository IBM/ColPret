from util.fit_utils import metric_per_column, get_perf_path, get_data_path, hist_fit, \
    hist_one_model_fit, get_perf_df, LossType, get_per_model_metadata, scale_fit_per_model
from fitting_funcs import MultFit, ChinchillaFit, PCAFit, Manual2Fit, ChinchillaTorchFit, ChinchillaTorchGuessFit
from experiments.predict_with_size_or_number import closer_in_scale_is_predictive, larger_is_predictable, \
    predict_smallest
from experiments.predict_last import PredictBest, PredictLast
from experiments.minimal_cut import minimal_cut
from experiments.num_to_size import iter_model_sizes
import os
from util.read_data import get_data


if __name__ == '__main__':
    # acquire data
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/test_cache/'
    data_path = get_data_path(cache_dir)

    loss_types = (LossType.PERP, LossType.LOSS)
    perf_path = get_perf_path(cache_dir, loss_types)
    all_df = get_data(save_in=data_path, force=force)
    df = get_perf_df(all_df, loss_types, save_in=perf_path, force=force)
    metadata = get_per_model_metadata(df)

    # fit

    # chinchila based (datablation guesses)
    force = True
    force = False
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    # df = df[df["model_type"].isin(["OPT"])]
    # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    abs_are = True
    verbose = False

    force = False
    fit_info = ChinchillaFit
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
    #          fit_info=fit_info, cut_beginning=10 ** 10, train_percentages=[1], iter_models=iter_model_sizes)
    test_percentage = 1
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_test1"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, cut_beginning=10 ** 10, verbose=verbose, test_percentage=test_percentage)

    force = False

    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "scale_down"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, verbose=verbose, experiment_name="scale_down", scale_down=True, annot=True)
    experiment_name = "num_to_size"
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, experiment_name), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, cut_beginning=10 ** 10, train_percentages=[1], iter_models=iter_model_sizes, experiment_name=experiment_name, iter_axis_name="Largest Model Parameters (Scale up predicted)")
    force = True
    closer_in_scale_is_predictive(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are,
                                  fit_info=fit_info, num_train_models=3)
    closer_in_scale_is_predictive(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are,
                                  fit_info=fit_info, num_train_models=4)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_no_cut"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, cut_beginning=0, verbose=verbose)
    predict_smallest(df, force=force, fig_dir=fig_dir,
                     at_least_loss=10, abs_are=abs_are, fit_info=fit_info)
    larger_is_predictable(df, force=force, fig_dir=fig_dir,
                          at_least_loss=10, abs_are=abs_are, fit_info=fit_info)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_best_seen"), at_least_loss=10, abs_are=abs_are,
             fit_info=PredictBest)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "last"), at_least_loss=10, abs_are=abs_are,
             fit_info=PredictLast, verbose=True)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "best"), at_least_loss=10, abs_are=abs_are,
             fit_info=PredictBest, verbose=True)

    force = False
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, cut_beginning=10 ** 10, verbose=verbose)
    minimal_cut(df, force=force, fig_dir=fig_dir, at_least_loss=10,
                abs_are=abs_are, fit_info=fit_info)
    scale_fit_per_model(df, force=force, fig_dir=os.path.join(
        fig_dir, "per_model"), at_least_loss=10, abs_are=abs_are)
    hist_one_model_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_1m"), at_least_loss=10, abs_are=abs_are,
                       verbose=verbose)
    # # fit_on_smaller(df,force)
    # # scaling_scaling_law(df, force)
    # # cross_validate(df,force)
    #

    # fit with torch
    force = False
    force = True
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/torch_fit'
    fit_func = ChinchillaTorchFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_func)
    force = False
    force = True
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/torch_guess_fit'
    fit_func = ChinchillaTorchGuessFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_func)

    # keep the connection between a,alpha and b,beta constant (or more specifically use the PCA)
    force = True
    force = False
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/pca_fit'
    fit_func = PCAFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_func)

    # Manual fit (fit a->alpha and b-> betha and fit
    force = True
    force = False
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/manual_fit'
    fit_func = Manual2Fit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_func)

    # Multiplicative fit
    force = True
    force = False
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/mult_fit'
    fit_func = MultFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_func)

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
    # abs_are = True
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are)
    # hist_one_model_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_1m"), at_least_loss=10, abs_are=abs_are)
    # scale_fit_per_model(df, force=force, fig_dir=os.path.join(fig_dir, "per_model"), at_least_loss=10, abs_are=abs_are)
    # minimal_cut(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
    # predict_smallest(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
    # closer_in_scale_is_predictive(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
    # larger_is_predictable(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
    # # fit_on_smaller(df,force)
    # # scaling_scaling_law(df, force)
    # # cross_validate(df,force)
    #
    # # data_aware_fit()
