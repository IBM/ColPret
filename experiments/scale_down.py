import os
import sys
sys.path.append("/Users/lc/PycharmProjects/CLPR")
os.chdir("/Users/lc/PycharmProjects/CLPR")
if True:
    from util.read_data import get_data
    from util.fit_utils import get_perf_df, \
        metric_per_column, get_perf_path, get_data_path, LossType, hist_fit
    from fitting_funcs import ChinchillaFit, FitInfo


def return_prediction(metadata, prediction):
    return [prediction] * len(metadata)


def fit_best_score(func, metadata, perf, p0=None, bounds=None, method=None, **kwargs):
    return [min(perf)], None


PredictBest = FitInfo(func=return_prediction,
                      guess=(),
                      bounds=(), name="best_seen",
                      fit_func=fit_best_score)


def fit_last_score(func, metadata, perf, p0=None, bounds=None, method=None, **kwargs):
    model_size = metadata["num_params"]
    metadata["perf"] = perf
    sub_metadata = metadata[metadata["num_params"]
                            == model_size].sort_values(by="tokens_seen")
    pred = sub_metadata["perf"].iloc[-5:].mean()
    return [pred], None


PredictLast = FitInfo(func=return_prediction,
                      guess=(),
                      bounds=(), name="best_seen",
                      fit_func=fit_last_score)

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

    # fit

    # chinchila based (datablation guesses)
    force = True
    force = False
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    abs_are = True
    verbose = False
    verbose = True
    fit_info = ChinchillaFit
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "scale_down"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, verbose=verbose, experiment_name="scale_down")
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "last"), at_least_loss=10, abs_are=abs_are,
    #          fit_info=PredictLast, verbose=True)
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "best"), at_least_loss=10, abs_are=abs_are,
    #  fit_info=PredictBest, verbose=True)
    # predict_best_seen(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
