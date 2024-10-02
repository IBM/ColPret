import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tqdm

sys.path.append("/Users/lc/PycharmProjects/CLPR")
os.chdir("/Users/lc/PycharmProjects/CLPR")
if True:
    from util.fit_utils import FIG_FORMAT, aggregate_hist, get_model_data, get_model_nums, get_per_model_metadata, nunique_model_size, opts_explained, plot_1popt, plot_2popt, plot_models_percentage_hist, single_scaling
    from util.cache import get_cache, save_cache
    from util.read_data import get_data
    from util.fit_utils import get_perf_df, \
        metric_per_column, get_perf_path, get_data_path, LossType, hist_fit
    from fitting_funcs import ChinchillaFit, FitInfo


def hist_fit(df, force=False, fig_dir=None, show=False, loss_types=(LossType.PERP, LossType.LOSS),
             at_least_loss=float("inf"),
             train_percentages=(0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.7, 0.8, 0.9, 1),
             experiment_name="", iter_models=None, iter_axis_name="model_selection_metadata",
             abs_are=True, cut_beginning=10 ** 10, fit_info: FitInfo = ChinchillaFit, scale_down=False, annot=False, verbose=False):
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
        abs_are:

    Returns:

    """

    orig_iter_models = iter_models
    if iter_models is None:
        # list of train_models and metadata tuples. The metadata is a number\string to plot against
        # (by default, one set of models is chosen: the largest models possible)
        iter_models = lambda train_models, num_train_models, *args, **kwargs:  [
            (train_models[:num_train_models], None)]
    else:
        assert experiment_name, "If non-standard experiment is done (e.g., iterating models not by default), and experiment name must be provided for caching."
    test_percentage = 0.7

    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"hist_fit_{abs_are}_" + LossType.join(
        loss_types, "_") + f"{fit_info.name}_cut{cut_beginning}_{experiment_name}_{scale_down}"
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []
    resulting_cols = ["scaled_set", "percentage", "mse", "are", "last_pred", "test_model",
                      "#Train models", "largest_train_model", "flops", iter_axis_name, "popt"]
    matches = []

    for scaled_set in tqdm.tqdm(df["scaled_set"].unique(), desc="Scaling families"):
        model_by_size = df.query("scaled_set==@scaled_set")[["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        if scale_down:
            model_by_size = model_by_size.iloc[::-1]
            if "pythia" in scaled_set and len(model_by_size):
                model_by_size = model_by_size[1:]

        if len(model_by_size["model_name"]) < 1:
            continue

        largest_model = model_by_size["model_name"].iloc[-1]
        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue

        target_model = model_by_size["model_name"].iloc[-1]
        smaller_models = model_by_size["model_name"].iloc[:-1]

        ##################
        worst_generalizing_model = ""
        worst_predicted_model = ""
        worst_generalization = 0
        worst_prediction = 0
        if 20 > len(smaller_models) > 3:
            print(scaled_set)
            for i in range(len(smaller_models)):

                if i == 0:
                    abl_smaller_models = list(smaller_models.iloc[1:])
                else:
                    abl_smaller_models = list(smaller_models.iloc[:i -
                                                                  1]) + list(smaller_models.iloc[i+1:])

                abl_test_df = get_model_data(
                    df=df, models=[smaller_models.iloc[i]], min_percentage=test_percentage)
                abl_train_df = get_model_data(df=df, models=abl_smaller_models,
                                              max_percentage=test_percentage,
                                              min_tokens=cut_beginning)
                mse, are, train_are, predicted, popt = single_scaling(abl_train_df, abl_test_df, fit_info, abs_are=abs_are,
                                                                      verbose=verbose)
                test_df = get_model_data(
                    df=df, models=[target_model], min_percentage=test_percentage)
                mse, test_are, train_are, predicted, popt = single_scaling(abl_train_df, test_df, fit_info, abs_are=abs_are,
                                                                           verbose=verbose)
                if test_are and worst_generalization < test_are:
                    worst_generalization = test_are
                    worst_generalizing_model = smaller_models.iloc[i]
                if are and worst_prediction < are:
                    worst_prediction = test_are
                    worst_predicted_model = smaller_models.iloc[i]
                print(i, are, test_are)
            matches.append(worst_predicted_model == worst_generalizing_model)
            print(worst_predicted_model, worst_generalizing_model,
                  worst_predicted_model == worst_generalizing_model)
            print("average accuracy", np.mean(matches))
        ###################

    #     for percentage in train_percentages:
    #         for num_train_models in get_model_nums(len(smaller_models)):
    #             for train_models, iter_data in iter_models(df=df, scaled_set=scaled_set, percentage=percentage, num_train_models=num_train_models, train_models=smaller_models, target_model=target_model):
    #                 train_models = list(train_models)
    #                 cache_id = scaled_set + \
    #                     str(num_train_models) + str(percentage) + \
    #                     str(iter_data)
    #                 if not force and cache_id in cache:
    #                     res = cache[cache_id]
    #                     assert len(res) == len(
    #                         resulting_cols), "columns mismatch, clean cache"
    #                 else:
    #                     test_df = get_model_data(
    #                         df=df, models=[target_model], min_percentage=test_percentage)
    #                     train_df = get_model_data(df=df, models=train_models,
    #                                               max_percentage=percentage,
    #                                               min_tokens=cut_beginning)
    #                     unique_model_sizes = nunique_model_size(train_df)
    #                     flops = train_df["flops"].sum()

    #                     mse, are, train_are, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_are=abs_are,
    #                                                                           verbose=verbose)

    #                     last_pred = predicted[-1] if predicted is not None else None
    #                     res = (
    #                         scaled_set, percentage, mse, are, last_pred, target_model, unique_model_sizes,
    #                         train_models[-1], flops, iter_data,
    #                         tuple(popt) if popt is not None else None)
    #                     if verbose:
    #                         print(
    #                             f"{scaled_set} {100 * percentage}% unique model sizes {unique_model_sizes}: mse {mse}, ARE {are}, train ARE{train_are}, popts {popt} predicted {np.mean(predicted) if predicted is not None else ''} actual {test_df['perf'].mean()}")
    #                     assert len(res) == len(
    #                         resulting_cols), "columns mismatch, ensure saved (res) and loaded (resulting_cols) values match"
    #                     cache[cache_id] = res
    #                 evals.append(res)
    #     save_cache(cache, cache_name)
    # save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=resulting_cols)
    evals["popt"] = evals["popt"].apply(np.asarray)
    # print(f"Mean guess: {evals.groupby('scaled_set')['popt'].mean()}")
    # print(f"Mean guess: {evals['popt'].mean()}")
    print(
        f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['scaled_set']}")
    eval = "are"
    plot_models_percentage_hist(
        evals, eval=eval, fig_dir=fig_dir, show=show, annot=annot)

    scaled_set_metadata = get_per_model_metadata(df, "scaled_set")
    model_name_metadata = get_per_model_metadata(df, "model_name")

    subfig_dir = os.path.join(fig_dir, "agg_hist_per_model_type")
    flops_subfig_dir = os.path.join(subfig_dir, "flops")
    if len(train_percentages) > 1:
        for model_type in df["model_type"].unique():
            sub_evals = evals[evals["scaled_set"].apply(
                lambda x: scaled_set_metadata["model_type"][x] == model_type)]
            if sub_evals.empty:
                continue
            aggregate_hist(sub_evals, eval=eval, iso="flops", eval_contours=[0.15, 0.10, 0.05],
                           fig_dir=os.path.join(subfig_dir),
                           exp_name=f"{model_type}",
                           show=show,   metadata=model_name_metadata, vmin=0, vmax=0.35, single_scale=True)
            aggregate_hist(sub_evals, eval="flops", fig_dir=flops_subfig_dir, exp_name=f"flops_{model_type}",
                           show=show, metadata=model_name_metadata, log_scale=True)
        # set_to_max_flops = evals.groupby("scaled_set")["flops"].max().to_dict()
        # evals["rel_flops"] = evals.apply(lambda row: row["flops"]/set_to_max_flops[row["scaled_set"]])
        aggregate_hist(evals, eval=eval, iso=None, eval_contours=[0.10, 0.05], fig_dir=fig_dir, show=show,
                       metadata=model_name_metadata, vmin=0, vmax=0.35)  # no iso as it aggregates on different model scales
    if orig_iter_models is not None:
        for model_type in df["model_type"].unique():
            sub_evals = evals[evals["scaled_set"].apply(
                lambda x: scaled_set_metadata["model_type"][x] == model_type)]
            if sub_evals.empty:
                continue
            aggregate_hist(sub_evals, eval=eval, iso="flops", eval_contours=[0.15, 0.10, 0.05],
                           fig_dir=subfig_dir,
                           exp_name=f"scale_{model_type}",
                           show=show,
                           metadata=model_name_metadata, vmin=0, vmax=0.35, column=iter_axis_name)
        aggregate_hist(evals, eval=eval, iso="flops", eval_contours=[0.10, 0.05], fig_dir=fig_dir, exp_name=f"scale", show=show,
                       metadata=model_name_metadata, vmin=0, vmax=0.35, column=iter_axis_name)
    if evals["popt"].dropna().empty:
        return
    plot_1popt(evals, popty=2, name="intercept", eval=eval, fig_dir=fig_dir, show=show, y_label="$E$",
               metadata=scaled_set_metadata, labels="arch")
    plot_1popt(evals, popty=2, name="intercept_model_type", eval=eval, fig_dir=fig_dir, show=show, y_label="$E$",
               metadata=scaled_set_metadata, labels="model_type")
    plot_1popt(evals, popty=2, name="intercept_none", eval=eval, fig_dir=fig_dir, show=show, y_label="$E$",
               metadata=scaled_set_metadata, labels="")
    if len(evals["popt"].dropna().iloc[0]) > 3:
        plot_2popt(evals, poptx=0, popty=3, name="scale", eval=eval, fig_dir=fig_dir, show=show, x_label="A", y_label="$\\alpha$",
                   metadata=scaled_set_metadata, labels="arch")
        plot_2popt(evals, poptx=0, popty=3, name="scale_model_type", eval=eval, fig_dir=fig_dir, show=show, x_label="A", y_label="$\\alpha$",
                   metadata=scaled_set_metadata, labels="model_type")
    if len(evals["popt"].dropna().iloc[0]) > 4:
        plot_2popt(evals, poptx=1, popty=4, name="tokens", eval=eval, fig_dir=fig_dir, show=show, x_label="B", y_label="$\\beta$",
                   metadata=scaled_set_metadata, labels="arch")
        plot_2popt(evals, poptx=1, popty=4, name="tokens_model_type", eval=eval, fig_dir=fig_dir, show=show, x_label="B", y_label="$\\beta$",
                   metadata=scaled_set_metadata, labels="model_type")
    if len(evals["popt"].dropna().iloc[0]) > 4:
        plot_2popt(evals, poptx=3, popty=4, name="alpha_beta", eval=eval, fig_dir=fig_dir, show=show, x_label="$\\alpha$", y_label="$\\beta$",
                   metadata=scaled_set_metadata, labels="arch")
        plot_2popt(evals, poptx=3, popty=4, name="alpha_beta_model_type", eval=eval, fig_dir=fig_dir, show=show, x_label="$\\alpha$", y_label="$\\beta$",
                   metadata=scaled_set_metadata, labels="model_type")
    opts_explained(evals, eval=eval, fig_dir=fig_dir, show=show,
                   metadata=scaled_set_metadata)

    # # plot on all models
    # for eval in ["mse", "are"]:
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
    #         plt.savefig(os.path.join(fig_dir, f"per_model_{eval}.{FIG_FORMAT}"), bbox_inches="tight")
    #     plt.show()

    # Group by characteristics
    def add_info(row):
        to_add = df[row["test_model"] == df["model_name"]].iloc[0, :]
        metric_map = metric_per_column(df)
        to_add = to_add[
            (col for col in to_add.index if
             col not in row.index and (col not in metric_map or metric_map[col] == LossType.UNK))]
        return pd.concat([row, to_add])

    evals = evals.apply(add_info, axis=1)
    # TODO model size, model family, final loss
    bins = np.percentile(evals["num_params"].unique(), np.linspace(20, 100, 5))
    # bins = [to_int(num) for num in ["100m", "500m", "1B", "5B", "10B"]]
    evals["model_size"] = np.digitize(evals["num_params"], bins)
    evals["best_loss"] = evals["test_model"].apply(
        lambda model: df.query("@model==model_name")["perf"].min())
    bins = np.percentile(evals["best_loss"].unique(), np.linspace(20, 100, 5))
    evals["best_loss_binned"] = np.digitize(evals["best_loss"], bins)

    evals["max_tokens"] = evals["test_model"].apply(
        lambda model: df.query("@model==model_name")["tokens_seen"].max())
    bins = np.percentile(evals["max_tokens"].unique(), np.linspace(20, 100, 5))
    evals["max_tokens_binned"] = np.digitize(evals["max_tokens"], bins)

    print("done")


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
    force = False
    force = True
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    # df = df[df["model_type"].isin(["pythia"])]
    # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    abs_are = True
    verbose = False
    fit_info = ChinchillaFit
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
    #          fit_info=fit_info, cut_beginning=10 ** 10, train_percentages=[1], iter_models=iter_model_sizes)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, cut_beginning=10 ** 10, verbose=verbose)
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "last"), at_least_loss=10, abs_are=abs_are,
    #          fit_info=PredictLast, verbose=True)
    # hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "best"), at_least_loss=10, abs_are=abs_are,
    #  fit_info=PredictBest, verbose=True)
    # predict_best_seen(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
