import seaborn as sns
from util.naming import to_int
from util.read_data import get_data
import ast
import math
import os
from enum import Enum
from typing import Dict, List, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import tqdm
from matplotlib.colors import LogNorm
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from fitting_funcs import DATA_FIT_COLS, TestFit, FitInfo, ChinchillaFit, DatablationsFit, bound_params
from util.cache import get_cache, save_cache
from util.plots import plot_pred_actual_compare, capitalize_fig
import matplotlib.font_manager as fm

matplotlib.use('QtAgg')
FIG_FORMAT = "pdf"
# # Ensure Times New Roman is available
# if "Times New Roman" in fm.findfont("Times New Roman"):
#     font = "Times New Roman"
# else:
#     font = "serif"  # Fallback to a generic serif font if Times New Roman is not available

# plt.rcParams.update({
#     'font.family': font,
#     'font.size': 11,
#     'axes.labelsize': 11,
#     'axes.titlesize': 11,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 11,
#     'legend.fontsize': 11,
#     'figure.titlesize': 13
# })
# # sns.set_theme(font_scale=1.5)
# sns.set_style("white")
# Set the style and font scale
sns.set_theme(style="white", font="Times New Roman", font_scale=1.5)

# If Times New Roman is not available, fall back to serif
if "Times New Roman" not in plt.rcParams["font.family"]:
    sns.set_theme(style="white", font="serif", font_scale=1.5)
plt.interactive(False)

# func = r"$L(N,D,R_N,R_D)=E + \frac{A}{(U_N + U_N * R_N^* * (1 - e^{(-1*R_N/(R_N^*))}))^\alpha} + \frac{B}{(U_D + U_D * R_D^* * (1 - e^{(-1*R_D/(R_D^*))}))^\beta}$"
# a, b, e, alpha, beta, rd_star, rn_star = [6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596, 15.387756, 5.309743]
epsilon = 1e-6


class LossType(Enum):
    NORM_ACC = "norm_acc"
    BPB = "bpb"
    LOSS = "loss"
    PERP = "perp"
    BYTE_PERP = "byte_perp"
    LIKELIHOOD_DIFFERENCE = "likelihood_difference"
    ACC1LIKE = "1acc_like"
    ACC100LIKE = "100acc_like"
    TIME = "time"
    UNK = "unk"

    def to_string(self):
        return f'{self.value})'

    @classmethod
    def join(cls, loss_types, sep):
        return sep.join([x.value for x in loss_types])


def single_scaling(train_df, test_df, fit_info, abs_are, verbose=False):
    if train_df.empty or test_df.empty:
        popt = None
    else:
        popt = fit(fit_info, train_df, train_df["perf"], verbose=verbose)

    mse = None
    are = None
    train_are = None
    predicted = None
    if not (popt is None or all(pd.isna(popt))):
        predicted = fit_info.func(test_df, *popt)
        if not all(pd.isna(predicted)):
            mse = mean_squared_error(predicted, test_df["perf"])
            are = mean_normalized_distance(predicted, test_df["perf"], abs_are)
            predicted_train = fit_info.func(train_df, *popt)
            train_are = mean_normalized_distance(
                predicted_train, train_df["perf"], abs_are)
    return mse, are, train_are, predicted, popt


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

    popt, pcov = curve_fit(fit_info.func, metadata, perf,
                           p0=fit_info.guess, bounds=fit_info.bounds)
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


def fit(fit_info: FitInfo, metadata, perf, default=None, verbose=True):
    try:
        try:
            popt, pcov = fit_info.fit_func(
                fit_info.func, metadata, perf, p0=fit_info.guess, bounds=fit_info.bounds)
        except RuntimeError as e:
            popt, pcov = fit_info.fit_func(fit_info.func, metadata, perf, p0=fit_info.guess, x_scale=fit_info.guess,
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
            if verbose:
                print(
                    f"Fit failed {metadata['model_name'].unique()[0]}, data size:{len(metadata)}, number of params to fit {len(fit_info.guess)}")
        if default is None:
            return default
        else:
            popt = np.zeros_like(fit_info.guess) + default
            pcov = None
    return popt


def aggregate_row_loss(row, func: Callable = np.mean, graceful=False):
    losses = []
    for x in ast.literal_eval(str(row["loss_cols"])):
        if not graceful or x in row:
            losses.append(row[x])
    return func(losses)


def normalize_metric(metric, score):
    if type(score) == type(str):
        score = float(score)
    if metric == LossType.PERP:
        return np.log2(score)
    if metric == LossType.LOSS:
        return score
    elif "acc" in metric.value:
        if "100" not in metric.value:
            score *= 100
        return math.log(score) if score else score
    elif "likelihood" in metric.value:
        return score
    elif metric == LossType.UNK:
        return score
    elif metric == LossType.TIME:
        return score
    else:
        raise ValueError((metric, score))


def aggregate_loss(subdf, func=np.mean, graceful=False):
    # col_to_metric = metric_per_column(subdf)
    # for col, metric in col_to_metric.items(): # assumes already normalized
    #     subdf[col] = subdf[col].apply(lambda x: normalize_metric(metric, x))
    return subdf.apply(lambda row: aggregate_row_loss(row, func=func, graceful=graceful), axis=1)


def fit_per_model(df, predict_with=0.3, force=False):
    fit_info = ChinchillaFit
    cut_beginning = 10 ** 8

    data = []
    cache_name = f"fit_per_model_{predict_with}"
    cache = get_cache(cache_name, force)
    for model_name in df["model_name"].unique():
        train_df = get_model_data(
            df=df, models=[model_name], max_percentage=predict_with, min_tokens=cut_beginning)
        cash_id = model_name + str(predict_with)
        popt = cached_fit(cache, cash_id, fit_info, train_df, train_df["perf"])

        if popt is None:
            continue
        test_df = get_model_data(df=df, models=[
                                 model_name], min_percentage=predict_with, min_tokens=cut_beginning)
        predicted = fit_info.func(test_df, *popt)
        subdf = pd.merge(train_df, test_df, how='right', indicator="in_fit")
        subdf["in_fit"] = subdf["in_fit"].apply(
            lambda x: True if x == "both" else False)
        subdf["pred"] = predicted
        data.append(subdf)
    save_cache(cache, cache_name)
    data = pd.concat(data)
    plot_pred_actual_compare(data, show=True)

    """/**
    * For the brave souls who get this far: You are the chosen ones,
    * the valiant knights of programming who toil away, without rest,
    * fixing our most awful code. To you, true saviors, kings of men,
    * I say this: never gonna give you up, never gonna let you down,
    * never gonna run around and desert you. Never gonna make you cry,
    * never gonna say goodbye. Never gonna tell a lie and hurt you.
    */"""


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
            df = df.query(
                "model_name != @model or tokens_seen >= @min_model_tokens")
        if max_percentage:
            max_model_tokens = tokens * max_percentage
            df = df.query(
                "model_name != @model or tokens_seen < @max_model_tokens")
    return df


def cached_fit(cache, cache_id, fit_info, metadata, perf, default=None):
    if cache_id in cache:
        popt = cache[cache_id]
    else:
        popt = fit(fit_info, metadata, perf, default)
        cache[cache_id] = [x for x in popt]
    return popt


def normalize_loss(df, loss_types: List[LossType] = None):
    metric_map = metric_per_column(df)
    if loss_types:
        drop_cols = [col for col, loss_type in metric_map.items() if
                     loss_type not in loss_types and loss_type != LossType.UNK]
        df = df.drop(columns=drop_cols)
    for col, loss_type in metric_map.items():
        if col not in df.columns:
            continue
        df[col] = df[col].apply(lambda x: normalize_metric(loss_type, x))
    return df


def get_perf_df(df, loss_types: List[LossType], graceful=True, force=False, save_in=None,
                losses_aggregation_func: Callable = np.mean):
    if save_in and not force:
        if os.path.isfile(save_in):
            return pd.read_csv(save_in)
    df = normalize_loss(df, loss_types=loss_types)
    perf = aggregate_loss(df, losses_aggregation_func, graceful=graceful)
    df = df.assign(perf=perf)
    df = df.dropna(subset=["perf"])
    if save_in:
        df.to_csv(save_in, index=False)
    return df


def scale_fit_per_model(df, force=False, fig_dir=None, show=False,
                        loss_types: List[LossType] = (
                            LossType.PERP, LossType.LOSS),
                        at_least_loss=float("inf"),
                        abs_are=True):
    """
    Predict each model with the begginning of its own training
    Args:
        df:
        force:
        fig_dir:
        show:
        loss_types:
        at_least_loss:
        abs_are:

    Returns:

    """
    cut_beginning = 10 ** 8
    fit_info = ChinchillaFit
    test_percentage = 0.7
    train_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"scale_fit_per_model_" + \
        str(abs_are) + LossType.join(loss_types=loss_types, sep="_")
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
                #     are = None
                #     predicted = None
                # else:
                #     popt = fit(fit_info, train_df, train_df["perf"])
                #     if popt is None:
                #         mse = None
                #         are = None
                #         predicted = None
                #     else:
                #         predicted = fit_info.func(test_df, *popt)
                #         mse = mean_squared_error(predicted, test_df["perf"])
                #         are = (test_df["perf"] - predicted) / test_df["perf"]
                #         if abs_are:
                #             are = np.abs(are)
                #         are = are.mean()
                mse, are, train_are, predicted, popt = single_scaling(
                    train_df, test_df, fit_info, abs_are=abs_are)

                last_pred = predicted[-1] if predicted is not None else None
                res = (model_name, percentage, mse, are, last_pred)
                cache[cache_id] = res
                print(f"{model_name} {percentage}: {mse}, {are}")
            evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(
        evals, columns=["model_name", "percentage", "mse", "are", "last_pred"])
    print(
        f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['model_name']}")

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
        to_add = df[row["model_name"] == df["model_name"]].iloc[0, :]
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
    evals["best_loss"] = evals["model_name"].apply(
        lambda model: df.query("@model==model_name")["perf"].min())
    bins = np.percentile(evals["best_loss"].unique(), np.linspace(20, 100, 5))
    evals["best_loss_binned"] = np.digitize(evals["best_loss"], bins)

    evals["max_tokens"] = evals["model_name"].apply(
        lambda model: df.query("@model==model_name")["tokens_seen"].max())
    bins = np.percentile(evals["max_tokens"].unique(), np.linspace(20, 100, 5))
    evals["max_tokens_binned"] = np.digitize(evals["max_tokens"], bins)

    eval = "are"
    for col_group in ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type",
                      "best_loss_binned"]:
        plt.clf()
        # ares = evals.groupby([col_group, "percentage"])["are"].mean().reset_index([1])
        for group in evals[col_group].unique():
            subdf = evals.query(f"{col_group}==@group").dropna(subset=[eval])
            subdf = subdf.groupby(["percentage"])[eval].mean().reset_index()
            if subdf.empty:
                continue
            x = subdf["percentage"]
            y = subdf[eval]
            sns.lineplot(x=x.tolist(), y=y.tolist(), label=group)

        if abs_are:
            plt.ylim(bottom=0)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title=col_group)
        plt.xlabel("Percentage of Training Trajectory Available")
        plt.ylabel("Mean Normalized Distance")
        if fig_dir:
            plt.savefig(os.path.join(
                fig_dir, f"per_model_{eval}_{col_group}.{FIG_FORMAT}"), bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()

    # compare predictions to actual
    eval = "are"
    for col_group in [
            "original_paper"]:  # ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type", "best_loss_binned"]
        for group in evals[col_group].unique():
            for model in evals[evals[col_group] == group]["model_name"].unique():
                # if "pythi" in group and "pythia-1.4b" not in model:
                #     continue
                subdf = df.query(
                    "model_name==@model & tokens_seen>=@cut_beginning")
                ax = sns.lineplot(
                    x=subdf["tokens_seen"], y=subdf["perf"], label=f"{group} {model}")
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
            plt.xlabel("Tokens seen")
            plt.ylabel("Loss")
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0,
                       title=f"{col_group}:{group}")
            sns.despine()
            # plt.legend().remove()
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            # ax.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))
            if fig_dir:
                os.makedirs(os.path.join(
                    fig_dir, "compare_predictions"), exist_ok=True)
                plt.savefig(os.path.join(fig_dir, "compare_predictions", f"{group}_{col_group}_{eval}.{FIG_FORMAT}"),
                            bbox_inches="tight")
            if show:
                plt.show()

            plt.clf()
    print("done")


def mean_normalized_distance(predicted, target, abs):
    are = (target - predicted) / target
    if abs:
        are = np.abs(are)
    are = are.mean()
    return are


def plot_models_percentage_hist(evals, eval, fig_dir, iterate_over="scaled_set", index="#Train models", annot=False,
                                columns="percentage", vmin=0, vmax=0.35, min_rows=2, eval_contours=None, contour_kind="steps", iso: str = None, metadata=None,
                                show=False, verbose=False):
    for scaled_set in evals[iterate_over].unique():
        subdf = evals.query(f"@{iterate_over}=={iterate_over}")
        plot_heatmap(evals=subdf, eval=eval, fig_dir=fig_dir, show=show, title="", metadata=metadata, annot=annot,
                     #  title=scaled_set,
                     fig_name=f"hist-perc-{eval}_{scaled_set}.{FIG_FORMAT}", column=columns, index=index, vmin=vmin, vmax=vmax,
                     eval_contours=eval_contours, iso=iso, contour_kind=contour_kind,
                     min_rows=min_rows, verbose=verbose)
    return


def get_per_model_metadata(df, index="model_name"):
    return df.groupby(index).head(1).set_index(index).to_dict()


def opts_explained(evals, eval, fig_dir, iterate_over="scaled_set", index="#Train models",
                   columns="percentage", metadata=None,
                   show=False):
    max_dict = evals.groupby(iterate_over)[[index, columns]].max().to_dict()
    relevant = evals[evals.apply(
        lambda row: row[index] == max_dict[index][row[iterate_over]] and row[columns] == max_dict[columns][
            row[iterate_over]], axis=1)]
    relevant = relevant.dropna(subset="popt")
    popts = relevant["popt"]

    my_model = PCA(n_components=len(popts.iloc[0]))
    my_model.fit_transform(popts.tolist())
    print("amount explained with N components:",
          my_model.explained_variance_ratio_.cumsum())


def hist_one_model_fit(df, force=False, fig_dir=None, show=False, loss_types=(LossType.PERP, LossType.LOSS),
                       at_least_loss=float("inf"),
                       train_percentages=(
                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
                       abs_are=True, verbose=False):
    cut_beginning = 10 ** 10
    fit_info = bound_params(ChinchillaFit, [6.255414, None, None, 0.3526596])
    # fit_info = Chinchilla1ModelFit
    test_percentage = 0.7

    os.makedirs(fig_dir, exist_ok=True)

    cache_name = f"hist_one_model_fit_{abs_are}_" + \
        LossType.join(loss_types=loss_types, sep="_")
    cache = get_cache(cache_name, force)
    df = df.dropna(subset=["scaled_set"])
    evals = []
    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    for scaled_set in df["scaled_set"].unique():
        model_by_size = df.query("scaled_set==@scaled_set")[["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")
        target_model = model_by_size["model_name"].iloc[-1]
        # to exclude the last one: .iloc[:-1]
        smaller_models = model_by_size["model_name"]
        if df.query("model_name==@target_model")["perf"].min() > at_least_loss:
            continue
        for percentage in train_percentages:
            for num_model in range(0, len(smaller_models)):
                cache_id = scaled_set + str(num_model) + str(percentage)
                current_model = smaller_models.to_list()[num_model]
                train_model_size = df.query(
                    "model_name==@current_model")["num_params"]
                assert train_model_size.nunique() == 1
                train_model_size = train_model_size.iloc[0] / 1e9
                if not force and cache_id in cache:
                    res = cache[cache_id]
                else:
                    train_df = get_model_data(df=df, models=[current_model],
                                              max_percentage=percentage,
                                              min_tokens=cut_beginning)
                    test_df = get_model_data(df=df, models=[target_model], min_percentage=test_percentage,
                                             min_tokens=cut_beginning)
                    # unique_model_sizes = num_model + 1
                    unique_model_sizes = nunique_model_size(train_df)
                    mse, are, train_are, predicted, popt = single_scaling(
                        train_df, test_df, fit_info, abs_are=abs_are)
                    last_pred = predicted[-1] if predicted is not None else None
                    res = (
                        scaled_set, percentage, mse, are, last_pred, target_model, unique_model_sizes, current_model,
                        train_model_size,
                        tuple(popt) if popt is not None else None)
                    cache[cache_id] = res
                    print(
                        f"{scaled_set} on {100 * percentage}% and {num_model} models: MSE {mse}, ARE {are}, training ARE {train_are}")
                evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

    # plot
    evals = pd.DataFrame(evals, columns=["scaled_set", "percentage", "mse", "are", "last_pred", "test_model",
                                         "#Train models", "smaller_model", "train_model_size", "popt"])
    evals["popt"] = evals["popt"].apply(np.asarray)
    # print(f"Mean guess: {evals.groupby('scaled_set')['popt'].mean()}")
    # print(f"Mean guess: {evals['popt'].mean()}")
    print(
        f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['scaled_set']}")
    eval = "are"
    plot_models_percentage_hist(evals, eval=eval, index="train_model_size", columns="percentage", fig_dir=fig_dir,
                                show=show, verbose=verbose)
    # plot_2popt(evals, poptx=0, popty=2, name="tokens", eval=eval, fig_dir=fig_dir, show=show,
    #    metadata=get_per_model_metadata(df, "scaled_set"))
    # opts_explained(evals, eval=eval, fig_dir=fig_dir, show=show,
    #    metadata=get_per_model_metadata(df, "scaled_set"))


def plot_2popt(evals, eval, fig_dir, poptx, popty, name, iterate_over="scaled_set", index="#Train models",
               columns="percentage", vmin=0, vmax=1, min_rows=2, metadata=None, labels="", x_label="", y_label="",
               show=False):
    sns.set_palette(sns.color_palette("colorblind"))
    max_dict = evals.groupby(iterate_over)[[index, columns]].max().to_dict()
    relevant = evals[evals.apply(
        lambda row: row[index] == max_dict[index][row[iterate_over]] and row[columns] == max_dict[columns][
            row[iterate_over]], axis=1)]
    relevant = relevant.dropna(subset="popt")

    if labels == "model_type":
        labels = [metadata["model_type"][model]
                  for model in relevant[iterate_over]]
        all_labels = list(sorted(set(labels)))
        markers_options = ['.', 'o', 's', '^', 'v', '<', '>', 'p', '*',
                           'h', 'H', '+', 'x', 'D', 'd']  # ["o", "v", "s", "P", "X","D"]
        markers_and_labels = [(markers_options[all_labels.index(label)], label)
                              for label in labels]
    else:
        def rename_label(label):
            if label == "dec":
                label = "Decoder"
            elif label == "enc":
                label = "Encoder"
            elif label == "enc-dec":
                label = "Encoder-Decoder"
            else:
                raise NotImplemented(
                    "choose how the name would apear in the legend")
            return label
        labels = [rename_label(metadata["arch"][model])
                  for model in relevant[iterate_over]]
        all_labels = list(sorted(set(labels)))
        markers_options = ["o", "v", "s", "P", "X"]
        markers_and_labels = [(markers_options[all_labels.index(label)], label)
                              for label in labels]

    # model scale params
    xs = [popt[poptx] for popt in relevant["popt"]]
    ys = [popt[popty] for popt in relevant["popt"]]
    reg = LinearRegression().fit(np.array([xs]).T, np.array([ys]).T)
    print(f"b = a*{reg.coef_}+{reg.intercept_} ")
    for marker, label in sorted(set(markers_and_labels)):
        x = [val for val, (mar, _) in zip(
            xs, markers_and_labels) if mar == marker]
        y = [val for val, (mar, _) in zip(
            ys, markers_and_labels) if mar == marker]
        plt.scatter(x=x, y=y, marker=marker, label=label)
    if x_label:
        plt.xlabel(xlabel=x_label)
    if y_label:
        plt.ylabel(ylabel=y_label)
    capitalize_fig()

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.despine()
    if fig_dir:
        plt.savefig(os.path.join(
            fig_dir, f"popts_{name}.{FIG_FORMAT}"), bbox_inches="tight")
    if show:
        plt.show()
    plt.clf()


def sort_pivot(pivot):
    try:
        col_nums = [float(col) for col in pivot.columns]
    except ValueError as e:
        col_nums = [to_int(col.split()[0]) for col in pivot.columns]
    sorted_cols = [col for _, col in sorted(zip(col_nums, pivot.columns))]
    return pivot.reindex(sorted_cols, axis=1)


def get_pivot(evals, eval, index, column, ascending_index):
    pivot = evals.pivot(index=index, columns=column, values=eval)
    pivot = pivot.sort_values(index, ascending=ascending_index)
    pivot = pivot.dropna(axis=0, how='all')
    pivot = sort_pivot(pivot)
    return pivot


def get_minimal_cell(pivot, cont, iso_pivot, consistency=0.75):
    lt = np.array(pivot) < cont
    min_per_row = []
    for row in range(len(lt)):
        for col in range(lt.shape[1]):
            if lt[row, col] and lt[row, col:].mean() > consistency:
                min_per_row.append((row, col))
                break
    min_iso = float("inf")
    min_cell = None
    for row, col in min_per_row:
        if iso_pivot.iloc[row, col] < min_iso:
            min_cell = (row, col)
            min_iso = iso_pivot.iloc[row, col]
    return min_cell


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def less_than(array, value):
    array = np.asarray(array)
    lt = (array <= value)
    if not any(lt):
        return -1, 0
    idx = lt.size - np.argmax(lt[::-1]) - 1
    return idx, array[idx]


def choose_contours(pivot, iso_pivot, eval_contours, choose_by=less_than):
    """
    choose_by: function from values(flops) and a value to the index and value of the last relevant cell, less_than,find_nearest
    """

    iso_contours = []
    if eval_contours:
        minimal_cells = []
        for cont in eval_contours:
            minimal_cell = get_minimal_cell(pivot, cont, iso_pivot)
            if minimal_cell:
                minimal_cells.append(minimal_cell)
        for cell in minimal_cells:
            iso_val = iso_pivot.iloc[cell]
            iso_contours.append(
                [(row, *choose_by(iso_pivot.iloc[row, :], iso_val)) for row in range(len(iso_pivot))])
    return iso_contours, minimal_cells


def cells_to_contours(x_iso, y_iso, contour_kind, x_shift=1, y_shift=1):
    if contour_kind == "spline":
        # x_iso = x_iso[::-1]
        # y_iso = y_iso[::-1]
        y_plot = np.linspace(y_iso[0], y_iso[-1] + 1, 1000)
        # with no 0.5 shift it passes in the middle of the cell in terms of y and before the cell in terms of x
        x_plot = make_interp_spline(
            np.array(y_iso) + x_shift, np.array(x_iso) + y_shift)(y_plot)

    elif contour_kind == "steps":
        x_plot = []
        y_plot = []
        for i, (x_val, y_val) in enumerate(zip(x_iso, y_iso)):
            next_y_val = y_iso[i + 1] if i + 1 < len(y_iso) else 0
            x_plot.append(x_val + x_shift)
            x_plot.append(x_val + x_shift)
            y_plot.append(y_val + y_shift)
            y_plot.append(next_y_val + y_shift)
        if not (x_plot[-1] == x_val + x_shift and y_plot[-1] == 0):
            x_plot.append(x_val + x_shift)
            y_plot.append(0)
        mon_x = []
        mon_y = []
        for x, y in zip(x_plot, y_plot):
            if mon_x:
                mon_x.append(min(mon_x[-1], x))
                mon_y.append(max(mon_y[-1], y))
            else:
                mon_x.append(x)
                mon_y.append(y)
        if mon_x != x_plot:
            x_plot = mon_x
        if mon_y != y_plot:
            y_plot = mon_y
    else:
        raise NotImplementedError
    return x_plot, y_plot


def get_common_subset(df1, df2):
    """
    Returns a subset of df2 with only the columns and index found in df1.

    Parameters:
    df1 (pandas.DataFrame): The reference DataFrame.
    df2 (pandas.DataFrame): The DataFrame to be subset.

    Returns:
    pandas.DataFrame: A subset of df2 with only the columns and index found in df1.
    """
    # Get the columns that are present in both DataFrames
    common_columns = [col for col in df1.columns if col in df2.columns]

    # Get the index values that are present in both DataFrames
    common_index = [idx for idx in df1.index if idx in df2.index]

    # Create a new DataFrame with only the common columns and index
    subset_df2 = df2[common_columns].loc[common_index]

    return subset_df2


def plot_heatmap(evals, eval, title, fig_dir, fig_name, show, metadata=None,
                 index: str = "#Train models",
                 column: str = "percentage", eval_contours=None, contour_kind="steps", iso: str = None, vmin: float = 0,
                 vmax: float = 0.35,
                 min_rows: int = None,
                 ascending_index: bool = True, annot: bool = True, log_scale: bool = False, verbose=False):
    print(f"plotting {os.path.join(fig_dir, fig_name)}")
    evals = evals.drop_duplicates(subset=[index, column])
    if evals.empty:
        return

    pivot = get_pivot(evals=evals, eval=eval, index=index,
                      column=column, ascending_index=ascending_index)
    iso_contours = []
    efficient_coiches = []
    if iso and len(pivot) > 2:
        iso_pivot = get_pivot(evals=evals, eval=iso, index=index,
                              column=column, ascending_index=ascending_index)
        iso_pivot = get_common_subset(pivot, iso_pivot)
        assert pivot.shape == iso_pivot.shape
        iso_contours, efficient_coiches = choose_contours(
            pivot=pivot, iso_pivot=iso_pivot, eval_contours=eval_contours)
    if pivot.dropna(axis=1, how="all").dropna(axis=0, how="all").empty:
        if verbose:
            print(f"Skipping {title}, empty: {pivot}")
        return
    if min_rows and len(pivot.dropna(axis=0, how='all')) < min_rows:
        if verbose:
            print(
                f"Skipping {title}, no different scales of smaller models {pivot}")
        return
    sns.set_palette(sns.color_palette("flare"))
    colors = sns.color_palette("flare", n_colors=len(
        eval_contours)) if eval_contours is not None else []
    heatmap = sns.heatmap(100 * pivot, annot=annot, vmin=100 * vmin if pd.notna(vmin) else None,
                          vmax=100 * vmax if pd.notna(vmax) else None,
                          norm=LogNorm() if log_scale else None)  # , cmap="crest")
    if index == "num_params" and metadata is not None:
        assert evals["test_model"].nunique() == 1
        tick_labels = [item.get_text() for item in heatmap.get_yticklabels()]
        test_model_params = float(
            metadata["num_params"][evals["test_model"].iloc[0]])
        # test_size =  test_model_params/1e9
        # test_size = round((test_size)/1e9, 1)

        def improve_param_naming(params):
            params = float(params)
            scale = test_model_params/params
            scale = int(scale) if scale > 1 else round(scale, 2)

            train_size = params/1e9
            largest_size = round((train_size), 1)
            return f"{largest_size}B \n(X{scale})"
        ticks = plt.yticks(rotation=0)
        ax = plt.gca()
        ax.set_ylabel("Largest Model Parameters (Scale up predicted)")
        ax.set_yticklabels([improve_param_naming(tick_label)
                            for tick_label in tick_labels])
    # plt.setp(ax.get_yticklabels(), rotation=90)
    # sns.set_palette(sns.color_palette("crest"))

    def adapt_cell(x, y):
        # x = pivot.shape[1] - x - 1
        # y = pivot.shape[0] - y - 1
        return x, y

    for choice in set(efficient_coiches):
        i = len(efficient_coiches) - 1 - efficient_coiches[::-1].index(choice)
        x, y = adapt_cell(choice[1], choice[0])
        x += 0.4
        y += 0.2
        ax = plt.scatter(x, y, marker="*")
        if not annot:
            plt.annotate(
                f"\n<{int(eval_contours[i]* 100)}", (x-0.2, y+0.25), color=colors[i], weight='bold')
    for line in iso_contours:
        x_iso = []
        y_iso = []
        for y, x, val in line:
            x, y = adapt_cell(x, y)
            x_iso.append(x)
            y_iso.append(y)

        if max(y_iso) != pivot.shape[0]:
            x_iso.append(x_iso[-1])
            y_iso.append(pivot.shape[0])

        if len(x_iso) > 3:
            x_plot, y_plot = cells_to_contours(
                x_iso, y_iso, contour_kind, 1, 0)
            plt.plot(x_plot, y_plot)

        # # Add data points to the plot
        # for x_val, y_val in zip(x_iso, y_iso):
        #     plt.text(x_val, y_val, str((x_val, y_val)), ha='left', va='bottom')

    if title:
        plt.gca().set_title(title)
    capitalize_fig(heatmap)
    if fig_dir and fig_name:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, fig_name), bbox_inches="tight")
        try:
            plt.gcf().axes[1].remove()
        except IndexError:
            pass
        no_leg_dir = os.path.join(fig_dir, "no_legend")
        os.makedirs(no_leg_dir, exist_ok=True)
        plt.savefig(os.path.join(no_leg_dir,
                    fig_name), bbox_inches="tight")

    if show:
        plt.show()
    plt.clf()
    plt.close()

    print(f"plotting {os.path.join(fig_dir, fig_name)} done")


def remove_outlier_scales(df):
    return df[
        # (df["scaled_set"] != "chinchilla") & # The model count here is a tad misleading
        (~ df["scaled_set"].str.contains("interv")) &
        (~ df["scaled_set"].str.contains("5sh")) &
        (~ df["scaled_set"].str.contains("v0"))
    ]


def fill_nas(evals, eval, index, column, verbose=False):
    if evals.empty:
        return evals

    column_vals = sorted(evals[column].unique())
    col_idxs = {val: i for i, val in enumerate(column_vals)}

    def fill(row):
        if pd.isna(row[eval]):
            row_col_val = row[column]
            if pd.isna(row[index]) and verbose:
                print(f"Empty index:{index} in {row}")
            equivalents = evals[(evals["scaled_set"] == row['scaled_set']) & (
                row[index] == evals[index])]
            if equivalents.empty:
                return row[eval]
            col_val_to_eval = equivalents.groupby(
                column)[eval].mean().to_dict()
            # there are duplicates of this status
            if pd.notna(col_val_to_eval[column_vals[col_idxs[row_col_val]]]):
                row[eval] = col_val_to_eval[column_vals[col_idxs[row_col_val]]]
            elif row_col_val == column_vals[-1]:
                return row[eval]
            else:
                replace_with_idx_after = col_idxs[row_col_val] + 1
                while replace_with_idx_after < len(column_vals) and pd.isna(
                        col_val_to_eval.get(column_vals[replace_with_idx_after], None)):
                    replace_with_idx_after += 1
                if replace_with_idx_after not in column_vals:
                    return row[eval]
                if replace_with_idx_after >= len(column_vals):
                    if verbose:
                        print(
                            f"All row is na:{col_val_to_eval},{evals['scaled_set']},{evals[index]}, {evals}")
                    return row[eval]
                replace_with_idx_before = col_idxs[row_col_val] - 1
                while replace_with_idx_before > 0 and pd.isna(
                        col_val_to_eval[column_vals[replace_with_idx_before]]):
                    replace_with_idx_before -= 1
                if replace_with_idx_before <= 0 or column_vals[replace_with_idx_before] not in col_val_to_eval:
                    replace_with_idx_before = replace_with_idx_after
                after_weight = abs(
                    col_idxs[row_col_val] - replace_with_idx_before)
                before_weight = abs(
                    col_idxs[row_col_val] - replace_with_idx_after)
                row[eval] = ((col_val_to_eval[column_vals[replace_with_idx_before]]) * before_weight + (col_val_to_eval[
                    column_vals[replace_with_idx_after]]) * after_weight) / (before_weight + after_weight)
        return row[eval]

    evals[eval] = evals.apply(fill, axis=1)
    return evals


def aggregate_hist(evals, eval, fig_dir, show, exp_name="", iso=None, eval_contours=None, metadata=None,
                   column="percentage", vmin=None, vmax=None, min_rows=0, log_scale=False, single_scale=False):
    evals = remove_outlier_scales(evals)
    other_dir = os.path.join(fig_dir, "other")
    os.makedirs(other_dir, exist_ok=True)
    # by num X percentage
    index = "#Train models"
    bins = [2, 3, 4, 5, 6, 7, 8, 1000]
    # bins do not include left barrier allowes, e.g., only 2 in the first
    bins = [bin - 0.1 for bin in bins]
    bin_labels = ["2", "3", "4", "5", "6", "7", "8+"]
    if len(evals["scaled_set"].unique()) == 1 and column == "percentage" or single_scale:
        old_index = index
        index = f"{old_index} (Scale up predicted)"
        evals = evals.rename(columns={old_index: index})
        largest = evals["test_model"].apply(
            lambda x: metadata["num_params"][x])
        largest_train = evals["largest_train_model"].apply(
            lambda x: metadata["num_params"][x])
        scales = np.unique(
            [test / train for test, train in zip(largest, largest_train)])
        scales = -np.sort(-scales)
        scales = [int(scale) if scale > 1 else round(scale, 2)
                  for scale in scales]
        special_last = [
            bin_labels[-1]+f" \n(<X{scales[len(bin_labels)-1]})"] if len(scales) >= len(bin_labels) else ["9empty_num"+str(x) for x in range(len(bin_labels) - len(scales))]
        bin_labels = [label+f" \n(X{scale})" for label,
                      scale in zip(bin_labels[:-1], scales)] + special_last

    evals[index] = pd.cut(evals[index],
                          bins=bins, labels=bin_labels)
    evals = fill_nas(evals, eval, index, column)
    evals = evals.dropna(subset=[eval])
    aggregate_by = [eval, iso] if iso else [eval]
    agg_scaled_reps = evals.groupby(["scaled_set", index, column], observed=False)[
        aggregate_by].mean().reset_index()  # macro average - don't overweight repetitions of different size etc. (like chinchilla)
    agg = agg_scaled_reps.groupby([index, column], observed=False)[
        aggregate_by].mean().reset_index()
    plot_heatmap(evals=agg, eval=eval, eval_contours=eval_contours, iso=iso, index=index, fig_dir=fig_dir, show=show,
                 metadata=metadata,
                 title=None, fig_name=f"hist-num-models-annot-agg-{exp_name}.{FIG_FORMAT}", column=column, vmin=vmin,
                 vmax=vmax, annot=True, log_scale=log_scale)
    plot_heatmap(evals=agg, eval=eval, eval_contours=eval_contours, iso=iso, index=index, fig_dir=fig_dir, show=show,
                 metadata=metadata,
                 title=None, fig_name=f"hist-num-models-agg-{exp_name}.{FIG_FORMAT}", column=column, vmin=vmin,
                 vmax=vmax, annot=False, log_scale=log_scale)
    plot_heatmap(evals=agg, eval=eval, eval_contours=eval_contours, iso=iso, index=index, fig_dir=other_dir, show=show,
                 metadata=metadata,
                 title=None, fig_name=f"hist-num-models-agg-spl{exp_name}.{FIG_FORMAT}", column=column, vmin=vmin,
                 vmax=vmax, annot=False, log_scale=log_scale, contour_kind="spline")

    # by scale X percentage
    index = "Scale up predicted"
    bins = [1, 2, 4, 8, 16, 32, 2000]
    bin_labels = [r"$1\times-2\times$", r"$4\times$",
                  r"$8\times$", r"$16\times$", r"$32\times$", r"$32+\times$"]
    largest = evals["test_model"].apply(lambda x: metadata["num_params"][x])
    largest_train = evals["largest_train_model"].apply(
        lambda x: metadata["num_params"][x])
    evals[index] = pd.cut([test / train for test, train in zip(largest, largest_train)], bins=bins,
                          labels=bin_labels)

    agg = evals.groupby([index, column], observed=False)[
        eval].mean().reset_index()

    plot_heatmap(evals=agg, eval=eval, index=index, fig_dir=other_dir, show=show, metadata=metadata, min_rows=min_rows,
                 title=None,
                 fig_name=f"hist-scale-agg_{exp_name}.{FIG_FORMAT}", column=column, vmin=vmin, vmax=vmax, ascending_index=False,
                 annot=False)
    plot_heatmap(evals=agg, eval=eval, index=index, fig_dir=other_dir, show=show, metadata=metadata, min_rows=min_rows,
                 title=None,
                 fig_name=f"hist-scale-agg-annot_{exp_name}.{FIG_FORMAT}", column=column, vmin=vmin, vmax=vmax,
                 ascending_index=False,
                 annot=True)


def nunique_model_size(df) -> int:
    return len(unique_model_size(df))


def unique_model_size(df) -> List[int]:
    max_training_steps = df.groupby(["num_params"])["perf"].count().mean()
    model_sizes = df["num_params"]
    if model_sizes.dropna().empty:
        return []
    # heuristic to separate checkpoints and training runs from 1 point per model (ee.g. because mode sizes are extracted and hence noisy)
    if max_training_steps > 5:
        model_sizes = [round(size, -6) for size in model_sizes]
    unique_model_sizes = np.unique(model_sizes)
    return unique_model_sizes


def get_model_nums(num_models):
    if num_models > 60:
        model_nums = range(2, num_models + 1, 10)
    elif num_models > 20:
        model_nums = range(2, num_models + 1,
                           int(num_models / 5))
    else:
        model_nums = range(2, num_models + 1)
    return model_nums


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
    for scaled_set in tqdm.tqdm(df["scaled_set"].unique(), desc="Scaling families"):
        model_by_size = df.query("scaled_set==@scaled_set")[["model_name", "num_params"]].drop_duplicates().sort_values(
            "num_params")

        largest_model = model_by_size["model_name"].iloc[-1]
        if scale_down:
            model_by_size = model_by_size.iloc[::-1]
            if "pythia" in scaled_set:
                model_by_size = model_by_size[1:]
        target_model = model_by_size["model_name"].iloc[-1]
        smaller_models = model_by_size["model_name"].iloc[:-1]
        if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
            continue
        for percentage in train_percentages:
            for num_train_models in get_model_nums(len(smaller_models)):
                for train_models, iter_data in iter_models(df=df, scaled_set=scaled_set, percentage=percentage, num_train_models=num_train_models, train_models=smaller_models, target_model=target_model):
                    train_models = list(train_models)
                    cache_id = scaled_set + \
                        str(num_train_models) + str(percentage) + \
                        str(iter_data)
                    if not force and cache_id in cache:
                        res = cache[cache_id]
                        assert len(res) == len(
                            resulting_cols), "columns mismatch, clean cache"
                    else:

                        train_df = get_model_data(df=df, models=train_models,
                                                  max_percentage=percentage,
                                                  min_tokens=cut_beginning)
                        test_df = get_model_data(
                            df=df, models=[target_model], min_percentage=test_percentage)
                        unique_model_sizes = nunique_model_size(train_df)
                        flops = train_df["flops"].sum()
                        mse, are, train_are, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_are=abs_are,
                                                                              verbose=verbose)

                        last_pred = predicted[-1] if predicted is not None else None
                        res = (
                            scaled_set, percentage, mse, are, last_pred, target_model, unique_model_sizes,
                            train_models[-1], flops, iter_data,
                            tuple(popt) if popt is not None else None)
                        if verbose:
                            print(
                                f"{scaled_set} {100 * percentage}% unique model sizes {unique_model_sizes}: mse {mse}, ARE {are}, train ARE{train_are}, popts {popt} predicted {np.mean(predicted) if predicted is not None else ''} actual {test_df['perf'].mean()}")
                        assert len(res) == len(
                            resulting_cols), "columns mismatch, ensure saved (res) and loaded (resulting_cols) values match"
                        cache[cache_id] = res
                    evals.append(res)
        save_cache(cache, cache_name)
    save_cache(cache, cache_name)

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
    if len(evals["popt"].dropna().iloc[0]) > 3:
        plot_2popt(evals, poptx=0, popty=3, name="scale", eval=eval, fig_dir=fig_dir, show=show, x_label="A", y_label="$\\alpha$",
                   metadata=scaled_set_metadata)
        plot_2popt(evals, poptx=0, popty=3, name="scale_model_type", eval=eval, fig_dir=fig_dir, show=show, x_label="A", y_label="$\\alpha$",
                   metadata=scaled_set_metadata, labels="model_type")
    if len(evals["popt"].dropna().iloc[0]) > 4:
        plot_2popt(evals, poptx=1, popty=4, name="tokens", eval=eval, fig_dir=fig_dir, show=show, x_label="B", y_label="$\\beta$",
                   metadata=scaled_set_metadata)
        plot_2popt(evals, poptx=1, popty=4, name="tokens_model_type", eval=eval, fig_dir=fig_dir, show=show, x_label="B", y_label="$\\beta$",
                   metadata=scaled_set_metadata, labels="model_type")
    if len(evals["popt"].dropna().iloc[0]) > 4:
        plot_2popt(evals, poptx=3, popty=4, name="alpha_beta", eval=eval, fig_dir=fig_dir, show=show, x_label="$\\alpha$", y_label="$\\beta$",
                   metadata=scaled_set_metadata)
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

    eval = "are"
    for col_group in ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type",
                      "best_loss_binned"]:
        plt.clf()
        # ares = evals.groupby([col_group, "percentage"])["are"].mean().reset_index([1])
        for group in evals[col_group].unique():
            subdf = evals.query(f"{col_group}==@group").dropna(subset=[eval])
            subdf = subdf.groupby(["percentage"])[eval].mean().reset_index()
            if df.dropna(axis=1, how='all').empty:
                continue
            x = subdf["percentage"]
            y = subdf[eval]
            sns.lineplot(x=x.tolist(), y=y.tolist(), label=group)

        if abs_are:
            plt.ylim(bottom=0)
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title=col_group)
        plt.xlabel("Percentage of Training Trajectory Available")
        plt.ylabel("Mean Normalized Distance")
        if fig_dir:
            plt.savefig(os.path.join(
                fig_dir, f"perc-are_{eval}_{col_group}.{FIG_FORMAT}"), bbox_inches="tight")
        if show:
            plt.show()
        plt.clf()

    # compare predictions to actual
    eval = "are"
    for col_group in [
            "original_paper"]:  # ["max_tokens_binned", "original_paper", "data", "arch", "model_size", "model_type", "best_loss_binned"]
        for group in evals[col_group].unique():
            for model in evals[evals[col_group] == group]["model_name"].unique():
                # if "pythi" in group and "pythia-1.4b" not in model:
                #     continue
                subdf = df.query(
                    "model_name==@model & tokens_seen>=@cut_beginning")
                if subdf.empty:
                    continue
                ax = sns.lineplot(
                    x=subdf["tokens_seen"], y=subdf["perf"], label=f"{group} {model}")
                color = ax.lines[-1].get_color()
                subdf = evals.query("model_name==@model")
                subdf = subdf.sort_values('are').drop_duplicates(
                    subset=['percentage', 'max_tokens', 'scaled_set', 'test_model', 'model_size'])
                x = subdf["max_tokens"]
                y = subdf["last_pred"]
                if y.dropna().empty:
                    continue
                sns.scatterplot(x=x, y=y, color=color)
                for i, txt in enumerate(train_percentages):
                    ax.annotate(txt, (x.iloc[i], y.iloc[i]))
            plt.xscale('log')
            plt.xlabel("Tokens seen")
            plt.ylabel("Loss")
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0,
                       title=f"{col_group}:{group}")
            sns.despine()
            # plt.legend().remove()
            # pos = ax.get_position()
            # ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
            # ax.legend(loc='center right', bbox_to_anchor=(2.25, 0.5))
            if fig_dir:
                os.makedirs(os.path.join(
                    fig_dir, "compare_predictions"), exist_ok=True)
                plt.savefig(os.path.join(fig_dir, "compare_predictions", f"{group}_{col_group}_{eval}.{FIG_FORMAT}"),
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
            indx = (tokens_per_epoch == metadata["tokens_per_epoch"]) & (
                num_params == metadata["num_params"])
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
            indx = (tokens_per_epoch == metadata["tokens_per_epoch"]) & (
                num_params == metadata["num_params"])
            x = metadata[indx]["tokens_seen"]
            ax = sns.lineplot(
                x=x, y=perf[indx], label=f"actual {num_params} {tokens_per_epoch}")
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
            is_predicted = np.concatenate(
                [np.ones_like(predicted), np.zeros_like(perf)])
            df = pd.DataFrame(
                {"perf": pred_n_perf, "is_predicted": is_predicted, trait: x})
            sns.boxplot(data=df, x=trait, y="perf",
                        hue="is_predicted", gap=0.1)
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


def metric_per_column(df) -> Dict[str, LossType]:
    if _metric_map:
        return _metric_map
    for column in df.select_dtypes(include='number'):
        norm_column = column.lower().strip()
        if column in _metric_map:
            continue
        elif contains(norm_column, ("norm", "quasi")) and "acc" in norm_column:
            _metric_map[column] = LossType("norm_acc")
        elif contains(norm_column, ("acc", "pct_ster", "exact_match", " em", "_em", "downstream_eval", "bpb")):
            _metric_map[column] = "1acc" if df[column].max() <= 1 else "100acc"
        elif contains(norm_column, "bits_per_byte"):
            _metric_map[column] = LossType("bpb")
        elif contains(norm_column, ("loss")):
            _metric_map[column] = LossType("loss")
        elif contains(norm_column, ("perp", "ppl")):
            _metric_map[column] = LossType("perp")
            if contains(norm_column, "byte"):
                _metric_map[column] = LossType("byte_perp")
        elif contains(norm_column, "likelihood_difference"):
            _metric_map[column] = LossType("likelihood_difference")
        elif contains(norm_column, "tim"):
            _metric_map[column] = LossType("time")
        else:
            min_score, max_score = (df[column].min(), df[column].max())
            if (min_score >= 0 and max_score <= 1):
                _metric_map[column] = LossType("1acc_like")
            elif (min_score >= 0 and 10 < max_score < 100):
                _metric_map[column] = LossType("100acc_like")
            else:
                _metric_map[column] = LossType("unk")
    return _metric_map


def get_data_path(cache_dir):
    return os.path.join(cache_dir, "data.csv.zst")


def get_perf_path(cache_dir: str, loss_types: List[LossType]):
    loss_types = [tp.value for tp in loss_types]
    return os.path.join(cache_dir, f"perfDF_{'_'.join(loss_types)}.csv.zst")
