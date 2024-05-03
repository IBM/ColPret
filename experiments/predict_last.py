import os

from fitting_funcs import FitInfo
from util.fit_utils import get_perf_df, \
    metric_per_column, get_perf_path, get_data_path, LossType, hist_fit
from util.read_data import get_data


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
    sub_metadata = metadata[metadata["num_params"] == model_size].sort_values(by="tokens_seen")
    pred = sub_metadata["perf"].iloc[-5:].mean()
    return [pred], None


PredictLast = FitInfo(func=return_prediction,
                      guess=(),
                      bounds=(), name="best_seen",
                      fit_func=fit_last_score)

#
# def predict_best_seen(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"),
#                       percentages=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99), abs_are=True):
#     cut_beginning = 10 ** 10
#     fit_info = PredictBest
#     test_percentage = 0.7
#     experiment_name = "predict_last"
#     fig_dir = os.path.join(fig_dir, experiment_name)
#     os.makedirs(fig_dir, exist_ok=True)
#     cache_name = experiment_name + "_" + str(abs_are) + "_".join(loss_types) + "_" + fit_info.name
#     cache = get_cache(cache_name, force)
#     df = df.dropna(subset=["scaled_set"])
#     evals = []
#     # # skip families with one model
#     # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
#     # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]
#
#     for scaled_set in df["scaled_set"].unique():
#         model_by_size = df.query("scaled_set==@scaled_set")[
#             ["model_name", "num_params"]].drop_duplicates().sort_values(
#             "num_params")
#         largest_model = model_by_size["model_name"].iloc[-1]
#         smaller_models = model_by_size["model_name"].iloc[:-1]
#         if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
#             continue
#         for min_percentage in percentages:
#             for num_train_models in range(2, len(smaller_models) + 1):
#                 cache_id = scaled_set + str(num_train_models) + str(min_percentage)
#                 if not force and cache_id in cache:
#                     res = cache[cache_id]
#                 else:
#                     train_df = get_model_data(df=df, models=smaller_models.to_list()[:num_train_models],
#                                               max_percentage=min_percentage,
#                                               min_tokens=cut_beginning)
#                     test_df = get_model_data(df=df, models=[largest_model], min_percentage=test_percentage,
#                                              min_tokens=cut_beginning)
#
#                     mse, are, train_are, predicted, popt = single_scaling(train_df, test_df, fit_info, abs_are=abs_are)
#                     last_pred = predicted[-1] if predicted is not None else None
#                     res = (scaled_set, min_percentage, mse, are, last_pred, largest_model, nunique_model_size(train_df),
#                            tuple(popt) if popt is not None else None)
#                     cache[cache_id] = res
#                     print(f"{scaled_set} {min_percentage} {num_train_models + 1}: {are}, {train_are}")
#                 evals.append(res)
#         save_cache(cache, cache_name)
#     save_cache(cache, cache_name)
#
#     evals = pd.DataFrame(evals, columns=["scaled_set", "min_percentage", "mse", "are", "last_pred", "largest_model",
#                                          "num_models", "params"])
#     print(f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['largest_model']}")
#     evals = evals.loc[:, ["scaled_set", "min_percentage", "are", "num_models"]]
#     c4_idx = evals["scaled_set"].apply(lambda x: "GPT2-c4" in x)
#     c4 = evals[c4_idx].groupby(["min_percentage", "num_models"])[
#         "are"].mean().apply(lambda x: x).reset_index()
#     evals = evals[~c4_idx]
#     c4["scaled_set"] = "GPT2-c4-all"
#     oscar_idx = evals["scaled_set"].apply(lambda x: "GPT2-oscar" in x)
#     oscar = evals[oscar_idx].groupby(["min_percentage", "num_models"])[
#         "are"].mean().apply(lambda x: x).reset_index()
#     evals = evals[~oscar_idx]
#     oscar["scaled_set"] = "GPT2-oscar-all"
#     evals = pd.concat(
#         [evals, oscar, c4])
#     plot_models_percentage_hist(evals, eval="are", index="num_models", columns="min_percentage", fig_dir=fig_dir,
#                                 min_rows=1, show=show)


if __name__ == '__main__':
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = (LossType.PERP, LossType.LOSS,)
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
    abs_are = True
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "last"), at_least_loss=10, abs_are=abs_are,
             fit_info=PredictLast, verbose=True)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "best"), at_least_loss=10, abs_are=abs_are,
             fit_info=PredictBest, verbose=True)
    # predict_best_seen(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_are=abs_are)
