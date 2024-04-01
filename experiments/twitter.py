import os

import pandas as pd
import sys
sys.path.append("/Users/lc/PycharmProjects/CLPR")
os.chdir("/Users/lc/PycharmProjects/CLPR")
if True:
    from fitting_funcs import ChinchillaFit
    from util.cache import save_cache, get_cache
    from util.fit_utils import plot_models_percentage_hist, get_model_data, get_perf_df, metric_per_column, get_perf_path, get_data_path, single_scaling, LossType
    from util.read_data import get_data


def cosine_test(df, force=False, fig_dir=None, show=False, loss_types=("perp"), at_least_loss=float("inf"), experiment_name="minimal_cut",
                minimal_cuts=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99), abs_are=True, fit_info=ChinchillaFit, cut_beginning=10 ** 10):

    test_percentage = 0.7

    fig_dir = os.path.join(fig_dir, experiment_name)
    os.makedirs(fig_dir, exist_ok=True)
    cache_name = experiment_name + "_" + \
        str(abs_are) + "_".join(loss_types) + "_" + fit_info.name
    cache = get_cache(cache_name, force)

    test_models = []
    train_models = []
    df = df[df["original_paper"] == 'datablations']
    for name in df["model_name"].unique():
        if name.count("-") > 1 and not (name.count("-") == 3 and "c4" in name):
            print(f"skipping {name}")
        elif df[df["model_name"] == name]["epochs"].max() > 1:
            print(f"skipping model with multiple epochs {name}")
        else:
            # df[df["model_name"] == name]["num_params"].max() == 8670000000:
            if "178" in name:
                test_models.append(name)
            else:
                # assert "867" not in name
                # assert "8b7" not in name
                train_models.append(name)
    evals = []
    train_models = ['GPT2-14m100m100m', 'GPT2-196m1b51b5', 'GPT2-1b1100m100m',
                    'GPT2-1b11b51b5', 'GPT2-20m400m400m', 'GPT2-25m400m400m',
                    'GPT2-2b8100m100m', 'GPT2-2b84b4b', 'GPT2-2b855b55b',
                    'GPT2-4b212b12b', 'GPT2-4b284b84b', 'GPT2-619m2b72b7',
                    'GPT2-7m100m100m', 'GPT2-83m1b51b5']

    # # skip families with one model
    # keep_family = df.groupby("scaled_set")["model_name"].unique().apply((lambda x: len(x) > 1)).to_dict()
    # df = df[df["scaled_set"].apply(lambda row: keep_family[row])]

    # for scaled_set in df["scaled_set"].unique():
    # model_by_size = df.query("scaled_set==@scaled_set")[
    #     ["model_name", "num_params"]].drop_duplicates().sort_values(
    #     "num_params")
    # largest_model = model_by_size["model_name"].iloc[-1]
    smaller_models = train_models
    # if df.query("model_name==@largest_model")["perf"].min() > at_least_loss:
    #     continue
    for min_percentage in minimal_cuts:
        cache_id = "cosine" + str(min_percentage)
        if not force and cache_id in cache:
            res = cache[cache_id]
        else:
            train_df = get_model_data(df=df, models=smaller_models)
            # ,
            #                           max_percentage=test_percentage,
            #                           min_tokens=cut_beginning, min_percentage=min_percentage * test_percentage)
            test_df = get_model_data(df=df, models=test_models, min_percentage=test_percentage,
                                     min_tokens=cut_beginning)
            num_models = len(train_df["model_name"].unique())
            mse, are, train_are, predicted, popt = single_scaling(
                train_df, test_df, fit_info, abs_are=abs_are)
            last_pred = predicted[-1] if predicted is not None else None
            res = ("", min_percentage, mse, are, last_pred, test_models[0], num_models,
                   tuple(popt) if popt is not None else None)
            cache[cache_id] = res
            print(
                f"cosine {min_percentage} {num_models}: {are}, {train_are}")
        evals.append(res)
    save_cache(cache, cache_name)

    evals = pd.DataFrame(evals, columns=["scaled_set", "Min. percentage", "mse", "are", "last_pred", "largest_model",
                                         "#Train models", "params"])
    print(
        f"models with max normalized distance: {evals.sort_values(by='are').dropna()[-10:]['largest_model']}")
    evals = evals.loc[:, ["scaled_set",
                          "Min. percentage", "are", "#Train models"]]
    # c4_idx = evals["scaled_set"].apply(lambda x: "GPT2-c4" in x)
    # c4 = evals[c4_idx].groupby(["Min. percentage", "#Train models"])[
    #     "are"].mean().apply(lambda x: x).reset_index()
    # evals = evals[~c4_idx]
    # c4["scaled_set"] = "GPT2-c4-all"
    # oscar_idx = evals["scaled_set"].apply(lambda x: "GPT2-oscar" in x)
    # oscar = evals[oscar_idx].groupby(["Min. percentage", "#Train models"])[
    #     "are"].mean().apply(lambda x: x).reset_index()
    # evals = evals[~oscar_idx]
    # oscar["scaled_set"] = "GPT2-oscar-all"
    # evals = pd.concat([evals, oscar, c4])

    plot_models_percentage_hist(evals, eval="are", index="#Train models", columns="Min. percentage", fig_dir=fig_dir, reverse_x=True,
                                min_rows=1, show=show)


if __name__ == '__main__':
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = (LossType.PERP, LossType.LOSS)
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
    cosine_test(df, force=force, fig_dir=fig_dir,
                at_least_loss=10, abs_are=abs_are)
