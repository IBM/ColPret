import os
import sys

sys.path.append("/Users/lc/PycharmProjects/CLPR")
os.chdir("/Users/lc/PycharmProjects/CLPR")
if True:
    from util.naming import to_int
    from util.fit_utils import get_perf_df, metric_per_column, get_perf_path, get_data_path, LossType, hist_fit
    from util.read_data import get_data
    from fitting_funcs import ChinchillaFit, FitInfo


def iter_model_sizes(df, scaled_set, percentage, num_train_models, train_models, largest_model):
    for last in range(num_train_models, len(train_models) + 1):
        model_list = list(train_models)[last-num_train_models:last]

        largest_train_model = model_list[-1]
        largest_train_size = df.query(
            "@largest_train_model==model_name")[["num_params"]].drop_duplicates().iloc[0].iloc[0]
        largest_train_size = (largest_train_size)/1e9
        largest_size = df.query(
            "@largest_model==model_name")[["num_params"]].drop_duplicates().iloc[0].iloc[0]
        largest_size = (largest_size)/1e9
        scale = float(largest_size)/float(largest_train_size)
        scale = int(scale) if scale > 1 else round(scale, 2)
        yield model_list, f"{largest_train_size}B \n(X{scale})"


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

    df = df[df["scaled_set"] != "LLaMA"]
    df = df[(df["original_paper"] != "pythia") |
            (df["num_params"] > to_int("0.17B"))]
    # fit

    # chinchila based (datablation guesses)
    force = False
    force = True
    abs_are = True
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    abs_are = True
    verbose = False
    verbose = True
    fit_info = ChinchillaFit
    experiment_name = "num_to_size"
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, experiment_name), at_least_loss=10, abs_are=abs_are,
             fit_info=fit_info, cut_beginning=10 ** 10, train_percentages=[1], iter_models=iter_model_sizes, experiment_name=experiment_name, iter_axis_name="Largest Model Parameters (Scale up predicted)")
