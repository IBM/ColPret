import os

from experiments.minimal_cut import minimal_cut
from util.fit_utils import metric_per_column, get_perf_path, get_data_path, scale_fit_per_model, hist_fit, \
    hist_one_model_fit, get_perf_df
from util.read_data import get_data

if __name__ == '__main__':
    force = False
    force = True
    data_path = '/Users/lc/PycharmProjects/CLPR/cache/data.csv'
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = ("perp",)
    perf_path = get_perf_path(cache_dir, loss_types)
    df = get_perf_df(get_data(save_in=data_path, force=force), loss_types, save_in=perf_path)
    force = True
    force = False
    fig_dir = '/Users/lc/PycharmProjects/CLPR/figs/'
    os.makedirs(fig_dir, exist_ok=True)

    metric_per_column(df)
    # df = df[df["model_type"].isin(["OPT"])]
    # df = df[(df["model_type"] .isin(["GPT2"])) & (df["code"].isna())]
    # df = df[df["original_paper"] == "pythia"]
    # df = df[df["domain"] == "LM"]
    abs_mnd = True
    hist_one_model_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist_1m"), at_least_loss=10, abs_mnd=abs_mnd)
    hist_fit(df, force=force, fig_dir=os.path.join(fig_dir, "hist"), at_least_loss=10, abs_mnd=abs_mnd)
    scale_fit_per_model(df, force=force, fig_dir=os.path.join(fig_dir, "per_model"), at_least_loss=10, abs_mnd=abs_mnd)
    minimal_cut(df, force=force, fig_dir=fig_dir, at_least_loss=10, abs_mnd=abs_mnd)
    # fit_on_smaller(df,force)
    # scaling_scaling_law(df, force)
    # cross_validate(df,force)

    # data_aware_fit()
