import os

from util.fit_utils import get_perf_df, get_perf_path, LossType, get_data_path, get_per_model_metadata
from util.read_data import get_data


def summary_df(all_df, save_in=None, cache_dir=None, force=True):
    sub_cols = ["scaled_set", "model_name", "num_params", "data", "loss_cols", "original_paper"]
    subdf = all_df.loc[:, sub_cols].drop_duplicates()

    max_flops_dict = all_df.groupby("model_name")["tokens_seen"].max().to_dict()
    subdf["tokens_seen"] = subdf["model_name"].apply(lambda name: max_flops_dict[name])

    max_flops_dict = all_df.groupby("model_name")["flops"].max().to_dict()
    subdf["flops"] = subdf["model_name"].apply(lambda name: max_flops_dict[name])

    down_loss_types = (LossType.ACC1LIKE, LossType.ACC100LIKE, LossType.NORM_ACC, LossType.LIKELIHOOD_DIFFERENCE)
    perf_path = get_perf_path(cache_dir, down_loss_types)
    down_df = get_perf_df(all_df, down_loss_types, save_in=perf_path, force=force)
    downstream_count_dict = down_df.groupby("model_name")["perf"].count().to_dict()
    subdf["downstream_containing_checkpoints"] = subdf["model_name"].apply(
        lambda name: downstream_count_dict.get(name, 0))
    subdf["downstream_containing_checkpoints"] = subdf.apply(
        lambda row: "TBD" if "over" in row["original_paper"] else row["downstream_containing_checkpoints"], axis=1)

    perp_loss_types = (LossType.PERP, LossType.LOSS)
    perf_path = get_perf_path(cache_dir, perp_loss_types)
    loss_df = get_perf_df(all_df, perp_loss_types, save_in=perf_path, force=force)
    loss_count_dict = loss_df.groupby("model_name")["perf"].count().to_dict()
    subdf["loss_containing_checkpoints"] = subdf["model_name"].apply(lambda name: loss_count_dict.get(name, 0))
    if save_in:
        os.makedirs(save_in, exist_ok=True)
        subdf.to_csv(os.path.join(save_in, "summary.csv"), index=False)


if __name__ == '__main__':
    # acquire data
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = (LossType.ACC1LIKE, LossType.ACC100LIKE, LossType.NORM_ACC, LossType.LIKELIHOOD_DIFFERENCE)
    perf_path = get_perf_path(cache_dir, loss_types)
    all_df = get_data(save_in=data_path, force=force)
    df = get_perf_df(all_df, loss_types, save_in=perf_path, force=force)
    metadata = get_per_model_metadata(df)

    fow_path = '/Users/lc/PycharmProjects/CLPR/fow'
    summary_df(all_df, save_in=fow_path, cache_dir=cache_dir, force=force)
