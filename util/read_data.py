import ast
import os.path
from functools import reduce
import pickle
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
import sys
sys.path.append("/Users/lc/PycharmProjects/CLPR")

if True:
    from util.naming import to_int, to_str

BASIC_DF_COLS = ["model_name", "model_type", "scaled_set", "tokens_seen", "flops", "num_params", "data", "checkpoint",
                 "loss_cols",
                 "original_paper", "seed"]
DATA_AWARE_DF_COLS = BASIC_DF_COLS + ["epochs"]
ARCH_AWARE_DF_COLS = BASIC_DF_COLS + ["arch"]
ARCHS = ["dec", "enc", "enc-dec", "moe", "ssm", np.nan]


def test_dfs(dfs, relevant_cols, supress_zero_tokens=False):
    for df in dfs:
        test_df(df, relevant_cols, supress_zero_tokens)


def test_df(df, relevant_cols, supress_zero_tokens=False):
    relevant_cols = set(relevant_cols)
    assert all(col in df.columns for col in
               relevant_cols), f"Missing columns: {[col for col in relevant_cols if col not in df.columns]}"
    loss_cols = set(col for cols in df["loss_cols"] for col in cols)
    missing_cols = [col for col in loss_cols if col not in df.columns]
    assert is_numeric_dtype(df["num_params"])
    assert not df["tokens_seen"].isnull().sum() > 0
    assert all(df["tokens_seen"] > 0), "tokens seen must be non-negative"
    assert supress_zero_tokens or all(df[
        "tokens_seen"] != 0), "values reported without tokens_seen, " \
        "if that is on purpose (randomly initialized score)," \
        " pass supress_zero_tokens=True, otherwise check data"
    assert not missing_cols, f"Loss column are stated in 'loss_col' but do not exist:{missing_cols}"
    assert not set(df["arch"].unique(
    )) - set(ARCHS), f"unexpexted arch types:{set(df['arch'].unique()) - set(ARCHS)}"
    if "GPT2-14m100m100m" in df["model_name"]:
        raise (
            "Why multiple models with the same training etc. but different losses? Seeds?")


def hf_checkpoint(name, revision):
    return {"pretrained_model_name_or_path": name, "revision": revision}


def get_bimix():
    dfs = []
    for dataset in ["Pile", "SlimPajama"]:
        with open(f"raw_data/bimix-law/{dataset}/ppls.pkl", "rb") as ppl_fl:
            ppls = pickle.load(ppl_fl)
        with open(f"raw_data/bimix-law/{dataset}/ratios.pkl", "rb") as ratio_fl:
            ratios = pickle.load(ratio_fl)
        for data_name, df in ppls.items():
            if data_name == "Baseline":
                new_data_name = dataset
            else:
                new_data_name = dataset+"_"+data_name + "_mix"
            df.columns = [x+"_ppl" if x !=
                          "global_step" else x for x in df.columns]
            df["data"] = new_data_name
            df["model_name"] = new_data_name
            df["model_type"] = "DoReMi"
            df["checkpoint"] = np.nan
            # scaled in the sense of data ratio not size
            df["scaled_set"] = df["model_name"]
            for ratio in ratios[data_name].index:
                df[ratio+"_data_ratio"] = ratios[data_name][ratio]
            dfs.append(df)
    df = pd.concat(dfs)
    df["original_paper"] = "DataMixing arxiv.org/abs/2405.14908"
    df = df.rename(columns={"global_step": "tokens_seen"})
    df["tokens_seen"] = df["tokens_seen"] * 512 * 1024

    # ignores downstream
    loss_columns = [col for col in df.columns if "ppl" in col]
    df["loss_cols"] = df.apply(
        lambda row: [col for col in loss_columns if pd.notna(row[col])], axis=1)
    df["model_type"] = "DoReMi"
    df["scaled_set"] = df["data"]
    df["epochs"] = 1
    df["arch"] = "dec"
    df["num_params"] = to_int("280M")
    df["seed"] = "0"

    df["flops"] = None
    return df


def get_chinchila():

    # TODO extract the full training losses
    df = pd.read_csv("raw_data/chinchila_extracted.csv")
    df["checkpoint"] = np.nan
    df["loss_cols"] = [["loss"]] * len(df)  # let later processing choose
    df["original_paper"] = "chinchilla"
    df["domain"] = "LM"
    df["scaled_set"] = "chinchilla"

    df["seed"] = "0"
    df = df.rename(columns={"Model Size": "num_params",
                   "steps": "tokens_seen", "Training FLOP": "flops"})
    df = df.drop(columns=["x", "y", "color", "hex_color"])
    df["num_params"] = df["num_params"].apply(int)
    df["num_params"] = df["num_params"].apply(lambda size: round(size, -7))
    df["model_name"] = df["num_params"].apply(lambda x: f"chin{x}")
    df["tokens_seen"] = df.apply(
        lambda row: row["flops"] / 6 / to_int(row["num_params"]), axis=1)
    df = df[df["tokens_seen"] != 0]
    df["arch"] = "dec"
    df["epochs"] = 1
    df["data"] = "MassiveText"  # unknown Google thing
    df["model_type"] = "Gopher"  # slight changes, adamW, slight tokenizer change
    return [df]


def get_overtrain():

    df = pd.read_csv("raw_data/overtrain/overtrain.csv")
    df = df.rename(columns={"loss": "openLM_loss",
                   "tokens": "tokens_seen", "name": "model_name"})
    df["loss_cols"] = [["train_loss"]] * len(df)
    name_to_params = {
        "d=96_l=8_h=4": "0.011B",
        "d=512_l=8_h=4": "0.079B",
        "d=576_l=24_h=8": "0.154B",
        "d=1024_l=24_h=8": "0.411B",
        "open_lm_1b": "1.4B",
        "open_lm_7b": "6.9B",
    }
    data_to_name = {
        "c4_original": "C4",
        "pile": "Pile",
        "rpj": "RedPajama",
        "rw_original": "RefinedWeb",
    }

    def data_from_name(name):
        for key, val in data_to_name.items():
            if key in name:
                return val
        raise ValueError

    def params_from_name(name):
        for key, val in name_to_params.items():
            if key in name:
                return val
        raise ValueError

    df["num_params"] = df["model_name"].apply(params_from_name).apply(to_int)
    df["data"] = df["model_name"].apply(data_from_name)
    df["model_type"] = "overtrain"
    df['arch'] = "dec"
    df["loss_cols"] = [["openLM_loss"]] * len(df)
    df['flops'] = np.nan

    downstream_df = pd.read_csv(
        "raw_data/overtrain/overtrain_downstreams.csv", index_col=0)
    downstream_df = downstream_df.rename(
        columns={"loss_openlm": "openLM_loss"})
    downstream_df["num_params"] = downstream_df["name"].apply(
        params_from_name).apply(to_int)
    downstream_df["tokens_seen"] = downstream_df.apply(lambda row: float(row["flops"]) / 6 / to_int(row["num_params"]),
                                                       axis=1)
    idxs = df.groupby(["model_name"])["openLM_loss"].idxmin().to_dict()
    downstream_df.index = [idxs.get(name, name)
                           for name in downstream_df["name"]]
    downstream_df = downstream_df.drop(
        columns=["model_name", "color", "D", "N"])
    downstream_df = downstream_df.rename(columns={"name": "model_name"})
    downstream_df.columns = [col.replace(
        "err", "acc") for col in downstream_df.columns]
    df = df.combine_first(downstream_df).reset_index(drop=True)
    loss_cols = [
        col for col in df.columns if ("loss" in col or "acc" in col) and "cols" not in col]
    df["loss_cols"] = df.apply(lambda row: tuple(
        col for col in loss_cols if pd.notna(row[col])), axis=1)

    max_tokens = df.groupby(["model_name"])["tokens_seen"].max().to_dict()
    df['checkpoint'] = df.apply(
        lambda row: f"https://huggingface.co/mlfoundations/scaling/tree/main/{row['model_name']}/checkpoints" if
        max_tokens[row["model_name"]] == row["tokens_seen"] else np.nan, axis=1)
    df['epochs'] = 1
    df['original_paper'] = "overtrain"  # arxiv.org/abs/2403.08540
    df["scaled_set"] = df.apply(lambda x: f"overtrain-{x['data']}", axis=1)
    df["compute_opt"] = df["model_name"].apply(
        lambda x: float(x.split("-")[-1]))
    compute_opt_per_model = df.groupby(["scaled_set", "num_params"])[
        "compute_opt"].max().to_dict()
    df["scaled_set"] = df.apply(
        lambda row: row["scaled_set"] if compute_opt_per_model[(row["scaled_set"], row["num_params"])] == row[
            "compute_opt"] else np.nan, axis=1)
    df["seed"] = "0"
    return [df]


def get_t5():

    def name_to_param(name):
        if "base" in name:
            return 247586304
        elif "large" in name:
            return 783173632
        elif "xxl" in name:
            return 11135426560
        elif "xl" in name:
            return 2849804288

    df = pd.read_csv("aggregated_eval/T5_pile.csv")
    df["checkpoint"] = df.apply(
        lambda row: hf_checkpoint(f"EleutherAI/pile-t5-{row['model_name']}", f"step_{row['steps']}"), axis=1)
    df["loss_cols"] = [["val_perplexity"]] * \
        len(df)  # let later processing choose
    df["original_paper"] = "t5-pile"
    df["domain"] = "LM"
    df["num_params"] = df["model_name"].apply(name_to_param)
    df["scaled_set"] = "t5-pile"

    df["seed"] = "0"
    df = df.rename(columns={"steps": "tokens_seen"})
    df["tokens_seen"] *= 1e6
    df = df[df["tokens_seen"] != 0]
    df["model_name"] = df.apply(
        lambda row: f"pile-t5-{row['model_name']}", axis=1)
    df["arch"] = "enc-dec"
    df["flops"] = None
    df["epochs"] = 1
    df["data"] = "pile"
    df["model_type"] = "t5-pile"
    return [df]


def get_datablations():
    dfs = []
    df = pd.read_csv(
        "aggregated_eval/datablations_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    df = df.rename(columns={"token_num": "tokens_seen"})
    df["model_name"] = df.apply(lambda x: f"GPT2-{to_str(x['num_params'])}-{x['data']}-{to_str(x['epochs'])}",
                                axis=1)
    df["scaled_set"] = df.apply(lambda x: f"GPT2-{x['data']}-{to_str(x['epochs'])}",
                                axis=1)
    assert len(df["model_name"].unique()) == len(
        set((x for x in df["model_args"])))
    df["original_paper"] = "datablations"
    df["model_type"] = "GPT2"
    df["domain"] = "LM"
    df["arch"] = "dec"
    # note, more checkpoints exist for seeds of the 2B models
    df["checkpoint"] = df["checkpoint"]
    df["loss_cols"] = [("loss",)] * len(df)
    df["seed"] = "0"

    # from util.fit_utils import LossType
    # loss_types = (LossType.PERP, LossType.LOSS)
    # from fit import get_perf_df
    # a = get_perf_df(df, loss_types)
    # print(a.groupby("model_name")["perf"].min().to_dict())

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv(
        "aggregated_eval/datablations_contour_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    df = df.rename(columns={"token_num": "tokens_seen"})
    df["scaled_set"] = np.nan

    def parse_name(row):
        name = row["model_args"]
        name = name.replace("{'name': '", "")
        name = name.replace("'}", "")
        return f"GPT2-{name}"

    df["model_name"] = df.apply(parse_name, axis=1)
    df["original_paper"] = "datablations"
    df["model_type"] = "GPT2"
    df["domain"] = "LM"
    df["arch"] = "dec"
    df["checkpoint"] = "missing? but should exist https://github.com/huggingface/datablations#download"
    df["loss_cols"] = [("openLM_loss",)] * len(df)
    df["seed"] = "0"

    # from util.fit_utils import LossType
    # loss_types = (LossType.PERP, LossType.LOSS)
    # from fit import get_perf_df
    # a = get_perf_df(df, loss_types)
    # print(a.groupby("model_name")[
    #           "perf"].min().to_dict())  # TODO problem with processing? why are losses all the same per model size?

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv(
        "aggregated_eval/datablations_code_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    df["tokens_per_epoch"] = df["tokens_per_epoch"].apply(to_int)
    df = df.rename(columns={"token_num": "tokens_seen",
                   'epochs': 'code', 'Seed': 'seed'})
    df["scaled_set"] = np.nan
    df["epochs"] = 1
    df["data"] = df.apply(
        lambda x: f"{x['data']}-{x['code']}%-code-{100 - x['code']}%", axis=1)
    df["model_name"] = df.apply(
        lambda x: f"GPT2-{to_str(x['num_params'])}-{x['data']}-{to_str(x['epochs'])}-{to_str(x['seed'])}",
        axis=1)
    assert len(df["model_name"].unique()) == len(
        set((x for x in df["model_args"]))) * 2
    df["original_paper"] = "datablations"
    df["model_type"] = "GPT2"
    df["domain"] = "LM"
    df["arch"] = "dec"
    df["checkpoint"] = np.nan

    df["gen_loss"] = df[df[
        "seed"] == 1]["loss"]
    df["code_loss"] = df[df[
        "seed"] == 2]["loss"]
    df["loss_cols"] = [("code_loss", "gen_loss")] * len(df)

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)
    return dfs


def get_opt_trajectories():

    def row_to_tokens_seen(row):
        model_size = to_int(row["num_params"])
        if model_size < 1e9:
            batch_size = 5e5
        elif model_size < 6e9:
            batch_size = 1e6
        elif model_size < 10e9:
            batch_size = 2e6
        elif model_size < 60e9:
            batch_size = 4e6
        else:
            batch_size = 2e6

        return row["steps"] * batch_size

    df = pd.read_csv("aggregated_eval/opt_trajectories.csv")
    df = df.rename(columns={"model_size": "num_params", "ppl": "loss"})
    df["tokens_seen"] = df.apply(row_to_tokens_seen, axis=1)
    df["num_params"] = df["num_params"].apply(to_int)
    df["model_type"] = df["model_name"]
    df["model_args"] = {}
    df["data"] = "OPT(ROBERTA_PILE_REDDIT)"
    df["tokens_per_epoch"] = 180e9
    df["epochs"] = df["tokens_seen"] / 180e9
    df["model_name"] = df.apply(lambda x: f"OPT-{to_str(x['num_params'])}",
                                axis=1)
    df["original_paper"] = "training_trajectories"
    df["model_type"] = "OPT"
    df["arch"] = "dec"
    df["checkpoint"] = "https://github.com/facebookresearch/metaseq/tree/main/projects/OPT"
    df["domain"] = "LM"
    df["loss_cols"] = [("loss",)] * len(df)
    df["scaled_set"] = "OPT"
    df["seed"] = "0"

    return [df]


def get_revisiting():
    dfs = []
    df = pd.read_csv("aggregated_eval/revisiting_lang.csv")
    df = df.rename(
        columns={"model": "model_type", "Seen Examples": "tokens_seen", "Model": "model_type", "Domain": "domain",
                 "Loss": "loss"})

    def to_num(str):
        try:
            num = float(str)
            return num
        except:
            return 0

    df["num_params"] = df["model_type"].apply(to_num)
    df["model_type"] = df["model_type"].apply(
        lambda x: "lambda" if to_num(x) else x)
    df["scaled_set"] = df["model_type"].apply(
        lambda x: "lambda" if to_num(x) else np.nan)
    df["num_params"] = df["num_params"].apply(to_int)
    df["arch"] = df["num_params"].apply(lambda x: "enc-dec" if x else np.nan)
    df["model_args"] = {}
    df["data"] = "LAMDA"
    df["tokens_per_epoch"] = 500e11  # assumed from their graph
    df[
        "flops"] = np.nan  # lamda paper only talks about the largest one, not clear it is worth the effort to deduce https://arxiv.org/pdf/2201.08239.pdf
    df["epochs"] = 1
    df["model_name"] = df.apply(lambda x: f"{x['model_type']}-{to_str(x['num_params'])}",
                                axis=1)
    df["original_paper"] = "Revisiting Neural Scaling Laws in Language and Vision"
    df["checkpoint"] = np.nan
    df["Task"] = df["Task"].apply(lambda x: "loss" if x in [
                                  "val_loss", "log_perplexity"] else x)
    df["loss_cols"] = df["Task"].apply(lambda x: (x,))
    df["seed"] = "0"
    # only one model had downstream
    downstream_indx = df["model_type"] == "262M"
    cur_df = df[~downstream_indx]

    test_df(cur_df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(cur_df)

    # add downstream evals on the 262M model
    tasks = df[downstream_indx]["Task"].unique()
    # no_lm_tasks = [task for task in tasks if "ppl" not in task and "loss" not in task and "lambada" not in task]
    downstream_dfs = [df[df["Task"] == task]
                      for task in df[downstream_indx]["Task"].unique()]
    dfs_changed = [d.rename(columns={'loss': d["Task"].iloc[0]}).drop(
        "Task", axis=1) for d in downstream_dfs]
    df = reduce(lambda left, right: pd.merge(
        left, right, how='outer'), dfs_changed)
    df["domain"] = "LM"
    # df["loss_cols"] = df.apply(lambda row: no_lm_tasks if "5shot" in row["model_name"] else tasks, axis=1)
    df["loss_cols"] = df.apply(lambda row: tasks, axis=1)
    df["checkpoint"] = np.nan
    df["model_type"] = "LAMDA"
    df["seed"] = "0"
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)
    return dfs


def get_pythia():
    dfs = []

    def switch_deduped(data_name):
        if "pile-deduped" in data_name:
            return "pile"
        if "pile" == data_name:
            return "pile-nodedup"
        return data_name

    def tokens_per_pythia_data(row):
        if row.data == "pile":
            return "334B"
        elif row.data == 'pile-deduped':
            return "207B"
        raise ValueError(f"Unexpected data type in pythia's csv:{row.data}")

    df = pd.read_csv("aggregated_eval/pythia.csv", low_memory=False)

    std_cols = [col for col in df.columns if col.endswith("stderr")]
    df = df.drop(columns=std_cols)
    df = df.drop(columns=["no_cache", "bootstrap_iters", "device", "limit", "description_dict",
                          "batch_size"])  # the original batch size is a leftover from the evaluation not training use the manually extracted tokens_per_training_batch

    def name_to_scaled_set(name):
        scaled_set = []
        for part in name.split("-"):
            if to_int(part, graceful=True, verbose=False) is None:
                scaled_set.append(part)
        return "-".join(scaled_set)

    df["seed"] = "0"
    df["scaled_set"] = df["model_name"].apply(name_to_scaled_set)
    df["tokens_per_epoch"] = df.apply(tokens_per_pythia_data, axis=1)
    df["tokens_per_epoch"] = df["tokens_per_epoch"].apply(to_int)
    df["tokens_seen"] = df["tokens_per_training_batch"] * df["steps"]
    df.drop(columns=["tokens_per_training_batch"])
    df["tokens_seen"] = df.apply(lambda x: to_int(
        "341B") if "bloom" in x["model_name"] else x["tokens_seen"], axis=1)
    df["tokens_seen"] = df.apply(lambda x: to_int(
        "300B") if "opt" in x["model_name"] else x["tokens_seen"], axis=1)
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    df["flops"] = None
    df["arch"] = "dec"
    df["loss_cols"] = df["loss_cols"].apply(ast.literal_eval)
    df["loss_cols"] = df.apply(lambda row: tuple(row["loss_cols"]
                                                 if "5shot" not in row["model_name"]
                                                 else tuple(task for task in row["loss_cols"]
                                                            if
                                                            "ppl" not in task and "loss" not in task and "lambada" not in task)),
                               axis=1)
    df["original_paper"] = "pythia"
    df["domain"] = "LM"
    df["num_params"] = df["num_params"].apply(to_int)
    df["data"] = df["data"].apply(switch_deduped)
    df = df.sort_values(by=["model_name", "steps"], ignore_index=True)
    df = df[df["tokens_seen"] != 0]
    df["seed"] = "0"

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)
    return dfs


def get_llm360():
    dfs = []
    df = pd.read_csv("aggregated_eval/llm360Mamba3B.csv")
    df["checkpoint"] = np.nan
    df["loss_cols"] = [["loss"]] * len(df)  # let later processing choose
    df["original_paper"] = "llm360-private"
    df["domain"] = "LM"
    df["num_params"] = to_int("3B")
    df["scaled_set"] = "llm360Mamba"

    df["seed"] = "0"
    df = df.rename(columns={"step": "tokens_seen"})
    df["tokens_seen"] *= to_int("500B") / 176703
    df = df[df["tokens_seen"] != 0]
    df["model_name"] = "LLM360Mamba3B"
    df["arch"] = "ssm"
    df["flops"] = None
    df["epochs"] = 1
    df["data"] = "LLM360/AmberDatasets"
    df["model_type"] = "LLM360Mamba"
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("aggregated_eval/llm360Mamba3B.csv")
    df["checkpoint"] = np.nan
    df["loss_cols"] = [["loss"]] * len(df)  # let later processing choose
    df["original_paper"] = "llm360-private"
    df["domain"] = "LM"
    df["num_params"] = to_int("3B")
    df["scaled_set"] = "llm360Mamba"

    df["seed"] = "0"
    df = df.rename(columns={"step": "tokens_seen"})
    df["tokens_seen"] *= to_int("500B") / 176703
    df = df[df["tokens_seen"] != 0]
    df["model_name"] = "LLM360Mamba3B"
    df["arch"] = "ssm"
    df["flops"] = None
    df["epochs"] = 1
    df["data"] = "LLM360/AmberDatasets"
    df["model_type"] = "LLM360Mamba"
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("aggregated_eval/k2.csv", index_col="index")
    min_max_cols = [col for col in df.columns if col.endswith(
        "_MIN") or col.endswith("_MAX")]
    spike_cols = [col for col in df.columns if
                  "spike" in col]
    ministage_cols = [col for col in df.columns if "minis" in col]
    # TODO add ministage_cols if adding the data from the last part of training
    df = df.drop(columns=min_max_cols+spike_cols+ministage_cols)
    df["tokens_per_epoch"] = to_int("1.4T")
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    loss_cols = []
    # chose to use downstream
    loss_cols += [x for x in df.columns if "eval" in x]
    # chose to use validation loss
    loss_cols += [x for x in df.columns if "loss" in x]
    df["loss_cols"] = [loss_cols] * len(df)  # let later processing choose
    df["original_paper"] = "LM360:arxiv-2312.06550"
    df["domain"] = "LM"
    df["num_params"] = df["num_params"].apply(to_int)
    df["scaled_set"] = np.nan
    df["seed"] = "0"
    df = df[df["tokens_seen"] != 0]
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("aggregated_eval/Amber.csv", index_col="index")
    min_max_cols = [col for col in df.columns if col.endswith(
        "_MIN") or col.endswith("_MAX")]
    df = df.drop(columns=min_max_cols)
    df["tokens_per_epoch"] = to_int("1.25T")
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    loss_cols = []
    # chose to use downstream
    loss_cols += [x for x in df.columns if "stream" in x]
    # chose to use validation loss
    loss_cols += [x for x in df.columns if "plex" in x]
    df["loss_cols"] = [loss_cols] * len(df)  # let later processing choose
    df["original_paper"] = "LM360:arxiv-2312.06550"
    df["domain"] = "LM"
    df["num_params"] = df["num_params"].apply(to_int)
    df["scaled_set"] = np.nan
    df["seed"] = "0"
    df = df[df["tokens_seen"] != 0]
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    return dfs


def get_extracted_gpt3():

    df = pd.read_csv("raw_data/gpt3/extracted.csv")
    meta_cols = ["model_name", "num_params", "num_layers",
                 "width", "num_heads", "head_dim", "batch_size", "lr"]
    metadata = pd.read_excel(
        "raw_data/gpt3/gpt3_metadata.xlsx", header=None, names=meta_cols)
    model_names = metadata["model_name"].unique()
    df["model_name"] = df["model"].apply(
        lambda x: [name for name in model_names if x.lower() in name.lower()][0])
    df = pd.merge(df, metadata, on="model_name")
    df["checkpoint"] = np.nan
    df["loss_cols"] = [["val_loss"]] * len(df)  # let later processing choose
    df["original_paper"] = "gpt3 - arxiv.org-abs-2005.14165"
    df["domain"] = "LM"
    df["scaled_set"] = "gpt3"
    df["seed"] = "0"
    df = df.rename(columns={"step": "tokens_seen"})
    df["tokens_seen"] *= to_int("1b")
    df = df[df["tokens_seen"] != 0]
    df["arch"] = "dec"
    df["flops"] = None
    df["epochs"] = 1
    df["data"] = "gpt3"
    df["model_type"] = "gpt3"
    df["num_params"] = df["num_params"].apply(to_int)
    return [df]


def get_moduleformer():
    dfs = []
    df = pd.read_csv("raw_data/IBM_MoE.csv")
    df = df.dropna(axis=0)
    df = df.rename(columns={"Model": "model_name",
                   "Effective Params": "num_params"})
    df = df.drop(columns=["Tokenizer"])

    df["num_params"] = df.apply(
        lambda row: row["num_params"] if row["num_params"] else row["Params"], axis=1)
    df["model_type"] = "ModuleFormer"
    df["arch"] = "moe"
    df = df[~df["tokens_seen"].isin(["Pile", "IBM Pile"])]
    tokens_df = df
    tokens_df["tokens_seen"] = tokens_df["tokens_seen"].apply(to_int)
    tokens_per_epoch_per_model = tokens_df.groupby(["model_name"]).max()[
        "tokens_seen"].to_dict()
    df["tokens_per_epoch"] = df["model_name"].apply(
        lambda x: tokens_per_epoch_per_model[x])
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    loss_columns = [col for col in df.columns if
                    " " in col and "Params" not in col and "wikitext" not in col and "ppl" not in col]  # chose downstream
    loss_columns += ['wikitext ppl', 'lambada_openai ppl']  # add perplexity
    # use validation loss
    loss_columns += [x for x in df.columns if "plex" in x]
    df["loss_cols"] = df.apply(
        lambda row: [x for x in loss_columns if row[x]], axis=1)
    # df["loss_cols"] = df["loss_cols"].apply([x for x in df.columns if "plex" in x]) # chose to use validation loss

    df["original_paper"] = "ModuleFormer"
    df["domain"] = "LM"
    df["checkpoint"] = np.nan
    df["num_params"] = df["num_params"].apply(to_int)
    df["flops"] = df["tokens_seen"] * 6 * df["num_params"].apply(
        to_int)  # note that this is the effective number of tokens not the overall ones (so efficient)
    df["scaled_set"] = np.nan
    df["seed"] = "0"

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)
    return dfs


def get_olmo():
    df = pd.read_csv("aggregated_eval/olmo.csv")

    # ignores downstream
    loss_columns = [col for col in df.columns if "CrossEntropyLoss" in col]
    df["loss_cols"] = df.apply(
        lambda row: [col for col in loss_columns if pd.notna(row[col])], axis=1)
    df["data"] = "dolma"
    df["original_paper"] = "olmo"
    df["tokens_seen"] = df["Step"] * 4e6
    df["model_type"] = "olmo"
    df["scaled_set"] = df.apply(
        lambda row: np.nan if row["model_name"] == "OLMo-7B-Twin-2T" else "olmo", axis=1)
    df["epochs"] = 1
    df["arch"] = "dec"
    df["num_params"] = df["num_params"].apply(to_int)
    df["seed"] = "0"

    df["flops"] = None
    return [df]


def get_mad():

    skipped = 0
    models_seen = 0
    numpy_dict = np.load(
        "raw_data/loss_size_flops_llama_sh.npy", allow_pickle=True)
    rows = []
    for model, mod_dict in numpy_dict.tolist().items():
        for flops, isoflop_dict in mod_dict.items():
            for model_data in zip(*isoflop_dict.values()):
                metadata = {key.lower(): val for key, val in zip(
                    isoflop_dict.keys(), model_data)}
                losses = metadata.pop("loss")
                models_seen += 1
                skip = False
                for loss, cur_flops in zip(losses, np.linspace(0, float(flops), len(losses) + 1)[1:]):
                    row = metadata.copy()
                    row["loss"] = loss
                    row["scaled_set"] = model
                    row["flops"] = cur_flops
                    row["max_flops"] = flops
                    if row["size"] is None:
                        skip = True
                        continue
                    rows.append(row)
                skipped += skip
    df = pd.DataFrame.from_records(rows)
    df = df.rename(columns={"size": "num_params"})
    df['original_paper'] = "MAD arxiv.org/2403.17844"
    df['tokens_seen'] = df.apply(
        lambda row: row["flops"] / 6 / to_int(row["num_params"]), axis=1)
    df['checkpoint'] = np.nan
    df['epochs'] = 1
    df["seed"] = "0"
    df["arch"] = "dec"
    df["model_name"] = df.apply(
        lambda row: f"{row['scaled_set']}_{to_str(row['num_params'])}_{to_str(row['max_flops'])}", axis=1)
    df["model_type"] = df.apply(lambda row: row['scaled_set'].lower(), axis=1)
    df["data"] = "pile"
    df["loss_cols"] = [["loss"]] * len(df)
    return [df]


def get_red_pajama():
    dfs = []
    df = pd.read_csv("raw_data/redPajama/2.4B.csv",
                     names=["tokens_seen", "train_loss"])
    df["tokens_seen"] = df["tokens_seen"].apply(lambda x: int(x * 1e9))
    df["loss_cols"] = [["train_loss"]] * len(df)
    df["data"] = "redPajama"
    df["model_name"] = "redPajama2.4"
    df["num_params"] = to_int("2.4B")
    df["model_type"] = "pythia"
    df['arch'] = "dec"
    df['flops'] = np.nan
    df['checkpoint'] = np.nan
    df['epochs'] = 1
    df['original_paper'] = "blog-redpajama-7b"
    df["scaled_set"] = "redPajama"
    df["seed"] = "0"

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("raw_data/redPajama/6.7B.csv",
                     names=["tokens_seen", "train_loss"])
    df["tokens_seen"] = df["tokens_seen"].apply(lambda x: int(x * 1e9))
    df["loss_cols"] = [["train_loss"]] * len(df)
    df["data"] = "redPajama"
    df["model_name"] = "redPajama6.7"
    df["num_params"] = to_int("6.7B")
    df["scaled_set"] = "redPajama"
    df["model_type"] = "pythia"
    df['arch'] = "dec"
    df["seed"] = "0"
    df['flops'] = np.nan
    available_checkpoints = [240, 280, 400, 440, 500, 600, 700, 720, 920]
    check_strings = [hf_checkpoint("togethercomputer/RedPajama-INCITE-7B-Base", f"{num}b_tokens") for num in
                     available_checkpoints]
    available_checkpoints = [num * 1e9 for num in available_checkpoints]
    checkpoints = []
    current = 0
    for num in df["tokens_seen"]:
        if len(available_checkpoints) > current and available_checkpoints[current] >= num:
            checkpoints.append(check_strings[current])
            current += 1
        else:
            checkpoints.append(np.nan)
    df['checkpoint'] = checkpoints
    df['epochs'] = 1
    df['original_paper'] = "blog-redpajama-7b"
    # test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)
    return dfs


def get_data(save_in=None, force=False) -> pd.DataFrame:
    """
    Returns: a Dataframe with performance per model
    columns:
    model_name: str, a unique identifier of the model (human interpretable)
    """
    if save_in and not force:
        os.makedirs(os.path.dirname(save_in), exist_ok=True)
        if os.path.isfile(save_in):
            return pd.read_csv(save_in)
    dfs = []

    df = get_bimix()
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    new_dfs = get_chinchila()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_overtrain()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_t5()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_datablations()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_opt_trajectories()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_revisiting()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_pythia()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_llm360()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_extracted_gpt3()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_olmo()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_mad()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    new_dfs = get_red_pajama()
    test_dfs(new_dfs,  DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs += new_dfs

    ### combine all ###
    res = pd.concat(dfs)
    res["loss_cols"] = res["loss_cols"].apply(tuple)
    res = res.sort_values(["model_type", "model_name", "tokens_seen"])

    res["flops"] = res.apply(
        lambda row: row["tokens_seen"] * 6 * to_int(row["num_params"]) if pd.isna(row["flops"]) else
        row["flops"], axis=1)
    if save_in:
        os.makedirs(os.path.dirname(save_in), exist_ok=True)
        res.to_csv(save_in, index=False)
    return res


if __name__ == '__main__':
    data = get_data(save_in=None)
    print()

# OLMO (https://arxiv.org/pdf/2402.00838.pdf) data from https://wandb.ai/ai2-llm/OLMo-7B/reports/OLMo-7B-Twin-2T--Vmlldzo2NzU0NTIz
# datablations (https://arxiv.org/pdf/2305.16264.pdf) through google drive files now in datablations dir.
# OPT evals found in pythia, loss extracted from training_trajectory_analysis and checkpoints found here (not HF friendly) https://github.com/facebookresearch/metaseq/tree/main/projects/OPT
# OPT loss extracted from training_trajectory_analysis paper https://arxiv.org/pdf/2212.09803.pdf
# revisiting_neural_scaling_laws (lambda and vision) https://github.com/google-research/google-research/tree/master/revisiting_neural_scaling_laws/data paper https://arxiv.org/abs/2209.06640
# LM360 Amber,K2 (the other one had too many phases) https://www.llm360.ai/
# Red Pajamas blog:https://www.together.ai/blog/redpajama-models-v1  model checkpoints here(other sizes elsewhere): https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base
# ModuleFormer MOE (https://arxiv.org/abs/2306.04640) (LLMs -0shot.csv) from https://docs.google.com/spreadsheets/d/1b_Em7HVESSExXCPvssJT7El5zc43KvysDJFJqVNM5jE/edit?usp=sharing
# overtrain: Language models scale reliably with over-training and on downstream tasks https://arxiv.org/abs/2403.08540 https://github.com/mlfoundations/scaling/blob/a003c4913793ac2ae7ef87b28ecb562955d026d5/scaling/constants.py#L139-L146
# bimix: DataMixing arxiv.org/abs/2405.14908
# chinchila (manually extracted) Training Compute-Optimal Large Language Models
# Gzip https://arxiv.org/abs/2405.16684 https://github.com/KhoomeiK/complexity-scaling/tree/main
# TODO
# Some granite training data: https://watsonx-data.cash.sl.cloud9.ibm.com/models/detail/5
# granite logs? https://ibm-research.slack.com/archives/C049F4GK05T/p1702911455907359?thread_ts=1702910609.328829&cid=C049F4GK05T
# ConvNets match https://arxiv.org/pdf/2310.16764.pdf
# ViT scaling laws arxiv.org/abs/2305.13035
# Stella's list might have more metadata? https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0
# hyperparameters for some of the models if useful https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0
# multi-modal scaling laws https://openreview.net/pdf?id=2n7dHVhwJf Scaling Laws for Generative Mixed-Modal Language Models
# (small) Moe scaling law https://arxiv.org/abs/2402.07871
# black mamba (MOW+mamba arch, might need to change current moe to dec moe or give moe a separate col) https://arxiv.org/abs/2402.01771
# extract more from chinchila
