import ast

import numpy as np
import pandas as pd
from functools import reduce

from pandas.core.dtypes.common import is_numeric_dtype

from util.naming import to_int, to_str

BASIC_DF_COLS = ["model_name", "model_type", "tokens_seen", "flops", "num_params", "data", "checkpoint", "loss_cols",
                 "original_paper"]
DATA_AWARE_DF_COLS = BASIC_DF_COLS + ["epochs"]
ARCH_AWARE_DF_COLS = BASIC_DF_COLS + ["arch"]
ARCHS = ["dec", "enc", "enc-dec", np.nan]


def test_df(df, relevant_cols):
    relevant_cols = set(relevant_cols)
    assert all(col in df.columns for col in
               relevant_cols), f"Missing columns: {[col for col in relevant_cols if col not in df.columns]}"
    loss_cols = set(col for cols in df["loss_cols"] for col in cols)
    missing_cols = [col for col in loss_cols if col not in df.columns]
    assert is_numeric_dtype(df["num_params"])
    assert not df["tokens_seen"].isnull().sum() > 0
    assert not missing_cols, f"Loss column are stated in 'loss_col' but do not exist:{missing_cols}"
    assert not set(df["arch"].unique()) - set(ARCHS), f"unexpexted arch types:{set(df['arch'].unique()) - set(ARCHS)}"


def hf_checkpoint(name, revision):
    return {"pretrained_model_name_or_path": name, "revision": revision}


def get_data() -> pd.DataFrame:
    """

    Returns: a Dataframe with performance per model
    columns:
    model_name: str, a unique identifier of the model (human interpretable)
    """
    dfs = []

    df = pd.read_csv("aggregated_eval/datablations_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    df = df.rename(columns={"token_num": "tokens_seen"})
    df["model_name"] = df.apply(lambda x: f"GPT2-{to_str(x['num_params'])}-{x['data']}-{to_str(x['epochs'])}",
                                axis=1)
    assert len(df["model_name"].unique()) == len(set((x for x in df["model_args"])))
    df["original_paper"] = "datablations"
    df["model_type"] = "GPT2"
    df["domain"] = "LM"
    df["arch"] = "dec"
    df["checkpoint"] = df["checkpoint"]  # note, more checkpoints exist for seeds of the 2B models
    df["loss_cols"] = [("loss",)] * len(df)
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("aggregated_eval/datablations_contour_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    df = df.rename(columns={"token_num": "tokens_seen"})

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
    df["loss_cols"] = [("loss",)] * len(df)

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("aggregated_eval/datablations_code_losses.csv", index_col="index")
    df["num_params"] = df["num_params"].apply(to_int)
    df["tokens_per_epoch"] = df["tokens_per_epoch"].apply(to_int)
    df = df.rename(columns={"token_num": "tokens_seen", 'epochs': 'code'})
    df["epochs"] = 1
    df["data"] = df.apply(lambda x: f"{x['data']}-{x['code']}%-code-{100 - x['code']}%", axis=1)
    df["model_name"] = df.apply(lambda x: f"GPT2-{to_str(x['num_params'])}-{x['data']}-{to_str(x['epochs'])}",
                                axis=1)
    assert len(df["model_name"].unique()) == len(set((x for x in df["model_args"])))
    df["original_paper"] = "datablations"
    df["model_type"] = "GPT2"
    df["domain"] = "LM"
    df["arch"] = "dec"
    df["checkpoint"] = np.nan
    df["loss_cols"] = [("loss",)] * len(df)

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

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

    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

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
    df["model_type"] = df["model_type"].apply(lambda x: "lambda" if to_num(x) else x)
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
    df["Task"] = df["Task"].apply(lambda x: "loss" if x in ["val_loss", "log_perplexity"] else x)
    df["loss_cols"] = df["Task"].apply(lambda x: (x,))
    downstream_indx = df["model_type"] == "262M"  # only one model had downstream
    cur_df = df[~downstream_indx]

    test_df(cur_df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(cur_df)

    # add downstream evals on the 262M model
    tasks = df[downstream_indx]["Task"].unique()
    downstream_dfs = [df[df["Task"] == task] for task in df[downstream_indx]["Task"].unique()]
    dfs_changed = [d.rename(columns={'loss': d["Task"].iloc[0]}).drop("Task", axis=1) for d in downstream_dfs]
    df = reduce(lambda left, right: pd.merge(left, right, how='outer'), dfs_changed)
    df["domain"] = "LM"
    df["loss_cols"] = [tasks] * len(df)
    df["checkpoint"] = np.nan
    df["model_type"] = "LAMDA"
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    def tokens_per_pythia_data(row):
        if row.data == "pile":
            return "334B"
        elif row.data == 'pile-deduped':
            return "207B"
        raise ValueError(f"Unexpected data type in pythia's csv:{row.data}")

    df = pd.read_csv("aggregated_eval/pythia.csv")

    std_cols = [col for col in df.columns if col.endswith("stderr")]
    df = df.drop(columns=std_cols)
    df = df.drop(columns=["no_cache", "bootstrap_iters", "device", "limit", "description_dict",
                          "batch_size"])  # the original batch size is a leftover from the evaluation not training use the manually extracted tokens_per_training_batch
    df["tokens_per_epoch"] = df.apply(tokens_per_pythia_data, axis=1)
    df["tokens_per_epoch"] = df["tokens_per_epoch"].apply(to_int)
    df["tokens_seen"] = df["tokens_per_training_batch"] * df["steps"]
    df.drop(columns=["tokens_per_training_batch"])
    df["tokens_seen"] = df.apply(lambda x: to_int("341B") if "bloom" in x["model_name"] else x["tokens_seen"], axis=1)
    df["tokens_seen"] = df.apply(lambda x: to_int("300B") if "opt" in x["model_name"] else x["tokens_seen"], axis=1)
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    df["flops"] = df["tokens_seen"] * 6 * df["num_params"].apply(to_int)
    df["arch"] = "dec"
    df["loss_cols"] = df["loss_cols"].apply(ast.literal_eval)
    df["original_paper"] = "pythia"
    df["domain"] = "LM"
    df["num_params"] = df["num_params"].apply(to_int)
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    res = pd.concat(dfs)
    res["loss_cols"] = res["loss_cols"].apply(tuple)

    df = pd.read_csv("aggregated_eval/Amber.csv", index_col="index")
    min_max_cols = [col for col in df.columns if col.endswith("_MIN") or col.endswith("_MAX")]
    df = df.drop(columns=min_max_cols)
    df["tokens_per_epoch"] = to_int("1.25T")
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    df["loss_cols"] = [[x for x in df.columns if "stream" in x]] * len(df)  # chose to use downstream
    # df["loss_cols"] = df["loss_cols"].apply([x for x in df.columns if "plex" in x]) # chose to use validation loss
    df["original_paper"] = "arxiv.org/abs/2312.06550"
    df["domain"] = "LM"
    df["num_params"] = df["num_params"].apply(to_int)
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    res = pd.concat(dfs)
    res["loss_cols"] = res["loss_cols"].apply(tuple)

    df = pd.read_csv("raw_data/IBM_MoE.csv")
    df = df.dropna(axis=0)
    df = df.rename(columns={"Model": "model_name", "Effective Params": "num_params"})
    df = df.drop(columns=["Tokenizer"])

    df["num_params"] = df.apply(lambda row: row["num_params"] if row["num_params"] else row["Params"], axis=1)
    df["model_type"] = "ModuleFormer"
    df["arch"] = "dec"
    df = df[~df["tokens_seen"].isin(["Pile", "IBM Pile"])]
    tokens_df = df
    tokens_df["tokens_seen"] = tokens_df["tokens_seen"].apply(to_int)
    tokens_per_epoch_per_model = tokens_df.groupby(["model_name"]).max()["tokens_seen"].to_dict()
    df["tokens_per_epoch"] = df["model_name"].apply(lambda x: tokens_per_epoch_per_model[x])
    df["epochs"] = df["tokens_seen"] / df["tokens_per_epoch"]
    loss_columns = [col for col in df.columns if
                    " " in col and "Params" not in col and "wikitext" not in col and "ppl" not in col]  # chose downstream
    # loss_columns = ['wikitext ppl', 'lambada_openai ppl'] # chose perplexity
    df["loss_cols"] = df.apply(lambda row: [x for x in loss_columns if row[x]], axis=1)
    # df["loss_cols"] = df["loss_cols"].apply([x for x in df.columns if "plex" in x]) # chose to use validation loss
    df["original_paper"] = "https://arxiv.org/abs/2306.04640"
    df["domain"] = "LM"
    df["checkpoint"] = np.nan
    df["num_params"] = df["num_params"].apply(to_int)
    df["flops"] = df["tokens_seen"] * 6 * df["num_params"].apply(
        to_int)  # note that this is the effective number of tokens not the overall ones (so efficient)
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("raw_data/redPajama/2.4B.csv", names=["tokens_seen", "train_loss"])
    df["tokens_seen"] = df["tokens_seen"].apply(lambda x: int(x * 1e9))
    df["loss_cols"] = [["train_loss"]] * len(df)
    df["data"] = "redPajama"
    df["model_name"] = "redPajama2.4"
    df["num_params"] = to_int("2.4B")
    df["model_type"] = "pythia"
    df['arch'] = "dec"
    df['flops'] = df["tokens_seen"] * 6 * df["num_params"].apply(to_int).apply(float)
    df['checkpoint'] = np.nan
    df['epochs'] = 1
    df['original_paper'] = "https://www.together.ai/blog/redpajama-7b"
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("raw_data/redPajama/6.7B.csv", names=["tokens_seen", "train_loss"])
    df["tokens_seen"] = df["tokens_seen"].apply(lambda x: int(x * 1e9))
    df["loss_cols"] = [["train_loss"]] * len(df)
    df["data"] = "redPajama"
    df["model_name"] = "redPajama6.7"
    df["num_params"] = to_int("6.7B")
    df["model_type"] = "pythia"
    df['arch'] = "dec"
    df['flops'] = df["tokens_seen"] * 6 * df["num_params"].apply(to_int).apply(float)
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
    df['original_paper'] = "https://www.together.ai/blog/redpajama-7b"
    res = pd.concat(dfs)
    res["loss_cols"] = res["loss_cols"].apply(tuple)
    return res

# datablations (https://arxiv.org/pdf/2305.16264.pdf) through google drive files now in datablations dir.
# OPT evals found in pythia, loss extracted from training_trajectory_analysis and checkpoints found here (not HF friendly) https://github.com/facebookresearch/metaseq/tree/main/projects/OPT
## OPT loss extracted from training_trajectory_analysis paper https://arxiv.org/pdf/2212.09803.pdf
# revisiting_neural_scaling_laws (lambda and vision) https://github.com/google-research/google-research/tree/master/revisiting_neural_scaling_laws/data paper https://arxiv.org/abs/2209.06640
# LM360 Amber (the other one had too many phases) https://www.llm360.ai/
# TODO
# Red Pajamas blog:https://www.together.ai/blog/redpajama-models-v1  model checkpoints here(other sizes elsewhere): https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base
# Some granite training data: https://watsonx-data.cash.sl.cloud9.ibm.com/models/detail/5
# granite logs? https://ibm-research.slack.com/archives/C049F4GK05T/p1702911455907359?thread_ts=1702910609.328829&cid=C049F4GK05T
# Yikang's MOE (LLMs -0shot.csv) from https://docs.google.com/spreadsheets/d/1b_Em7HVESSExXCPvssJT7El5zc43KvysDJFJqVNM5jE/edit?usp=sharing
# ConvNets match https://arxiv.org/pdf/2310.16764.pdf
# ViT scaling laws arxiv.org/abs/2305.13035
# Stella's list might have more? https://docs.google.com/spreadsheets/d/1gc6yse74XCwBx028HV_cvdxwXkmXejVjkO-Mz2uwE0k/edit#gid=0
# hyperparameters for some of the models if useful https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0
