import pandas as pd
from functools import reduce
from util.naming import to_int, to_str

BASIC_DF_COLS = ["model_name", "model_type", "tokens_seen", "flops", "num_params", "data", "checkpoint", "loss_cols"]
DATA_AWARE_DF_COLS = BASIC_DF_COLS + ["epochs"]
ARCH_AWARE_DF_COLS = BASIC_DF_COLS + ["arch"]
ARCHS = ["dec", "enc", "enc-dec"]


def test_df(df, relevant_cols):
    relevant_cols = set(relevant_cols)
    assert all(col in df.columns for col in
               relevant_cols), f"Missing columns: {[col for col in relevant_cols if col not in df.columns]}"
    loss_cols = set(col for cols in df["loss_cols"] for col in cols)
    missing_cols = [col for col in loss_cols if col not in df.columns]
    assert not missing_cols, f"Loss column are stated in 'loss_col' but do not exist:{missing_cols}"


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
    df["checkpoint"] = None
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
    df["arch"] = df["num_params"].apply(lambda x: "enc-dec" if x else None)
    df["model_args"] = {}
    df["data"] = "LAMDA"
    df["tokens_per_epoch"] = 500e11  # assumed from their graph
    df[
        "flops"] = None  # lamda paper only talks about the largest one, not clear it is worth the effort to deduce https://arxiv.org/pdf/2201.08239.pdf
    df["epochs"] = 1
    df["model_name"] = df.apply(lambda x: f"{x['model_type']}-{to_str(x['num_params'])}",
                                axis=1)
    df["original_paper"] = "Revisiting Neural Scaling Laws in Language and Vision"
    df["checkpoint"] = None
    df["Task"] = df["Task"].apply(lambda x: "loss" if x in ["val_loss", "log_perplexity"] else x)
    df["loss_cols"] = df["Task"].apply(lambda x: (x,))
    downstream_indx = df["model_type"] == "262M"  # only one model had downstream
    cur_df = df[~downstream_indx]

    test_df(cur_df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(cur_df)

    # add downstream evals on the 262M model
    tasks = df[downstream_indx]["Task"].unique()
    dfs = [df[df["Task"] == task] for task in df[downstream_indx]["Task"].unique()]
    dfs_changed = [d.rename(columns={'loss': d["Task"].iloc[0]}).drop("Task", axis=1) for d in dfs]
    df = reduce(lambda left, right: pd.merge(left, right, how='outer'), dfs_changed)
    df["domain"] = "LM"
    df["loss_cols"] = [tasks] * len(df)
    df["checkpoint"] = None
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)

    df = pd.read_csv("aggregated_eval/pythia.csv")
    df = df.rename(columns={"model_size": "num_params", "ppl": "loss"})
    df["arch"] = df.apply(row_to_tokens_seen, axis=1)
    df["data"] = df["num_params"].apply(to_int)
    df["epochs"] = df["model_name"]
    df["flops"] = {}
    df["data"] = "OPT(ROBERTA_PILE_REDDIT)"
    df["loss_cols"] = 180e9
    df["model_name"] = df["tokens_seen"] / 180e9
    df["num_params"] = df.apply(lambda x: f"OPT-{to_str(x['num_params'])}", axis=1)
    df["tokens_seen"] = ""
    df["checkpoint"] =""
    df["domain"] = "LM"
    test_df(df, DATA_AWARE_DF_COLS + ARCH_AWARE_DF_COLS)
    dfs.append(df)
    return pd.concat(dfs)

# datablations (https://arxiv.org/pdf/2305.16264.pdf) through google drive files now in datablations dir.
# OPT evals found in pythia, loss extracted from training_trajectory_analysis and checkpoints found here (not HF friendly) https://github.com/facebookresearch/metaseq/tree/main/projects/OPT
## OPT loss extracted from training_trajectory_analysis paper https://arxiv.org/pdf/2212.09803.pdf
# revisiting_neural_scaling_laws (lambda and vision) https://github.com/google-research/google-research/tree/master/revisiting_neural_scaling_laws/data paper https://arxiv.org/abs/2209.06640
# TODO
# shayne https://arxiv.org/pdf/2305.13169.pdf
# Some granite training data: https://watsonx-data.cash.sl.cloud9.ibm.com/models/detail/5
# Yikang's MOE (LLMs -0shot.csv) from https://docs.google.com/spreadsheets/d/1b_Em7HVESSExXCPvssJT7El5zc43KvysDJFJqVNM5jE/edit?usp=sharing
# ConvNets match https://arxiv.org/pdf/2310.16764.pdf
# ViT scaling laws arxiv.org/abs/2305.13035
