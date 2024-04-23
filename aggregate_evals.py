import json
import math
import os
import re
from functools import reduce

import numpy as np
import pandas as pd

from util.naming import to_int
from util.read_data import hf_checkpoint

steps_reg = re.compile(r"[-_/\\]?steps?[_-]?(\d+)")


def parse_winobias(eval_res):
    values_to_extract = ["em", "em_stderr"]
    if "config" in eval_res:
        config = eval_res["config"]
        results_in = "results"
        results_gen = eval_res[results_in]

    else:
        config = {"model_args": eval_res["model_args"]}
        results_in = "table_results"
        results_gen = eval_res[results_in].values()

    res = {}
    stop = False
    for dataset_res in results_gen:
        for extracted_val in values_to_extract:
            if isinstance(dataset_res, str) and dataset_res not in extracted_val:
                stop = True
                break
            key = dataset_res["task_name"] + "-" + dataset_res["prompt_name"].replace(" ", "_") + " " + extracted_val
            res[key] = dataset_res[extracted_val]
        if stop:
            res = {key + " " + subkey: val
                   for key, dataset_res in eval_res["results"].items()
                   for subkey, val in dataset_res.items()}
            break
    steps = config["model_args"]["iteration"]
    return steps, res, config


def normalize_pythia_model_name(model_type, model_size, data, num_shots):
    name = f"{model_type}-{model_size}-{data.replace('pile', '').replace('-', '')}".lower().strip("-_ ")
    if num_shots != 0:
        name += f"-{num_shots}shot"
    if "v0" in model_type:
        # move the v0 to the end of the whole name, this is how it was origivanlly saved by the authors
        name = name.replace("-v0", "") + "-v0"
    return name


def normalize_pythia_model_type(model_type):
    if "21" in model_type or "long" in model_type:
        model_type = "pythia-long-intervention"
    elif "7" in model_type or "inter" in model_type:
        model_type = "pythia-intervention"
    return model_type


def aggregate_pythia(path, save_dir):
    config_cols = set()
    res_cols = set()
    rows = []
    results = []
    configs = []
    for root, dirnames, filenames in os.walk(path):
        if "csv" in root:
            continue

        for filename in filenames:
            if filename.endswith("json"):
                model_name = os.path.splitext(os.path.basename(filename))[0]
                steps_match = re.search(steps_reg, model_name)
                if steps_match:
                    model_name = model_name.replace(steps_match[0], "", 1)
                    steps = int(steps_match[1])
                    steps_start = steps_match.start()
                else:
                    steps = pd.NA
                    steps_start = -1

                dirs = root.split(os.sep)
                test_type = dirs[-1]
                test_subtype = dirs[-2]
                model_type = dirs[-3]
                if test_subtype == "evals":
                    test_subtype = filename[:steps_start]
                    model_type = filename[:filename.index("-")]

                with open(os.path.join(root, filename)) as fl:
                    eval_res = json.load(fl)

                if test_subtype == "winobias":
                    model_size, test_type = test_type, model_type
                    model_type = "pythia"
                    data = "pile-deduped" if "deduped" in eval_res['config']['model_args']['train_data_paths'][
                        0] else "pile"
                    steps, res, config = parse_winobias(eval_res)
                    num_shots = config["num_fewshot"]
                    model_type = normalize_pythia_model_type(model_type)
                    model_name = normalize_pythia_model_name(model_type, model_size, data, num_shots)
                elif model_type == "winobias":
                    model_name = model_name.split("_")[2]
                    if "long" in model_name:
                        model_type = "pythia-long-intervention"
                    elif "inter" in model_name:
                        model_type = "pythia-intervention"
                    else:
                        model_type = "pythia"
                    data = "pile-deduped" if "deduped" in model_name else "pile"
                    model_size, test_subtype, test_type = test_type, model_type, "bias-evals"
                    steps, res, config = parse_winobias(eval_res)
                    if "num_shots" in config:
                        num_shots = config["num_shots"]
                    elif "num_fewshot" in config:
                        num_shots = config["num_fewshot"]
                    else:
                        num_shots = 0
                    model_type = normalize_pythia_model_type(model_type)
                    model_name = normalize_pythia_model_name(model_type, model_size, data, num_shots)
                elif test_type == "bias-evals":
                    data = "pile-deduped" if "deduped" in model_name else "pile"
                    model_type = filename.split("-")[0]
                    if "long" in model_type:
                        model_type = "long-intervention"
                    model_size = filename.split(f"{model_type}-")[1].split("-step")[0].split("-global")[0].replace(
                        "-deduped", "")
                    test_subtype, test_type = "winobias", "bias-evals"
                    steps, res, config = parse_winobias(eval_res)
                    num_shots = config["num_fewshot"]
                    model_type = normalize_pythia_model_type(model_type)
                    model_name = normalize_pythia_model_name(model_type, model_size, data, num_shots)
                elif test_type == "winobias":
                    raise
                else:  # parse main experiments
                    data = "pile-deduped" if "deduped" in model_name else "pile"
                    model_name = model_name.split("_")[0].replace("-global", "")
                    if "tok" in model_name:  # baselines for the interventions, just like checkpoints
                        continue
                    if "bf16" in model_name:  # there is only one:
                        data = "pile"
                        model_type = "pythia-bf16"
                        model_size = "1b"
                        num_shots = 0 if "zero" in root else 5
                        model_name = normalize_pythia_model_name(model_type, model_size, data, num_shots)
                    # original paper did not need to specify it is pythia...
                    if not [model for model in ("pythia", "bloom", "opt") if model in model_name]:
                        model_name = "pythia-" + model_name
                    shortname = model_name.replace("-deduped", "")
                    if "0shot" in model_name:
                        test_subtype = "zero-shot"
                        num_shots = 0
                        shortname = shortname.replace("-0shot", "")
                        # model_name += "-0"
                    elif "5shot" in model_name:
                        test_subtype = "five-shot"
                        num_shots = 5
                        shortname = shortname.replace("-5shot", "")
                        # model_name += "-5shot"
                    else:
                        num_shots = 5 if "five" in root else 0
                    shortname = shortname.replace("-long", "")
                    model_size = shortname.split("-")[-1]
                    model_type = "-".join(shortname.split("-")[:-1])
                    if model_name.endswith("long"):
                        model_name = normalize_pythia_model_name(model_type, model_size, data, num_shots)
                    if "v0" in root:
                        model_type = model_type.replace("pythia", "pythia-v0")
                        model_name = model_name.replace("-0shot", "") + "-v0"
                        assert "v0" in model_type
                    model_type = normalize_pythia_model_type(model_type)

                    config = eval_res["config"]
                    if not [x for x in ["160", "410", "1.4"] if x in root]:  # they have an error
                        assert config["num_fewshot"] == num_shots
                    if "iteration" in config["model_args"]:
                        if pd.notna(steps):
                            assert steps == config["model_args"]["iteration"]
                        else:
                            steps = config["model_args"]["iteration"]
                    res = {key + " " + subkey: val
                           for key, dataset_res in eval_res["results"].items()
                           for subkey, val in dataset_res.items()}
                if "pythia" in model_name:
                    if "1.3b" in model_name.lower():  # older models, with the old name
                        checkpoint = np.nan
                    elif "350m" in model_name.lower():  # older models, with the old name
                        checkpoint = np.nan
                    elif steps in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000] or steps % 1000 == 0:
                        checkpoint = hf_checkpoint(f"EleutherAI/{model_name.split('_')[0]}", steps)
                    else:
                        checkpoint = np.nan

                else:
                    checkpoint = np.nan
                model_size_int = to_int(model_size)
                if "pythia" in model_type:
                    tokens_per_training_batch = 2097152
                elif model_type == "bloom":
                    if model_size_int < 1.5e9:
                        tokens_per_training_batch = 256 * 1024
                    elif model_size_int < 1e10:
                        tokens_per_training_batch = 512 * 2048
                    else:
                        tokens_per_training_batch = 2048 * 2048
                elif model_type == "opt":
                    if model_size_int < 1e9:
                        tokens_per_training_batch = 5e5
                    elif model_size_int < 6e9:
                        tokens_per_training_batch = 1e6
                    elif model_size_int < 10e9:
                        tokens_per_training_batch = 2e6
                    elif model_size_int < 60e9:
                        tokens_per_training_batch = 4e6
                    else:
                        tokens_per_training_batch = 2e6
                else:
                    raise ValueError("unexpected model")
                loss_columns = [x for x in res.keys() if "std" not in x]
                assert to_int(model_size), f"{model_size}, is not a number"
                assert normalize_pythia_model_name(model_type, model_size, data, num_shots) == model_name
                assert normalize_pythia_model_type(model_type) == model_type

                rows.append(
                    [model_name, steps, model_size, model_type, test_subtype, test_type, checkpoint, loss_columns,
                     data, tokens_per_training_batch])

                config_cols |= config.keys()
                res_cols |= res.keys()
                configs.append(config)
                results.append(res)

    for row, config, res in zip(rows, configs, results):
        row += [config.get(config_col) for config_col in config_cols]
        row += [res.get(res_col) for res_col in res_cols]
    df = pd.DataFrame(data=rows,
                      columns=["model_name", "steps", "num_params", "model_type", "test_subtype", "test_type",
                               "checkpoint", "loss_cols", "data", "tokens_per_training_batch"] +
                              list(config_cols) + list(res_cols))
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "pythia.csv"), index=False)


def aggregate_mamballm360(save_dir):
    dfs = []
    df = pd.read_csv("raw_data/lm360/MAMBA3B/wandb_export_2024-04-21T20_29_45.965-04_00_loss_curve.csv")
    cols = [col for col in df.columns if
            "loss" in col and "MIN" not in col and "MAX" not in col]  # note early means a later run, prefer later runs when duplicate losses (means a restart)

    def extract_val(row):
        for col in cols:
            val = row[col]
            if pd.notna(val):
                return val

    df["loss"] = df.apply(extract_val, axis=1)
    df = df.drop(columns=[col for col in df.columns if col not in ["step", "loss"]])
    dfs.append(df)

    df = pd.read_csv("raw_data/lm360/MAMBA3B/wandb_export_2024-04-21T20_29_57.317-04_00_learning_rate.csv")
    cols = [col for col in df.columns if
            "rate" in col and "MIN" not in col and "MAX" not in col]  # note early means a later run, prefer later runs when duplicate  (means a restart)
    df["lr"] = df.apply(extract_val, axis=1)
    df = df.drop(columns=[col for col in df.columns if col not in ["step", "lr"]])
    dfs.append(df)

    df = pd.read_csv("raw_data/lm360/MAMBA3B/wandb_export_2024-04-21T20_30_05.126-04_00_grad_norm.csv")
    cols = [col for col in df.columns if
            "grad" in col and "MIN" not in col and "MAX" not in col]  # note early means a later run, prefer later runs when duplicate  (means a restart)
    df["grad_norm"] = df.apply(extract_val, axis=1)
    df = df.drop(columns=[col for col in df.columns if col not in ["step", "grad_norm"]])
    dfs.append(df)

    import functools as ft
    df = ft.reduce(lambda left, right: pd.merge(left, right, on='step'), dfs)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "llm360Mamba3B.csv"), index=False)


def aggregate_lm360(save_dir):
    # read from HF
    #     import urllib.request
    # for checkpoint in checkpoints:
    #     for filename in filenames:
    #         path = os.path.join(save_dir, "lm360", filename)
    #         urllib.request.urlretrieve(f"https://huggingface.co/LLM360/Amber/raw/{checkpoint}/{filename}",path)
    #         with open(path) as js:
    #             js = json.load(js)
    # checkpoint = int(checkpoint.split("_")[-1])
    # checkpoints = ["ckpt_{i:03d}" for i in range(358)] + ["main"]
    # filenames = ["eval_arc.json", ]
    # BASIC_DF_COLS = ["model_name", "model_type", "tokens_seen", "flops", "num_params", "data", "checkpoint",
    #                  "loss_cols",
    #                  "original_paper"]
    # DATA_AWARE_DF_COLS = BASIC_DF_COLS + ["epochs"]
    # ARCH_AWARE_DF_COLS = BASIC_DF_COLS + ["arch"]
    # ARCHS = ["dec", "enc", "enc-dec"]
    # rows = ["model_name", "steps", "num_params", "model_type", "test_subtype", "test_type",
    #         "checkpoint", "loss_cols", "data", "tokens_per_training_batch"] + list(config_cols) + list(res_cols)
    loss_cols = []
    for root, _, filenames in os.walk(
            os.path.join("raw_data", "raw_data/lm360")):  # downloaded manually from https://wandb.ai/llm360/projects
        dirname = root.split(os.sep)[-1]
        model_name = dirname
        num_params = "6.7B"
        data = f"LLM360/{model_name}Datasets"
        eval_dfs = []
        for filename in filenames:
            if filename.startswith("."):
                continue
            eval_df = pd.read_csv(os.path.join(root, filename))
            if "train - loss" in eval_df.columns:
                # mapping = dict(zip(np.linspace(0, len(eval_df) - 1,len(eval_df) - 1),np.linspace(0,354,2))
                checkpoint_steps = math.floor((len(eval_df) - 1) / 356)
                checkpoints = [x / checkpoint_steps for x in range(len(eval_df))]
                checkpoints[-1] = 356  # becase of the rounding the loss steps do not exactly match
                eval_df["checkpoint"] = checkpoints
                eval_dfs.insert(0, eval_df)
            else:
                eval_df["checkpoint"] = eval_df["Step"]
                eval_df = eval_df.drop(columns=["Step"])
                eval_dfs.append(eval_df)
        if not eval_dfs:
            continue
        model_df = reduce(lambda left, right: pd.merge(left, right, on=["checkpoint"],
                                                       how="outer"), eval_dfs)
        model_df = model_df.reset_index()
        # model_df["steps"] = np.arrange(len(model_df)) * 2240
        model_df["tokens_seen"] = np.arange(len(model_df)) * to_int("1.25T") / (len(model_df) - 1)
        model_df["flops"] = model_df["tokens_seen"] * 6 * to_int(num_params)
        model_df["checkpoint"] = model_df["checkpoint"].apply(
            lambda i: hf_checkpoint(f"LLM360/{model_name}", f"ckpt_{i}") if i == int(
                i) else np.nan)
        model_df["model_type"] = "Lamma"
        model_df["arch"] = "dec"
        model_df["data"] = data
        model_df["model_name"] = model_name
        model_df["num_params"] = num_params
        os.makedirs(save_dir, exist_ok=True)
        model_df.to_csv(os.path.join(save_dir, f"{model_name}.csv"), index=False)


def aggregate_olmo(path, save_dir):
    path = os.path.join("raw_data", path)
    res_df = {}
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith("csv"):
                single_eval_df = pd.read_csv(os.path.join(root, filename))
                drop_columns = [col for col in single_eval_df.columns if col.endswith("MAX") or col.endswith("MIN")]
                single_eval_df = single_eval_df.drop(columns=drop_columns)

                model_name, test_name = single_eval_df.columns[-1].replace("Group:", "").strip().split("- eval/")
                if "run" in model_name:
                    model_name = model_name.split("-run")[0].strip()

                single_eval_df[test_name] = single_eval_df.iloc[:, 1:].sum(axis=1)
                single_eval_df = single_eval_df.drop(columns=single_eval_df.columns[1:-1])

                single_eval_df["model_name"] = model_name

                single_eval_df["num_params"] = model_name.split("B")[0].split("-")[-1] + "B"
                # dfs.append(single_eval_df)
                if model_name not in res_df:
                    res_df[model_name] = single_eval_df
                elif test_name in res_df[model_name].columns:
                    continue  # data downloaded twice by mistake
                else:
                    res_df[model_name] = pd.DataFrame.merge(res_df[model_name], single_eval_df,
                                                            on=["Step", "model_name", "num_params"],
                                                            how="outer")
    res_df = pd.concat(res_df.values())

    checkpoints = {}
    for model in res_df["model_name"].unique():
        from huggingface_hub import HfApi
        api = HfApi()
        refs = api.list_repo_refs("allenai/OLMo-7B")
        checkpoints[model] = {}
        for branch in refs.branches:
            if branch.name == "main":
                continue
            step = int(branch.name.replace("step", "").split("-")[0])
            checkpoints[model][step] = branch.name

    def olmo_checkpoint(row):
        model_checkpoints = checkpoints[row["model_name"]]
        if row["Step"] in model_checkpoints:
            revision = model_checkpoints[row["Step"]]
            return hf_checkpoint(row["model_name"], revision)
        else:
            return np.nan

    res_df["checkpoint"] = res_df.apply(olmo_checkpoint, axis=1)

    os.makedirs(save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(save_dir, f"olmo.csv"), index=False)


def t5_root_to_name(root):
    if "base" in root:
        return "base"
    elif "large" in root:
        return "large"
    elif "xxl" in root:
        return "xxl"
    elif "xl" in root:
        return "xl"


def aggregate_t5_pile(path, save_dir):
    import tensorflow as tf
    from tensorflow.python.summary.summary_iterator import summary_iterator as si
    path = os.path.join("raw_data", path)
    train = {}
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            model_name = t5_root_to_name(root)
            print(f"processing {os.path.join(root, filename)}")
            if "infer" in root:
                with open(os.path.join(root, filename)) as fl:
                    data = fl.readlines()
                for line in data:
                    info = json.loads(line)
                    key = (model_name, info["step"])
                    if key not in train:
                        train[key] = {}
                    if info["perplexity"]:
                        train[key]["val_perplexity"] = info[
                            "perplexity"]  # note that we pick the last when repeated entries exists (due to reinitialized)
            if "train" in root:
                try:
                    sum_it = si(os.path.join(root, filename))
                    for line in sum_it:
                        key = (model_name, line.step)
                        if key not in train:
                            train[key] = {}
                        for subline in line.summary.value:
                            train[key][subline.tag] = tf.make_ndarray(subline.tensor)
                except Exception as e:
                    print(f"Corrupted file {os.path.join(root, filename)}")
    res_df = pd.DataFrame(train).transpose().reset_index(names=["model_name", "steps"])
    os.makedirs(save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(save_dir, f"t5_pile.csv"), index=False)


if __name__ == '__main__':
    save_dir = "aggregated_eval"
    aggregate_t5_pile("t5_pile", save_dir=save_dir)
    aggregate_olmo("OLMO", save_dir=save_dir)
    aggregate_pythia("pythiarch/evals", save_dir=save_dir)
    aggregate_lm360(save_dir=save_dir)
    aggregate_mamballm360(save_dir=save_dir)