import os
import json
import pandas as pd
import re

from util.naming import to_int

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
                    if "1.3b" in model_name.lower():
                        checkpoint = None
                    elif "350m" in model_name.lower():
                        checkpoint = None
                    else:
                        checkpoint = f"EleutherAI/{model_name.split('_')[0]}"

                else:
                    checkpoint = None

                loss_columns = list(res.keys())
                assert to_int(model_size), f"{model_size}, is not a number"
                assert normalize_pythia_model_name(model_type, model_size, data, num_shots) == model_name
                assert normalize_pythia_model_type(model_type) == model_type

                rows.append(
                    [model_name, steps, model_size, model_type, test_subtype, test_type, checkpoint, loss_columns,
                     data])

                config_cols |= config.keys()
                res_cols |= res.keys()
                configs.append(config)
                results.append(res)

    for row, config, res in zip(rows, configs, results):
        row += [config.get(config_col) for config_col in config_cols]
        row += [res.get(res_col) for res_col in res_cols]
    df = pd.DataFrame(data=rows,
                      columns=["model_name", "steps", "num_params", "model_type", "test_subtype", "test_type",
                               "checkpoint", "loss_cols", "data"] +
                              list(config_cols) + list(res_cols))
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "pythia.csv"), index=False)


if __name__ == '__main__':
    aggregate_pythia("pythiarch/evals", save_dir="aggregated_eval")
