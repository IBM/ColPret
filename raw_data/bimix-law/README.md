# Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining

## Setup

To train models from scratch, follow the instructions in [Getting Started](https://github.com/sangmichaelxie/doremi/blob/main/README.md#getting-started) to install DoReMi and preprocess datasets.


Here, we provide a standalone script to collect conditonal entropy from the processed datasets:

```bash
python collect_ce.py /path/to/a/processed/domain/ 2
```

## Running

Before training, transfer the entire [configs](configs) folder into the cloned DoReMi codebase.

To train a 120M model using data mixture driven by conditional entropy, run the following command within the DoReMi directory:

```bash
bash scripts/runs/run_pile_baseline120M.sh sp_ce
```

To evaluate the trained model on downstream tasks, run the following command within the DoReMi directory:

```bash
bash scripts/runs/run_pile_baseline120M.sh sp_ce eval
```

## Visualization

To visualize scaling behaviors, refer to `scaling_prop.ipynb` and `scaling_steps.ipynb` within the [Pile](Pile) and [SlimPajama](SlimPajama) directories.


## Fitting

To fit the coefficients of the bivariate mixing law, refer to `fit_mixing_law.ipynb` within the [Pile](Pile) and [SlimPajama](SlimPajama) directories.
