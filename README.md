# ColPret
Efficient Scaling laws and collaborative pretraining.

# data
To just get the data use
```
import get_data from read_data 
all_df = get_data()
```
The columns you may expect in it are DATA_AWARE_DF_COLS and ARCH_AWARE_DF_COLS in read_data.py

# aggregate performance on kinds of losses you care about
```
# choose losses
loss_types = (LossType.PERP, LossType.LOSS)
get_perf_df(all_df, loss_types)
```
# fit
If you want something initial that predicts, you have fit.py
It has `fit_per_model()` that just fits a scaling law for every model on the beginnig of the training.
It also has `data_aware_fit()` which tries to fit a function to all the data.
Note that the current function fit is not a reasonable one, it does not care about the model type (e.g. OPT\GPT) or on the loss (e.g. training loss or an aggregation over some tasks).
