import os

import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import seaborn as sns


def plot_pred_actual_compare(metadata, preds=None, perfs=None, show=False):
    # compare predictions to actual
    for model_name in metadata["model_name"].unique():
        sub_data = metadata[metadata["model_name"] == model_name]
        if perfs is None:
            if "perf" in sub_data.columns:
                perf = sub_data["perf"]
            else:
                perf = sub_data["loss"]
        else:
            perf = perfs["model_name"]
        if preds is None:
            pred = sub_data["pred"]
        else:
            pred = preds["model_name"]
            raise NotImplementedError

        x = sub_data["tokens_seen"]
        model_name = sub_data['model_name'].iloc[0]
        assert len(sub_data['model_name'].unique()) == 1, f"assumes one model was given in this dataframe, got {len(sub_data['model_name'].unique())}"
        ax = sns.lineplot(x=x, y=perf, label=f"{model_name}")
        color = ax.lines[-1].get_color()
        sns.lineplot(x=x, y=pred, color=color, #label=f"predicted {model_name}",
                     linestyle='dashed')
        in_fit = sub_data[sub_data["in_fit"] == True]
        if in_fit.empty:
            idxmax = 0
        else:
            idxmax = in_fit["tokens_seen"].idxmax()
        sns.scatterplot(x=[x.iloc[idxmax]], y=[pred.iloc[idxmax]], markers="|", color=color, s=100)
        # if show:# TODO delete
        #     plt.show()
    plt.xscale('log')
    plt.xlabel("tokens_seen")
    # plt.legend().remove()
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.7, pos.height])
    ax.legend(loc='center right', bbox_to_anchor=(1.6, 0.5))
    plt.savefig("graphs/compare_predictions.pdf")
    if show:
        plt.show()
    os.makedirs("graphs", exist_ok=True)
    plt.clf()
