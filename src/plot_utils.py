import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from wrapper_model import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
}


def basic_plot(metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def parse_eval_name(eval_name):
    parts = eval_name.rsplit("=", 1)
    if len(parts) == 2:
        param_name = parts[0]
        try:
            param_val = float(parts[1])
            return param_name, param_val
        except ValueError:
            return None, None
    return None, None


def plot_param_sweep(all_metrics, models_to_plot=None):
    # Group experiments by parameter name
    param_groups = {}
    
    # First, find all models present in the metrics
    all_models = set()
    for eval_name, results in all_metrics.items():
        for model_name in results.keys():
            all_models.add(model_name)

    if models_to_plot is None:
        models_to_plot = sorted(list(all_models))

    for eval_name, results in all_metrics.items():
        param_name, param_val = parse_eval_name(eval_name)
        if param_name:
            if param_name not in param_groups:
                param_groups[param_name] = {}
            for model_name in models_to_plot:
                if model_name in results:
                    if model_name not in param_groups[param_name]:
                        param_groups[param_name][model_name] = []
                    param_groups[param_name][model_name].append((param_val, results[model_name]["mean"][0]))

    # Create a plot for each parameter sweep
    figs = []
    for param_name, model_results in param_groups.items():
        if not any(len(vals) > 1 for vals in model_results.values()):
            continue

        fig, ax = plt.subplots(1, 1)
        color_idx = 0
        for model_name, values in model_results.items():
            if len(values) < 2:
                continue
            
            values.sort()
            param_vals = [v[0] for v in values]
            mean_errors = [v[1] for v in values]

            ax.plot(param_vals, mean_errors, "o-", label=model_name, color=palette[color_idx % 10], lw=2)
            color_idx += 1

        ax.set_xlabel(param_name.replace("_", " "))
        ax.set_ylabel("squared error")
        ax.set_title(f"Effect of {param_name.replace('_', ' ')}")
        ax.legend()
        fig.set_size_inches(5, 4)
        figs.append(fig)
    
    return figs


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics
