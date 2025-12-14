import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from wrapper_model import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

# Hardcoded mapping from parameter sweep keys to display titles
PARAM_TITLE_MAP = {
    "scaled_query_scale": "Scaling Query Vectors",
    "opposite_quadrants_num_flipped": "Transformation to Opposite Quadrant",
    "random_quadrants_num_constrained": "Transformation to Random Quadrant",
    "orthogonal_train_test_num_orthogonal": "Transformation to Orthogonal Subspace",
    "subspace_dim": "Subspace Projection",
    "skewed_exponent": "Skewed Variance with Exponential Scaling",
    "noisyLR_std": "Noisy Linear Regression",
    "affineLR_std": "Affine Linear Regression",
    "sparseLR_k": "Sparse Linear Regression",
}

# Hardcoded mapping from parameter keys to x-axis label text
PARAM_XLABEL_MAP = {
    "scaled_query_scale": "Scale",
    "opposite_quadrants_num_flipped": "Number of Flipped Dimensions",
    "random_quadrants_num_constrained": "Number of Constrained Points",
    "orthogonal_train_test_num_orthogonal": "Number of Orthogonal Vectors",
    "subspace_dim": "Number of Zeroed Eigenvalues",
    "skewed_exponent": "Eigenvalue Scaling Exponent",
    "noisyLR_std": "Noise Standard Deviation",
    "affineLR_std": "Bias Standard Deviation",
    "sparseLR_k": "Sparsity Factor",
}


def get_display_name(model_name):
    """Extract a clean display name from model identifier, focusing on attention type."""
    # Handle specific run_ids explicitly
    if model_name == "nanogpt_local_100k_5":
        return "Local ($\ell_{window}$=5)"
    elif model_name == "nanogpt_local_100k_8":
        return "Local ($\ell_{window}$=8)"
    elif model_name == "nanogpt_local_100k_15":
        return "Local ($\ell_{window}$=15)"
    elif model_name == "nanogpt_mqa_100k":
        return "MQA"
    elif model_name == "nanogpt_rela_100k":
        return "ReLA"
    elif model_name == "nanogpt_favor_100k":
        return "FAVOR"
    elif model_name == "nanogpt_rebased_100k":
        return "ReBased"
    elif model_name == "nanogpt_softmax_100k":
        return "Softmax"
    elif model_name == "nanogpt_softmax_100k_large":
        return "Softmax (Large)"
    elif model_name == "nanogpt_softmax_100k_small":
        return "Softmax (Small)"
    elif model_name == "nanogpt_softmax_100k_tiny":
        return "Softmax (Tiny)"
    elif model_name == "nanogpt_softmax_test":
        return "Softmax"
    elif model_name == "pretrained":
        return "Pretrained"
    elif "favor" in model_name.lower():
        return "FAVOR"
    elif "rebased" in model_name.lower():
        return "ReBased"
    elif "softmax" in model_name.lower():
        return "Softmax"
    elif "rela" in model_name.lower():
        return "ReLA"
    elif "mqa" in model_name.lower():
        return "MQA"
    elif "local" in model_name.lower():
        return "Local Attention"
    elif "linear" in model_name.lower() and "attention" in model_name.lower():
        return "Linear Attention"
    elif "reformer" in model_name.lower():
        return "Reformer"
    elif "performer" in model_name.lower():
        return "Performer"
    elif "pretrained" in model_name.lower():
        return "Pretrained"
    # Otherwise return the model name as is
    return model_name


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
        metrics = {k: metrics[k] for k in models if k in metrics}

    color = 0
    ax.axhline(trivial, ls="--", color="gray", label="zero estimator")
    for name in sorted(metrics.keys()):  # Sort for consistent ordering
        vs = metrics[name]
        display_name = get_display_name(name)
        ax.plot(vs["mean"], "-", label=display_name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3, color=palette[color % 10])
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    if len(metrics) > 0:
        first_metric = list(metrics.values())[0]
        ax.set_xlim(-1, len(first_metric["mean"]) + 0.1)
    ax.set_ylim(-0.1, 1.25)

    legend = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Attention")
    fig.set_size_inches(6, 4)
    plt.tight_layout()
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
                    
                    # Store mean, std, and bootstrap limits
                    metric_data = {
                        "val": param_val,
                        "mean": results[model_name]["mean"][0],
                        "std": results[model_name]["std"][0],
                        "low": results[model_name]["bootstrap_low"][0],
                        "high": results[model_name]["bootstrap_high"][0]
                    }
                    param_groups[param_name][model_name].append(metric_data)

    # Create a plot for each parameter sweep
    figs = []
    for param_name, model_results in param_groups.items():
        if not any(len(vals) > 1 for vals in model_results.values()):
            continue

        fig, ax = plt.subplots(1, 1)
        color_idx = 0
        for model_name in sorted(model_results.keys()):  # Sort for consistent ordering
            values = model_results[model_name]
            if len(values) < 2:
                continue
            
            values.sort(key=lambda x: x["val"])
            
            param_vals = [v["val"] for v in values]
            mean_errors = [v["mean"] for v in values]
            
            lower_errors = [v["mean"] - v["low"] for v in values]
            upper_errors = [v["high"] - v["mean"] for v in values]
            yerr = [lower_errors, upper_errors]
            
            display_name = get_display_name(model_name)
            
            ax.errorbar(param_vals, mean_errors, yerr=yerr, fmt="o-", 
                       label=display_name, color=palette[color_idx % 10], 
                       lw=2, markersize=6, capsize=5)
            color_idx += 1

        ax.set_xlabel(PARAM_XLABEL_MAP.get(param_name, param_name.replace("_", " ")))
        ax.set_ylabel("Mean Squared Error")
        # Use hardcoded title mapping if available; fallback to formatted key
        display_title = PARAM_TITLE_MAP.get(param_name, f"Effect of {param_name.replace('_', ' ')}")
        ax.set_title(display_title if display_title else f"Effect of {param_name.replace('_', ' ')}")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Attention")
        ax.grid(True, alpha=0.3)
        fig.set_size_inches(7, 4)
        plt.tight_layout()
        figs.append(fig)
    
    return figs


def plot_experiment(all_metrics, experiment_name, models_to_plot=None, trivial=1.0):
    """
    Plot a specific experiment comparing multiple models.
    
    Args:
        all_metrics: Dictionary with structure {experiment_name: {model_name: metrics}}
        experiment_name: Name of the experiment to plot (e.g., "standard", "noisyLR_std=1.0")
        models_to_plot: List of model names to include, or None for all models
        trivial: Baseline value to plot as horizontal line
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    if experiment_name not in all_metrics:
        print(f"Experiment '{experiment_name}' not found in metrics.")
        print(f"Available experiments: {list(all_metrics.keys())}")
        return None, None
    
    experiment_metrics = all_metrics[experiment_name]
    
    if models_to_plot is not None:
        experiment_metrics = {k: v for k, v in experiment_metrics.items() if k in models_to_plot}
    
    fig, ax = basic_plot(experiment_metrics, trivial=trivial)
    ax.set_title(f"Experiment: {experiment_name}")
    
    return fig, ax


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
