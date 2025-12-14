import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import wrapper_model as models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    print(config_path)
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        if torch.cuda.is_available():
            state = torch.load(state_path)
        else:
            state = torch.load(state_path, map_location=torch.device('cpu'))
        print(state["model_state_dict"].keys())
        print(os.path.join(run_path, "state.pt"))
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state)

    return model, conf


# Functions for evaluation


def eval_batch(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    context_length = xs.shape[1] - 1
    query_index = context_length 
    
    xs_context = xs[:, :context_length, :]
    
    if xs_p is None:
        xs_query = xs[:, query_index:query_index+1, :]
    else:
        xs_query = xs_p[:, query_index:query_index+1, :]

    xs_comb = torch.cat((xs_context, xs_query), dim=1)
    
    ys = task.evaluate(xs_comb)
    
    pred = model(xs_comb.to(device), ys.to(device), inds=[query_index]).detach()
    
    metrics = task.get_metric()(pred.cpu(), ys)[:, query_index]

    return metrics.unsqueeze(1)


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None

### QUERY LEVEL SHIFTS

def gen_scaled_query(data_sampler, n_points, b_size, scale=2.0):
    xs = data_sampler.sample_xs(n_points, b_size)

    xs_train_pre = xs
    xs_test_post = xs * scale

    return xs_train_pre, xs_test_post


def gen_opposite_quadrants(data_sampler, n_points, b_size, num_flipped_dims=0):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dims = xs.shape[2]
    pattern = torch.randn([b_size, 1, n_dims]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    for i in range(b_size):
        flipped_dims = torch.randperm(n_dims)[:num_flipped_dims] # TODO: make sure this works
        xs_test_post[i, :, flipped_dims] *= -1

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size, num_constrained_points=0):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dims = xs.shape[2]
    pattern = torch.randn([b_size, 1, n_dims]).sign()

    xs_train_pre = xs.clone()
    
    if num_constrained_points > 0:
        for i in range(b_size):
            constrained_point_indices = torch.randperm(n_points)[:num_constrained_points]
            
            xs_train_pre[i, constrained_point_indices, :] = xs[i, constrained_point_indices, :].abs() * pattern[i, :, :]

    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size, num_orthogonal_vectors=0):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    
    if num_orthogonal_vectors == 0:
        return xs_train_pre, xs_test_post
    
    num_orthogonal_vectors = min(num_orthogonal_vectors, n_points)
    
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        
        random_indices = torch.randperm(i)[:num_orthogonal_vectors]
        xs_train_pre_i = xs[:, random_indices, :]
        
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size): # TODO: This may not be a good shift
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


### PROMPT LEVEL SHIFTS

def gen_subspace(data_sampler, n_points, b_size, num_dim=1):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dims = xs.shape[2]
    k = max(1, int(n_dims - num_dim))
    eigenvals = torch.zeros(n_dims)
    eigenvals[:k] = 1
    scale = sample_transformation(eigenvals, normalize=True)
    xs_train_pre = xs @ scale
    return xs_train_pre, None


def gen_skewed(data_sampler, n_points, b_size, exponent=1.0):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dims = xs.shape[2]
    idx = torch.arange(n_dims, dtype=xs.dtype, device=xs.device) + 1.0
    eigenvals = 1.0 / (idx ** float(exponent))
    scale = sample_transformation(eigenvals, normalize=True)
    xs_train_pre = xs @ scale
    return xs_train_pre, None

# TODO: get rid of this one
# def gen_scale_x(data_sampler, n_points, b_size, scale=1.0):
#     xs = data_sampler.sample_xs(n_points, b_size)
#     n_dims = xs.shape[2]
#     eigenvals = scale * torch.ones(n_dims, dtype=xs.dtype, device=xs.device)
#     t = sample_transformation(eigenvals, normalize=True)
#     xs_train_pre = xs @ t
#     return xs_train_pre, None









def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    prompting_strategy_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )
    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size, **prompting_strategy_kwargs)
        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    # Query-level shifts
    for scale_val in [0.1 * i for i in range(1, 51, 2)]:
        evaluation_kwargs[f"scaled_query_scale={scale_val}"] = {
            "prompting_strategy": "scaled_query",
            "prompting_strategy_kwargs": {"scale": scale_val},
        }

    for num_flipped in [i for i in range(1, n_dims, 1)]:
        evaluation_kwargs[f"opposite_quadrants_num_flipped={num_flipped}"] = {
            "prompting_strategy": "opposite_quadrants",
            "prompting_strategy_kwargs": {"num_flipped_dims": num_flipped},
        }
    
    for num_constrained in [i for i in range(1, n_points, 2)]:
        evaluation_kwargs[f"random_quadrants_num_constrained={num_constrained}"] = {
            "prompting_strategy": "random_quadrants",
            "prompting_strategy_kwargs": {"num_constrained_points": num_constrained},
        }

    for num_orthogonal in [i for i in range(1, min(n_points, n_dims), 1)]:
        evaluation_kwargs[f"orthogonal_train_test_num_orthogonal={num_orthogonal}"] = {
            "prompting_strategy": "orthogonal_train_test",
            "prompting_strategy_kwargs": {"num_orthogonal_vectors": num_orthogonal},
        }

    evaluation_kwargs["overlapping_train_test"] = {
        "prompting_strategy": "overlapping_train_test",
    }

    # Prompt-level shifts
    for num_dim in [i for i in range(0, n_dims, 1)]:
        evaluation_kwargs[f"subspace_dim={num_dim}"] = {
            "prompting_strategy": "subspace",
            "prompting_strategy_kwargs": {"num_dim": num_dim},
        }
    
    for exp_val in [0.1 * i for i in range(1, 30, 2)]:
        evaluation_kwargs[f"skewed_exponent={exp_val}"] = {
            "prompting_strategy": "skewed",
            "prompting_strategy_kwargs": {"exponent": exp_val},
        }

    # TODO: get rid of this one
    # for scale_val in [0.5, 2.0, 5.0]:
    #     evaluation_kwargs[f"scale_x_scale={scale_val}"] = {
    #         "prompting_strategy": "scale_x",
    #         "prompting_strategy_kwargs": {"scale": scale_val},
    #     }
    
    # Task-level shifts
    for noise_val in [i * 0.1 for i in range(1, 50, 2)]:
        evaluation_kwargs[f"noisyLR_std={noise_val}"] = {
            "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": noise_val},
            "task_name": "noisy_linear_regression",
        }

    for bias_val in [i * 0.1 for i in range(1, 50, 2)]:
        evaluation_kwargs[f"affineLR_std={bias_val}"] = {
            "task_sampler_kwargs": {"bias_std": bias_val},
            "task_name": "affine_regression",
        }

    for sparsity_val in [i for i in range(1, n_dims, 1)]:
        evaluation_kwargs[f"sparseLR_k={sparsity_val}"] = {
            "task_sampler_kwargs": {"sparsity": sparsity_val},
            "task_name": "sparse_linear_regression",
        }

    evaluation_kwargs["uniform_w_dist"] = {
        "task_name": "uniform_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    # try:
    #     with open(save_path) as fp:
    #         all_metrics = json.load(fp)
    # except Exception:
    all_metrics = {}

    i = 0
    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        # 2-14 is query level
        # 15-23 is prompt level
        # 24-33 is task level
        i += 1
        # if i <= 23:
        #     continue
        # if i >= 24:
        #     continue

        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            # if model.name in metrics and not recompute:
            #     continue
            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        if torch.cuda.is_available():
            model = model.cuda().eval()
        else:
            model = model.eval()
        all_models = [model]
        # if not skip_baselines:
        #     all_models += models.get_relevant_baselines(conf.training.task)
    evaluation_kwargs = build_evals(conf)
    print(evaluation_kwargs)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    print(all_metrics)
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        # Skip non-directory files (like .DS_Store on macOS)
        if not os.path.isdir(task_dir):
            continue
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            # Skip non-directory files
            if not os.path.isdir(run_path):
                continue
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
