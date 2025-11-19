from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names, plot_param_sweep

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "/home/kenzhengjk/182/limits-of-icl/models/"

df = read_run_dir(run_dir)

task = "linear_regression"
#task = "sparse_linear_regression"
#task = "decision_tree"
#task = "relu_2nn_regression"

run_id = "nanogpt_softmax_test"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = True

if recompute_metrics:
    results = get_run_metrics(run_path)  # these are normally precomputed at the end of training

import json
with open("/home/kenzhengjk/182/limits-of-icl/results.json", "w") as f:
    json.dump(results, f, indent=4)

state = torch.load("/home/kenzhengjk/182/limits-of-icl/models/linear_regression/nanogpt_softmax_test/state.pt", map_location=torch.device('cuda'))

# state['model_state_dict'].keys()

with open("/home/kenzhengjk/182/limits-of-icl/results.json", "r") as f:
    results = json.load(f)

figs = plot_param_sweep(results)
for fig in figs:
    plt.show()

