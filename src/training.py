import torch
import torch.nn as nn
import os
import sys
import uuid
import yaml
from munch import Munch
from types import SimpleNamespace

# --- 1. Setup Environment ---
# Ensure Python can find your modules.
if '.' not in sys.path:
    sys.path.append('.')

# Import the necessary functions from your project
from train import train
from wrapper_model import build_model
# All other necessary modules (tasks, samplers, etc.) will be imported by the above.

# --- 2. Define the Configuration as a Python Dictionary ---
# This replaces the need for YAML files and the quinine library.
# It combines the settings from softmax_test.yaml and models/standard.yaml.
config_dict = {
    # "out_dir": "/home/kenzhengjk/182/limits-of-icl/models/nanogpt_softmax_100k/",
    "out_dir": "/home/kenzhengjk/182/limits-of-icl/models/nanogpt_local_100k/",
    "test_run": False,
    "model": {
        "family": "nanogpt",
        # "attention_type": "softmax_causal",
        "attention_type": "local_global",
        "attention_kwargs": {},
        "n_dims": 20,
        "n_positions": 41,
        "n_embd": 256,
        "n_layer": 12,
        "n_head": 8,
    },
    "training": {
        "task": "linear_regression",
        "task_kwargs": {},
        "data": "gaussian",
        "batch_size": 64,
        "learning_rate": 0.0001,
        "train_steps": 100000,
        "save_every_steps": 10000,
        # "train_steps": 5000,
        # "save_every_steps": 1000,
        "curriculum": {
            "dims": { "start": 5, "end": 20, "inc": 1, "interval": 2000 },
            "points": { "start": 11, "end": 41, "inc": 2, "interval": 2000 },
        },
    },
    "wandb": {
        "project": "in-context-training-colab",
        "entity": "your-wandb-entity", # CHANGE THIS if using wandb
        "name": "nanogpt-softmax-test-run-no-quinine",
        "log_every_steps": 10,
    },
}

# --- 3. Prepare for Training ---
# Convert the dictionary to a Munch object to allow dot notation (e.g., args.model.family)
args = SimpleNamespace()
args.out_dir = config_dict["out_dir"]
args.test_run = config_dict["test_run"]
args.wandb = SimpleNamespace(**config_dict["wandb"])
args.training = SimpleNamespace(**config_dict["training"])
args.training.curriculum = SimpleNamespace(**config_dict["training"]["curriculum"])
args.training.curriculum.dims = SimpleNamespace(**config_dict["training"]["curriculum"]["dims"])
args.training.curriculum.points = SimpleNamespace(**config_dict["training"]["curriculum"]["points"])
# The 'model' attribute is itself a namespace object
args.model = SimpleNamespace(**config_dict["model"])

#args.training.keep_every_steps = 1000

print("Checking for and applying default values...")

# Top-level defaults
if not hasattr(args, 'test_run'):
    args.test_run = False

# Model defaults
if not hasattr(args.model, 'attention_type'):
    args.model.attention_type = 'softmax_causal'
if not hasattr(args.model, 'attention_kwargs'):
    args.model.attention_kwargs = {}

# Training defaults
if not hasattr(args.training, 'num_tasks'):
    args.training.num_tasks = None
if not hasattr(args.training, 'num_training_examples'):
    args.training.num_training_examples = None
if not hasattr(args.training, 'batch_size'):
    args.training.batch_size = 64
if not hasattr(args.training, 'learning_rate'):
    args.training.learning_rate = 3e-4
if not hasattr(args.training, 'train_steps'):
    args.training.train_steps = 1000
if not hasattr(args.training, 'save_every_steps'):
    args.training.save_every_steps = 1000
if not hasattr(args.training, 'keep_every_steps'):
    args.training.keep_every_steps = -1
if not hasattr(args.training, 'resume_id'):
    args.training.resume_id = None

# Wandb defaults
if not hasattr(args.wandb, 'project'):
    args.wandb.project = 'in-context-training'
if not hasattr(args.wandb, 'entity'):
    args.wandb.entity = 'in-context'
if not hasattr(args.wandb, 'notes'):
    args.wandb.notes = ''
if not hasattr(args.wandb, 'name'):
    args.wandb.name = None
if not hasattr(args.wandb, 'log_every_steps'):
    args.wandb.log_every_steps = 10

print("Defaults applied successfully.")
print("-" * 40)

# Create output directory
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
print(f"Output directory set to: {os.path.abspath(args.out_dir)}")
print("-" * 40)

# --- 4. Run the Training ---
print("Building model...")
model = build_model(args.model)

# Move model to GPU if available
if torch.cuda.is_available():
    model.cuda()
model.train()

print("\nStarting training...")
train(model, args)

print("\nTraining finished successfully!")
