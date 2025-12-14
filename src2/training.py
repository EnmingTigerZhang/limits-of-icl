import os
import sys
import torch
import yaml
import argparse
from types import SimpleNamespace

# Ensure local imports work
if '.' not in sys.path:
    sys.path.append('.')

from train_utils import train
from wrapper_model import build_model
from plot_loss import plot_full_loss_curve


def load_config(path: str) -> SimpleNamespace:
    """Load YAML config and convert to nested SimpleNamespace."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        return d

    return to_namespace(cfg)


def namespace_to_dict(x):
    if isinstance(x, dict):
        return {k: namespace_to_dict(v) for k, v in x.items()}
    if isinstance(x, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(x).items()}
    return x


def apply_defaults(args: SimpleNamespace) -> None:
    args.test_run = getattr(args, "test_run", False)

    args.model.attention_type = getattr(args.model, "attention_type", "softmax_causal")

    args.model.attention_kwargs = namespace_to_dict(
        getattr(args.model, "attention_kwargs", {})
    )

    tr = args.training
    tr.task_kwargs = namespace_to_dict(getattr(tr, "task_kwargs", {}))

    tr.num_tasks = getattr(tr, "num_tasks", None)
    tr.num_training_examples = getattr(tr, "num_training_examples", None)
    tr.batch_size = getattr(tr, "batch_size", 64)
    tr.learning_rate = getattr(tr, "learning_rate", 3e-4)
    tr.train_steps = getattr(tr, "train_steps", 1000)
    tr.save_every_steps = getattr(tr, "save_every_steps", 1000)
    tr.keep_every_steps = getattr(tr, "keep_every_steps", -1)
    tr.resume_id = getattr(tr, "resume_id", None)


def main(config_path: str):
    print(f"Loading config from {config_path}")
    args = load_config(config_path)

    print("Applying defaults...")
    apply_defaults(args)
    print("Defaults applied.")
    print("-" * 40)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.out_dir)}")
    print("-" * 40)

    # Build model
    print("Building model...")
    model = build_model(args.model)

    if torch.cuda.is_available():
        model.cuda()
    model.train()

    # Train
    print("Starting training...")
    train(model, args)

    # Plot loss
    plot_full_loss_curve(args.out_dir, args.model.attention_type)

    print("Training finished successfully!")


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Run training with a YAML config")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to YAML configuration file"
        )
        return parser.parse_args()

    args = parse_args()
    main(args.config)