import os
from random import randint
from tqdm import tqdm
import torch
import csv

from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    # --- CSV Logging Setup ---
    log_file_path = os.path.join(args.out_dir, "training_log.csv")
    log_interval = 500  # Log every 500 iterations
    
    # Check if the log file needs a header
    write_header = not os.path.exists(log_file_path)

    # Open the file in append mode. It will be created if it doesn't exist.
    log_file = open(log_file_path, 'a', newline='')
    csv_writer = csv.writer(log_file)

    if write_header:
        csv_writer.writerow(["step", "loss", "n_points", "n_dims"])
    # --- End CSV Logging Setup ---

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        curriculum.update()

        # --- CSV Logging ---
        if i % log_interval == 0:
            csv_writer.writerow([i, loss, curriculum.n_points, curriculum.n_dims_truncated])
            log_file.flush() # Ensure data is written to disk
        # --- End CSV Logging ---

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            #print("Got here")
            #print(f"trying to save to {state_path}")
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))
