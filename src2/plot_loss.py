import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def plot_full_loss_curve(BASE_DIR, ATTENTION_TYPE, CSV_NAME='training_log.csv', window_size=15, dims_int=2000):
    """
    Plots a loss curve from the training log and adds curriculum visualization.
    
    Args:
        BASE_DIR (str): Path to the directory containing the CSV.
        CSV_NAME (str): Name of the CSV file containing the training log.
        ATTENTION_TYPE (str): Type of attention ('local-global', etc.) to be used in the title.
        window_size (int): Window size for subsetting the data (default is 15).
        dims_int (int): Step size for curriculum visualization (default is 2000).
    """
    
    # --- Settings for publication-quality plots ---
    mpl.rcParams.update({
        "font.size": 12,
        "figure.figsize": (6, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # --- Load CSV ---
    subsample_step = 1
    df = pd.read_csv(os.path.join(BASE_DIR, CSV_NAME))[::subsample_step]

    # --- Create figure ---
    fig, ax = plt.subplots()

    ax.plot(df["step"], df["loss"], linewidth=1.5, color="black")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{ATTENTION_TYPE[0].upper() + ATTENTION_TYPE[1:]} Attention Loss Curve")

    # --- Curriculum visualization ---
    curriculum_steps = list(range(0, max(df["step"]) + dims_int, dims_int))

    for s in curriculum_steps:
        ax.axvline(s, color="gray", linestyle="--", linewidth=0.4)

    # Legend entry for curriculum lines
    ax.plot([], [], '--', color="gray", linewidth=0.8, label="Curriculum Step Boundary")
    ax.legend(frameon=False)

    # --- Save the plot ---
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, f"{ATTENTION_TYPE}_loss.jpg")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

def main():
    BASE_DIR = "/limits-of-icl/models/nanogpt_softmax_100k"
    ATTENTION_TYPE = "softmax_causal"
    plot_full_loss_curve(BASE_DIR, ATTENTION_TYPE, dims_int=2000)

if __name__ == '__main__':
    main()