import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# --- Settings for publication-quality plots ---
mpl.rcParams.update({
    "font.size": 12,
    "figure.figsize": (6, 4),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Load CSV --- 
BASE_DIR = "/home/kenzhengjk/182/limits-of-icl/models/nanogpt_softmax_100k"
CSV_NAME = "training_log.csv"
ATTENTION_TYPE = "softmax"

df = pd.read_csv(os.path.join(BASE_DIR, CSV_NAME))

# --- Create figure ---
fig, ax = plt.subplots()

ax.plot(df["step"], df["loss"], linewidth=1.5, color="black")
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title(f"{ATTENTION_TYPE[0].upper() + ATTENTION_TYPE[1:]} Attention Loss Curve")

# --- Curriculum visualization ---
dims_int = 2000

curriculum_steps = list(range(0, max(df["step"]) + dims_int, dims_int))

for s in curriculum_steps:
    ax.axvline(s, color="gray", linestyle="--", linewidth=0.4)

# Legend entry for curriculum lines
ax.plot([], [], '--', color="gray", linewidth=0.8, label="Curriculum Step Boundary")
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, f"{ATTENTION_TYPE}_loss.jpg"), bbox_inches="tight")
