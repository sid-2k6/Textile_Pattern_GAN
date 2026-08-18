"""
Publication-quality plotting (font size 20, dpi 300) and sample-grid saving.

All curves read from the epoch history DataFrame. Metrics that don't exist for a
given model (or are entirely NaN) are SKIPPED — never fabricated / zero-filled.
"""
from __future__ import annotations
import os


FONT_SIZE = 20
DPI = 300


def set_pub_style(font_size: int = FONT_SIZE):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "savefig.dpi": DPI,
        "figure.dpi": 120,
    })
    return plt


def _has(df, col):
    import pandas as pd  # noqa
    return col in df.columns and df[col].notna().any()


def line_plot(df, x, ys, title, ylabel, out_path, labels=None):
    """Plot one or more y-series vs x; skip absent/empty columns."""
    plt = set_pub_style()
    ys = [y for y in ys if _has(df, y)]
    if not ys:
        print(f"  [skip] {title}: no data")
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, y in enumerate(ys):
        lab = (labels[i] if labels and i < len(labels) else y)
        ax.plot(df[x], df[y], marker="o", markersize=3, linewidth=2, label=lab)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    if len(ys) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_history(df, out_dir, model_name, has_gp=False):
    """Generate the standard set of training curves into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    made = []
    P = lambda n: os.path.join(out_dir, n)
    made.append(line_plot(df, "epoch", ["generator_loss"],
                          f"{model_name}: Generator Loss vs Epoch", "Generator Loss",
                          P("generator_loss.png")))
    # discriminator vs critic loss (whichever exists)
    made.append(line_plot(df, "epoch", ["discriminator_loss", "critic_loss"],
                          f"{model_name}: Discriminator/Critic Loss vs Epoch",
                          "Loss", P("discriminator_loss.png")))
    made.append(line_plot(df, "epoch", ["real_accuracy"],
                          f"{model_name}: Discriminator Real Accuracy vs Epoch",
                          "Discriminator Real Accuracy", P("disc_real_accuracy.png")))
    made.append(line_plot(df, "epoch", ["fake_accuracy"],
                          f"{model_name}: Discriminator Fake Accuracy vs Epoch",
                          "Discriminator Fake Accuracy", P("disc_fake_accuracy.png")))
    made.append(line_plot(df, "epoch", ["fid"], f"{model_name}: FID vs Epoch",
                          "FID (lower is better)", P("fid.png")))
    made.append(line_plot(df, "epoch", ["kid_mean"], f"{model_name}: KID vs Epoch",
                          "KID mean (lower is better)", P("kid.png")))
    made.append(line_plot(df, "epoch", ["diversity"],
                          f"{model_name}: Diversity vs Epoch",
                          "Feature diversity", P("diversity.png")))
    made.append(line_plot(df, "epoch", ["lr_g", "lr_d"],
                          f"{model_name}: Learning Rate vs Epoch", "Learning Rate",
                          P("learning_rate.png"), labels=["Generator", "Discriminator/Critic"]))
    made.append(line_plot(df, "epoch", ["epoch_time"],
                          f"{model_name}: Epoch Time vs Epoch", "Seconds",
                          P("epoch_time.png")))
    if has_gp:
        made.append(line_plot(df, "epoch", ["gradient_penalty"],
                              f"{model_name}: Gradient Penalty vs Epoch",
                              "Gradient Penalty", P("gradient_penalty.png")))
    return [m for m in made if m]


def save_sample_grid(images, out_path, title="", nrow=None):
    """Save a grid of images. `images` is a float tensor (B,3,H,W) in [-1,1] or [0,1]."""
    import numpy as np
    import torch
    from torchvision.utils import make_grid
    plt = set_pub_style()
    x = images.detach().float().cpu()
    if x.min() < -0.01:
        x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(x.size(0))))
    grid = make_grid(x, nrow=nrow)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    if title:
        plt.title(title)
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
