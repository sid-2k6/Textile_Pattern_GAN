#!/usr/bin/env python3
"""
Generate EDA artifacts for the Textile Pattern GAN datasets.

This script runs the *pixel-level* EDA on whatever images are available locally
under --img-root (default: the Gunung Ringgit sample captured from Drive) and
writes the *structural / composition* EDA from the parsed Drive listing.

Outputs (into eda/outputs/):
  - dataset_composition.csv         full class table (counts + provenance)
  - fig_composition.png             bar chart of images per class
  - pixel_stats.csv                 per-image pixel statistics (available imgs)
  - fig_sample_grid.png             grid of sample images
  - fig_dimensions.png              width/height scatter + counts
  - fig_filesize_vs_pixels.png      metadata-bloat evidence
  - fig_brightness_hist.png         brightness distribution
  - fig_rgb_channels.png            per-channel intensity distributions
  - fig_mean_color.png              per-image mean colour swatches
"""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------
# 1. STRUCTURAL / COMPOSITION EDA  (accurate counts from folder-name labels)
# --------------------------------------------------------------------------
# Provenance legend:
#   name   -> exact count embedded in the Drive folder name, e.g. "all_motifs (864)"
#   listed -> count observed in gdown listing (gdown caps at 50 files/folder,
#             so these are lower bounds / estimates)
COMPOSITION = [
    # dataset,               class,                 images, unit,    provenance, include
    ("Batik_Lasem",          "Gunung Ringgit",        864, "motifs", "name",   True),
    ("Batik_Lasem",          "Kricak / Watu Pecah",  1004, "motifs", "name",   True),
    ("Batik_Lasem",          "Latohan",              1373, "motifs", "name",   True),
    ("Batik_Lasem",          "Nyuk Pitu",            1291, "motifs", "name",   True),
    ("Batik_Lasem",          "Seritan",              1328, "motifs", "name",   True),
    ("Batik_Nitik_960",      "Batik Nitik 960",       960, "images", "name",   True),
    ("Batik_Nitik_Sarimbit", "Sarimbit 120",          120, "images", "name",   True),
    ("Original Images",      "All Images (sources)",   40, "images", "listed", True),
    ("NeuralLoom",           "Mekhela chador",         33, "groups", "listed", True),
    ("NeuralLoom",           "Mekhela cropped",       165, "images", "listed", True),  # ~33 groups x 5
    ("NeuralLoom",           "Saree body",             49, "images", "listed", True),
    ("DeepFashion2",         "(excluded)",              0, "images", "excluded", False),
    ("DeepFashion3D",        "(excluded)",              0, "images", "excluded", False),
]
comp = pd.DataFrame(COMPOSITION, columns=[
    "dataset", "class", "images", "unit", "provenance", "included_in_eda"])
comp.to_csv(os.path.join(OUT, "dataset_composition.csv"), index=False)

inc = comp[comp.included_in_eda]
plt.figure(figsize=(10, 6))
colors = {"Batik_Lasem": "#c0392b", "Batik_Nitik_960": "#e67e22",
          "Batik_Nitik_Sarimbit": "#f1c40f", "Original Images": "#7f8c8d",
          "NeuralLoom": "#2980b9"}
bar_colors = [colors.get(d, "#333") for d in inc.dataset]
labels = [f"{d}\n{c}" for d, c in zip(inc.dataset, inc["class"])]
plt.bar(range(len(inc)), inc.images, color=bar_colors)
plt.xticks(range(len(inc)), labels, rotation=45, ha="right", fontsize=8)
plt.ylabel("image / motif count")
plt.title("Dataset composition (Batik + NeuralLoom, DeepFashion excluded)")
for i, v in enumerate(inc.images):
    plt.text(i, v + 10, str(v), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_composition.png"), dpi=130)
plt.close()
print("Composition table + chart written. Total included images (approx):",
      int(inc.images.sum()))

# --------------------------------------------------------------------------
# 2. PIXEL-LEVEL EDA on locally available images
# --------------------------------------------------------------------------
def run_pixel_eda(img_root):
    paths = sorted(glob.glob(os.path.join(img_root, "**", "*.jpg"), recursive=True))
    paths += sorted(glob.glob(os.path.join(img_root, "**", "*.png"), recursive=True))
    if not paths:
        print(f"[pixel EDA] no images found under {img_root}; skipping.")
        return None
    rows = []
    for p in paths:
        try:
            im = Image.open(p)
            im.load()
        except Exception as e:
            rows.append(dict(path=p, corrupt=True))
            continue
        rgb = im.convert("RGB")
        a = np.asarray(rgb).astype(np.float32)
        w, h = im.size
        rel = os.path.relpath(p, img_root)
        subset = rel.split(os.sep)[-2] if os.sep in rel else "root"
        n_meta = len([k for k in im.info.keys()
                      if k in ("exif", "photoshop", "xmp", "icc_profile")])
        rows.append(dict(
            path=rel, subset=subset, corrupt=False,
            width=w, height=h, pixels=w * h,
            filesize_kb=os.path.getsize(p) / 1024.0,
            mean_r=a[..., 0].mean(), mean_g=a[..., 1].mean(), mean_b=a[..., 2].mean(),
            brightness=a.mean(), contrast_std=a.std(),
            metadata_blocks=n_meta,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pixel_stats.csv"), index=False)
    good = df[~df.corrupt].copy()
    print(f"[pixel EDA] analysed {len(good)} images "
          f"({df.corrupt.sum()} corrupt) from {img_root}")

    # ---- sample grid ----
    n = min(36, len(good))
    cols = 6
    rows_g = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 1.4, rows_g * 1.4))
    for i, p in enumerate(good.path.head(n)):
        ax = plt.subplot(rows_g, cols, i + 1)
        ax.imshow(Image.open(os.path.join(img_root, p)).convert("RGB"))
        ax.axis("off")
    plt.suptitle(f"Sample images (n={n})", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_sample_grid.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # ---- dimensions ----
    dim_counts = good.groupby(["width", "height"]).size().reset_index(name="n")
    plt.figure(figsize=(6, 4))
    plt.scatter(good.width, good.height, alpha=0.3, s=40)
    plt.xlabel("width (px)"); plt.ylabel("height (px)")
    plt.title("Image dimensions")
    txt = "\n".join(f"{int(r.width)}x{int(r.height)}: {int(r.n)}"
                    for _, r in dim_counts.iterrows())
    plt.gca().text(0.98, 0.02, txt, transform=plt.gca().transAxes,
                   ha="right", va="bottom", fontsize=8,
                   bbox=dict(boxstyle="round", fc="w", alpha=.7))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_dimensions.png"), dpi=130)
    plt.close()

    # ---- filesize vs pixels (metadata bloat) ----
    plt.figure(figsize=(6, 4))
    plt.scatter(good.pixels, good.filesize_kb, alpha=0.4)
    plt.xlabel("pixel count (w*h)"); plt.ylabel("file size (KB)")
    plt.title("File size vs pixels — metadata bloat check")
    bytes_per_px = (good.filesize_kb * 1024 / good.pixels).mean()
    plt.gca().text(0.5, 0.95,
                   f"mean {bytes_per_px:.1f} bytes/pixel\n(raw RGB = 3 B/px)",
                   transform=plt.gca().transAxes, ha="center", va="top",
                   fontsize=9, bbox=dict(boxstyle="round", fc="#ffe", alpha=.8))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_filesize_vs_pixels.png"), dpi=130)
    plt.close()

    # ---- brightness histogram ----
    plt.figure(figsize=(6, 4))
    plt.hist(good.brightness, bins=25, color="#34495e")
    plt.xlabel("mean brightness (0-255)"); plt.ylabel("images")
    plt.title("Brightness distribution")
    plt.axvline(good.brightness.mean(), color="r", ls="--",
                label=f"mean={good.brightness.mean():.1f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_brightness_hist.png"), dpi=130)
    plt.close()

    # ---- per-channel intensity distribution ----
    plt.figure(figsize=(6, 4))
    for ch, col in [("mean_r", "r"), ("mean_g", "g"), ("mean_b", "b")]:
        plt.hist(good[ch], bins=25, alpha=0.5, color=col, label=ch)
    plt.xlabel("per-image channel mean (0-255)"); plt.ylabel("images")
    plt.title("RGB channel means")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_rgb_channels.png"), dpi=130)
    plt.close()

    # ---- mean colour swatches ----
    m = min(60, len(good))
    sw = good.head(m)
    cols = 12
    rows_s = int(np.ceil(m / cols))
    swatch = np.zeros((rows_s, cols, 3), dtype=np.uint8)
    for i, (_, r) in enumerate(sw.iterrows()):
        swatch[i // cols, i % cols] = [r.mean_r, r.mean_g, r.mean_b]
    plt.figure(figsize=(cols * 0.4, rows_s * 0.4))
    plt.imshow(swatch); plt.axis("off")
    plt.title("Per-image mean colour", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_mean_color.png"), dpi=130,
                bbox_inches="tight")
    plt.close()

    # ---- summary json ----
    summ = dict(
        n_images=int(len(good)),
        n_corrupt=int(df.corrupt.sum()),
        dims=dim_counts.to_dict("records"),
        brightness_mean=float(good.brightness.mean()),
        brightness_std=float(good.brightness.std()),
        contrast_std_mean=float(good.contrast_std.mean()),
        mean_rgb=[float(good.mean_r.mean()), float(good.mean_g.mean()),
                  float(good.mean_b.mean())],
        filesize_kb_mean=float(good.filesize_kb.mean()),
        bytes_per_pixel_mean=float(bytes_per_px),
        metadata_blocks_mean=float(good.metadata_blocks.mean()),
    )
    json.dump(summ, open(os.path.join(OUT, "pixel_summary.json"), "w"), indent=2)
    print("[pixel EDA] summary:", json.dumps(summ, indent=2)[:600])
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root",
                    default="/projects/sandbox/drive_data/Datasets/Batik_Lasem")
    args = ap.parse_args()
    run_pixel_eda(args.img_root)
