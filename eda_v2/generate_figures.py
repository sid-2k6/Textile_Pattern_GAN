#!/usr/bin/env python3
"""Publication figures (Part W). Metadata-driven figures are COMPLETE (5860
records); image-derived figures are SAMPLE-based (160 imgs, 2 datasets) and are
titled as such so no figure is mistaken for a full-dataset census."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
META = "/projects/sandbox/data/_metadata/metadata_motifs.csv"

m = pd.read_csv(META, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False)
m = m.loc[:, [c for c in m.columns if not c.startswith("Unnamed") and c.strip()]]
m.columns = [c.strip() for c in m.columns]
for c in m.columns: m[c] = m[c].str.strip()

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=130,
                                    bbox_inches="tight"); plt.close(fig)

# 1. class distribution (COMPLETE)
cls = m.motif_name.value_counts()
fig, ax = plt.subplots(figsize=(8, 4.5))
cls.plot(kind="bar", color="#c0392b", ax=ax)
ax.set_title(f"Batik_Lasem motif class distribution (COMPLETE metadata, n={len(m)})")
ax.set_ylabel("motif crop records"); ax.set_xlabel("motif_name")
for i, v in enumerate(cls): ax.text(i, v+10, str(v), ha="center", fontsize=8)
save(fig, "01_class_distribution_metadata.png")

# 2. resolution split per class (COMPLETE)
ct = pd.crosstab(m.motif_name, m.image_resolution)
fig, ax = plt.subplots(figsize=(8, 4.5))
ct.plot(kind="bar", stacked=True, ax=ax, color=["#2980b9", "#e67e22"])
ax.set_title("Batik_Lasem resolution per class (COMPLETE metadata)")
ax.set_ylabel("records"); ax.legend(title="image_resolution")
save(fig, "02_resolution_per_class_metadata.png")

# 3. origin-image leakage: crops per origin (COMPLETE)
po = m.origin_images.value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(po)), po.values, color="#8e44ad")
ax.set_title(f"LEAKAGE: crops per origin image — {len(po)} origins generate {len(m)} crops")
ax.set_xlabel("origin image (rank)"); ax.set_ylabel("# crops derived")
ax.axhline(po.mean(), color="k", ls="--", label=f"mean={po.mean():.0f}")
ax.text(len(po)*0.4, po.max()*0.8,
        f"top origin = {po.iloc[0]} crops\n({100*po.iloc[0]/len(m):.1f}% of all)",
        bbox=dict(boxstyle="round", fc="#ffe"))
ax.legend(); save(fig, "03_origin_leakage_metadata.png")

# 4. motif x origin heatmap (COMPLETE)
mo = pd.crosstab(m.motif_name, m.origin_images)
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(mo.values, aspect="auto", cmap="magma")
ax.set_yticks(range(len(mo.index))); ax.set_yticklabels(mo.index, fontsize=8)
ax.set_xticks([]); ax.set_xlabel(f"{mo.shape[1]} distinct origin images")
ax.set_title("motif_name x origin_image (crop counts) — source concentration")
fig.colorbar(im, ax=ax, label="# crops"); save(fig, "04_motif_x_origin_heatmap.png")

# 5. workshop distribution (COMPLETE)
w = m["Source Workshop"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4.5))
w.plot(kind="barh", color="#16a085", ax=ax)
ax.set_title("Batik_Lasem source workshop distribution (COMPLETE)")
ax.set_xlabel("records"); save(fig, "05_workshop_distribution_metadata.png")

# 6. year distribution (COMPLETE)
fig, ax = plt.subplots(figsize=(5, 4))
m.year_collected.value_counts().sort_index().plot(kind="bar", color="#34495e", ax=ax)
ax.set_title("collection year (COMPLETE)"); ax.set_ylabel("records")
save(fig, "06_year_distribution_metadata.png")

# ---- image-sample-based figures ----
try:
    q = pd.read_csv(os.path.join(OUT, "image_quality.csv"))
    col = pd.read_csv(os.path.join(OUT, "color_statistics.csv"))
    tex = pd.read_csv(os.path.join(OUT, "texture_features.csv"))

    # 7. brightness / saturation / colorfulness by dataset
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ds, g in q.groupby("dataset"):
        axs[0].hist(g.brightness.dropna(), bins=20, alpha=.55, label=ds)
    axs[0].set_title("Brightness (SAMPLE)"); axs[0].legend(fontsize=7)
    for ds, g in col.groupby("dataset"):
        axs[1].hist(g.mean_saturation.dropna(), bins=20, alpha=.55, label=ds)
    axs[1].set_title("Mean saturation (SAMPLE)"); axs[1].legend(fontsize=7)
    for ds, g in col.groupby("dataset"):
        axs[2].hist(g.colorfulness.dropna(), bins=20, alpha=.55, label=ds)
    axs[2].set_title("Colorfulness (SAMPLE)"); axs[2].legend(fontsize=7)
    save(fig, "07_color_by_dataset_sample.png")

    # 8. texture by dataset (edge density, glcm contrast) reliability-flagged
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ds, g in tex.groupby("dataset"):
        axs[0].hist(g.edge_density.dropna(), bins=20, alpha=.55, label=ds)
    axs[0].set_title("Edge density (SAMPLE)"); axs[0].legend(fontsize=7)
    rel = tex[tex.texture_reliable == True]
    for ds, g in tex.groupby("dataset"):
        axs[1].hist(g.glcm_contrast.dropna(), bins=20, alpha=.55, label=ds)
    axs[1].set_title("GLCM contrast (SAMPLE; unreliable <64px)"); axs[1].legend(fontsize=7)
    save(fig, "08_texture_by_dataset_sample.png")

    # 9. PCA of aesthetic features colored by dataset
    aes = pd.read_csv(os.path.join(OUT, "aesthetic_features.csv"))
    feats = ["mean_hue","hue_entropy","mean_saturation","mean_value","contrast",
             "colorfulness","edge_density","gray_entropy","glcm_contrast",
             "glcm_homogeneity","glcm_energy","lbp_entropy","gabor_mean",
             "fft_highfreq_ratio","symmetry_h","pattern_density","spatial_entropy"]
    feats = [f for f in feats if f in aes.columns]
    X = aes[feats].fillna(aes[feats].mean())
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    U, S, Vt = np.linalg.svd(Xs.values, full_matrices=False)
    pc = U[:, :2] * S[:2]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for ds in aes.dataset.unique():
        mask = (aes.dataset == ds).values
        ax.scatter(pc[mask, 0], pc[mask, 1], alpha=.6, label=ds, s=25)
    ax.set_title("Aesthetic-feature space (PCA, SAMPLE) — domain separation")
    ax.set_xlabel(f"PC1 ({100*S[0]**2/ (S**2).sum():.0f}% var)")
    ax.set_ylabel(f"PC2 ({100*S[1]**2/ (S**2).sum():.0f}% var)")
    ax.legend(); save(fig, "09_feature_space_pca_sample.png")

    # 10. correlation heatmap
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("Aesthetic-feature correlation (SAMPLE)")
    fig.colorbar(im, ax=ax, fraction=.046); save(fig, "10_feature_correlation_sample.png")
except Exception as e:
    print("image-based figures skipped:", e)

# 11. sample grids
def grid(paths, title, name, cell=1.4):
    paths = paths[:16]
    if not paths: return
    cols = 8; rows = int(np.ceil(len(paths)/cols))
    fig = plt.figure(figsize=(cols*cell, rows*cell))
    for i, p in enumerate(paths):
        ax = fig.add_subplot(rows, cols, i+1)
        try: ax.imshow(Image.open(p).convert("RGB"))
        except Exception: pass
        ax.axis("off")
    fig.suptitle(title); save(fig, name)

import glob
gr = sorted(glob.glob("/projects/sandbox/drive_data/Datasets/Batik_Lasem/**/all_motifs*/*.jpg", recursive=True))
grid(gr, "Batik_Lasem / Gunung Ringgit (28x28 crops, SAMPLE)", "11_grid_batik_gunungringgit.png")
mek = sorted(glob.glob("/projects/sandbox/data/NeuralLoom/**/Mekhela chador/*.jpg", recursive=True))
grid(mek, "NeuralLoom / Mekhela source images (SAMPLE)", "12_grid_neuralloom_mekhela.png")
# 13. one mekhela source and its 5 crops (lineage)
crops = sorted(glob.glob("/projects/sandbox/data/NeuralLoom/**/mekhela_cropped_imgs2018*/*", recursive=True))
if crops:
    one = os.path.dirname(crops[0])
    cp = sorted(glob.glob(os.path.join(one, "*")))
    grid(cp, f"NeuralLoom lineage: 5 crops of one source\n({os.path.basename(one)})",
         "13_neuralloom_lineage_5crops.png")
print("figures written to", FIG)
print(sorted(os.listdir(FIG)))
