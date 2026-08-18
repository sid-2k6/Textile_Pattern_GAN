# Textile Pattern GAN

Research code for **"Deep Generative Adversarial Networks for Aesthetic-Driven
Apparel Pattern Synthesis and Interactive Visualization."**

| Area | Location |
|---|---|
| Exploratory data analysis (v1) | [`eda/`](eda/) |
| Research-grade dataset audit (v2) | [`eda_v2/`](eda_v2/) — see `RESEARCH_EDA_REPORT.md` |
| Shared utilities | [`src/batik_gan/`](src/batik_gan/) |
| Split manifests | [`data/splits/`](data/splits/) |
| **GAN baseline notebooks** | [`notebooks/`](notebooks/) |

---

## GAN Baseline Experiments

Three independent, reproducible GAN baselines on **Batik_Lasem**, implemented as
separate Colab notebooks and sharing one canonical dataset, split and evaluation
protocol so their results are directly comparable. These are the baselines for the
proposed aesthetic-driven GAN — **no results are claimed here; the notebooks must
be trained first.**

### 1. Dataset (canonical, non-redundant)
Batik_Lasem stores each motif *instance* several times: `all_motifs (N)` (the
union, at native resolution), plus `28x28_images` and `142x142_images` (which are
exact-MD5 subsets of `all_motifs`). Training on all folders would 2–3× count the
same instances. We therefore use the **`all_motifs` folder as the single canonical
representation**: one file per instance at its **native resolution** (4,949 at
28×28, 911 at 142×142; total **5,860** instances / 5 motif classes). Verified in
the EDA (`eda_v2/`).

### 2. 50% subset strategy
One shared, deterministic (`SEED = 42`) **stratified-by-motif** 50% subset
(**2,930** instances) is used identically by all three models
(`data/splits/batik_lasem_50pct_manifest.csv`). It is generated once by
`src/batik_gan/manifest.py` and committed, so every baseline sees the exact same
samples.

### 3. Train/test methodology (leakage-safe)
Because only **39 origin photographs** produced all 5,860 crops (one origin can
account for >50% of a class), a random per-image split would leak highly similar
crops across train/test. We use a **group-aware split by `origin_images`**: whole
origins go entirely to train or test (verified **0 shared origins/filenames**). A
deterministic randomized search picks the group assignment that best preserves the
motif distribution (achieved **≤2.1% max per-motif gap** at ~78/22 train/test:
**2,290 train / 640 test**). The split report (train/test groups, sample counts,
per-motif distribution, unavoidable gaps) is printed by the notebooks and encoded
in the manifests.

> **Held-out discipline.** The test set is used **only** for final/epoch FID/KID
> evaluation. The dataset (39 origins) does not permit a separate validation set,
> so `best.pt` is chosen by lowest **test-set FID** — this is documented as a
> limitation (final test numbers are an optimistic bound; use a fresh hold-out for
> publication-grade selection).

### 4. Baseline 1 — DCGAN  (`notebooks/01_DCGAN_BatikLasem_50pct.ipynb`)
Unconditional DCGAN at **128×128** (transposed-conv generator, BatchNorm + ReLU,
Tanh; strided-conv discriminator with LeakyReLU, logits + `BCEWithLogitsLoss`),
Adam, mixed precision (AMP).

### 5. Baseline 2 — Conditional WGAN-GP  (`notebooks/02_cWGAN_GP_BatikLasem_50pct.ipynb`)
Conditional WGAN-GP at 128×128, conditioned on `motif_name`. Wasserstein loss +
correct gradient penalty (λ = 10), critic uses **InstanceNorm (no BatchNorm)**,
`N_CRITIC = 5` (configurable), fp32 (GP double-backward stability). Reports
critic loss / gradient penalty / Wasserstein estimate — **discriminator accuracy is
N/A for a WGAN critic and is never fabricated**.

### 6. Baseline 3 — StyleGAN2-ADA  (`notebooks/03_StyleGAN2_ADA_BatikLasem_50pct.ipynb`)
Official **NVlabs/stylegan2-ada-pytorch** (pinned commit), trained on the same 50%
train split (pre-resized to 128×128 with identical preprocessing, packed via
`dataset_tool.py`). Uses native ADA training/logging and resume; we **additionally
recompute FID/KID against the held-out test set** with the same evaluator as the
other baselines for a fair comparison. (Native `fid50k_full` uses training reals
and is reported separately.)

### 7. Comparison  (`notebooks/04_Baseline_Comparison.ipynb`)
Reads saved `history.csv` / `final_metrics.*` / `config.json` (no retraining) and
produces the comparison table + 300-dpi plots.

### 8. Google Colab + NVIDIA L4
Every notebook prints an environment banner (PyTorch / CUDA / GPU / VRAM), warns if
no GPU, uses AMP where stable, `pin_memory`, and reproducible seeding. Designed for
Colab + L4 (24 GB); nothing is hard-coded to a single runtime.

### 9. How to resume training
Set `RESUME = True` (default). With `RESUME_CHECKPOINT = None` the notebook
auto-discovers the latest checkpoint and prints *"Checkpoint found. Resuming from
epoch X."* (or *"Starting from epoch 0."*). Checkpoints, `history.csv`, samples and
plots are written to **Google Drive** (`OUTPUT_ROOT`) so a Colab disconnect loses no
progress — just re-run. StyleGAN2-ADA resumes natively from its latest snapshot.

### 10. Output directory (not committed — see `.gitignore`)
```
outputs/<MODEL>/
  checkpoints/  (latest.pt, best.pt, epoch_XXXX.pt | StyleGAN: network-snapshot-*.pkl)
  generated_samples/  (fixed_noise.pt, epoch_XXXX.png, final_grid.png, per-motif)
  plots/  logs/
  history.csv   config.json   final_metrics.csv   final_metrics.json
```

### 11. Evaluation metrics
- **FID** and **KID** (torchmetrics, Inception-V3 pool3 / 2048-d), **real = held-out
  test set**, identical protocol and generated-sample count for all three models.
- **Diversity** = mean pairwise Inception-feature distance (a statistic, not
  accuracy); NaN if it cannot be computed (never fabricated).
- DCGAN also logs **Discriminator Real/Fake Accuracy** (clearly labelled — *not*
  generator accuracy). Generator accuracy/precision/recall/F1 are **not applicable**
  to GANs and are explicitly omitted.

### Setup
```bash
pip install -r requirements.txt   # torch/torchvision come preinstalled on Colab
```
Open a notebook in Colab (GPU/L4 runtime), edit the clearly-marked **DATASET_ROOT**
in the path cell to point at your Google-Drive copy of `Datasets/`, and run.
