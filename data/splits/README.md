# Batik_Lasem canonical split manifests

Deterministic (`SEED = 42`), committed so all three GAN baselines use the **exact
same** samples and train/test split. Regenerate with:

```bash
python src/batik_gan/manifest.py "<DATASET_ROOT>/Batik_Lasem/motifs (isen-isen)/metadata motifs.csv" data/splits
```

| File | Rows | Description |
|---|--:|---|
| `batik_lasem_canonical_manifest.csv` | 5860 | one row per motif instance (canonical = `all_motifs`, native resolution) |
| `batik_lasem_50pct_manifest.csv` | 2930 | stratified-by-motif 50% subset (the shared experiment set) |
| `batik_lasem_50pct_train.csv` | 2290 | group-aware TRAIN (origins disjoint from test) |
| `batik_lasem_50pct_test.csv` | 640 | held-out TEST (never used for training/selection during training) |

**Columns:** `instance_id, filename, rel_path, motif_name, motif_folder,
all_motifs_dir, origin_images, Source Workshop, year_collected, image_type,
image_resolution, color_profile, annotator_initials`.

**Leakage guarantee:** 0 `origin_images` and 0 `filename` shared between train and
test. Group-aware split → 27 train groups / 12 test groups; max per-motif
train/test percentage gap ≈ 2.0%. `rel_path` is resolved robustly at load time
(`src/batik_gan/paths.resolve_image_path`), so it tolerates folder-name variations.
