# Research-Grade Dataset Audit (v2) — Aesthetic-Driven Textile GAN

This directory is a **rigorous, anti-hallucination dataset audit** for the project
*"Deep GANs for Aesthetic-Driven Apparel Pattern Synthesis and Interactive
Visualization."* It audits **Batik_Lasem, Batik_Nitik_960, Batik_Nitik_Sarimbit_120,
NeuralLoom, and the Batik_Lasem Original Images**, and marks **DeepFashion2 /
DeepFashion3D** as *pending access*.

> **Read `RESEARCH_EDA_REPORT.md` first** — it is the 27-section report and the
> executive decision. Every claim is tagged **[OBS]** directly observed, **[CMP]**
> computed, **[INF]** inferred, **[UNK]** unknown, **[USR]** user-reported (not
> reproduced here), or **[REC]** recommendation.

## ⚠️ Scope & honesty statement
- The complete **`metadata_motifs.csv` (5,860 rows)** and two metadata PDFs were
  fully obtained → the **Batik_Lasem structural/lineage/leakage audit is complete**.
- **Image analysis is SAMPLE-based**: Google Drive anonymous access lists only ~50
  files/folder and throttles bulk downloads, so only **94 Batik_Lasem/Gunung-Ringgit**
  + **66 NeuralLoom** images were analysed here. Image-derived stats are labelled
  *SAMPLE*. Re-run on the full local dataset before publication.
- No human aesthetic labels exist in any dataset; none were fabricated.

## ▶ Run on the FULL local dataset — Windows (one command)

You already have the data at `D:\Textile_Pattern_GAN\Datasets\`. From the repo root:

```powershell
# 1. get this eda_v2 folder into your working copy
git fetch origin
git checkout eda/research-audit-v2      # brings in eda_v2\ (your Datasets\ is gitignored, stays put)

# 2. run the whole audit (auto-finds the metadata CSV, excludes DeepFashion)
cd eda_v2
powershell -ExecutionPolicy Bypass -File .\run_full_audit.ps1
```
Outputs land in `eda_v2\outputs\` (CSV/JSON) and `eda_v2\figures\` (PNG). Edit the
`$DATA` line in `run_full_audit.ps1` if your `Datasets` folder is elsewhere.
`DeepFashion2/3D` are skipped on purpose (pending access).

> The runner sets two env vars for you: `TEXTILE_METADATA_CSV` (the Batik_Lasem
> `metadata motifs.csv`) and `DATASET_ROOT`. Near-duplicate detection on the full
> Batik_Lasem set (all three representation folders) can take **10–20 min**.

### Manual equivalent (any OS)
```powershell
$env:TEXTILE_METADATA_CSV = "D:\Textile_Pattern_GAN\Datasets\Batik_Lasem\motifs (isen-isen)\metadata motifs.csv"
$env:DATASET_ROOT = "D:\Textile_Pattern_GAN\Datasets"
python metadata_audit.py $env:TEXTILE_METADATA_CSV
python image_audit.py --datasets "Batik_Lasem=$env:DATASET_ROOT\Batik_Lasem;Batik_Nitik_960=$env:DATASET_ROOT\Batik_Nitik_960;Batik_Nitik_Sarimbit_120=$env:DATASET_ROOT\Batik_Nitik_Sarimbit_120;NeuralLoom=$env:DATASET_ROOT\NeuralLoom"
python generate_figures.py
python build_plans.py
```

## Reproduce (generic)
```bash
pip install pillow numpy pandas matplotlib scipy scikit-image pypdf gdown

# 1) Complete metadata audit (works fully offline once you have the CSV)
python metadata_audit.py /path/to/metadata_motifs.csv

# 2) Image audit — point at your local dataset roots (semicolon-separated name=path)
python image_audit.py --datasets \
  "Batik_Lasem=/data/Batik_Lasem;Batik_Nitik_960=/data/Batik_Nitik_960;\
NeuralLoom=/data/NeuralLoom;Original_Images=/data/Original Images;\
Batik_Nitik_Sarimbit_120=/data/Batik_Nitik_Sarimbit_120"
#   (DeepFashion2/3D intentionally omitted until access is granted)

# 3) Figures + strategy/plan files
python generate_figures.py
python build_plans.py
```

## Scripts
| File | Part(s) | Produces |
|---|---|---|
| `metadata_audit.py` | J, K, M | metadata_summary, class/origin/workshop/year distributions, motif×{origin,workshop,year} matrices, `metadata_audit.json` |
| `image_audit.py` | B, C, D, E, F, G, H, O, U, V | inventory, duplicates, lineage, quality, resolution, color, texture, aesthetic features, correlation |
| `generate_figures.py` | W | 13 publication figures (metadata-complete + image-sample) |
| `build_plans.py` | N, R, S, T, Y, Z, AB, AC, AG | readiness, role mapping, compatibility, gan requirements, split manifest, plans |

## Key machine-readable outputs (`outputs/`)
`dataset_inventory.csv`, `image_inventory.csv`, `source_lineage.csv`,
`duplicate_groups.csv`, `near_duplicate_groups.csv`, `image_quality.csv`,
`resolution_statistics.csv`, `color_statistics.csv`, `texture_features.csv`,
`aesthetic_features.csv` (+ `_summary`), `metadata_summary.csv`,
`class_distribution.csv`, `cross_dataset_statistics.csv`,
`cross_dataset_compatibility.csv`, `feature_correlation.csv`,
`dataset_readiness.csv`, `dataset_role_mapping.csv`, `gan_data_requirements.csv`,
`aesthetic_attribute_availability.csv`, `split_manifest.csv` (+ `_spec.json`),
`dataset_summary.json`, `gan_data_plan.json`, plus the batik_lasem_* matrices.

## Headline verified facts (Batik_Lasem, COMPLETE metadata)
- 5,860 crop records; classes Latohan 1373 / Seritan 1328 / Nyuk Pitu 1291 /
  Kricak 1004 / Gunung Ringgit 864.
- **39 origin images**; one origin → **697 crops (52.5 % of Seritan)**; 6 origins
  span multiple motif classes → **origin-level global split is mandatory**.
- `28x28_images` is an **exact-MD5 subset** of `all_motifs`; `all_motifs =
  28×28 ∪ 142×142`.
- 4,949 @ 28×28 + 911 @ 142×142; 7 workshops; years 2021/2022; 100 % sRGB; 100 %
  `raw (crop)`.
- Batik_Nitik_960 is **rotation-augmented** (4 motifs × {0,90,180,270}° per
  category) per its own documentation → not independent.
```
```
