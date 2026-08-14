# EDA — Textile Pattern GAN

Exploratory data analysis of the **Batik** datasets and the **NeuralLoom** dataset
from the shared Google Drive folder. `DeepFashion2` / `DeepFashion3D` are **excluded**.

## TL;DR
- **Read [`EDA_REPORT.md`](./EDA_REPORT.md)** for findings, tables and figures.
- **Run [`Textile_EDA.ipynb`](./Textile_EDA.ipynb)** (Colab) to reproduce the full
  pixel-level EDA on the complete dataset (mount Drive, set `ROOT`, run all).

## Why a notebook instead of a full local run?
Google Drive anonymous access (a) lists only ~50 files per folder and (b)
rate-limits bulk downloads. The precomputed figures/CSVs in `outputs/` were made
from the 94 `Gunung Ringgit` images that downloaded before throttling; the notebook
sidesteps both limits by reading the folder directly once it's mounted in your Drive.

## Reproduce locally
```bash
pip install pillow numpy pandas matplotlib gdown

# (optional) re-enumerate the Drive folder to rebuild the count table:
gdown --folder "https://drive.google.com/drive/folders/16vAxXOXUPkcDld5Hf-oJthYnRhUaVMYH" \
      --remaining-ok -O ./dl > listing.txt 2>&1
python parse_gdrive_listing.py listing.txt

# regenerate figures/CSVs from a local image folder:
python generate_eda.py --img-root /path/to/Batik_Lasem
```

## Layout
```
eda/
├── EDA_REPORT.md            # findings & recommendations
├── Textile_EDA.ipynb        # full reproducible EDA (Colab)
├── generate_eda.py          # figure/CSV generator (local sample)
├── parse_gdrive_listing.py  # gdown log -> per-folder count table
└── outputs/
    ├── dataset_composition.csv
    ├── pixel_stats.csv
    ├── pixel_summary.json
    └── figures/*.png
```
