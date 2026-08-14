# Exploratory Data Analysis — Textile Pattern GAN

**Datasets analysed:** Batik (`Batik_Lasem`, `Batik_Nitik_960`, `Batik_Nitik_Sarimbit_120`,
`Original Images`) **and NeuralLoom.**
**Explicitly excluded:** `DeepFashion2`, `DeepFashion3D` (per request).

Source: shared Google Drive folder
`16vAxXOXUPkcDld5Hf-oJthYnRhUaVMYH`.

---

## 1. How this EDA was produced (and an important caveat)

The data lives in a **public Google Drive folder**. Two facts shaped this analysis:

1. **Anonymous Drive access only returns the first ~50 files per folder.**
   `gdown --folder` therefore reports at most 50 files/folder, which *undercounts*
   the large classes. Luckily the dataset authors **embed the true count in each
   folder name** (e.g. `all_motifs (864)`), so class sizes below are taken from
   those names where available (`provenance = name`) and from the capped listing
   otherwise (`provenance = listed`, treat as a lower bound).
2. **Bulk per-file download is rate-limited** ("Cannot retrieve the public link …
   too many accesses"). A first burst captured **94 real `Gunung Ringgit` images**,
   on which the pixel-level section below is computed. The
   [`Textile_EDA.ipynb`](./Textile_EDA.ipynb) notebook reproduces the **full**
   pixel EDA once the folder is mounted in Colab / downloaded locally.

---

## 2. Dataset composition

| Dataset | Class | Count | Unit | Source of count |
|---|---|--:|---|---|
| Batik_Lasem | Gunung Ringgit | 864 | motifs | folder name |
| Batik_Lasem | Kricak / Watu Pecah | 1004 | motifs | folder name |
| Batik_Lasem | Latohan | 1373 | motifs | folder name |
| Batik_Lasem | Nyuk Pitu | 1291 | motifs | folder name |
| Batik_Lasem | Seritan | 1328 | motifs | folder name |
| **Batik_Lasem total** | 5 *isen-isen* motif classes | **5 860** | motifs | |
| Batik_Nitik_960 | Batik Nitik | 960 | images | folder name (+ `Metadata`) |
| Batik_Nitik_Sarimbit_120 | Sarimbit | 120 | images | folder name |
| Original Images | All Images (source photos) | ≥40 | images | listing (capped) |
| NeuralLoom | Mekhela chador | ~33 groups / ~165 crops | images | listing (capped) |
| NeuralLoom | Saree body | ~49 | images | listing (capped) |
| ~~DeepFashion2~~ | excluded | — | | |
| ~~DeepFashion3D~~ | excluded | — | | |

*(See [`outputs/dataset_composition.csv`](./outputs/dataset_composition.csv) and
[`outputs/figures/fig_composition.png`](./outputs/figures/fig_composition.png).)*

### Folder hierarchy (included parts)

```
Datasets/
├── Batik_Lasem/
│   └── motifs (isen-isen)/
│       ├── Gunung Ringgit/      { 28x28_images, all_motifs (864) }
│       ├── Kricak_Watu Pecah/   { 28x28_images, all_motifs (1004) }
│       ├── Latohan/             { 28x28_images, 142x142_images, all_motifs (1373) }
│       ├── Nyuk Pitu/           { 28x28_images, 142x142_images, all_motifs (1291) }
│       └── Seritan/             { 28x28_images, 142x142_images, all_motifs (1328) }
├── Original Images/  └── All Images/        (source photographs)
├── Batik_Nitik_960/  { Batik Nitik 960 Images, Metadata }
├── Batik_Nitik_Sarimbit_120/  { Dataset, Original images high-resolution }
└── NeuralLoom/
    ├── Mekhela chador body/ …  (garment photos + mekhela_cropped_imgs/*)
    └── saree body/ …           ("New folder - Copy (N)" one image each)
   (DeepFashion2/ and DeepFashion3D/ present in Drive — NOT analysed)
```

### Observations
- **Class imbalance** inside Batik_Lasem: Latohan (1373) and Seritan (1328) have
  ~1.6× the samples of Gunung Ringgit (864). A conditional GAN should account for
  this (class weighting / oversampling) or it will under-represent minority motifs.
- **Multiple pre-rendered resolutions** exist per motif: `28x28_images` (all
  classes), `142x142_images` (only Latohan, Nyuk Pitu, Seritan), and `all_motifs`.
  Pick **one** target resolution; don't mix silently.
- **NeuralLoom is a different domain** from Batik_Lasem: full colour photographs of
  Assamese *Mekhela Chador* / *Saree* textiles vs. tiny cropped batik motifs. Its
  folder structure is messy (`New folder - Copy (N)`, and crop-folders named like
  `*.jpg`). Expect heavy cleanup and treat it as a separate training domain.

---

## 3. Pixel-level EDA (Gunung Ringgit sample, n = 94)

Computed on the 94 images retrieved before rate-limiting. Full stats in
[`outputs/pixel_stats.csv`](./outputs/pixel_stats.csv);
figures in [`outputs/figures/`](./outputs/figures/).

| Metric | Value |
|---|--:|
| Images analysed | 94 (0 corrupt) |
| Resolution | **28 × 28 px, RGB** (100% uniform) |
| Mean brightness (0–255) | 203.6  (σ ≈ 10.5) → bright, light background |
| Mean RGB | (206.7, 201.3, 202.7) → warm near-neutral cream |
| Mean per-image contrast (σ) | 60.2 |
| Mean file size | 16.8 KB |
| **Bytes per pixel** | **≈ 22.0**  (raw RGB is 3 B/px!) |
| Metadata blocks/image | 4.0 (EXIF + Photoshop + XMP + ICC on *every* file) |

### Key findings
1. **Uniform 28×28 resolution** — the `all_motifs` folder is *also* 28×28, not
   full resolution as the name might suggest. This is a MNIST-style GAN dataset.
2. **Severe metadata bloat.** A 28×28 image holds 784 pixels yet averages ~17 KB
   because each file carries Adobe Photoshop CS6 EXIF/XMP/ICC blocks
   (~22 bytes/pixel vs. the ~3 B/px of raw RGB). This is pure overhead — **strip
   metadata** before building training shards to cut dataset size ~5–7×.
   See [`fig_filesize_vs_pixels.png`](./outputs/figures/fig_filesize_vs_pixels.png).
3. **Consistent motif rendering** (see
   [`fig_sample_grid.png`](./outputs/figures/fig_sample_grid.png)): the Gunung
   Ringgit *isen-isen* is a dotted concentric fan/arc ("mountain range") drawn in
   dark brown/indigo ink on cream cloth, with fairly consistent orientation —
   favourable for a GAN, though low intra-class pose variety may limit diversity.
4. **Bright, low-key-variance palette** (brightness σ ≈ 10) → little exposure
   variation; per-pixel normalisation with a fixed mean/std is fine.

Figures: `fig_sample_grid`, `fig_dimensions`, `fig_filesize_vs_pixels`,
`fig_brightness_hist`, `fig_rgb_channels`, `fig_mean_color`, `fig_composition`.

---

## 4. Data-quality checklist
- [x] No corrupt files in the 94-image sample.
- [ ] Duplicate / near-duplicate scan across full classes — run notebook §4 (aHash).
- [ ] Strip EXIF/XMP/ICC metadata (large win, see §3).
- [ ] Verify `all_motifs` vs `28x28_images` are not duplicates of each other.
- [ ] Standardise NeuralLoom folder layout and file naming.
- [ ] Confirm exact counts for `Original Images`, `Batik_Nitik_*` and NeuralLoom by
      running the notebook on the mounted Drive (bypasses gdown's 50-file cap).

---

## 5. Recommendations for the GAN
1. **Target resolution:** train batik at 28×28 (baseline, matches the bulk of
   `Batik_Lasem`) or upscale to 64×64; if higher fidelity is needed use the
   `142x142_images` subset (but only 3 of 5 motifs have it).
2. **Conditional generation:** motif class label is a natural condition
   (cGAN / class-conditional). Handle the 864→1373 imbalance.
3. **Keep domains separate:** don't mix 28×28 batik crops with NeuralLoom
   photographs in one unconditional model.
4. **Preprocess once:** strip metadata, convert to a packed tensor/`.npy`/WebDataset
   shard, normalise to [-1, 1].

---

## 6. Files in this folder
| File | Purpose |
|---|---|
| `EDA_REPORT.md` | this report |
| `Textile_EDA.ipynb` | reproducible full-dataset EDA (mount Drive & run) |
| `generate_eda.py` | script that produced the figures/CSVs from a local sample |
| `parse_gdrive_listing.py` | turns a `gdown --folder` log into a count table |
| `outputs/dataset_composition.csv` | class table with count provenance |
| `outputs/pixel_stats.csv` | per-image pixel statistics (sample) |
| `outputs/pixel_summary.json` | aggregate pixel summary |
| `outputs/figures/*.png` | all charts |
