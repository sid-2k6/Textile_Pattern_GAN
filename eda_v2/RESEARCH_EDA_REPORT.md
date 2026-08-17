# Research-Grade Dataset Audit & GAN Data-Readiness Analysis

## Project: *Deep Generative Adversarial Networks for Aesthetic-Driven Apparel Pattern Synthesis and Interactive Visualization*

**Document type:** dataset audit + data-to-model feasibility analysis (pre-training).
**Status:** GAN training NOT started (per instruction). This document establishes
what can *legitimately* be extracted from the data we actually possess.

---

### Evidence tagging convention (used throughout)

| Tag | Meaning |
|---|---|
| **[OBS]** | DIRECTLY OBSERVED — verified from actual files/images/metadata in this environment |
| **[CMP]** | COMPUTED — derived programmatically from the data here |
| **[INF]** | INFERRED — reasonable interpretation, not explicitly provided |
| **[UNK]** | UNKNOWN / NOT AVAILABLE — cannot be established from available data |
| **[USR]** | USER-REPORTED — provided in the task brief from the user's *local* machine; **not** independently reproducible in this sandbox |
| **[REC]** | RECOMMENDATION — design proposal, not a data fact |

> **Environment constraint (read first).** The datasets live in a public Google
> Drive folder. Two hard limits shaped this audit: (1) anonymous Drive enumeration
> returns only ~50 items per folder; (2) per-IP bulk download is aggressively
> throttled. Consequently the **complete `metadata_motifs.csv` (5,860 rows) and two
> metadata PDFs were fully obtained and analysed**, but **image acquisition is a
> partial sample**: 94 Batik_Lasem/Gunung-Ringgit crops and 66 NeuralLoom images
> (33 Mekhela sources + 33 Mekhela crops). Every image-derived statistic is
> therefore **sample-based** and labelled as such. Nothing here is extrapolated to
> a full-dataset census without saying so.

---

## 1. Executive Summary

- **Batik_Lasem is structurally well-documented and, at the metadata level, fully
  audited.** [OBS] 5,860 motif-crop records across 5 motif classes, but **every
  record is a crop (`image_type = raw (crop)`, 100%)** derived from only **39
  origin photographs**. [OBS/CMP]
- **Severe source concentration → the effective independent sample size is ≈ 39,
  not 5,860.** A single origin photo (`Batik Artisan with Seritan Background.jpg`)
  produced **697 crops = 52.5 % of the entire Seritan class**. [CMP] This is the
  single most important finding for train/val/test design: **crop-level random
  splitting would catastrophically leak.**
- **Representation folders overlap.** `28x28_images` is a **byte-identical subset**
  of `all_motifs` (44/44 sampled Gunung-Ringgit files are exact MD5 duplicates).
  [OBS] `all_motifs (N)` = `28x28_images ∪ 142x142_images` [INF, from metadata
  counts]. Counting the three folders separately triple-counts the data.
- **Batik_Nitik_960 is rotation-augmented**: 60 categories × 16 = 4 motifs × {0,90,
  180,270}° [OBS, from the dataset's own PDF]. The 960 images are **not
  independent**; effective base ≈ 240 motifs / 60 fabric pieces.
- **NeuralLoom is a different visual domain** from batik (brightness 129 vs 204,
  saturation 0.41 vs 0.10, colorfulness 48 vs 21 — [CMP] on the sample) and its
  Mekhela images are organised as **5 source-linked crops per source garment**
  [OBS]. These 5 crops are *not* duplicates and *not* independent designs.
- **No dataset contains human aesthetic labels.** [OBS] "Aesthetic-driven" must
  therefore be built on (a) computable objective visual descriptors and/or (b) an
  explicitly-external pretrained aesthetic model — never on fabricated scores.
- **DeepFashion2 / DeepFashion3D are PENDING** (present only as compressed
  archives; not extracted/analysed). [OBS] True 3D garment visualization depends
  on DeepFashion3D access. [INF]

---

## 2. Research Objective (data framing)

The proposed system is an aesthetic-conditioned, multi-scale, multi-source textile
GAN with downstream garment visualization. This audit answers, for every link in
the pipeline **data → features → aesthetic representation → conditioning →
generator → discriminator → pattern synthesis → garment visualization →
evaluation**, whether the required information is [OBS]/[CMP]/[INF]/[UNK] (see §27
and the data-flow table at the end).

---

## 3. Dataset Inventory

### 3.1 Currently audited (physically present here)

| Dataset | Root (here) | Access | Files here | Images here | Non-image | Notes |
|---|---|---|--:|--:|--:|---|
| Batik_Lasem | `drive_data/.../Batik_Lasem` | ✅ | 94 | 94 | 0 | Gunung-Ringgit only (28×28); metadata CSV complete |
| NeuralLoom | `data/NeuralLoom` | ✅ (partial) | 66 | 66 | 0 | 33 Mekhela sources + 33 crops |
| _metadata | `data/_metadata` | ✅ | 3 | 0 | 3 | `metadata_motifs.csv` (5,860), 2 PDFs |

`[OBS]` from `outputs/dataset_inventory.csv`. Extensions observed: `.jpg` only for
images; `.csv`, `.pdf` for metadata. **`is_sample = True`** for both image
datasets.

### 3.2 Documented but not pulled here (throttled)

| Dataset | Evidence | Nominal size | Independence caveat |
|---|---|--:|---|
| Batik_Nitik_960 | PDF documentation [OBS] | 960 (60×16) | rotation-augmented [OBS] |
| Original Images (Batik_Lasem origins) | 39 via metadata [CMP]; 27-pg PDF [OBS] | 39 cloths | these *are* the source units |
| Batik_Nitik_Sarimbit_120 | folder name only | **[UNK]** — "120" NOT verified | do not assume 120 |

### 3.3 Pending access (excluded)

| Dataset | Evidence here | Status |
|---|---|---|
| DeepFashion2 | `train.zip`, `test-001.zip`, `drive-download-*.zip` present [OBS] | **PENDING** — not extracted |
| DeepFashion3D | `filtered_registered_mesh-002.rar`, `point_cloud-001.zip` [OBS] | **PENDING** — 3D data, not extracted |

> Missing datasets are **not** treated as empty. Their statistics are **[UNK]**.

Machine-readable: `outputs/dataset_inventory.csv`, `image_inventory.csv`.

---

## 4. Dataset Provenance

- **Batik_Lasem** [OBS from `metadata_motifs.csv` + `Short_Description...pdf`]:
  hand-drawn *batik tulis* cloths from Lasem, Central Java; 7 named workshops
  (Pesona Canting, KUB Srikandi, Pusaka Beruang, VJ Basiroen artwork, Sekar
  Kencana, Katrin Bee, Lumintu Batik); collected 2021–2022; motif crops annotated
  by initials (`annotator_initials`); color profile 100 % `sRGB IEC61966-2.1`.
- **Batik_Nitik_960** [OBS from its PDF]: published on Mendeley Data (Minarno,
  Nugroho, Soesanti, 2022); Sony A6400, 6024×4024, studio lighting; 60 categories
  from the Winotosastro Batik collection; **each category = 4 motifs × 4 rotations**.
- **NeuralLoom** [OBS folder structure + [USR]]: Assamese Mekhela-Chador and Saree
  textiles; filenames encode capture timestamps (e.g. `IMG_20180824_*`), i.e. field
  photographs. Semantic labels are **folder-derived only**, not dataset-provided.
- **DeepFashion2 / 3D**: provenance **[UNK]** here (archives unopened).

---

## 5. Data Lineage

**Batik_Lasem** [OBS/CMP/INF]:
```
39 origin photographs (Original Images / "Short Description" PDF)
        │  (hand-crop, per metadata image_type = "raw (crop)")
        ▼
5,860 motif crops  ──► motif_class (5)          [OBS via metadata]
        │
        ├─ 4,949 at 28×28  ─► 28x28_images  (⊆ all_motifs, exact MD5 dup [OBS])
        └─   911 at 142×142 ─► 142x142_images
        └─ all_motifs (N) = union of the two resolutions per class   [INF]
```
**NeuralLoom** [OBS]:
```
Mekhela source image  ──►  folder named after the source (…jpg/)
        └─ exactly 5 crops per source (verified on downloaded sources)
Saree: one image per "New folder…" (source folder = unit)
```
Machine-readable: `outputs/source_lineage.csv`.

**Provenance classification of each image** (roles used): `crop` (all Batik_Lasem;
Mekhela crops), `source` (Mekhela sources, Saree bodies). No image in the sample is
a *resized* copy **except** the `28x28_images`↔`all_motifs` exact duplicates. [OBS]

---

## 6. Duplicate Analysis (Part D)

Method [CMP]: MD5 + SHA-256 (exact), and aHash/dHash/pHash (perceptual, Hamming ≤ 6
within dataset).

| Finding | Value | Evidence |
|---|--:|---|
| Exact-duplicate groups (sample) | 44 | [CMP] `duplicate_groups.csv` |
| …all spanning `28x28_images` ↔ `all_motifs` | 44/44 | [OBS] — representation overlap, not accidental dup |
| Near-duplicate groups (Batik sample) | 43 | [CMP] — same pairs; `same_source = True` |
| NeuralLoom 5-crop groups flagged as near-dup | 0 | [CMP] — crops are *different regions*, correctly **not** duplicates |
| Corrupt/unreadable images (sample) | 0 | [CMP] |

**User-reported NeuralLoom exact-dup audit** [USR, not reproduced here]: 353 files,
346 unique hashes, 7 redundant copies of `IMG_20180824_200147.jpg`. This is
plausible and consistent with field-capture workflows; **flagged for local
re-verification** (the 155-image Saree tree could not be fully pulled here).

Handling classes assigned: exact dups → **B (retain one canonical)**; source-linked
crops → **C (keep as source-linked views)**; unresolved → **E (manual)**.

---

## 7. Image Quality (Part E) — sample

`[CMP]` from `image_quality.csv` (160 images). Dataset-level means:

| Dataset | n | W×H | Brightness | Contrast | Saturation | Colorfulness | Edge density | Gray entropy |
|---|--:|---|--:|--:|--:|--:|--:|--:|
| Batik_Lasem | 94 | 28×28 | 203.6 | 0.24 | 0.10 | 21.1 | 0.246 | 6.14 |
| NeuralLoom | 66 | 454–3023 | 129.0 | 0.11 | 0.41 | 48.2 | 0.051 | 5.70 |

Detectors run: extremely dark/bright (clipping fractions), blur (Laplacian
variance `sharpness_lapvar`), low-information (entropy), unusual aspect ratio,
inconsistent color space. **No corrupt or blank images in the sample.** [CMP]
Note the Batik_Lasem crops carry **Adobe Photoshop EXIF/XMP/ICC metadata (~4 blocks
each)** inflating 28×28 files to ~17 KB (≈22 B/px vs 3 B/px raw) [CMP] — strip
before packing.

---

## 8. Resolution Analysis (Part F)

**Batik_Lasem (COMPLETE, from metadata `image_resolution`)** [OBS]:

| Motif | 28×28 | 142×142 | total |
|---|--:|--:|--:|
| Gunung Ringgit | 864 | 0 | 864 |
| Kricak / Watu Pecah | 1004 | 0 | 1004 |
| Latohan | 1075 | 298 | 1373 |
| Nyuk Pitu | 986 | 305 | 1291 |
| Seritan | 1020 | 308 | 1328 |
| **Total** | **4949** | **911** | **5860** |

- Dominant resolution **28×28** (84.5 %); higher-res **142×142** exists for only
  **3 of 5** motifs. [OBS]
- The 142×142 rows are **separate crop records**, not guaranteed resizes of the
  28×28 rows; pixel-level correspondence is **[UNK]** (those images were not
  downloadable here). Do **not** assume they are the same crops upscaled.
- **Batik_Nitik_960**: 6024×4024 originals [OBS from PDF]. **NeuralLoom**: 454–3023
  px, ~square, highly variable [CMP sample]. **Sarimbit / Original Images**: [UNK].

**Resizing implication [INF/REC]:** at 28×28 fine isen-isen dot structure is already
near the sampling limit — upscaling 28→64/128 **cannot recover** lost motif detail;
it only interpolates. Genuine high-resolution training requires the 142×142 subset
(only 911 imgs) or Batik_Nitik_960 (24 MP, but rotation-augmented). Machine-readable:
`resolution_statistics.csv`.

---

## 9. Color Analysis (Part G) — objective descriptors (NOT aesthetic labels)

`[CMP]` `color_statistics.csv`, `cross_dataset_statistics.csv`. Computed: RGB, HSV,
LAB means/std, hue entropy, saturation, colorfulness (Hasler–Süsstrunk).

- **Batik_Lasem** occupies a **bright, low-saturation, warm-neutral** region
  (mean RGB ≈ (207,201,203), sat 0.10) — cream cloth + dark ink.
- **NeuralLoom** is **darker and far more saturated/colorful** (sat 0.41,
  colorfulness 48) — dyed garments photographed in the field.
- The two datasets clearly occupy **different color distributions** (see PCA,
  §15 / `figures/09_feature_space_pca_sample.png`). These are **objective
  visual/color descriptors**, explicitly *not* aesthetic scores.

---

## 10. Texture / Pattern Analysis (Part H) — reliability-flagged

`[CMP]` `texture_features.csv`: GLCM (contrast/homogeneity/energy/correlation),
LBP (entropy/uniformity), Gabor (mean/std), FFT high-frequency ratio, edge density,
symmetry (h/v), spatial entropy, pattern density.

> **Critical reliability caveat [CMP/INF]:** every Batik_Lasem sample image is
> 28×28, i.e. below the `texture_reliable = 64 px` threshold. **GLCM/Gabor/FFT
> periodicity computed at 28×28 are recorded but must be treated as unreliable.**
> Reliable texture descriptors are only available from the 142×142 batik subset,
> Batik_Nitik_960, and NeuralLoom — none of which could be fully downloaded here.

Batik crops show high edge density (0.25) and low symmetry; NeuralLoom shows low
edge density (0.05) with larger coherent regions — consistent with the domain gap.

---

## 11. Motif / Semantic Analysis (Part I)

**Batik_Lasem** — 5 metadata-backed motif classes [OBS], imbalanced 864→1373
(1.6×). Class counts, resolution availability, and origin relationships in
`class_distribution.csv`, `batik_lasem_motif_x_origin.csv`. Within-class visual
diversity is **bounded by 8–10 origins per class** [CMP].

**NeuralLoom** — the *only* defensible labels are the **folder-derived**
`Mekhela` vs `Saree` grouping [OBS]; there are **no dataset-provided annotations**
[OBS]. Finer textile-level grouping is possible via the source-image / crop-folder
structure, but any semantic garment taxonomy beyond Mekhela/Saree is **[UNK]**.

---

## 12. Metadata Analysis (Part J)

`metadata_motifs.csv` [OBS]: semicolon-delimited, UTF-8-BOM, 5,860 rows, 10 columns
(`filename, motif_name, origin_images, Origin_images_Type, Source Workshop,
year_collected, image_type, image_resolution, color_profile, annotator_initials`).

| Check | Result | Evidence |
|---|---|---|
| Duplicate filenames | 0 | [CMP] |
| filename prefix ↔ motif_name agreement | 100 % | [CMP] |
| Distinct origin images | **39** (verifies the "39 origins" claim) | [CMP] |
| Origin_images_Type | Textile (+ few variants) | `batik_lasem_origintype_meta.csv` |
| color_profile | 100 % sRGB | [OBS] |
| image_type | 100 % `raw (crop)` | [OBS] |
| Every metadata filename maps to a real image? | **[UNK]** — only Gunung-Ringgit images present here to cross-check (those matched) | partial [OBS] |

The 27-page `Short Description…pdf` documents the **39 origin cloths** narratively
(title, symbolism, size, workshop, technique) [OBS]. These are **descriptive texts,
not aesthetic scores** — must not be used as aesthetic labels.
Machine-readable: `metadata_summary.csv`, `metadata_audit.json`, the distribution
and cross-tab CSVs.

---

## 13. NeuralLoom Deep Audit (Part L)

| Item | Here [OBS/CMP] | User-reported [USR] |
|---|---|---|
| Mekhela source images | 33 (downloaded, verified) | 33 |
| Crops per Mekhela source | **exactly 5** (verified on pulled sources; folder named after source) | 5 |
| Mekhela crops total | 33 pulled (of 165) | 165 |
| Saree images | not fully pulled (throttled) | 155 in 154 folders |
| Folder with 2 images | **not located here** — do not assume duplicate | 1 folder, 2 images |
| Exact-dup audit | not reproduced | 346 unique / 7 dup copies of `IMG_20180824_200147.jpg` |

Lineage verified visually (`figures/13_neuralloom_lineage_5crops.png`): the 5 crops
are **different regions of the same garment** — source-linked, non-duplicate,
non-independent. The "folder with 2 images" and the exact-dup count **require local
verification** and must **not** be assumed to be duplicates. [flagged]

---

## 14. Batik_Lasem Deep Audit (Part K)

- **Class balance** [OBS]: Latohan 1373, Seritan 1328, Nyuk Pitu 1291, Kricak 1004,
  Gunung Ringgit 864 (imbalance ratio 1.59).
- **Workshop distribution** [OBS]: Pesona Canting 2253, KUB Srikandi 1438, Pusaka
  Beruang 747, VJ Basiroen 682, Sekar Kencana 444, Katrin Bee 166, Lumintu 130.
- **Year** [OBS]: 2022 = 5448, 2021 = 412 (only Gunung Ringgit spans both).
- **Origin-image distribution / leakage** [CMP]: 39 origins; crops-per-origin
  min 7, median 105, mean 150, **max 697**; largest single origin = 52.5 % of
  Seritan, 47.8 % of Gunung Ringgit.
- **6 origins span multiple motif classes** (e.g. `Isen Cloth 4.jpg` → 3 classes)
  [CMP] → the split unit must be the **origin image globally**, not per class.
- Matrices delivered: `batik_lasem_motif_x_origin.csv`, `_x_workshop.csv`,
  `_x_year.csv`.

**Source-image leakage conclusion [CMP/REC]:** multiple motif crops demonstrably
come from the same origin photograph; random or class-stratified crop splitting is
invalid. Use the **origin-level split** in `split_manifest.csv`.

---

## 15. Cross-Dataset Compatibility (Part N)

`outputs/cross_dataset_compatibility.csv`. Verdicts [CMP where color/scale cited,
else INF]:

| A | B | Verdict | Why |
|---|---|---|---|
| Batik_Lasem | Batik_Nitik_960 | **Partially compatible** | both flat batik motifs; but 28 px vs 24 MP scale gap + Nitik rotation-aug |
| Batik_Lasem | NeuralLoom | **Incompatible (raw joint)** | 28 px isolated motifs vs multi-MP garment photos; color stats differ sharply |
| Batik_Nitik_960 | NeuralLoom | **Incompatible (raw joint)** | studio swatch vs field garment |
| Batik_Lasem | Original Images | **Complementary** | originals are the *sources* of the crops |
| Batik_* | DeepFashion2/3D | **Incompatible / pending** | fashion garments vs textile motifs |

**Do not concatenate all datasets into one raw training pool.** [REC]

---

## 16. Aesthetic Attribute Availability (Part O)

`outputs/aesthetic_attribute_availability.csv`. Summary:

| Category | Computable here | Reliability | Needs annotation |
|---|---|---|---|
| Color (hue entropy, saturation, LAB, colorfulness, palette) | **YES** | med–high | No |
| Composition (symmetry, spatial entropy, balance) | **YES** | med | No |
| Texture (GLCM/LBP/Gabor/FFT) | **PARTIAL** | **LOW at 28×28** | No |
| Human aesthetic preference score | **NO** | — | **YES** |
| Learned aesthetic embedding | **EXTERNAL only** | — | external model |

**Four-way distinction enforced:** (A) objective visual attributes = **available**
[CMP]; (B) learned aesthetic representation = only via **external** pretrained model;
(C) human aesthetic preference = **not present** [OBS]; (D) aesthetic score = **must
not be fabricated** [OBS]. Per-image descriptors in `aesthetic_features.csv`;
summary in `aesthetic_feature_summary.csv`.

---

## 17. Aesthetic Supervision Strategy (Part P)

| Option | Data required | Validity | Cost | Feasibility |
|---|---|---|---|---|
| **1. External pretrained aesthetic model** (NIMA / LAION-aesthetic / CLIP-based) | none new | valid **iff** reported as external, domain-shift acknowledged | low | **High** |
| 2. Small human-rated textile set | new annotation (rubric, raters) | highest for "aesthetic" claims | high | medium |
| 3. Weak/self-supervised descriptors | computable now [CMP] | valid as *objective* proxies, not "preference" | low | high |

**[REC] Recommended:** combine **Option 3 (now)** as objective conditioning +
**Option 2 (small, later)** to obtain genuine human aesthetic supervision for the
paper's central claim; use **Option 1** only as an explicitly-external auxiliary
signal. Never label descriptor values as "aesthetic scores."

---

## 18. GAN Data Requirements (Parts Q/R)

`outputs/gan_data_requirements.csv` maps each component → required input → source →
**actual availability** → preprocessing → condition → limitation. Highlights:

- Pattern/Generator/basic Discriminator: **available** from Batik_Lasem (dedup,
  28→64). 
- Multi-scale (≥64 px): **partial** — only 911 hi-res batik + (un-pulled) Nitik.
- Aesthetic encoder/aux head: descriptors **available**; human scores **missing**;
  external model = external.
- Garment/Visualization: 2D partial (NeuralLoom); **3D pending** (DeepFashion3D).

---

## 19. Proposed GAN Data Mapping (Part S)

`outputs/dataset_role_mapping.csv`. Verified, not blindly accepted:

| Dataset | Pattern | Aesthetic | Conditioning | Garment ctx | Visualization | Eval |
|---|---|---|---|---|---|---|
| Batik_Lasem | **primary** | proxy | motif/origin/workshop/year | no | no | motif fidelity |
| Batik_Nitik_960 | yes (Nitik) | proxy | 60 categories | no | no | classification |
| Sarimbit_120 | TBD | proxy | [UNK] | no | no | TBD |
| Original Images | global layout ctx | descriptive text | 39 cloths | partial | no | qualitative |
| NeuralLoom | secondary | proxy | Mekhela/Saree | **yes** | 2D | domain-shift |
| DeepFashion2 | no | no | landmarks* | structure* | try-on* | external* |
| DeepFashion3D | no | no | 3D* | 3D garment* | **3D*** | external* |
*pending access.

---

## 20. Training Strategy (Part T)

`outputs/gan_data_plan.json` (5 staged phases). Per stage: dataset, resolution,
labels, split unit, augmentation, normalization, leakage prevention.

1. **Pattern representation** — Batik_Lasem all_motifs (deduped), 28→64,
   origin-level split.
2. **Conditional pretraining** — +Batik_Nitik_960 (rotations grouped), 64,
   category/source split.
3. **Aesthetic conditioning (proxy)** — computed descriptors (+optional external).
4. **Cross-domain adaptation** — NeuralLoom, ≥128, Mekhela-source split.
5. **Garment visualization** — NeuralLoom 2D now; 3D only if DeepFashion3D accessible.

**Hard rules:** dedup representations; source-aware splits only; never mix batik
crops with garment photos in one raw pool; descriptors ≠ human aesthetics; no 3D
claim without DeepFashion3D.

---

## 21. Baselines (Part X) — [REC]

| Baseline | Suitable? | Resolution | Data need | Role |
|---|---|---|---|---|
| DCGAN | yes (28–64) | low | modest | lower bound at native 28×28 |
| Conditional GAN (cGAN/ACGAN) | yes | 64 | class labels ✓ | class-conditional reference |
| WGAN-GP | yes | 64–128 | modest | stability baseline on imbalanced data |
| StyleGAN2(-ADA) | **only with caution** | ≥128 | **ADA needed** (few real sources) | high-fidelity ref *iff* enough independent sources |
| Textile/pattern-specific GAN | if justified by lit review | — | — | domain baseline |

StyleGAN2 is **not** recommended purely for popularity: with **≈39 independent
Batik_Lasem sources**, only StyleGAN2-**ADA** (limited-data) is defensible, and even
then diversity is source-bounded. [REC]

---

## 22. Evaluation Metrics (Part Y) — [REC]

| Metric | Meaningful for textile? | Note |
|---|---|---|
| FID | yes, with caution | needs enough reals; report per-class; sensitive at 28×28 |
| KID | **preferred at small n** | unbiased for small samples |
| LPIPS diversity | yes | intra-condition diversity / mode collapse |
| SSIM/PSNR | only for reconstruction | **not** for unconditional realism |
| Inception Score | weak | ImageNet features ill-suited to batik |
| CLIP similarity | conditional adherence | mark CLIP as external |
| Class/condition consistency | **yes** | train a motif classifier; measure adherence |
| Color/palette consistency | **yes** | compare descriptor distributions |
| Pattern/texture similarity | yes (≥64 px) | GLCM/FFT distance; unreliable at 28 |
| Aesthetic alignment | proxy or external | must state which |

**Textile-specific [REC]:** motif-classifier consistency, palette-distribution
distance, periodicity/repetition match, and source-held-out generalization.

---

## 23. Ablation Plan (Part Z) — [REC]

`Baseline cGAN` → +multi-scale → +aesthetic conditioning (proxy) → +multi-condition
fusion → +aesthetic-aware discriminator → +pattern-consistency → +garment
visualization. For each: component removed, expected effect, metric, research
question. **Only components the data supports are marked implementable now**
(garment-viz 3D = pending; aesthetic-preference head = needs human labels/external).

---

## 24. Dataset Limitations

- **≈39 independent Batik_Lasem sources** despite 5,860 crops → low true diversity;
  coarse splits (the biggest origin alone can dominate a split). [CMP]
- 28×28 dominant resolution → texture descriptors unreliable; detail unrecoverable.
- Batik_Nitik_960 rotation augmentation inflates apparent size 4×. [OBS]
- No human aesthetic labels anywhere. [OBS]
- Sarimbit_120 count unverified. [UNK]
- NeuralLoom full census (353) and dup audit unverified here. [USR]
- DeepFashion2/3D unavailable → garment structure & 3D viz unverified. [UNK]

---

## 25. Research Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Source leakage inflates metrics | **High** | origin-level split (`split_manifest.csv`) |
| "Aesthetic" claim unsupported by data | **High** | objective proxies + human study + label external models |
| Over-claiming 3D visualization | High | gate on DeepFashion3D access |
| Texture features trusted at 28 px | Medium | reliability flag; use ≥64 px only |
| Rotation-aug treated as real diversity (Nitik) | Medium | group rotations; dedup |
| Domain mixing degrades training | Medium | staged, domain-separated training |

---

## 26. Final Dataset Preparation Plan (Part AC)

Per dataset: integrity → **store provenance (hashes, origin, metadata) BEFORE any
transform** → dedup (`28x28 ⊆ all_motifs`; Nitik rotations; NeuralLoom source-links)
→ source grouping → metadata reconciliation → resolution standardization → color
standardization (strip EXIF/ICC; sRGB→linear as needed) → crop/pad → augmentation
(that does not break source grouping) → feature extraction (`aesthetic_features.csv`)
→ conditioning-vector creation → **source-level train/val/test split** → training
shards. No irreversible transform before provenance is persisted.

---

## 27. Final Recommendations — Executive Decision (Part AG)

| # | Decision | Answer |
|---|---|---|
| A | Use which datasets | Batik_Lasem (primary), Batik_Nitik_960 (secondary, dedup rotations), NeuralLoom (domain/appearance) |
| B | Exclude | DeepFashion2/3D until access; Sarimbit_120 until inventoried |
| C | Pretraining | Batik_Lasem all_motifs (deduped) → then +Nitik_960 |
| D | Aesthetic feature extraction | all image datasets (objective descriptors, `aesthetic_features.csv`) |
| E | Conditional generation | Batik_Lasem (motif+origin+palette+density); Nitik categories |
| F | Garment visualization | NeuralLoom 2D now; DeepFashion3D **only if** access granted |
| G | Missing data | full images for Nitik/Sarimbit/Original/NeuralLoom-saree; DeepFashion2/3D |
| H | Manually annotate | a small human textile-aesthetic set (Option 2) |
| I | From external model | aesthetic score proxy (NIMA/LAION) — **label as external** |
| J | Do NOT fabricate | aesthetic scores, labels, counts, 3D capability |
| K | Resolution per stage | 28→64 (batik base), 128+ (multi-scale/NeuralLoom) |
| L | Safest split | **origin-image-level global** (Batik), source/category (Nitik), Mekhela-source (NeuralLoom) — see `split_manifest.csv` |
| M | Next steps | (1) pull full datasets locally & re-run this pipeline; (2) verify Nitik/Sarimbit/NeuralLoom counts; (3) obtain DeepFashion access; (4) design human-aesthetic mini-study; (5) build deduped, source-split shards |

---

### Data-flow feasibility (the "most important" requirement)

| Arrow | Data required | Have it now? | How obtained | Tag |
|---|---|---|---|---|
| DATA → FEATURES | pixels + metadata | Batik(94)+NL(66)+meta(5860) | done here (sample) | [OBS/CMP] |
| FEATURES → AESTHETIC REP. | objective descriptors | yes (proxy) | `aesthetic_features.csv` | [CMP] |
| AESTHETIC REP. → CONDITIONING | descriptor/label vectors | partial (no human labels) | proxies + external/human study | [CMP]+[REC] |
| CONDITIONING → GENERATOR | conditioned exemplars | yes (Batik) | staged training | [OBS]+[REC] |
| GENERATOR → DISCRIMINATOR | real multi-res images | partial (hi-res scarce) | pull 142/Nitik | [OBS]+[UNK] |
| DISCR. → PATTERN SYNTHESIS | adversarial + pattern-consistency targets | partial (texture unreliable @28) | ≥64 px data | [CMP]+[UNK] |
| PATTERN → GARMENT VIS. | garment/3D | 2D partial; 3D none | NeuralLoom / DeepFashion3D(pending) | [UNK] |
| VIS. → EVALUATION | held-out sources + metrics | design ready | KID/FID/consistency, source-held-out | [REC] |

---

*All machine-readable artefacts are in `eda_v2/outputs/`; figures in
`eda_v2/figures/`; reproduce with `metadata_audit.py`, `image_audit.py`,
`generate_figures.py`, `build_plans.py`. Image-derived numbers are sample-based
under the documented Google-Drive enumeration/throttling limits and must be
re-run on the full local dataset before publication.*
