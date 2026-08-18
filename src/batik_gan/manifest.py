"""
Canonical manifest + reproducible 50% subset + group-aware (origin) split for
Batik_Lasem. This module is intentionally TORCH-FREE (only pandas/numpy) so it
can run anywhere (incl. this repo's CI / a plain Colab cell) to regenerate the
committed manifests deterministically.

Canonical representation decision (see EDA / README):
  * The metadata file has exactly one row per motif *instance* (5,860 rows).
  * On disk each instance is duplicated across folders:
        all_motifs (N)      -> the UNION (every instance, at its NATIVE resolution)
        28x28_images        -> the 28x28 instances only
        142x142_images      -> the 142x142 instances only
    (all_motifs == 28x28_images ∪ 142x142_images ; the per-resolution folders are
     exact-MD5 subsets of all_motifs — verified in the EDA.)
  * Therefore the CANONICAL, non-redundant source = the `all_motifs` folder:
    one file per instance at the HIGHEST-quality (native) resolution available
    for that instance (142x142 for 911 instances, 28x28 for the other 4,949).
    We never pool the three folders (that would 2x–3x count the same instances).

Reproducibility: SEED controls the stratified 50% subset and the group split.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

SEED = 42

# metadata motif_name  ->  on-disk motif folder name (from the Drive structure)
MOTIF_FOLDER = {
    "Gunung Ringgit": "Gunung Ringgit",
    "Kricak / Watu Pecah": "Kricak_Watu Pecah",
    "Latohan": "Latohan",
    "Nyuk Pitu": "Nyuk Pitu",
    "Seritan": "Seritan",
}
BATIK_SUBDIR = os.path.join("Batik_Lasem", "motifs (isen-isen)")
MOTIF_ORDER = ["Gunung Ringgit", "Kricak / Watu Pecah", "Latohan", "Nyuk Pitu", "Seritan"]
META_COLUMNS = ["filename", "motif_name", "origin_images", "Origin_images_Type",
                "Source Workshop", "year_collected", "image_type",
                "image_resolution", "color_profile", "annotator_initials"]


# --------------------------------------------------------------------------- #
def load_metadata(metadata_csv: str) -> pd.DataFrame:
    """Load the semicolon-delimited, UTF-8-BOM metadata; strip cols/values."""
    m = pd.read_csv(metadata_csv, sep=";", dtype=str,
                    encoding="utf-8-sig", keep_default_na=False)
    m = m.loc[:, [c for c in m.columns if not c.startswith("Unnamed") and c.strip()]]
    m.columns = [c.strip() for c in m.columns]
    for c in m.columns:
        m[c] = m[c].str.strip()
    return m


def build_canonical_manifest(metadata_csv: str) -> pd.DataFrame:
    """One canonical row per instance, pointing at the all_motifs native file."""
    m = load_metadata(metadata_csv)
    counts = m["motif_name"].value_counts().to_dict()      # class totals -> folder name
    rows = []
    for _, r in m.iterrows():
        motif = r["motif_name"]
        folder = MOTIF_FOLDER.get(motif)
        if folder is None:
            raise ValueError(f"Unknown motif_name in metadata: {motif!r}")
        all_dir = f"all_motifs ({counts[motif]})"
        rel = os.path.join(BATIK_SUBDIR, folder, all_dir, r["filename"]).replace("\\", "/")
        rows.append(dict(
            filename=r["filename"],
            rel_path=rel,
            motif_name=motif,
            motif_folder=folder,
            all_motifs_dir=all_dir,
            origin_images=r["origin_images"],
            **{"Source Workshop": r.get("Source Workshop", "")},
            year_collected=r.get("year_collected", ""),
            image_type=r.get("image_type", ""),
            image_resolution=r.get("image_resolution", ""),   # native/canonical res
            color_profile=r.get("color_profile", ""),
            annotator_initials=r.get("annotator_initials", ""),
        ))
    df = pd.DataFrame(rows)
    df.insert(0, "instance_id", range(len(df)))
    return df


def stratified_subset(manifest: pd.DataFrame, frac: float = 0.5,
                      seed: int = SEED) -> pd.DataFrame:
    """Deterministic per-motif stratified subset (round to nearest, seed-fixed)."""
    parts = []
    for motif in MOTIF_ORDER:
        g = manifest[manifest.motif_name == motif].sort_values("filename")
        k = int(round(len(g) * frac))
        rng = np.random.default_rng(seed + MOTIF_ORDER.index(motif))
        idx = rng.choice(len(g), size=k, replace=False)
        parts.append(g.iloc[np.sort(idx)])
    out = pd.concat(parts).sort_values("instance_id").reset_index(drop=True)
    return out


def group_aware_split(subset: pd.DataFrame, test_frac: float = 0.2,
                      seed: int = SEED, n_candidates: int = 20000,
                      frac_lo: float = 0.18, frac_hi: float = 0.22):
    """Assign whole origin_images groups to train/test (no origin shared).

    Because the 39 origins are few, large and motif-specific, a purely greedy
    group split badly skews the motif distribution. Instead we run a *fully
    deterministic* randomized search (fixed `seed`): sample many candidate sets
    of test-origins whose combined size falls in [frac_lo, frac_hi]·N, require
    every motif present in both splits, and keep the candidate that MINIMISES the
    maximum per-motif train/test percentage gap. This preserves the motif
    distribution as closely as the grouping constraint allows.
    Returns (subset_with_split_col, report_dict).
    """
    sub = subset.copy()
    total = len(sub)
    # per-origin motif-count vectors (numpy, for fast scoring)
    origins = list(sub.groupby("origin_images"))
    names = [o for o, _ in origins]
    vecs = np.array([[int((g.motif_name == m).sum()) for m in MOTIF_ORDER]
                     for _, g in origins], dtype=float)      # (n_origins, 5)
    sizes = vecs.sum(axis=1)
    motif_tot = np.array([(sub.motif_name == m).sum() for m in MOTIF_ORDER], float)

    def gap_for(mask):
        test_vec = vecs[mask].sum(axis=0)
        train_vec = motif_tot - test_vec
        nt, ntr = test_vec.sum(), train_vec.sum()
        if nt == 0 or ntr == 0 or (test_vec == 0).any() or (train_vec == 0).any():
            return np.inf
        return float(np.max(np.abs(100 * train_vec / ntr - 100 * test_vec / nt)))

    rng = np.random.default_rng(seed)
    lo, hi = frac_lo * total, frac_hi * total
    n = len(names)
    best_mask, best_gap = None, np.inf
    for _ in range(n_candidates):
        perm = rng.permutation(n)
        mask = np.zeros(n, dtype=bool)
        tot = 0.0
        for i in perm:
            if tot + sizes[i] <= hi:
                mask[i] = True
                tot += sizes[i]
            if tot >= lo and rng.random() < 0.3:
                break
        if lo <= tot <= hi:
            g = gap_for(mask)
            if g < best_gap:
                best_gap, best_mask = g, mask.copy()

    moved = []
    if best_mask is None:      # fallback: smallest-origins-first greedy
        order = np.argsort(sizes)
        best_mask = np.zeros(n, dtype=bool); tot = 0.0
        for i in order:
            if tot < test_frac * total:
                best_mask[i] = True; tot += sizes[i]
        moved = ["fallback_greedy"]

    assign = {names[i]: ("test" if best_mask[i] else "train") for i in range(n)}
    sub["split"] = sub.origin_images.map(assign)
    train, test = sub[sub.split == "train"], sub[sub.split == "test"]

    def dist(df):
        return {m: int((df.motif_name == m).sum()) for m in MOTIF_ORDER}

    report = dict(
        n_train_groups=int((pd.Series(assign) == "train").sum()),
        n_test_groups=int((pd.Series(assign) == "test").sum()),
        train_samples=int(len(train)),
        test_samples=int(len(test)),
        test_fraction=round(len(test) / total, 4),
        train_motif_distribution=dist(train),
        test_motif_distribution=dist(test),
        train_motif_pct={m: round(100 * v / max(1, len(train)), 2) for m, v in dist(train).items()},
        test_motif_pct={m: round(100 * v / max(1, len(test)), 2) for m, v in dist(test).items()},
        origins_shared_between_train_test=[],   # guaranteed empty by construction
        multi_motif_origins=[o for o, g in sub.groupby("origin_images")
                             if g.motif_name.nunique() > 1],
        guarantee_moves=moved,
        max_abs_motif_pct_gap=None,
    )
    gap = max(abs(report["train_motif_pct"][m] - report["test_motif_pct"][m])
              for m in MOTIF_ORDER)
    report["max_abs_motif_pct_gap"] = round(gap, 2)
    return sub, report


def verify_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """True iff no origin_images value is shared across train and test."""
    return len(set(train.origin_images) & set(test.origin_images)) == 0


def build_all(metadata_csv: str, out_dir: str, frac: float = 0.5,
              test_frac: float = 0.2, seed: int = SEED):
    """Generate + save canonical manifest, 50% subset, and group-aware split."""
    os.makedirs(out_dir, exist_ok=True)
    canonical = build_canonical_manifest(metadata_csv)
    subset = stratified_subset(canonical, frac=frac, seed=seed)
    split, report = group_aware_split(subset, test_frac=test_frac, seed=seed)
    train = split[split.split == "train"].drop(columns=["split"])
    test = split[split.split == "test"].drop(columns=["split"])

    canonical.to_csv(os.path.join(out_dir, "batik_lasem_canonical_manifest.csv"), index=False)
    subset.to_csv(os.path.join(out_dir, "batik_lasem_50pct_manifest.csv"), index=False)
    train.to_csv(os.path.join(out_dir, "batik_lasem_50pct_train.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "batik_lasem_50pct_test.csv"), index=False)
    report["canonical_total"] = int(len(canonical))
    report["subset_total"] = int(len(subset))
    report["no_leakage"] = verify_no_leakage(train, test)
    report["seed"] = seed
    return canonical, subset, train, test, report


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("usage: python -m batik_gan.manifest "
              "<path/to/metadata motifs.csv> [out_dir=data/splits]")
        raise SystemExit(1)
    meta = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/splits"
    c, s, tr, te, rep = build_all(meta, out)
    print(json.dumps(rep, indent=2))
