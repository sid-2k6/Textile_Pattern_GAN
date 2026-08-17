#!/usr/bin/env python3
"""
Batik_Lasem metadata audit  (Parts J, K, M).

Operates on the COMPLETE, directly-observed metadata file
`metadata_motifs.csv` (semicolon-delimited, UTF-8-BOM). Because this file is
complete (not subject to the gdown 50-file/folder enumeration cap), every count
produced here is DIRECTLY OBSERVED / COMPUTED, not a sample.

Outputs (into outputs/):
  metadata_summary.csv            column-level profile
  class_distribution.csv          motif_name counts
  batik_lasem_origin_distribution.csv
  batik_lasem_workshop_distribution.csv
  batik_lasem_year_distribution.csv
  batik_lasem_motif_x_origin.csv  (leakage matrix)
  batik_lasem_motif_x_workshop.csv
  batik_lasem_motif_x_year.csv
  metadata_audit.json             headline verified facts
"""
import os, json, sys
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/projects/sandbox/data/_metadata/metadata_motifs.csv"

# --- load -----------------------------------------------------------------
raw = pd.read_csv(CSV, sep=";", dtype=str, encoding="utf-8-sig", keep_default_na=False)
# drop fully-empty trailing columns produced by trailing ';;'
raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed") and c.strip() != ""]]
raw.columns = [c.strip() for c in raw.columns]
n_rows = len(raw)
print(f"[metadata] rows (excl. header) = {n_rows}; columns = {list(raw.columns)}")

# --- column profile -------------------------------------------------------
prof = []
for c in raw.columns:
    s = raw[c]
    blank = (s.str.strip() == "").sum()
    prof.append(dict(column=c, dtype="string", n=n_rows,
                     non_blank=int(n_rows - blank), blank=int(blank),
                     n_unique=int(s.nunique()),
                     example=s[s.str.strip() != ""].iloc[0] if (s.str.strip() != "").any() else ""))
pd.DataFrame(prof).to_csv(os.path.join(OUT, "metadata_summary.csv"), index=False)

def dist(col, fname):
    if col not in raw.columns:
        print(f"[metadata] column '{col}' not present"); return None
    d = raw[col].str.strip().value_counts(dropna=False).rename_axis(col)\
                 .reset_index(name="count")
    d.to_csv(os.path.join(OUT, fname), index=False)
    return d

cls = dist("motif_name", "class_distribution.csv")
org = dist("origin_images", "batik_lasem_origin_distribution.csv")
wsp = dist("Source Workshop", "batik_lasem_workshop_distribution.csv")
yr = dist("year_collected", "batik_lasem_year_distribution.csv")
dist("image_resolution", "batik_lasem_resolution_meta.csv")
dist("color_profile", "batik_lasem_colorprofile_meta.csv")
dist("image_type", "batik_lasem_imagetype_meta.csv")
dist("annotator_initials", "batik_lasem_annotator_meta.csv")
dist("Origin_images_Type", "batik_lasem_origintype_meta.csv")

# --- cross matrices (leakage-relevant) ------------------------------------
def crosstab(a, b, fname):
    if a in raw.columns and b in raw.columns:
        ct = pd.crosstab(raw[a].str.strip(), raw[b].str.strip())
        ct.to_csv(os.path.join(OUT, fname))
        return ct
crosstab("motif_name", "origin_images", "batik_lasem_motif_x_origin.csv")
crosstab("motif_name", "Source Workshop", "batik_lasem_motif_x_workshop.csv")
crosstab("motif_name", "year_collected", "batik_lasem_motif_x_year.csv")

# --- leakage: crops per origin image --------------------------------------
per_origin = raw.groupby(raw["origin_images"].str.strip()).size()
per_origin_by_class = raw.groupby([raw["motif_name"].str.strip(),
                                   raw["origin_images"].str.strip()]).size()

# --- filename integrity ---------------------------------------------------
fn = raw["filename"].str.strip()
dup_fn = fn[fn.duplicated(keep=False)]
# does filename prefix agree with motif_name?
def prefix_ok(row):
    f = row["filename"].strip().lower().replace(" ", "")
    m = row["motif_name"].strip().lower().replace(" ", "").replace("/", "")
    # motif tokens: use first word of motif
    token = row["motif_name"].strip().split()[0].lower()
    return token[:4] in f
prefix_match = raw.apply(prefix_ok, axis=1).mean()

facts = {
    "source_file": os.path.basename(CSV),
    "delimiter": ";", "encoding": "utf-8-sig",
    "n_records_excl_header": int(n_rows),
    "columns": list(raw.columns),
    "motif_classes": cls.set_index("motif_name")["count"].to_dict() if cls is not None else {},
    "motif_class_total": int(cls["count"].sum()) if cls is not None else None,
    "n_unique_origin_images": int(org.shape[0]) if org is not None else None,
    "origin_images_min_crops": int(per_origin.min()),
    "origin_images_max_crops": int(per_origin.max()),
    "origin_images_median_crops": float(per_origin.median()),
    "origin_images_mean_crops": float(per_origin.mean()),
    "n_workshops": int(wsp.shape[0]) if wsp is not None else None,
    "workshops": wsp.set_index("Source Workshop")["count"].to_dict() if wsp is not None else {},
    "years": yr.set_index("year_collected")["count"].to_dict() if yr is not None else {},
    "n_duplicate_filenames": int(dup_fn.nunique()),
    "filename_prefix_matches_motif_frac": round(float(prefix_match), 4),
    "image_resolution_values": raw["image_resolution"].str.strip().value_counts().to_dict()
        if "image_resolution" in raw.columns else {},
    "color_profile_values": raw["color_profile"].str.strip().value_counts().to_dict()
        if "color_profile" in raw.columns else {},
    "image_type_values": raw["image_type"].str.strip().value_counts().to_dict()
        if "image_type" in raw.columns else {},
}
json.dump(facts, open(os.path.join(OUT, "metadata_audit.json"), "w"), indent=2)
print(json.dumps(facts, indent=2))
