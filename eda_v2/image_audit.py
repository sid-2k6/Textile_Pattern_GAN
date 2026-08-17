#!/usr/bin/env python3
"""
Research-grade image audit pipeline (Parts B, C, D, E, F, G, H, O, U, V).

Scans one or more (dataset_name, root_path) pairs and computes, PER IMAGE:
  - inventory (path, ext, size, format, mode, channels)
  - integrity (corrupt flag)
  - exact hashes (md5, sha256) and perceptual hashes (aHash, dHash, pHash)
  - geometry (w, h, aspect)
  - quality (brightness, contrast, saturation, hue stats, RGB, LAB, grayscale
    entropy, edge density, sharpness / Laplacian variance, clipping, exposure)
  - texture (GLCM contrast/homogeneity/energy/correlation, LBP uniformity/entropy,
    Gabor magnitude stats, FFT spectral fineness, symmetry, periodicity)
  - lineage (dataset, semantic group, representation, class, source_id, role)

IMPORTANT (anti-hallucination):
  * Everything here is COMPUTED from actual pixels of images physically present
    on disk. It is NOT a full-dataset census: image acquisition from the source
    Google Drive is subject to a ~50-file/folder enumeration cap and per-IP
    download throttling, so counts here are SAMPLE counts for large folders.
    The `is_sample` field in dataset_inventory.csv flags this.
  * `texture_reliable` is False when min(w,h) < 64: GLCM/Gabor/FFT periodicity
    are not trustworthy at 28x28. Such values are still recorded but must not be
    treated as reliable descriptors.
"""
import os, sys, json, hashlib, glob, argparse, warnings
import numpy as np
import pandas as pd
from PIL import Image
warnings.filterwarnings("ignore")
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, canny
from skimage.color import rgb2lab, rgb2hsv, rgb2gray
from skimage.measure import shannon_entropy
from skimage.filters import gabor
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
TEXTURE_MIN = 64        # below this, texture descriptors are unreliable

# ----------------------------- hashing -----------------------------------
def file_hashes(path):
    md5 = hashlib.md5(); sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk); sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()

def ahash(gray8):
    s = np.asarray(Image.fromarray(gray8).resize((8, 8)))
    return (s > s.mean()).flatten()

def dhash(gray8):
    s = np.asarray(Image.fromarray(gray8).resize((9, 8)))
    return (s[:, 1:] > s[:, :-1]).flatten()

def phash(gray8):
    s = np.asarray(Image.fromarray(gray8).resize((32, 32)), dtype=np.float32)
    from scipy.fftpack import dct
    d = dct(dct(s, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return (d > np.median(d)).flatten()

def bits_to_hex(bits):
    return "%016x" % int("".join("1" if b else "0" for b in bits), 2)

# ----------------------------- lineage -----------------------------------
def classify(dataset, root, path):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    parent = parts[-2] if len(parts) >= 2 else ""
    fname = parts[-1]
    info = dict(dataset=dataset, rel_path=rel, folder=parent, filename=fname,
                representation="", motif_class="", source_id="", role="unknown",
                semantic_group="")
    if dataset == "Batik_Lasem":
        # representation folder
        rep = next((p for p in parts if p in
                    ("28x28_images", "142x142_images") or p.startswith("all_motifs")), "")
        info["representation"] = rep
        # motif class from filename prefix
        pref = fname.split("_")[0].lower()
        cmap = {"gunungringgit": "Gunung Ringgit", "kricak": "Kricak / Watu Pecah",
                "latohan": "Latohan", "nyukpitu": "Nyuk Pitu", "seritan": "Seritan"}
        info["motif_class"] = cmap.get(pref, "")
        info["semantic_group"] = info["motif_class"] or "Original Images"
        info["role"] = "crop"           # metadata: 100% raw (crop)
        info["source_id"] = ""          # filled later from metadata (origin_images)
    elif dataset == "NeuralLoom":
        low = rel.lower()
        if "mekhela_cropped" in low:
            info["role"] = "crop"
            info["representation"] = "mekhela_crop"
            info["source_id"] = parent      # crop folder is named after source image
            info["semantic_group"] = "Mekhela"
        elif "mekhela chador" in low:
            info["role"] = "source"
            info["representation"] = "mekhela_source"
            info["source_id"] = fname
            info["semantic_group"] = "Mekhela"
        elif "saree" in low or "new folder" in low:
            info["role"] = "source"
            info["representation"] = "saree_body"
            info["source_id"] = parent
            info["semantic_group"] = "Saree"
    else:
        info["semantic_group"] = dataset
        info["source_id"] = fname
    return info

# ----------------------------- per-image ----------------------------------
def analyze(path, dataset, root, meta_origin):
    row = classify(dataset, root, path)
    row["path"] = path
    row["ext"] = os.path.splitext(path)[1].lower()
    row["filesize_bytes"] = os.path.getsize(path)
    try:
        im = Image.open(path); im.load()
    except Exception as e:
        row["corrupt"] = True; row["error"] = str(e); return row
    row["corrupt"] = False
    row["format"] = im.format
    row["mode"] = im.mode
    w, h = im.size
    row.update(width=w, height=h, pixels=w * h,
               aspect_ratio=round(w / h, 4) if h else np.nan)
    rgb = im.convert("RGB")
    a_full = np.asarray(rgb)
    row["channels"] = a_full.shape[2] if a_full.ndim == 3 else 1
    # Downscale a WORKING copy for heavy pixel statistics so multi-megapixel
    # NeuralLoom photos stay tractable. Geometry/filesize/hashes use the original.
    MAXDIM = 384
    if max(w, h) > MAXDIM:
        scale = MAXDIM / max(w, h)
        work = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        work = rgb
    a = np.asarray(work).astype(np.float32)
    # metadata blocks
    row["metadata_blocks"] = sum(k in im.info for k in
                                 ("exif", "photoshop", "xmp", "icc_profile"))
    # hashes
    md5, sha = file_hashes(path)
    gray8 = np.asarray(work.convert("L"))   # working-resolution grayscale
    row.update(md5=md5, sha256=sha,
               ahash=bits_to_hex(ahash(gray8)),
               dhash=bits_to_hex(dhash(gray8)),
               phash=bits_to_hex(phash(gray8)))
    # ---- quality / color ----
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    gray = rgb2gray(a / 255.0)
    hsv = rgb2hsv(a / 255.0)
    lab = rgb2lab(a / 255.0)
    row.update(
        mean_r=float(R.mean()), mean_g=float(G.mean()), mean_b=float(B.mean()),
        std_r=float(R.std()), std_g=float(G.std()), std_b=float(B.std()),
        brightness=float(a.mean()), brightness_std=float(a.std()),
        contrast=float(gray.std()),
        mean_hue=float(hsv[..., 0].mean()), std_hue=float(hsv[..., 0].std()),
        mean_saturation=float(hsv[..., 1].mean()),
        mean_value=float(hsv[..., 2].mean()),
        lab_L=float(lab[..., 0].mean()), lab_a=float(lab[..., 1].mean()),
        lab_b=float(lab[..., 2].mean()),
        gray_entropy=float(shannon_entropy(gray8)),
        clip_low_frac=float((gray8 <= 2).mean()),
        clip_high_frac=float((gray8 >= 253).mean()),
    )
    # hue entropy (16-bin)
    hh, _ = np.histogram(hsv[..., 0], bins=16, range=(0, 1), density=True)
    hh = hh[hh > 0]
    row["hue_entropy"] = float(-(hh * np.log2(hh)).sum() / 4.0) if hh.size else 0.0
    # colorfulness (Hasler & Susstrunk 2003)
    rg = R - G; yb = 0.5 * (R + G) - B
    row["colorfulness"] = float(np.sqrt(rg.std()**2 + yb.std()**2)
                                + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))
    # sharpness / blur (variance of Laplacian)
    row["sharpness_lapvar"] = float(ndimage.laplace(gray).var())
    # edge density (Canny)
    try:
        row["edge_density"] = float(canny(gray, sigma=1.0).mean())
    except Exception:
        row["edge_density"] = np.nan
    # symmetry
    row["symmetry_h"] = float(1 - np.abs(gray - gray[:, ::-1]).mean())
    row["symmetry_v"] = float(1 - np.abs(gray - gray[::-1, :]).mean())
    # spatial entropy: entropy of edge distribution over 4x4 grid
    try:
        e = canny(gray, sigma=1.0).astype(float)
        gh = np.array_split(e, 4, 0); cells = []
        for band in gh: cells += [c.mean() for c in np.array_split(band, 4, 1)]
        cells = np.array(cells); cells = cells / (cells.sum() + 1e-9)
        cells = cells[cells > 0]
        row["spatial_entropy"] = float(-(cells * np.log2(cells)).sum()) if cells.size else 0.0
        row["pattern_density"] = float(e.mean())
    except Exception:
        row["spatial_entropy"] = np.nan; row["pattern_density"] = np.nan

    # ---- texture (reliability-flagged) ----
    reliable = min(w, h) >= TEXTURE_MIN
    row["texture_reliable"] = bool(reliable)
    try:
        q = (gray8 // 8).astype(np.uint8)   # 32 levels
        glcm = graycomatrix(q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=32, symmetric=True, normed=True)
        for p in ("contrast", "homogeneity", "energy", "correlation"):
            row[f"glcm_{p}"] = float(graycoprops(glcm, p).mean())
    except Exception:
        for p in ("contrast", "homogeneity", "energy", "correlation"):
            row[f"glcm_{p}"] = np.nan
    try:
        lbp = local_binary_pattern(gray8, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        hist = hist[hist > 0]
        row["lbp_entropy"] = float(-(hist * np.log2(hist)).sum())
        row["lbp_uniformity"] = float((hist**2).sum())
    except Exception:
        row["lbp_entropy"] = np.nan; row["lbp_uniformity"] = np.nan
    try:
        mags = []
        for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
            fr, fi = gabor(gray, frequency=0.3, theta=theta)
            mags.append(np.sqrt(fr**2 + fi**2).mean())
        row["gabor_mean"] = float(np.mean(mags))
        row["gabor_std"] = float(np.std(mags))
    except Exception:
        row["gabor_mean"] = np.nan; row["gabor_std"] = np.nan
    # FFT spectral fineness: high-freq energy ratio
    try:
        F = np.fft.fftshift(np.abs(np.fft.fft2(gray - gray.mean())))
        cy, cx = np.array(F.shape) // 2
        yy, xx = np.ogrid[:F.shape[0], :F.shape[1]]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        rmax = r.max()
        hi = F[r > 0.5 * rmax].sum(); tot = F.sum() + 1e-9
        row["fft_highfreq_ratio"] = float(hi / tot)
    except Exception:
        row["fft_highfreq_ratio"] = np.nan
    return row


def hamming_hex(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="semicolon list of name=path pairs")
    ap.add_argument("--metadata",
                    default="/projects/sandbox/data/_metadata/metadata_motifs.csv")
    ap.add_argument("--near-dup-threshold", type=int, default=6)
    args = ap.parse_args()

    # metadata origin lookup (Batik_Lasem)
    meta_origin = {}
    if os.path.exists(args.metadata):
        m = pd.read_csv(args.metadata, sep=";", dtype=str,
                        encoding="utf-8-sig", keep_default_na=False)
        m = m.loc[:, [c for c in m.columns if not c.startswith("Unnamed") and c.strip()]]
        m.columns = [c.strip() for c in m.columns]
        meta_origin = dict(zip(m["filename"].str.strip(),
                               m["origin_images"].str.strip()))

    pairs = []
    for tok in args.datasets.split(";"):
        if "=" in tok:
            name, p = tok.split("=", 1); pairs.append((name.strip(), p.strip()))

    rows = []
    ds_inv = []
    for name, root in pairs:
        paths = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                 if os.path.isfile(p)]
        imgs = [p for p in paths if p.lower().endswith(IMG_EXT)]
        nonimgs = [p for p in paths if not p.lower().endswith(IMG_EXT)]
        exts = {}
        for p in paths:
            e = os.path.splitext(p)[1].lower(); exts[e] = exts.get(e, 0) + 1
        nfolders = len({os.path.dirname(p) for p in paths})
        ds_inv.append(dict(dataset=name, root=root, accessible=True,
                           total_files=len(paths), image_files=len(imgs),
                           nonimage_files=len(nonimgs), folder_count=nfolders,
                           extensions=json.dumps(exts),
                           is_sample=True))
        for p in imgs:
            r = analyze(p, name, root, meta_origin)
            if name == "Batik_Lasem" and not r.get("source_id"):
                r["source_id"] = meta_origin.get(r.get("filename", ""), "")
            rows.append(r)
        print(f"[{name}] files={len(paths)} images={len(imgs)} analysed")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "image_inventory.csv"), index=False)
    pd.DataFrame(ds_inv).to_csv(os.path.join(OUT, "dataset_inventory.csv"), index=False)

    good = df[~df.corrupt].copy() if "corrupt" in df else df.copy()

    # ---- lineage ----
    good[["dataset", "rel_path", "semantic_group", "representation",
          "motif_class", "source_id", "role", "width", "height", "md5"]]\
        .to_csv(os.path.join(OUT, "source_lineage.csv"), index=False)

    # ---- exact duplicate groups ----
    dupe_rows = []
    for h, grp in good.groupby("md5"):
        if len(grp) > 1:
            dupe_rows.append(dict(group_id=f"md5_{h[:10]}", n=len(grp), md5=h,
                                  files=" | ".join(grp["rel_path"]),
                                  datasets=" | ".join(sorted(grp["dataset"].unique())),
                                  handling="retain one canonical copy (B)"))
    pd.DataFrame(dupe_rows).to_csv(os.path.join(OUT, "duplicate_groups.csv"), index=False)

    # ---- near-duplicate groups (pHash Hamming within dataset) ----
    near = []
    for ds, grp in good.groupby("dataset"):
        recs = grp[["rel_path", "phash", "source_id", "width", "height"]].to_dict("records")
        used = set()
        for i in range(len(recs)):
            if i in used: continue
            cluster = [recs[i]]
            for j in range(i + 1, len(recs)):
                if j in used: continue
                if hamming_hex(recs[i]["phash"], recs[j]["phash"]) <= args.near_dup_threshold:
                    cluster.append(recs[j]); used.add(j)
            if len(cluster) > 1:
                used.add(i)
                same_src = len({c["source_id"] for c in cluster}) == 1
                near.append(dict(
                    group_id=f"{ds}_nd_{i}", dataset=ds, n=len(cluster),
                    files=" | ".join(c["rel_path"] for c in cluster),
                    same_source=same_src,
                    handling=("treat as source-linked views (C)" if same_src
                              else "needs manual inspection (E)")))
    pd.DataFrame(near).to_csv(os.path.join(OUT, "near_duplicate_groups.csv"), index=False)

    # ---- quality / resolution / color / texture / aesthetic tables ----
    qcols = ["dataset","rel_path","width","height","aspect_ratio","mode","channels",
             "format","filesize_bytes","brightness","brightness_std","contrast",
             "mean_saturation","clip_low_frac","clip_high_frac","sharpness_lapvar",
             "edge_density","gray_entropy","metadata_blocks","corrupt"]
    df[[c for c in qcols if c in df.columns]].to_csv(
        os.path.join(OUT, "image_quality.csv"), index=False)

    res = good.groupby(["dataset","width","height"]).size().reset_index(name="count")
    res.to_csv(os.path.join(OUT, "resolution_statistics.csv"), index=False)

    ccols = ["dataset","rel_path","mean_r","mean_g","mean_b","std_r","std_g","std_b",
             "mean_hue","std_hue","hue_entropy","mean_saturation","mean_value",
             "lab_L","lab_a","lab_b","colorfulness"]
    good[[c for c in ccols if c in good.columns]].to_csv(
        os.path.join(OUT, "color_statistics.csv"), index=False)

    tcols = ["dataset","rel_path","texture_reliable","glcm_contrast","glcm_homogeneity",
             "glcm_energy","glcm_correlation","lbp_entropy","lbp_uniformity",
             "gabor_mean","gabor_std","fft_highfreq_ratio","edge_density",
             "spatial_entropy","pattern_density","symmetry_h","symmetry_v"]
    good[[c for c in tcols if c in good.columns]].to_csv(
        os.path.join(OUT, "texture_features.csv"), index=False)

    # aesthetic feature table (Part U spec) + per-image availability
    acols = ["dataset","semantic_group","motif_class","source_id","rel_path",
             "width","height","texture_reliable",
             "mean_hue","hue_entropy","mean_saturation","mean_value","lab_L",
             "contrast","colorfulness","hue_entropy",
             "edge_density","gray_entropy","glcm_contrast","glcm_homogeneity",
             "glcm_energy","glcm_correlation","lbp_entropy","gabor_mean","gabor_std",
             "fft_highfreq_ratio","symmetry_h","symmetry_v","pattern_density",
             "spatial_entropy"]
    aes = good[[c for c in dict.fromkeys(acols) if c in good.columns]].copy()
    aes.insert(0, "image_id", range(len(aes)))
    aes.to_csv(os.path.join(OUT, "aesthetic_features.csv"), index=False)

    # aesthetic feature summary per dataset
    num = aes.select_dtypes(include=[np.number]).columns.drop("image_id", errors="ignore")
    summ = aes.groupby("dataset")[list(num)].agg(["mean","std","min","max"])
    summ.to_csv(os.path.join(OUT, "aesthetic_feature_summary.csv"))

    # cross-dataset stats
    cds = good.groupby("dataset").agg(
        n=("rel_path","count"),
        mean_w=("width","mean"), mean_h=("height","mean"),
        brightness=("brightness","mean"), contrast=("contrast","mean"),
        saturation=("mean_saturation","mean"), colorfulness=("colorfulness","mean"),
        edge_density=("edge_density","mean"), gray_entropy=("gray_entropy","mean"),
    ).reset_index()
    cds.to_csv(os.path.join(OUT, "cross_dataset_statistics.csv"), index=False)

    # class distribution (image-level, sample)
    good.groupby(["dataset","semantic_group"]).size().reset_index(name="count")\
        .to_csv(os.path.join(OUT, "class_distribution_images.csv"), index=False)

    # ---- feature correlation (Part V) ----
    feat = good.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["pixels","width","height"] if c in good], errors="ignore")
    feat = feat.dropna(axis=1, how="all")
    corr = feat.corr()
    corr.to_csv(os.path.join(OUT, "feature_correlation.csv"))

    summary = dict(
        datasets=[d["dataset"] for d in ds_inv],
        n_images_analysed=int(len(good)),
        n_corrupt=int(df["corrupt"].sum()) if "corrupt" in df else 0,
        n_exact_dup_groups=len(dupe_rows),
        n_near_dup_groups=len(near),
        resolutions=res.to_dict("records"),
        note="SAMPLE-BASED for large folders (gdown 50-file/folder cap + throttling)",
    )
    json.dump(summary, open(os.path.join(OUT, "image_audit_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
