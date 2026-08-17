#!/usr/bin/env python3
"""Emit strategy/plan machine-readable files (Parts N, R, S, T, Y, Z, AB, AC, AG).

Every field is either a directly-observed/computed fact from this audit or an
explicitly-labelled RECOMMENDATION/INFERENCE. Nothing here fabricates dataset
contents; pending datasets are marked as such.
"""
import os, json, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
META = os.environ.get("TEXTILE_METADATA_CSV",
                      "/projects/sandbox/data/_metadata/metadata_motifs.csv")

# ---------------------------------------------------------------- dataset readiness (Part AB)
readiness = [
 # dataset, access, evidence_level, images_here, quality, label, source_indep, resolution, aesthetic_superv, garment_relevance, gan_suitability, status, notes
 ["Batik_Lasem", "accessible", "metadata COMPLETE + 94 imgs (Gunung Ringgit) OBSERVED",
  "94 of 5860", "high (sRGB, 0 corrupt in sample)", "strong (metadata: motif+origin+workshop+year)",
  "LOW (39 origins -> 5860 crops; 1 origin=52.5% of a class)", "28x28 (4949) / 142x142 (911)",
  "none (no scores; descriptive PDF only)", "low (isolated motifs, no garment)", "GOOD for motif/pattern learning",
  "YELLOW", "Must dedup 28x28⊆all_motifs; source-aware split on origin_images."],
 ["Batik_Nitik_960", "accessible (not pulled here)", "PDF doc COMPLETE; images NOT downloaded",
  "0 of 960", "high (24MP studio capture per PDF)", "strong (60 categories x16)",
  "MEDIUM (60 pieces; 16/cat = 4 motifs x4 rotations => augmented, not independent)", "6024x4024 originals",
  "none", "low (fabric swatches)", "GOOD for Nitik pattern diversity",
  "YELLOW", "Rotational augmentation documented: 90/180/270. Deduplicate rotations for eval."],
 ["Batik_Nitik_Sarimbit_120", "accessible (not pulled here)", "name only; count UNVERIFIED",
  "0 of ~120?", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN (has 'Original images high-resolution')",
  "none", "low-med (Sarimbit = paired set context)", "UNKNOWN until inspected",
  "RED", "Do NOT assume 120. Inventory before use."],
 ["Original Images (Batik_Lasem origins)", "accessible (not pulled here)", "39 origins per metadata; 27-pg PDF descriptions",
  "0 of 39", "UNKNOWN (likely high-res cloth photos)", "these ARE the sources",
  "n/a (are the source units)", "UNKNOWN (full cloth photos)",
  "descriptive text only (NOT scores)", "medium (full patterned cloth)", "context/provenance, evaluation",
  "YELLOW", "39 origin cloths; use as source-split key and for global-pattern context."],
 ["NeuralLoom", "accessible (partial)", "66 imgs OBSERVED here; 353 total USER-REPORTED",
  "66 of ~353 (user)", "med (phone photos, variable)", "folder-derived only (Mekhela/Saree)",
  "MEDIUM (33 Mekhela sources x5 crops; saree per-folder)", "454-3023 px (observed range)",
  "none", "HIGH (worn/lay garment textiles)", "GOOD for apparel/textile appearance",
  "YELLOW", "5 crops/source verified; keep source-linked. Verify 353/dup audit locally."],
 ["DeepFashion2", "PENDING ACCESS", "zip archives present (train.zip/test-001.zip); NOT extracted",
  "0", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "none", "HIGH (garment structure/landmarks)",
  "TBD", "RED/PENDING", "Access/password pending. Do not fabricate stats."],
 ["DeepFashion3D", "PENDING ACCESS", "rar/zip (filtered_registered_mesh, point_cloud); NOT extracted",
  "0", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN (3D meshes/point clouds)", "none", "3D garment",
  "TBD", "RED/PENDING", "3D data; only source that could support true 3D visualization."],
]
pd.DataFrame(readiness, columns=[
 "dataset","access","evidence_level","images_available_here","data_quality","label_quality",
 "source_independence","resolution","aesthetic_supervision","garment_relevance",
 "gan_suitability","readiness","notes"]).to_csv(os.path.join(OUT,"dataset_readiness.csv"), index=False)

# ---------------------------------------------------------------- dataset role mapping (Part S)
roles = [
 ["Batik_Lasem","YES (motif/isen-isen)","proxy only (visual descriptors)","motif class, origin, workshop, year, resolution","NO","NO","held-out motif fidelity","Primary pattern/motif source (28->64px)."],
 ["Batik_Nitik_960","YES (Nitik patterns)","proxy only","60 categories","NO","NO","classification/retrieval baseline","Pattern diversity; handle rotation aug."],
 ["Batik_Nitik_Sarimbit_120","TBD (inspect)","proxy only","UNKNOWN","weak","NO","TBD","Verify before any use."],
 ["Original Images","context (global layout)","descriptive text (not scores)","39 cloth descriptions","NO","partial (cloth photos)","qualitative","Source lineage + global pattern context."],
 ["NeuralLoom","secondary (woven textile appearance)","proxy only","Mekhela/Saree (folder)","NO","YES (garment textile)","domain-shift test","Apparel/textile appearance; different domain."],
 ["DeepFashion2","NO (pending)","NO","landmarks/pose (if extracted)","garment structure","render/try-on","external generalization","PENDING; garment structure only if accessible."],
 ["DeepFashion3D","NO (pending)","NO","3D meshes/pointclouds","NO","3D garment try-on","external","PENDING; only 3D-visualization candidate."],
]
pd.DataFrame(roles, columns=[
 "dataset","pattern_learning","aesthetic_learning","conditioning_available","garment_context",
 "visualization","evaluation_role","recommended_usage"]).to_csv(os.path.join(OUT,"dataset_role_mapping.csv"), index=False)

# ---------------------------------------------------------------- GAN data requirements (Part Q/R)
gan_req = [
 ["Pattern Encoder","motif/pattern images","Batik_Lasem all_motifs (28/142)","AVAILABLE (metadata) / images SAMPLE here","dedup 28x28⊆all_motifs; resize","motif_class","28x28 limits fine texture; upscale cautiously"],
 ["Multi-scale Feature Encoder","same at >=64px","Batik_Lasem 142x142 subset (911) + Nitik 24MP","PARTIAL: only 911 hi-res batik; Nitik not pulled","tile/resize","-","Only 3/5 motifs have 142; Nitik has true hi-res"],
 ["Aesthetic Attribute Encoder","objective visual descriptors","computed from any image","COMPUTED (aesthetic_features.csv)","standardize; flag <64px","color/texture/symmetry vectors","descriptors != human aesthetic; texture unreliable <64px"],
 ["Condition Fusion","condition vectors","metadata (class/origin) + computed descriptors","AVAILABLE for Batik_Lasem","concat/embed","motif+palette+density","no human aesthetic label exists"],
 ["Generator","target-res exemplars","Batik_Lasem (28/64)","AVAILABLE","normalize [-1,1]","conditioning","start 28-64px"],
 ["Multi-scale Discriminator","real images multi-res","Batik_Lasem + Nitik","PARTIAL hi-res","pyramid","-","hi-res batik scarce"],
 ["Aesthetic Discriminator/Aux head","descriptor targets or pretrained aesthetic model","computed descriptors (weak) OR EXTERNAL NIMA/LAION-aesthetic","descriptors AVAILABLE; human scores MISSING","-","aesthetic reward","must mark external model as external"],
 ["Pattern Consistency","repetition/periodicity targets","FFT/GLCM descriptors","COMPUTED (reliable only >=64px)","-","periodicity","unreliable at 28x28"],
 ["Garment Visualization","garment images/3D","NeuralLoom (2D) / DeepFashion2 (2D) / DeepFashion3D (3D PENDING)","2D partial; 3D PENDING","texture mapping","-","true 3D needs DeepFashion3D access"],
]
pd.DataFrame(gan_req, columns=[
 "component","required_input","dataset_source","actual_availability","preprocessing",
 "potential_condition","limitation"]).to_csv(os.path.join(OUT,"gan_data_requirements.csv"), index=False)

# ---------------------------------------------------------------- aesthetic attribute availability (Part O)
aes_avail = [
 ["hue diversity / hue_entropy","all image datasets","YES","HSV hist entropy","medium","NO"],
 ["mean saturation / value","all","YES","HSV mean","high","NO"],
 ["colorfulness (Hasler-Susstrunk)","all","YES","std/mean of rg,yb","high","NO"],
 ["LAB statistics","all","YES","skimage rgb2lab","high","NO"],
 ["dominant palette","all (>=64px best)","YES","k-means on pixels","medium","NO"],
 ["edge density","all","YES","Canny","medium (noisy 28px)","NO"],
 ["GLCM (contrast/homogeneity/energy/corr)",">=64px","PARTIAL","graycomatrix","LOW at 28px","NO"],
 ["LBP statistics",">=64px","PARTIAL","local_binary_pattern","LOW at 28px","NO"],
 ["Gabor statistics",">=64px","PARTIAL","gabor bank","LOW at 28px","NO"],
 ["FFT periodicity/repetition",">=128px ideally","PARTIAL","2D FFT radial","LOW at 28px","NO"],
 ["symmetry (h/v)","all","YES","flip diff","medium","NO"],
 ["spatial entropy / balance","all","YES","grid edge entropy","medium","NO"],
 ["human aesthetic preference score","NONE","NO","requires human rating or external model","-","YES"],
 ["learned aesthetic embedding","NONE natively","EXTERNAL only","pretrained NIMA/CLIP-aesthetic","-","external model"],
]
pd.DataFrame(aes_avail, columns=[
 "attribute","dataset_scope","computable","method","reliability","requires_annotation"]
 ).to_csv(os.path.join(OUT,"aesthetic_attribute_availability.csv"), index=False)

# ---------------------------------------------------------------- cross-dataset compatibility (Part N)
compat = [
 ["Batik_Lasem","Batik_Nitik_960","PARTIALLY COMPATIBLE","both flat batik motifs; but 28px vs 24MP scale gap + Nitik rotation aug"],
 ["Batik_Lasem","NeuralLoom","INCOMPATIBLE (raw joint)","28px isolated motifs vs multi-MP garment photos; brightness 204 vs 129, saturation .10 vs .41 (OBSERVED)"],
 ["Batik_Nitik_960","NeuralLoom","INCOMPATIBLE (raw joint)","studio swatch vs field garment photo; scale/lighting differ"],
 ["Batik_Lasem","Original Images","COMPLEMENTARY","originals are the SOURCE cloths of the crops; use for context not mixing"],
 ["Batik_*","DeepFashion2/3D","INCOMPATIBLE domains / PENDING","fashion garments vs textile motifs; use only for garment structure/vis"],
]
pd.DataFrame(compat, columns=["dataset_a","dataset_b","verdict","reason"]
 ).to_csv(os.path.join(OUT,"cross_dataset_compatibility.csv"), index=False)

# ---------------------------------------------------------------- split manifest (Part M/T) — SPEC, source-aware
split_spec = {
 "principle": "SOURCE-LEVEL isolation. Never split source-linked images across train/val/test.",
 "Batik_Lasem": {
   "split_unit": "origin_images (39 units) — NOT individual crops",
   "reason": "one origin -> up to 697 crops (52.5% of a class); crop-level split leaks",
   "caveat": "only 39 origin units total (8-10/class) => very coarse; consider grouped CV",
   "recommended": {"train_origins": "~27", "val_origins": "~6", "test_origins": "~6",
                   "constraint": "stratify by motif_class; keep each origin wholly in one split"},
   "dedup_before_split": "drop 28x28_images copies (⊆ all_motifs, exact MD5 dup)"
 },
 "Batik_Nitik_960": {
   "split_unit": "fabric piece / motif base (before 90/180/270 rotation)",
   "reason": "16 imgs/category = 4 motifs x 4 rotations => rotations must stay together",
   "recommended": "split on the 60 categories or 240 motif-bases; keep rotations grouped"
 },
 "NeuralLoom": {
   "split_unit": "Mekhela source image (33) / Saree source folder",
   "reason": "5 crops per Mekhela source are source-linked views",
   "recommended": "keep all crops of a source in the same split"
 },
 "status": "SPEC ONLY — actual manifest requires full local dataset (throttled here)."
}
json.dump(split_spec, open(os.path.join(OUT,"split_manifest_spec.json"),"w"), indent=2)

# concrete partial split manifest for the 39 Batik_Lasem origins (from metadata)
try:
    m = pd.read_csv(META, sep=";",
                    dtype=str, encoding="utf-8-sig", keep_default_na=False)
    m = m.loc[:, [c for c in m.columns if not c.startswith("Unnamed") and c.strip()]]
    m.columns=[c.strip() for c in m.columns]
    for c in m.columns: m[c]=m[c].str.strip()
    # GLOBAL per-origin unit (39). 6 origins span multiple motif classes, so the
    # safe split unit is the ORIGIN IMAGE globally, never per (class,origin).
    per = m.groupby("origin_images").agg(
        n_crops=("filename","count"),
        motif_classes=("motif_name", lambda s: "|".join(sorted(s.unique()))),
        n_classes=("motif_name","nunique")).reset_index()
    per = per.sort_values("n_crops", ascending=False).reset_index(drop=True)
    # greedy balance: assign biggest origins round-robin to keep crop counts even
    tally={"train":0,"val":0,"test":0}; target={"train":0.8,"val":0.1,"test":0.1}
    rows=[]
    for _,r in per.iterrows():
        # choose split whose (current fraction) is furthest below its target
        tot=sum(tally.values())+1
        deficit={k:target[k]-tally[k]/tot for k in tally}
        split=max(deficit, key=deficit.get)
        tally[split]+=r.n_crops
        rows.append(dict(origin_images=r.origin_images, n_crops=int(r.n_crops),
                         n_motif_classes=int(r.n_classes), motif_classes=r.motif_classes,
                         split=split))
    sm=pd.DataFrame(rows)
    sm.to_csv(os.path.join(OUT,"split_manifest.csv"), index=False)
    print("split_manifest.csv:", len(sm), "GLOBAL origin units (39 expected); crops per split:")
    print(sm.groupby("split").n_crops.sum().to_string())
    print("multi-class origins:", int((sm.n_motif_classes>1).sum()))
except Exception as e:
    print("split manifest partial skipped:", e)

# ---------------------------------------------------------------- headline dataset_summary + gan_data_plan
dataset_summary = {
 "audited_here": ["Batik_Lasem (metadata COMPLETE; 94 imgs sample)","NeuralLoom (66 imgs sample)"],
 "documented_not_pulled": ["Batik_Nitik_960 (PDF)","Original Images (39 origins via metadata)","Batik_Nitik_Sarimbit_120 (unverified)"],
 "pending_access": ["DeepFashion2 (zip)","DeepFashion3D (rar/zip 3D)"],
 "batik_lasem_verified": {
   "records": 5860, "classes": {"Latohan":1373,"Seritan":1328,"Nyuk Pitu":1291,
     "Kricak / Watu Pecah":1004,"Gunung Ringgit":864},
   "origin_images": 39, "max_crops_from_one_origin": 697,
   "resolution_records": {"28 x 28":4949,"142 x 142":911},
   "workshops": 7, "years": {"2022":5448,"2021":412},
   "representation_relationship": "28x28_images ⊆ all_motifs (exact MD5 dup, OBSERVED on Gunung Ringgit); all_motifs = 28x28 ∪ 142x142 (INFERRED from counts)",
   "color_profile":"100% sRGB", "image_type":"100% raw (crop)"
 },
 "neuralloom_verified_here": {"mekhela_sources_observed":33,"crops_per_source_observed":5,
   "crop_folder_named_after_source": True,
   "user_reported_unverified_here": {"total_files":353,"saree_images":155,"unique_hashes":346,
     "exact_dup_copies":7,"dup_file":"IMG_20180824_200147.jpg"}},
 "critical_caveats": [
   "Effective independent Batik_Lasem sources = 39, not 5860 (severe source concentration).",
   "Batik_Nitik_960 contains 90/180/270 rotational augmentation (documented).",
   "No human aesthetic labels exist in any dataset.",
   "28x28 resolution makes GLCM/Gabor/FFT texture descriptors unreliable.",
   "True 3D garment visualization requires DeepFashion3D (PENDING access)."]
}
json.dump(dataset_summary, open(os.path.join(OUT,"dataset_summary.json"),"w"), indent=2)

gan_data_plan = {
 "stages": [
  {"stage":1,"name":"Pattern representation learning","dataset":"Batik_Lasem all_motifs (dedup)",
   "resolution":"28->64","labels":"motif_class","split":"origin-level","note":"foundation"},
  {"stage":2,"name":"Conditional generative pretraining","dataset":"Batik_Lasem (+Nitik_960 if pulled)",
   "resolution":"64","labels":"motif/category + palette + density","split":"source/category-level",
   "note":"handle Nitik rotation grouping"},
  {"stage":3,"name":"Aesthetic conditioning (proxy)","dataset":"computed aesthetic_features.csv",
   "resolution":"64","labels":"objective descriptors (NOT human scores)","split":"same",
   "note":"optionally external NIMA/CLIP-aesthetic reward (mark EXTERNAL)"},
  {"stage":4,"name":"Cross-domain adaptation","dataset":"NeuralLoom (textile appearance)",
   "resolution":">=128","labels":"folder domain","split":"Mekhela-source-level",
   "note":"domain shift; keep separate from batik pool"},
  {"stage":5,"name":"Garment visualization","dataset":"NeuralLoom 2D / DeepFashion2 (PENDING) / DeepFashion3D (PENDING)",
   "resolution":"native","labels":"-","split":"-","note":"2D texture-mapping now; 3D only if DeepFashion3D accessible"}
 ],
 "hard_rules": [
   "Deduplicate 28x28_images vs all_motifs before counting/splitting.",
   "Source-aware splits only (origin_images / fabric piece / Mekhela source).",
   "Do NOT concatenate batik crops with garment photos in one raw pool.",
   "Do NOT claim human aesthetic supervision; descriptors are proxies.",
   "Do NOT claim 3D visualization without DeepFashion3D."]
}
json.dump(gan_data_plan, open(os.path.join(OUT,"gan_data_plan.json"),"w"), indent=2)
print("plan files written.")
