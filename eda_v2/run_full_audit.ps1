# run_full_audit.ps1  —  Full-dataset EDA for the Textile Pattern GAN project (Windows/PowerShell)
#
# USAGE (from the eda_v2 folder):
#   cd D:\Textile_Pattern_GAN\eda_v2
#   powershell -ExecutionPolicy Bypass -File .\run_full_audit.ps1
#
# Edit $DATA below if your Datasets folder is elsewhere.
# DeepFashion2 / DeepFashion3D are intentionally EXCLUDED (pending access).

$ErrorActionPreference = "Stop"

# ---- 1. Dataset root -------------------------------------------------------
$DATA = "D:\Textile_Pattern_GAN\Datasets"
if (-not (Test-Path $DATA)) { Write-Error "Datasets folder not found: $DATA"; exit 1 }

# ---- 2. Auto-locate the Batik_Lasem metadata CSV --------------------------
$csv = Get-ChildItem -Path $DATA -Recurse -Filter "metadata motifs.csv" -ErrorAction SilentlyContinue |
       Select-Object -First 1
if (-not $csv) {
  Write-Warning "Could not find 'metadata motifs.csv' under $DATA — metadata audit will be skipped."
} else {
  $env:TEXTILE_METADATA_CSV = $csv.FullName
  Write-Host "Metadata CSV: $($csv.FullName)"
}
$env:DATASET_ROOT = $DATA

# ---- 3. Install dependencies (safe to re-run) -----------------------------
python -m pip install --quiet pillow numpy pandas matplotlib scipy scikit-image pypdf

# ---- 4. Complete metadata audit (Parts J/K/M) -----------------------------
if ($csv) { python metadata_audit.py "$($csv.FullName)" }

# ---- 5. Image audit over all datasets EXCEPT DeepFashion (Parts B-H,O,U,V)-
$pairs = @(
  "Batik_Lasem=$DATA\Batik_Lasem",
  "Batik_Nitik_960=$DATA\Batik_Nitik_960",
  "Batik_Nitik_Sarimbit_120=$DATA\Batik_Nitik_Sarimbit_120",
  "NeuralLoom=$DATA\NeuralLoom"
)
# include Original Images if present (top-level or nested under Batik_Lasem)
foreach ($cand in @("$DATA\Original Images", "$DATA\Batik_Lasem\Original Images")) {
  if (Test-Path $cand) { $pairs += "Original_Images=$cand"; break }
}
$datasets = $pairs -join ";"
Write-Host "`nAuditing datasets:`n$datasets`n"
Write-Host "(Note: near-duplicate detection on the full Batik_Lasem set can take 10-20 min.)"
python image_audit.py --datasets $datasets

# ---- 6. Figures (Part W) + strategy/plan files (Parts N,R,S,T,AB,AC,AG) ----
python generate_figures.py
python build_plans.py

Write-Host "`n==================================================================="
Write-Host "DONE."
Write-Host "  Machine-readable outputs : eda_v2\outputs\*.csv / *.json"
Write-Host "  Figures                  : eda_v2\figures\*.png"
Write-Host "  Report                   : eda_v2\RESEARCH_EDA_REPORT.md"
Write-Host "==================================================================="
