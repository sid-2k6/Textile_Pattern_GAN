"""
Path configuration & validation (TORCH-FREE).

Nothing here is hard-coded to a specific machine: the notebook fills in
PROJECT_ROOT / DATASET_ROOT (typically under a mounted Google Drive) and this
module builds the standard output tree and validates that the dataset exists.

Standard output layout (per model):
    <OUTPUT_ROOT>/<MODEL>/
        checkpoints/  generated_samples/  plots/  logs/
        history.csv   config.json   final_metrics.csv   final_metrics.json
"""
from __future__ import annotations
import os
import glob
from dataclasses import dataclass, field, asdict


@dataclass
class ProjectPaths:
    project_root: str
    dataset_root: str            # folder that CONTAINS Batik_Lasem/ ...
    metadata_path: str
    splits_dir: str
    output_root: str
    model_name: str
    checkpoint_dir: str = ""
    samples_dir: str = ""
    plots_dir: str = ""
    logs_dir: str = ""
    history_csv: str = ""
    config_json: str = ""
    final_metrics_csv: str = ""
    final_metrics_json: str = ""

    def __post_init__(self):
        base = os.path.join(self.output_root, self.model_name)
        self.checkpoint_dir = os.path.join(base, "checkpoints")
        self.samples_dir = os.path.join(base, "generated_samples")
        self.plots_dir = os.path.join(base, "plots")
        self.logs_dir = os.path.join(base, "logs")
        self.history_csv = os.path.join(base, "history.csv")
        self.config_json = os.path.join(base, "config.json")
        self.final_metrics_csv = os.path.join(base, "final_metrics.csv")
        self.final_metrics_json = os.path.join(base, "final_metrics.json")

    def make_dirs(self):
        for d in (self.checkpoint_dir, self.samples_dir, self.plots_dir, self.logs_dir):
            os.makedirs(d, exist_ok=True)
        return self

    def as_dict(self):
        return asdict(self)


def expected_structure_message(dataset_root: str) -> str:
    return (
        "Expected dataset structure under DATASET_ROOT:\n"
        f"  {dataset_root}/\n"
        "    Batik_Lasem/\n"
        "      motifs (isen-isen)/\n"
        "        Gunung Ringgit/all_motifs (864)/*.jpg\n"
        "        Kricak_Watu Pecah/all_motifs (1004)/*.jpg\n"
        "        Latohan/all_motifs (1373)/*.jpg\n"
        "        Nyuk Pitu/all_motifs (1291)/*.jpg\n"
        "        Seritan/all_motifs (1328)/*.jpg\n"
        "        metadata motifs.csv\n"
    )


def validate_paths(paths: ProjectPaths, require_metadata: bool = True) -> None:
    """Print a report and raise FileNotFoundError with guidance if invalid.

    Never silently creates an incorrect dataset path.
    """
    print("=" * 60)
    print("PATH VALIDATION")
    print("=" * 60)
    ok = True
    for label, p, must in [
        ("PROJECT_ROOT", paths.project_root, True),
        ("DATASET_ROOT", paths.dataset_root, True),
        ("METADATA_PATH", paths.metadata_path, require_metadata),
        ("SPLITS_DIR", paths.splits_dir, False),
    ]:
        exists = bool(p) and os.path.exists(p)
        print(f"  {label:14s}: {p}  ->  {'OK' if exists else 'MISSING'}")
        if must and not exists:
            ok = False
    batik = os.path.join(paths.dataset_root, "Batik_Lasem")
    batik_ok = os.path.isdir(batik)
    print(f"  {'Batik_Lasem':14s}: {batik}  ->  {'OK' if batik_ok else 'MISSING'}")
    print("=" * 60)
    if not ok or not batik_ok:
        print(expected_structure_message(paths.dataset_root))
        raise FileNotFoundError(
            "Dataset/metadata paths invalid. Fix DATASET_ROOT/METADATA_PATH in the "
            "configuration cell (see expected structure above). Execution stopped.")
    print("All required paths found.\n")


# ------------------------------------------------------------------ resolution
def resolve_image_path(dataset_root: str, rel_path: str,
                       filename: str = "", motif_folder: str = "") -> str | None:
    """Robustly locate a canonical image file on disk.

    1. Try DATASET_ROOT / rel_path (fast path — the committed manifest layout).
    2. Fall back to globbing `**/all_motifs*/<filename>` (folder-count agnostic).
    3. Final fallback: any `**/<filename>` under Batik_Lasem.
    Returns an absolute path or None.
    """
    if rel_path:
        cand = os.path.join(dataset_root, rel_path)
        if os.path.isfile(cand):
            return cand
    if not filename:
        filename = os.path.basename(rel_path) if rel_path else ""
    if not filename:
        return None
    batik = os.path.join(dataset_root, "Batik_Lasem")
    search_roots = [batik] if os.path.isdir(batik) else [dataset_root]
    for root in search_roots:
        hits = glob.glob(os.path.join(root, "**", "all_motifs*", filename), recursive=True)
        if hits:
            return hits[0]
    for root in search_roots:
        hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if hits:
            return hits[0]
    return None
