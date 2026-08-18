"""
Dataset, preprocessing, audit and visualisation for the canonical Batik_Lasem
50% subset. Shared by ALL three baselines so preprocessing is identical.

Preprocessing decision (documented):
  * Canonical images are square (28x28 or 142x142 native). We resize directly to
    IMAGE_SIZE x IMAGE_SIZE with high-quality interpolation (bicubic + antialias),
    so aspect ratio is preserved and no cropping distortion occurs.
  * Upscaling 28->128 only interpolates (cannot invent detail) — documented, not
    hidden. 142->128 is a mild downscale.
  * RGB, float32, normalised to [-1, 1] (matches Tanh generator output).
  * Default augmentation = NONE (safest for motif semantics). Optional horizontal
    flip is exposed but OFF by default.
"""
from __future__ import annotations
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from .paths import resolve_image_path

MOTIF_ORDER = ["Gunung Ringgit", "Kricak / Watu Pecah", "Latohan",
               "Nyuk Pitu", "Seritan"]
MOTIF_TO_IDX = {m: i for i, m in enumerate(MOTIF_ORDER)}
IDX_TO_MOTIF = {i: m for m, i in MOTIF_TO_IDX.items()}


# --------------------------------------------------------------------------- #
def build_transform(image_size: int = 128, augment: bool = False,
                    hflip: bool = False):
    """Deterministic preprocessing pipeline -> tensor in [-1, 1]."""
    from torchvision import transforms
    ops = [
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
    ]
    if augment and hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [
        transforms.ToTensor(),                              # [0,1], CxHxW, RGB
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # -> [-1,1]
    ]
    return transforms.Compose(ops)


def denormalize(t):
    """[-1,1] tensor -> [0,1] tensor for viewing/metrics."""
    return (t * 0.5 + 0.5).clamp(0, 1)


class BatikCanonicalDataset:
    """torch Dataset over a manifest DataFrame (canonical all_motifs images)."""

    def __init__(self, manifest: pd.DataFrame, dataset_root: str,
                 image_size: int = 128, transform=None, conditional: bool = False,
                 verify: bool = True):
        from torch.utils.data import Dataset  # noqa: F401  (import-time check)
        self.df = manifest.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.conditional = conditional
        self.transform = transform or build_transform(image_size)
        # resolve absolute paths once (robust to folder-name variations)
        self.paths = []
        missing = []
        for _, r in self.df.iterrows():
            ap = resolve_image_path(dataset_root, r.get("rel_path", ""),
                                    r.get("filename", ""), r.get("motif_folder", ""))
            self.paths.append(ap)
            if ap is None:
                missing.append(r.get("filename", ""))
        self.missing = missing
        if verify and missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest images could not be located under "
                f"{dataset_root} (e.g. {missing[:5]}). Check DATASET_ROOT.")
        self.labels = [MOTIF_TO_IDX[m] for m in self.df["motif_name"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch
        ap = self.paths[idx]
        img = Image.open(ap).convert("RGB")
        x = self.transform(img)
        y = self.labels[idx]
        if self.conditional:
            return x, torch.tensor(y, dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.long)   # label always returned; ignore if unconditional


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2,
                    seed: int = 42, drop_last: bool = True):
    """DataLoader with L4-friendly, reproducible settings."""
    import torch
    from torch.utils.data import DataLoader
    from .env import seed_worker
    g = torch.Generator()
    g.manual_seed(seed)
    persistent = num_workers > 0
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=drop_last, worker_init_fn=seed_worker, generator=g,
        persistent_workers=persistent,
    )


# --------------------------------------------------------------------------- #
def audit_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  dataset_root: str, sample_check: int = 400,
                  stop_on_error: bool = True) -> dict:
    """Quick dataset audit; prints a report and (optionally) raises on problems.

    Checks: missing files, unreadable images, RGB/non-RGB, size distribution,
    duplicate filenames, and TRAIN/TEST leakage (shared origin_images/filenames).
    """
    print("=" * 60)
    print("DATASET AUDIT")
    print("=" * 60)
    total = len(train_df) + len(test_df)
    print(f"Canonical 50% subset samples : {total}")
    print(f"Train samples                : {len(train_df)}")
    print(f"Test samples                 : {len(test_df)}")

    def dist(df, name):
        print(f"\n{name} motif distribution:")
        for m in MOTIF_ORDER:
            n = int((df.motif_name == m).sum())
            print(f"  {m:22s}: {n:5d}  ({100*n/max(1,len(df)):5.2f}%)")
    dist(train_df, "Train"); dist(test_df, "Test")

    # leakage
    shared_origin = set(train_df.origin_images) & set(test_df.origin_images)
    shared_files = set(train_df.filename) & set(test_df.filename)
    print(f"\nLeakage check:")
    print(f"  origin_images shared (train∩test): {len(shared_origin)}")
    print(f"  filenames shared    (train∩test): {len(shared_files)}")

    # duplicate filenames within subset
    allfn = pd.concat([train_df.filename, test_df.filename])
    dup = int(allfn.duplicated().sum())
    print(f"  duplicate filenames within subset: {dup}")

    # file existence + readability (sample)
    both = pd.concat([train_df, test_df]).reset_index(drop=True)
    missing, unreadable, non_rgb = [], [], []
    sizes = Counter()
    check_idx = range(len(both)) if sample_check <= 0 else \
        np.linspace(0, len(both) - 1, min(sample_check, len(both)), dtype=int)
    for i in check_idx:
        r = both.iloc[int(i)]
        ap = resolve_image_path(dataset_root, r.get("rel_path", ""),
                                r.get("filename", ""), r.get("motif_folder", ""))
        if ap is None:
            missing.append(r.filename); continue
        try:
            im = Image.open(ap); im.verify()
            im = Image.open(ap)
            if im.mode != "RGB":
                non_rgb.append((r.filename, im.mode))
            sizes[im.size] += 1
        except Exception as e:
            unreadable.append((r.filename, str(e)))
    print(f"\nIntegrity (checked {len(list(check_idx))} of {len(both)}):")
    print(f"  missing files    : {len(missing)}")
    print(f"  unreadable images: {len(unreadable)}")
    print(f"  non-RGB images   : {len(non_rgb)}  (converted to RGB on load)")
    print(f"  size distribution: {dict(sizes)}")
    print("=" * 60)

    report = dict(total=total, train=len(train_df), test=len(test_df),
                  shared_origin=len(shared_origin), shared_files=len(shared_files),
                  duplicate_filenames=dup, missing=len(missing),
                  unreadable=len(unreadable), non_rgb=len(non_rgb),
                  size_distribution={str(k): v for k, v in sizes.items()})
    problems = []
    if len(shared_origin) or len(shared_files):
        problems.append("TRAIN/TEST LEAKAGE detected")
    if len(missing):
        problems.append("missing files")
    if len(unreadable):
        problems.append("unreadable images")
    report["problems"] = problems
    if problems and stop_on_error:
        raise RuntimeError("Dataset audit failed: " + "; ".join(problems) +
                           ". Fix before training (execution stopped).")
    if not problems:
        print("Audit passed: no leakage, no missing/unreadable files in the check.\n")
    return report


# --------------------------------------------------------------------------- #
def show_sample_grid(dataset, n: int = 16, title: str = "Training samples",
                     out_path: str = None, seed: int = 42):
    """Display/save a grid of dataset images (denormalised)."""
    import torch
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    imgs = torch.stack([dataset[int(i)][0] for i in idxs])
    grid = make_grid(denormalize(imgs), nrow=int(np.ceil(np.sqrt(len(idxs)))))
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title, fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def show_one_per_motif(train_df: pd.DataFrame, dataset_root: str,
                       image_size: int = 128, out_path: str = None):
    """Display one example image per motif class to verify correct data."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(MOTIF_ORDER), figsize=(4 * len(MOTIF_ORDER), 4))
    for ax, motif in zip(axs, MOTIF_ORDER):
        sub = train_df[train_df.motif_name == motif]
        ax.axis("off")
        if len(sub) == 0:
            ax.set_title(f"{motif}\n(none)", fontsize=14); continue
        r = sub.iloc[0]
        ap = resolve_image_path(dataset_root, r.get("rel_path", ""), r.filename)
        if ap:
            im = Image.open(ap).convert("RGB").resize((image_size, image_size))
            ax.imshow(im)
        ax.set_title(f"{motif}\n{r.image_resolution}", fontsize=14)
    plt.suptitle("One canonical example per motif", fontsize=20)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
