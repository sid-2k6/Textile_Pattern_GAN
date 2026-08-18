"""
Generative-model evaluation — IDENTICAL protocol for all three baselines.

FID / KID use torchmetrics (Inception-V3 pool3, 2048-d features).
  * REAL images  = the held-out TEST set (never the train set).
  * FAKE images  = `n_gen` generated samples.
  * Both are provided as uint8 [0,255], 3-channel, resized to 128 (the training
    resolution) — torchmetrics resizes to 299 internally for Inception.
Diversity = mean pairwise Euclidean distance between generated images' Inception
  features (higher = more diverse). Clearly a *diversity statistic*, not accuracy.

We deliberately DO NOT report accuracy/precision/recall for the generator (not
meaningful for GANs). Discriminator accuracy is reported only for the DCGAN
discriminator and is explicitly labelled as such.
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------ helpers
def to_uint8(imgs):
    """[-1,1] or [0,1] float tensor (B,3,H,W) -> uint8 [0,255] tensor."""
    import torch
    x = imgs.detach().float()
    if x.min() < -0.01:            # assume [-1,1]
        x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    return (x * 255).to(torch.uint8)


class GenerativeEvaluator:
    """Holds real test images and computes FID/KID/diversity for a generator.

    Parameters
    ----------
    real_images_uint8 : (N,3,H,W) uint8 tensor of the TEST set (held-out).
    kid_subset_size   : KID subset size (must be <= min(#real,#fake)).
    """
    def __init__(self, real_images_uint8, device, kid_subset_size: int = 50,
                 feature: int = 2048):
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
        self.device = device
        self.real = real_images_uint8
        self.n_real = real_images_uint8.shape[0]
        self.kid_subset_size = min(kid_subset_size, self.n_real)
        self.feature = feature
        self._fid = FrechetInceptionDistance(feature=feature, normalize=False)
        self._kid = KernelInceptionDistance(feature=feature, normalize=False,
                                            subset_size=self.kid_subset_size)
        self._fid = self._fid.to(device)
        self._kid = self._kid.to(device)
        self._inception = None

    def _inception_features(self, imgs_uint8):
        """Inception pool3 features (for the diversity statistic)."""
        import torch
        import torch.nn.functional as F
        if self._inception is None:
            self._inception = self._fid.inception    # reuse torchmetrics' net
        x = imgs_uint8.to(self.device).float()
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self._inception(x)
        return feats

    @staticmethod
    def _batched(gen_fn, n, batch, device):
        """Yield uint8 image batches from a generation function."""
        import torch
        done = 0
        while done < n:
            b = min(batch, n - done)
            with torch.no_grad():
                imgs = gen_fn(b)          # expected float (B,3,H,W)
            yield to_uint8(imgs).to(device)
            done += b

    def evaluate(self, gen_fn, n_gen: int = 640, batch: int = 64,
                 compute_diversity: bool = True) -> dict:
        """Compute FID, KID(mean/std), and a diversity statistic.

        gen_fn(batch_size) -> float image tensor (B,3,H,W) in [-1,1].
        """
        import torch
        self._fid.reset(); self._kid.reset()
        # real
        self._fid.update(self.real.to(self.device), real=True)
        self._kid.update(self.real.to(self.device), real=True)
        # fake
        fake_feats = []
        for fb in self._batched(gen_fn, n_gen, batch, self.device):
            self._fid.update(fb, real=False)
            self._kid.update(fb, real=False)
            if compute_diversity:
                try:
                    fake_feats.append(self._inception_features(fb).cpu())
                except Exception:
                    fake_feats = None
                    compute_diversity = False
        fid = float(self._fid.compute().item())
        kid_mean, kid_std = self._kid.compute()
        out = {"fid": fid, "kid_mean": float(kid_mean.item()),
               "kid_std": float(kid_std.item()),
               "n_real": int(self.n_real), "n_gen": int(n_gen),
               "kid_subset_size": int(self.kid_subset_size)}
        # Diversity is a *statistic*, never fabricated: NaN if it cannot be computed.
        if compute_diversity and fake_feats:
            try:
                feats = torch.cat(fake_feats, dim=0)
                out["diversity"] = float(pairwise_feature_diversity(feats))
            except Exception:
                out["diversity"] = float("nan")
        else:
            out["diversity"] = float("nan")
        return out


def pairwise_feature_diversity(feats, max_n: int = 512) -> float:
    """Mean pairwise Euclidean distance among feature vectors (sampled)."""
    import torch
    n = feats.shape[0]
    if n > max_n:
        idx = torch.linspace(0, n - 1, max_n).long()
        feats = feats[idx]
    d = torch.cdist(feats, feats, p=2)
    n = feats.shape[0]
    if n < 2:
        return float("nan")
    return float(d.sum().item() / (n * (n - 1)))


def build_real_uint8_from_loader(loader, n_max: int = None):
    """Collect a uint8 [0,255] tensor of real images from a dataloader."""
    import torch
    xs, total = [], 0
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        xs.append(to_uint8(imgs))
        total += imgs.shape[0]
        if n_max and total >= n_max:
            break
    out = torch.cat(xs, dim=0)
    return out[:n_max] if n_max else out


# ---------------------------------------------------- discriminator accuracy
def discriminator_accuracy(logits_real, logits_fake) -> dict:
    """Binary accuracy of the DCGAN discriminator (LOGITS in).

    Clearly labelled: this is *Discriminator* real/fake accuracy, NOT a
    generator/model classification accuracy.
    """
    import torch
    with torch.no_grad():
        real_pred = (torch.sigmoid(logits_real) > 0.5).float()
        fake_pred = (torch.sigmoid(logits_fake) <= 0.5).float()
        real_acc = float(real_pred.mean().item())
        fake_acc = float(fake_pred.mean().item())
    return {"disc_real_accuracy": real_acc, "disc_fake_accuracy": fake_acc,
            "disc_total_accuracy": 0.5 * (real_acc + fake_acc)}
