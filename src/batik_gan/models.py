"""
GAN architectures for 128x128 RGB textile-motif generation.

  DCGANGenerator / DCGANDiscriminator      -> Baseline 1 (unconditional)
  CWGANGPGenerator / CWGANGPCritic         -> Baseline 2 (conditional WGAN-GP)

Design notes
------------
* DCGAN: DCGAN-style transposed-conv generator (BatchNorm + ReLU, final Tanh) and
  strided-conv discriminator (LeakyReLU, BatchNorm). Discriminator returns raw
  LOGITS (use BCEWithLogitsLoss) for AMP stability.
* WGAN-GP critic MUST NOT use BatchNorm (it breaks the per-sample gradient
  penalty); we use InstanceNorm2d in the critic. The critic returns a raw score
  (no sigmoid). Conditioning is done by concatenating a learned, spatially
  broadcast label embedding as extra input channels (generator and critic).
"""
from __future__ import annotations
import torch
import torch.nn as nn


def weights_init(m):
    """DCGAN weight init (N(0,0.02) conv, N(1,0.02) norm)."""
    cn = m.__class__.__name__
    if "Conv" in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cn or "InstanceNorm" in cn:
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _num_blocks(image_size: int) -> int:
    """Number of x2 up/down blocks from 4x4 to image_size (128 -> 5)."""
    assert image_size in (32, 64, 128, 256), "supported: 32/64/128/256"
    n = 0
    s = 4
    while s < image_size:
        s *= 2
        n += 1
    return n


# =============================== DCGAN ===================================== #
class DCGANGenerator(nn.Module):
    def __init__(self, nz: int = 128, ngf: int = 64, nc: int = 3, image_size: int = 128):
        super().__init__()
        nblocks = _num_blocks(image_size)              # 128 -> 5
        mult = 2 ** (nblocks - 1)                      # channel multiplier at 4x4
        layers = [
            nn.ConvTranspose2d(nz, ngf * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * mult), nn.ReLU(True),
        ]
        for i in range(nblocks - 1):
            in_c = ngf * mult
            mult //= 2
            layers += [
                nn.ConvTranspose2d(in_c, ngf * mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * mult), nn.ReLU(True),
            ]
        layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """Returns raw logits (use BCEWithLogitsLoss)."""
    def __init__(self, ndf: int = 64, nc: int = 3, image_size: int = 128):
        super().__init__()
        nblocks = _num_blocks(image_size)
        layers = [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                  nn.LeakyReLU(0.2, inplace=True)]
        mult = 1
        for i in range(nblocks - 1):
            in_c = ndf * mult
            mult *= 2
            layers += [
                nn.Conv2d(in_c, ndf * mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=False)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(-1)


# ========================= Conditional WGAN-GP ============================= #
class CWGANGPGenerator(nn.Module):
    def __init__(self, nz: int = 128, num_classes: int = 5, embed_dim: int = 64,
                 ngf: int = 64, nc: int = 3, image_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        nblocks = _num_blocks(image_size)
        mult = 2 ** (nblocks - 1)
        layers = [
            nn.ConvTranspose2d(nz + embed_dim, ngf * mult, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * mult), nn.ReLU(True),
        ]
        for i in range(nblocks - 1):
            in_c = ngf * mult
            mult //= 2
            layers += [
                nn.ConvTranspose2d(in_c, ngf * mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * mult), nn.ReLU(True),
            ]
        layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, z, labels):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        e = self.embed(labels).unsqueeze(-1).unsqueeze(-1)   # (B, embed, 1, 1)
        return self.main(torch.cat([z, e], dim=1))


class CWGANGPCritic(nn.Module):
    """WGAN-GP critic (NO BatchNorm -> InstanceNorm). Returns raw score.

    Label conditioning: embedding spatially broadcast to an extra input channel
    block concatenated with the image.
    """
    def __init__(self, num_classes: int = 5, embed_dim: int = 64, ndf: int = 64,
                 nc: int = 3, image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.label_channels = 1
        self.embed = nn.Embedding(num_classes, image_size * image_size)
        nblocks = _num_blocks(image_size)
        layers = [nn.Conv2d(nc + self.label_channels, ndf, 4, 2, 1),
                  nn.LeakyReLU(0.2, inplace=True)]
        mult = 1
        for i in range(nblocks - 1):
            in_c = ndf * mult
            mult *= 2
            layers += [
                nn.Conv2d(in_c, ndf * mult, 4, 2, 1),
                nn.InstanceNorm2d(ndf * mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(ndf * mult, 1, 4, 1, 0)]
        self.main = nn.Sequential(*layers)

    def forward(self, x, labels):
        b = x.size(0)
        lab = self.embed(labels).view(b, 1, self.image_size, self.image_size)
        return self.main(torch.cat([x, lab], dim=1)).view(-1)


def build_models(kind: str, cfg: dict, device):
    """Factory. kind in {'dcgan','cwgan_gp'}. Returns (G, D)."""
    if kind == "dcgan":
        G = DCGANGenerator(cfg["latent_dim"], cfg.get("ngf", 64), 3, cfg["image_size"])
        D = DCGANDiscriminator(cfg.get("ndf", 64), 3, cfg["image_size"])
    elif kind == "cwgan_gp":
        G = CWGANGPGenerator(cfg["latent_dim"], cfg["num_classes"], cfg["embedding_dim"],
                             cfg.get("ngf", 64), 3, cfg["image_size"])
        D = CWGANGPCritic(cfg["num_classes"], cfg["embedding_dim"], cfg.get("ndf", 64),
                          3, cfg["image_size"])
    else:
        raise ValueError(f"unknown kind {kind}")
    G.apply(weights_init); D.apply(weights_init)
    return G.to(device), D.to(device)
