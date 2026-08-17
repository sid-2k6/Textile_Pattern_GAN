"""
batik_gan — shared utilities for the Batik_Lasem GAN baseline experiments.

Modules
-------
manifest    : (torch-free) canonical manifest + 50% subset + group-aware split
paths       : (torch-free) project/Colab path config, validation, image resolution
env         : environment banner, seeding, device, AMP helpers
data        : Dataset, transforms, dataset audit, visualisation helpers
models      : DCGAN and conditional WGAN-GP generators/discriminators
metrics     : FID / KID (torchmetrics) + generative diversity, discriminator accuracy
checkpoint  : checkpoint save/load/resume + append-only history.csv logger
viz         : publication plots (font 20, dpi 300) + sample-grid saving
train       : reusable training pieces (gradient penalty, epoch evaluation)

All three baseline notebooks import from this package so the canonical dataset,
preprocessing, split, evaluation protocol and logging are IDENTICAL across
DCGAN, cWGAN-GP and StyleGAN2-ADA (only the model/algorithm differs).
"""
__version__ = "0.1.0"
__all__ = ["manifest", "paths", "env", "data", "models", "metrics",
           "checkpoint", "viz", "train"]
