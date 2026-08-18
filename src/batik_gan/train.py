"""
Reusable training pieces shared by the baselines.

  * gradient_penalty      : correct WGAN-GP gradient penalty (interpolated real/fake)
  * make_fixed_noise      : fixed latent (+ optional labels) for epoch sample grids
  * epoch_summary_print   : the standard per-epoch structured print block

The actual epoch loops live in the notebooks (kept explicit and readable) but
call these helpers so the two custom GANs share identical, correct components.
"""
from __future__ import annotations


def gradient_penalty(critic, real, fake, device, labels=None):
    """WGAN-GP gradient penalty: E[(||grad C(x_hat)||_2 - 1)^2],
    x_hat = eps*real + (1-eps)*fake, eps ~ U(0,1) per-sample."""
    import torch
    b = real.size(0)
    eps = torch.rand(b, 1, 1, 1, device=device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(x_hat, labels) if labels is not None else critic(x_hat)
    grads = torch.autograd.grad(
        outputs=scores, inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(b, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp


def make_fixed_noise(n: int, nz: int, device, seed: int = 42, num_classes: int = None):
    """Fixed latent vectors (and balanced labels if conditional) for eval grids."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, nz, generator=g).to(device)
    if num_classes is None:
        return z, None
    labels = torch.arange(n, device=device) % num_classes    # balanced, deterministic
    return z, labels


def epoch_summary_print(model_name, epoch, epochs, train_metrics, eval_metrics,
                        lrs, epoch_time, ckpt_paths, gpu_mb=None):
    """Standard structured per-epoch summary block."""
    line = "=" * 60
    print(line)
    print(f"{model_name} | Epoch {epoch}/{epochs}")
    print(line)
    print("Train:")
    for k, v in train_metrics.items():
        print(f"  {k:26s}: {v}")
    print("Held-out Evaluation:")
    for k in ("fid", "kid_mean", "kid_std", "diversity"):
        if k in eval_metrics:
            print(f"  {k:26s}: {eval_metrics[k]}")
    print("Learning Rate:")
    for k, v in lrs.items():
        print(f"  {k:26s}: {v}")
    if gpu_mb is not None:
        print(f"GPU Memory (peak MB)         : {gpu_mb}")
    print(f"Epoch Time (sec)             : {epoch_time:.1f}")
    print("Checkpoint:")
    for k, v in (ckpt_paths or {}).items():
        print(f"  {k:26s}: {v}")
    print(line + "\n")
