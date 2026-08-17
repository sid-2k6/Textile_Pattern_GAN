"""
Environment banner, deterministic seeding, device selection, AMP helpers.
Designed for Google Colab + NVIDIA L4 (24 GB) but never hard-codes a runtime.
"""
from __future__ import annotations
import os
import random
import platform


def print_environment() -> dict:
    """Print the standard ENVIRONMENT banner and return an info dict."""
    import torch
    info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": None,
        "gpu_memory_gb": None,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
    print("=" * 50)
    print("ENVIRONMENT")
    print("=" * 50)
    print(f"PyTorch    : {info['pytorch']}")
    print(f"CUDA       : {info['cuda_version']}  (available={info['cuda_available']})")
    print(f"cuDNN      : {info['cudnn']}")
    print(f"GPU        : {info['gpu']}")
    print(f"GPU Memory : {info['gpu_memory_gb']} GB")
    print(f"Python     : {info['python']}")
    print("=" * 50)
    if not info["cuda_available"]:
        print("WARNING: CUDA GPU NOT available. Training will be extremely slow on CPU.")
        print("         On Colab: Runtime > Change runtime type > GPU (L4 recommended).")
    return info


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed python / numpy / torch / cuda. Returns the seed."""
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    print(f"SEED = {seed}  (deterministic={deterministic})")
    return seed


def seed_worker(worker_id: int):
    """DataLoader worker seeding for reproducibility."""
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_memory_mb() -> float:
    """Current max GPU memory allocated (MB), or 0 on CPU."""
    import torch
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    return 0.0


def reset_peak_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def software_versions() -> dict:
    """Best-effort versions of key libraries for config.json reproducibility."""
    vers = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "torchvision", "torchmetrics", "numpy", "pandas",
                "PIL", "matplotlib"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            vers[mod] = "not installed"
    return vers
