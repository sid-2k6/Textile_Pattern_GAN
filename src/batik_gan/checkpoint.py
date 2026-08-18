"""
Checkpointing / resume + append-only history.csv.

CheckpointManager writes:
    checkpoints/latest.pt        (every epoch, atomic)
    checkpoints/best.pt          (when best metric improves; lower FID = better)
    checkpoints/epoch_XXXX.pt    (every `save_interval` epochs)

A checkpoint payload contains: epoch, model/optim/scheduler/scaler states,
python/numpy/torch/cuda RNG states, training history, config, best metric.

HistoryLogger appends one row per epoch and rewrites the CSV each time (so an
interrupted Colab session never loses history). On resume it LOADS the existing
history and continues (never resets).
"""
from __future__ import annotations
import os
import glob
import shutil


# --------------------------------------------------------------- RNG states
def capture_rng_states() -> dict:
    import random
    import numpy as np
    import torch
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict):
    import random
    import numpy as np
    import torch
    if not states:
        return
    try:
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"].cpu() if hasattr(states["torch"], "cpu")
                            else states["torch"])
        if states.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])
    except Exception as e:
        print(f"WARNING: could not fully restore RNG states: {e}")


class CheckpointManager:
    def __init__(self, ckpt_dir: str, save_interval: int = 10,
                 best_mode: str = "min"):
        self.dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.save_interval = save_interval
        self.best_mode = best_mode
        self.best_metric = float("inf") if best_mode == "min" else float("-inf")

    # ---- paths
    @property
    def latest_path(self):
        return os.path.join(self.dir, "latest.pt")

    @property
    def best_path(self):
        return os.path.join(self.dir, "best.pt")

    def epoch_path(self, epoch):
        return os.path.join(self.dir, f"epoch_{epoch:04d}.pt")

    # ---- save
    def _atomic_save(self, payload, path):
        import torch
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)          # atomic; never corrupts existing ckpt

    def save(self, epoch: int, payload: dict, metric: float = None) -> dict:
        """Save latest (+ periodic epoch snapshot). Returns dict of saved paths."""
        saved = {}
        self._atomic_save(payload, self.latest_path)
        saved["latest"] = self.latest_path
        if self.save_interval and (epoch % self.save_interval == 0):
            self._atomic_save(payload, self.epoch_path(epoch))
            saved["epoch"] = self.epoch_path(epoch)
        if metric is not None and self._is_better(metric):
            self.best_metric = metric
            self._atomic_save(payload, self.best_path)
            saved["best"] = self.best_path
        return saved

    def _is_better(self, metric) -> bool:
        import math
        if metric is None or (isinstance(metric, float) and math.isnan(metric)):
            return False
        return metric < self.best_metric if self.best_mode == "min" \
            else metric > self.best_metric

    # ---- resume
    def find_latest(self):
        if os.path.isfile(self.latest_path):
            return self.latest_path
        eps = sorted(glob.glob(os.path.join(self.dir, "epoch_*.pt")))
        return eps[-1] if eps else None

    def load(self, path=None, map_location="cpu"):
        import torch
        if path is None:
            path = self.find_latest()
        if path is None or not os.path.isfile(path):
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

    def resolve_resume(self, resume: bool, resume_checkpoint=None):
        """Return a checkpoint path to resume from, or None (start fresh)."""
        if not resume:
            print("RESUME=False -> starting training from epoch 0.")
            return None
        path = resume_checkpoint or self.find_latest()
        if path and os.path.isfile(path):
            print(f"Checkpoint found. Resuming from: {path}")
            return path
        print("No checkpoint found. Starting training from epoch 0.")
        return None


class HistoryLogger:
    """Append-only epoch metrics -> history.csv (safe on resume)."""
    def __init__(self, csv_path: str, columns):
        self.csv_path = csv_path
        self.columns = list(columns)
        self.rows = []
        if os.path.isfile(csv_path):
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                self.rows = df.to_dict("records")
                for c in df.columns:
                    if c not in self.columns:
                        self.columns.append(c)
                print(f"Loaded existing history ({len(self.rows)} epochs) from {csv_path}")
            except Exception as e:
                print(f"WARNING: could not read existing history ({e}); starting new.")

    def last_epoch(self) -> int:
        return int(self.rows[-1]["epoch"]) if self.rows else 0

    def append(self, row: dict):
        import pandas as pd
        for c in row:
            if c not in self.columns:
                self.columns.append(c)
        self.rows.append(row)
        pd.DataFrame(self.rows).reindex(columns=self.columns).to_csv(
            self.csv_path, index=False)

    def as_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows).reindex(columns=self.columns)
