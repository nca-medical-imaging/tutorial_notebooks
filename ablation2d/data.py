"""Loading the corpus. Small enough to live in RAM, so it does.

3000 samples of 8 x 128 x 128 uint8 is 393 MB as inputs and 49 MB as targets.
That fits, and holding it resident removes the DataLoader from the picture
entirely — no workers, no collate, no surprise that the "training loop" was
actually a disk benchmark. In a 30-minute hands-on that matters more than it
looks: the single most common way a live notebook stalls is I/O.

**It stays uint8 on the device and is dequantised per batch.** As float32 the
training split alone is 1.77 GB, which on a 6 GB laptop GPU is the difference
between a batch of 8 and a batch of 32 — measured: 3.85 GB peak and 57 s/epoch
at batch 8 with float32 storage, against a batch four times larger once the
corpus costs 0.44 GB instead. The dequantisation is one fused op on a slice
that is already being copied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .channels import CHANNEL_NAMES, N_INPUT_CHANNELS


class AblationDataset:
    """One split, resident, on `device`.

    `inputs` are (N, 8, H, W) and `cell_death` is (N, 1, H, W), both float32 in
    [0, 1]. Dequantisation happens once, at load.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cpu",
                 limit: int | None = None):
        z = np.load(Path(path))
        x = z["inputs"]
        # v2 stores TWO targets; v1 stored one under its own name.
        y = z["targets"] if "targets" in z else z["cell_death"][:, None]
        if limit is not None:
            x, y = x[:limit], y[:limit]
        # `_deq` does `.float().div_(255)`, which is safe ONLY because these are
        # uint8: `.float()` on an already-float tensor is a no-op that returns the
        # SAME tensor, and the in-place divide would then corrupt the corpus on
        # every batch. Assert rather than comment.
        if x.dtype != np.uint8 or y.dtype != np.uint8:
            raise TypeError(f"corpus must be uint8, got {x.dtype}/{y.dtype}")
        self._x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self._y = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        self.device = device
        if self._x.shape[1] != N_INPUT_CHANNELS:
            raise ValueError(f"corpus has {self._x.shape[1]} channels, "
                             f"model expects {N_INPUT_CHANNELS} ({CHANNEL_NAMES})")

    @staticmethod
    def _deq(t):
        return t.float().div_(255.0)

    @property
    def torch_device(self) -> torch.device:
        """Where the tensors actually are. `train()` needs this to seed its
        generator, and reaching into `._x` from outside is how a view that is
        not this class (see `multires`) breaks the training loop."""
        return self._x.device

    @property
    def inputs(self) -> torch.Tensor:
        """The whole split as float32. Materialises a copy — use `batches()` in
        a training loop and this only for small splits or one-off figures."""
        return self._deq(self._x)

    @property
    def targets(self) -> torch.Tensor:
        """(N, 2, H, W) — necrosis then vessel."""
        return self._deq(self._y)

    @property
    def cell_death(self) -> torch.Tensor:
        """Just the necrosis head, for code that only wants the ablation."""
        return self._deq(self._y[:, 0:1])

    def __len__(self) -> int:
        return self._x.shape[0]

    @property
    def shape(self):
        return tuple(self._x.shape[2:])

    def sample(self, idx):
        """One or more cases as float32, without materialising the split."""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self._deq(self._x[idx]), self._deq(self._y[idx])

    def batches(self, batch_size: int, generator: torch.Generator | None = None,
                shuffle: bool = True):
        n = len(self)
        order = torch.randperm(n, generator=generator, device=self._x.device) \
            if shuffle else torch.arange(n, device=self._x.device)
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            yield self._deq(self._x[sel]), self._deq(self._y[sel])

    def stats(self) -> dict:
        frac = (self._y[:, 0] >= 252).flatten(1).float().mean(1)   # 0.99 * 255
        ves = (self._y[:, 1] > 127).flatten(1).float().mean(1) if self._y.shape[1] > 1 \
            else None
        out = {
            "n": len(self),
            "grid": self.shape,
            "lesion_frac_pct": [round(float(frac.min()) * 100, 2),
                                round(float(frac.median()) * 100, 2),
                                round(float(frac.max()) * 100, 2)],
            "channel_max": {n: round(float(self._x[:, i].max()) / 255.0, 3)
                            for i, n in enumerate(CHANNEL_NAMES)},
        }
        if ves is not None:
            out["vessel_frac_pct"] = round(float(ves.mean()) * 100, 2)
        return out


def load_splits(root: str | Path, device="cpu", limit=None):
    root = Path(root)
    return {s: AblationDataset(root / f"{s}.npz", device, limit)
            for s in ("train", "val", "test") if (root / f"{s}.npz").exists()}
