"""Train coarse, finish fine — and the patch alternative.

An NCA's weights are a 3x3 rule. Nothing in them refers to the grid size, so a
rule learned on a small grid applies to a large one; the only thing that changes
is how many steps it takes to cross the picture. That is the loophole this
module exploits, and it comes in two flavours that are easy to confuse:

**Downscale (`CoarseView`).** Halve the resolution: 128x128 at 2 mm becomes
64x64 at 4 mm. A step is 4x cheaper AND one cell now spans 4 mm, so the same
physical reach needs half as many steps — the saving compounds to ~8x. The
catch is that the rule is now a *4 mm* rule, and it has to transfer back.

**Patch (`PatchView`).** Keep 2 mm and train on random 64x64 crops. A step is
4x cheaper and the rule never changes scale, so there is nothing to transfer.
The catch is the crop boundary: zero padding at a cut edge is a wall the model
never sees at inference, and a lesion sliced in half is a label that disagrees
with its own inputs.

Which one wins is a measurement, not an opinion — `scripts/run_speed_study.py`
runs both against a single-stage baseline on the same budget.

**The pooling rule is not invented here.** It is nca-forge's, restated in 2-D:

* `{needle, applicator_activation, applicator_duration}` are **max**-pooled.
  They are sparse. The needle is a line, so an f x f block holds f of its cells
  out of f^2 and mean-pooling drops the amplitude as 1/f — to 0.5 at 4 mm. The
  applicator slot is closer to a point and falls faster. In 3-D that mistake
  was measured at **0.24 DSC and 44 mm of HD95**.
* the physics channels are **mean**-pooled, because that is what homogenisation
  means: the effective property of a coarse cell IS the average of what it
  contains. Max would let a cell holding one vessel cell behave entirely like
  blood.
* **Neither target** is max-pooled. Max-pooling a target dilates it — a coarse
  cell holding one dead fine cell becomes fully dead — and the model then trains
  toward a systematically larger lesion than it is judged on. The vessel target
  is worse: vessels are thin, so max-pooling turns a 3 mm vein into a 4 mm one
  and the segmentation is scored against a mask nobody drew.

The documented caveat applies here too, and harder: mean-pooling
`vessel_sink_rate` scales a thin vessel's sink down by its area fraction, so at
4 mm the model may stop seeing the vessel at all. Hence `policy="sparse+vessel"`,
which keeps the sink peak-preserving. At 2 mm the sink channel is already
close to inert (see `device.VESSEL_SINK_MIN_RADIUS_MM`); do not assume it
survives a halving.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .channels import CHANNEL_NAMES

#: Sparse channels, max-pooled. nca-forge's `SPARSE_CHANNELS`, by name. In v2
#: the physics channels are gone, so the split is simply: the CT is a dense
#: field and is averaged (that is what a coarser voxel of tissue MEANS); the
#: needle and the applicator slot are sparse and are max-pooled, or they vanish.
SPARSE_CHANNELS = frozenset({"needle", "applicator_activation",
                             "applicator_duration"})

#: Named policies, matching nca-forge's `POOLING_POLICIES`.
POOLING_POLICIES = {
    "sparse": SPARSE_CHANNELS,
    "sparse+vessel": SPARSE_CHANNELS | {"vessel_sink_rate"},
    "all": frozenset(CHANNEL_NAMES),
}


def _max_pooled_mask(policy: str) -> torch.Tensor:
    try:
        names = POOLING_POLICIES[policy]
    except KeyError:
        raise ValueError(f"unknown pooling policy {policy!r}; "
                         f"expected one of {sorted(POOLING_POLICIES)}")
    return torch.tensor([n in names for n in CHANNEL_NAMES], dtype=torch.bool)


def downsample_inputs(x: torch.Tensor, factor: int,
                      policy: str = "sparse+vessel") -> torch.Tensor:
    """(B, 8, H, W) -> (B, 8, H/f, W/f), each channel by its own rule."""
    if factor <= 1:
        return x
    mask = _max_pooled_mask(policy).to(x.device)
    mean = F.avg_pool2d(x, factor)
    mx = F.max_pool2d(x, factor)
    return torch.where(mask.view(1, -1, 1, 1), mx, mean)


def downsample_target(y: torch.Tensor, factor: int) -> torch.Tensor:
    """Always mean, both heads. A max-pooled target is a dilated target."""
    return y if factor <= 1 else F.avg_pool2d(y, factor)


class CoarseView:
    """An `AblationDataset` seen at 1/factor resolution.

    Pools ONCE at construction and holds the result, rather than pooling every
    batch: the coarse corpus is 1/4 the size and pooling it repeatedly would
    spend the saving on the thing that was supposed to be saved.
    """

    def __init__(self, ds, factor: int = 2, policy: str = "sparse+vessel"):
        self.factor, self.policy, self.base = factor, policy, ds
        with torch.no_grad():
            self._x = downsample_inputs(ds.inputs, factor, policy)
            # BOTH targets. Pooling only the necrosis head silently trained the
            # vessel head against nothing.
            self._y = downsample_target(
                ds.targets if hasattr(ds, "targets") else ds.cell_death, factor)

    def __len__(self):
        return self._x.shape[0]

    @property
    def torch_device(self):
        return self._x.device

    @property
    def shape(self):
        return tuple(self._x.shape[2:])

    @property
    def inputs(self):
        return self._x

    @property
    def targets(self):
        return self._y

    @property
    def cell_death(self):
        return self._y[:, 0:1]

    def sample(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self._x[idx], self._y[idx]

    def batches(self, batch_size, generator=None, shuffle=True):
        n = len(self)
        order = (torch.randperm(n, generator=generator, device=self._x.device)
                 if shuffle else torch.arange(n, device=self._x.device))
        for i in range(0, n, batch_size):
            sel = order[i:i + batch_size]
            yield self._x[sel], self._y[sel]

    def stats(self):
        return {"n": len(self), "grid": self.shape, "factor": self.factor,
                "policy": self.policy}


class PatchView:
    """Random crops at the ORIGINAL resolution. No scale transfer to worry about.

    Crops are drawn per batch rather than fixed, so the model sees every part of
    every case over an epoch — a fixed crop set would be a smaller corpus, not a
    cheaper one.
    """

    def __init__(self, ds, size: int = 64, seed: int = 0):
        self.base, self.size = ds, size
        self._g = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        return len(self.base)

    @property
    def torch_device(self):
        return self.base.torch_device

    @property
    def shape(self):
        return (self.size, self.size)

    @property
    def inputs(self):
        return self.base.inputs

    @property
    def targets(self):
        return self.base.targets

    @property
    def cell_death(self):
        return self.base.cell_death

    def sample(self, idx):
        return self.base.sample(idx)

    def batches(self, batch_size, generator=None, shuffle=True):
        for x, y in self.base.batches(batch_size, generator, shuffle):
            h, w = x.shape[-2:]
            s = self.size
            i = int(torch.randint(0, h - s + 1, (1,), generator=self._g))
            j = int(torch.randint(0, w - s + 1, (1,), generator=self._g))
            yield x[..., i:i + s, j:j + s], y[..., i:i + s, j:j + s]

    def stats(self):
        return {"n": len(self), "grid": self.shape, "crop": self.size}


def scale_steps(steps: int, factor: int, floor: int = 3) -> int:
    """Rollout length for a coarser grid.

    A cell is `factor` times wider, so the same physical reach needs `factor`
    times fewer steps. nca-forge does this with
    `train_step_scale_with_resolution` anchored at 2 mm; not scaling it is how
    a coarse stage ends up cheaper per step and no cheaper overall.
    """
    return max(floor, int(round(steps / factor)))
