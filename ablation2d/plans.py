"""Needle plans, and how they become input channels.

The sampling marginals are the ones ThermoNav's 3-D corpus was built to, which
are in turn taken from the clinical literature quoted in its datasheet:

| quantity  | target                                        | achieved here      |
|-----------|-----------------------------------------------|--------------------|
| antennas  | 55 / 25 / 13 / 7 % for 1 / 2 / 3 / 4          | exact (PMF)        |
| power     | median 75 W dial, 60 % in 50-100 W, 5 % off   | 75 W, 59 %, 5.0 %  |
| duration  | median 5.23 min, 65 % in 2-8 min, range 3-10  | 5.23 min, 80 %     |

(Clinical references: KLCA 2024 practice guideline; a 2025 series of 133
lesions. The duration marginal cannot be hit exactly — "65 % in 2-8 min" and a
hard 3-10 min range are not consistent with each other, since 3-10 min is
already almost all of 2-8 min. The median and the range are honoured and the
spread is reported rather than claimed: 13 % of needles sit on the 180 s floor
and 10 % on the 600 s ceiling.)

The 5 % of needles at exactly 0 W are not a mistake. They are the only examples
that teach the network that a needle drawn on the picture with the power off
kills nothing — without them it learns "needle => lesion" and the interactive
demo lies the moment somebody drags the power slider to zero.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .anatomy import Anatomy, LBL_LIVER
from .device import EMPRINT_HP, Device

#: P(n antennas), n = 1..4.
N_NEEDLE_PMF = np.array([0.55, 0.25, 0.13, 0.07])

DIAL_MIN_W, DIAL_MAX_W = 30.0, 150.0
DIAL_MEDIAN_W, DIAL_LOG_SIGMA = 75.0, 0.42
P_ZERO_POWER = 0.05

DUR_MIN_S, DUR_MAX_S = 180.0, 600.0
DUR_MEDIAN_S, DUR_LOG_SIGMA = 314.0, 0.50

NEEDLE_LEN_MIN_MM, NEEDLE_LEN_MAX_MM = 40.0, 200.0


@dataclass
class Needle:
    """One applicator. `tip` and `base` are (x, y) in mm, in the slice plane."""

    tip_x_mm: float
    tip_y_mm: float
    base_x_mm: float
    base_y_mm: float
    dial_power_w: float
    duration_s: float

    @property
    def tip(self) -> np.ndarray:
        return np.array([self.tip_x_mm, self.tip_y_mm], np.float32)

    @property
    def base(self) -> np.ndarray:
        return np.array([self.base_x_mm, self.base_y_mm], np.float32)

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.base - self.tip))

    @property
    def axis(self) -> np.ndarray:
        """Unit vector pointing from the tip back towards the base."""
        v = self.base - self.tip
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.array([0.0, 1.0], np.float32)

    def slot_endpoints(self, device: Device = EMPRINT_HP):
        """The two ends of the radiating slot, in mm."""
        u = self.axis
        return (self.tip + u * device.emission_start_mm,
                self.tip + u * device.emission_end_mm)

    def deposited_w(self, device: Device = EMPRINT_HP) -> float:
        return device.deposited_w(self.dial_power_w)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    needles: list[Needle]

    @property
    def n(self) -> int:
        return len(self.needles)

    def as_dict(self) -> dict:
        return {"needles": [n.as_dict() for n in self.needles]}

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        return Plan([Needle(**n) for n in d["needles"]])


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def _lognormal_clipped(rng, median, sigma, lo, hi, size=None):
    x = median * np.exp(rng.normal(0.0, sigma, size))
    return np.clip(x, lo, hi)


def sample_power_w(rng, size=None):
    p = _lognormal_clipped(rng, DIAL_MEDIAN_W, DIAL_LOG_SIGMA,
                           DIAL_MIN_W, DIAL_MAX_W, size)
    off = rng.random(size) < P_ZERO_POWER
    return np.where(off, 0.0, p)


#: v2 fixes the burn time. It is no longer an input channel, so a corpus that
#: still varied it would be teaching the model to predict the average of every
#: duration it might have been — the "ground truth is not a function of the
#: inputs" failure, in its purest form.
FIXED_DURATION_S = 300.0
VARY_DURATION = False


def sample_duration_s(rng, size=None):
    if not VARY_DURATION:
        return (np.full(size, FIXED_DURATION_S) if size is not None
                else FIXED_DURATION_S)
    return _lognormal_clipped(rng, DUR_MEDIAN_S, DUR_LOG_SIGMA,
                              DUR_MIN_S, DUR_MAX_S, size)


def sample_plan(rng: np.random.Generator, anat: Anatomy, *,
                device: Device = EMPRINT_HP,
                n_needles: int | None = None,
                max_tries: int = 400) -> Plan:
    """A geometrically feasible plan for this anatomy.

    Hard constraints, matching the 3-D corpus:

    * the tip is inside the liver and NOT inside a vessel lumen;
    * the far end of the radiating slot is inside the liver — otherwise the
      device deposits into connective tissue and the label is a burn in the
      wrong organ;
    * needle length in [40, 200] mm, entering from outside the liver.

    **Vessel crossing stays allowed.** That is the heat-sink signal the corpus
    exists to teach, and forbidding it would remove the most interesting thing
    in the picture.
    """
    if n_needles is None:
        n_needles = int(rng.choice(np.arange(1, 5), p=N_NEEDLE_PMF))
    dx = anat.dx_mm
    liver = anat.liver_mask
    vessel = anat.vessel_mask
    ys, xs = np.nonzero(liver)
    h, w = anat.shape
    fov_x, fov_y = w * dx, h * dx

    def inside_liver(pt_mm) -> bool:
        j, i = int(pt_mm[0] / dx), int(pt_mm[1] / dx)
        return 0 <= i < h and 0 <= j < w and bool(liver[i, j])

    needles: list[Needle] = []
    tries = 0
    while len(needles) < n_needles and tries < max_tries:
        tries += 1
        k = rng.integers(len(xs))
        iy, ix = int(ys[k]), int(xs[k])
        if vessel[iy, ix]:
            continue                                   # tip in a lumen
        # Jitter inside the voxel so tips are not all on the lattice.
        tip = np.array([(ix + rng.random()) * dx, (iy + rng.random()) * dx], np.float32)
        theta = rng.uniform(0.0, 2 * np.pi)
        u = np.array([np.cos(theta), np.sin(theta)], np.float32)  # tip -> base
        if not inside_liver(tip + u * device.emission_end_mm):
            continue                                   # slot leaves the organ
        length = float(rng.uniform(NEEDLE_LEN_MIN_MM, NEEDLE_LEN_MAX_MM))
        base = tip + u * length
        # The entry point has to be on the picture: a base outside the field of
        # view is a needle whose shaft channel is truncated, and the network
        # would be shown a plan it cannot see.
        if not (0.0 <= base[0] <= fov_x and 0.0 <= base[1] <= fov_y):
            length = _clip_ray_to_box(tip, u, fov_x, fov_y)
            if length < NEEDLE_LEN_MIN_MM:
                continue
            base = tip + u * length
        needles.append(Needle(
            tip_x_mm=float(tip[0]), tip_y_mm=float(tip[1]),
            base_x_mm=float(base[0]), base_y_mm=float(base[1]),
            dial_power_w=float(sample_power_w(rng)),
            duration_s=float(sample_duration_s(rng)),
        ))
    if not needles:
        raise RuntimeError("no feasible needle placement for this anatomy")
    return Plan(needles)


def _clip_ray_to_box(origin, direction, fov_x, fov_y) -> float:
    """How far the ray travels before it leaves [0, fov_x] x [0, fov_y]."""
    t = np.inf
    for o, d, hi in ((origin[0], direction[0], fov_x), (origin[1], direction[1], fov_y)):
        if abs(d) < 1e-9:
            continue
        t = min(t, ((hi if d > 0 else 0.0) - o) / d)
    return float(max(t, 0.0))
