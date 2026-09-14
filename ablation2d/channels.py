"""What the network is shown — v2, the CT layout.

**v1 gave the model the physics; v2 gives it the picture.**

v1's eight inputs were the per-voxel terms of the bioheat equation: rho*c,
perfusion, k, sigma, the vessel sink rate, plus the applicator dose and the
needle. That is the *right* input if you already have a segmentation and a
material table — which is to say, if somebody has already done the hard part.

v2 hands the model what a radiologist actually has:

    inputs   0  hounsfield              the CT slice
             1  needle                  the applicator shaft
             2  applicator_activation   delivered power at the radiating slot

    targets  0  cell_death              the necrosis field
             1  vessel                  where the vessels are

and asks it to do **both jobs at once** — find the vasculature and predict the
burn — from three channels instead of eight.

**Duration is no longer an input.** Every plan in the v2 corpus fires for
exactly 5 minutes, so burn time is a constant of the problem rather than a
degree of freedom. That removes a channel and removes the most confounded axis
in the v1 corpus; the cost is that the model can no longer be asked "what if I
burn for eight minutes".

**Why this is not obviously going to work.** Measured on the real demo slice
(case-1001, axial 154, portal-venous phase): liver parenchyma is 127 +/- 27 HU
and vessel is 136 +/- 48. That is **d' = 0.23**, and the best possible single HU
threshold scores **F1 0.245** across the real-CT test split. Per voxel, a CT very nearly does not know where
the vessels are. What separates them is shape — bright, round-or-tubular,
connected — which is a question a local rule iterated fifteen times can ask and
a threshold cannot. Whether it does is the measurement, and F1 0.245 is the
number to beat.
"""
from __future__ import annotations

import numpy as np

from .anatomy import Anatomy, LBL_VESSEL, normalise_hu
from .device import EMPRINT_HP, Device
from .plans import Plan

#: The three things the model is given.
CHANNEL_NAMES = ["hounsfield", "needle", "applicator_activation"]
N_INPUT_CHANNELS = len(CHANNEL_NAMES)

#: The two things it must produce. Order matters: the model reads them off the
#: first channels of its state, in this order.
TARGET_NAMES = ["cell_death", "vessel"]
N_TARGET_CHANNELS = len(TARGET_NAMES)

#: Every plan in the v2 corpus burns for exactly this long.
FIXED_DURATION_S = 300.0

#: v1's layout, kept so the older corpus and checkpoint stay readable and the
#: two designs can be compared rather than merely swapped.
CHANNEL_NAMES_V1 = [
    "applicator_activation", "applicator_duration", "volumetric_heat_capacity",
    "perfusion", "thermal_conductivity", "electrical_conductivity",
    "vessel_sink_rate", "needle",
]

NORM = {
    "hounsfield": (0.0, 1.0),          # already normalised by anatomy.normalise_hu
    "needle": (0.0, 1.0),
    "applicator_activation": (0.0, 1.0),
}


# --------------------------------------------------------------------------- #
# painting the applicator
# --------------------------------------------------------------------------- #
def _segment_distance(shape, dx, a_mm, b_mm) -> np.ndarray:
    """Distance in mm from every cell centre to the segment a-b."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    px = (xx + 0.5) * dx
    py = (yy + 0.5) * dx
    ab = np.asarray(b_mm, np.float32) - np.asarray(a_mm, np.float32)
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return np.hypot(px - a_mm[0], py - a_mm[1]).astype(np.float32)
    t = ((px - a_mm[0]) * ab[0] + (py - a_mm[1]) * ab[1]) / L2
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (a_mm[0] + t * ab[0]),
                    py - (a_mm[1] + t * ab[1])).astype(np.float32)


def paint_plan(plan: Plan, shape, dx: float,
               device: Device = EMPRINT_HP) -> dict[str, np.ndarray]:
    """The two plan-dependent channels, unnormalised.

    A 14G antenna is 2.11 mm across and a cell is 2 mm, so the shaft is
    sub-voxel: the paint radius is floored at half a cell, otherwise a needle
    running between two cell centres marks NOTHING and the network is shown an
    empty picture for a plan that ablates.

    Overlapping needles take the MAX, not the sum — two antennas firing side by
    side do deposit more power, but that is the SOLVER's job; the channel states
    each applicator's own setting, and summing would make two 50 W needles
    indistinguishable from one at 100 W.
    """
    r_paint = max(0.5 * device.diameter_mm, 0.5 * dx)
    act = np.zeros(shape, np.float32)
    ndl = np.zeros(shape, np.float32)
    for n in plan.needles:
        d_shaft = _segment_distance(shape, dx, n.tip, n.base)
        ndl = np.maximum(ndl, (d_shaft <= r_paint).astype(np.float32))
        s0, s1 = n.slot_endpoints(device)
        d_slot = _segment_distance(shape, dx, s0, s1)
        slot = d_slot <= r_paint
        if not slot.any():                       # slot shorter than a cell
            slot = d_slot <= (d_slot.min() + 1e-6)
        act[slot] = np.maximum(act[slot], n.dial_power_w / device.max_power_w)
    return {"applicator_activation": act, "needle": ndl}


# --------------------------------------------------------------------------- #
# the input stack and the targets
# --------------------------------------------------------------------------- #
def build_inputs(anat: Anatomy, plan: Plan, *, device: Device = EMPRINT_HP,
                 normalise: bool = True, mask_ct: bool = False) -> np.ndarray:
    """(3, NY, NX) float32 — exactly what the network sees.

    `mask_ct=True` zeroes the CT outside the organ (liver | vessels), which is
    the input encoding of `data_liver/`. A model trained on that corpus MUST be
    fed this way: it learned that outside the liver is exactly zero, and handing
    it a full abdominal CT is a distribution shift, not a test of the model.
    Measured against the stored masked corpus: at most 1 uint8 level of
    difference, the same rounding floor the unmasked path has against `data/`.
    """
    if anat.hounsfield is None:
        raise ValueError(
            "this anatomy carries no Hounsfield channel, and v2 is built on it. "
            "Procedural anatomies get one from `sample_anatomy`; a patient slice "
            "needs the CT passed to `from_segmentations(..., hounsfield=...)`.")
    painted = paint_plan(plan, anat.shape, anat.dx_mm, device)
    hu = normalise_hu(anat.hounsfield)
    if mask_ct:
        hu = hu * (anat.liver_mask | anat.vessel_mask)
    stack = [hu,
             painted["needle"],
             painted["applicator_activation"]]
    return np.stack(stack).astype(np.float32)


def build_targets(anat: Anatomy, cell_death: np.ndarray) -> np.ndarray:
    """(2, NY, NX) float32 — necrosis, and where the vessels are."""
    vessel = (anat.label == LBL_VESSEL).astype(np.float32)
    return np.stack([np.asarray(cell_death, np.float32), vessel]).astype(np.float32)


def channel_index(name: str) -> int:
    return CHANNEL_NAMES.index(name)


def target_index(name: str) -> int:
    return TARGET_NAMES.index(name)
