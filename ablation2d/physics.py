"""Ground truth: UniPhys Pennes bioheat + microwave SAR + Arrhenius cell death.

**The label is solved in 3-D and then sliced.** That deserves the paragraph it
gets in the notebook, because the obvious alternative is wrong in a way that
does not look wrong.

A one-cell-deep grid (`nz = 1`) is a genuine 2-D heat equation, and it is the
physics of an infinite slab: no heat leaves through the faces. Measured on this
grid, 90 W for 5 min, the 2-D slab gives a 40 x 48 mm lesion where the mid-plane
of the same needle solved in 3-D gives 40 x 42 mm. Same transverse extent, 14 %
too long axially — the missing out-of-plane conduction has nowhere to go.

So the anatomy is EXTRUDED along z and the full 3-D bioheat problem is solved,
then the mid-plane is taken as the label. Because the anatomy does not vary in
z and the needles lie in the mid-plane, that mid-plane is a deterministic
function of the 2-D picture the network is shown — which is what makes the
learning problem well posed. Depth costs almost nothing: the mid-plane is
bit-identical at nz = 32, 48, 64 and 96, so the slab is only 32 cells deep and a
solve costs about what the 2-D one did.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .anatomy import Anatomy
from .channels import build_inputs
from .device import EMPRINT_HP, VESSEL_SINK_MIN_RADIUS_MM, Device
from .plans import Plan
from .uniphys_env import require_uniphys

#: Slab depth for the 3-D solve. 32 cells at 2 mm = 64 mm. The mid-plane is
#: BIT-IDENTICAL at nz = 32, 48, 64 and 96, so the adiabatic wall is provably
#: out of reach and the extra depth was pure cost.
NZ = 32

#: Simulated seconds to keep running after the last applicator switches off.
#: The lesion keeps growing once the power is gone — that is the whole point of
#: an Arrhenius damage integral. Measured against a 3600 s tail: 300 s is still
#: 0.045 off and two cells short, 900 s is 0.004 off and exact on the contour,
#: 1800 s is bit-identical. `dynamic_stop` usually returns long before it.
#:
#: 900 s is what the corpus is generated at. Against (nz=40, 1800 s) over ten
#: varied plans it is 1.64x faster for a mean necrosis DSC of 0.9995 (worst
#: 0.9978, worst disagreement 3 cells out of 782) — two orders below the
#: disagreement between the solver's own resolutions, so it is a saving, not an
#: approximation anyone will notice.
COOLDOWN_S = 900.0

#: The corpus's necrosis contour: the smallest of the four endpoints, and the
#: only one that means what a planner is asked to guarantee.
DEAD_CONTOUR = 0.99


@dataclass
class Sample:
    inputs: np.ndarray        # (8, NY, NX) float32, normalised
    cell_death: np.ndarray    # (NY, NX)   float32 in [0, 1]
    anatomy: Anatomy
    plan: Plan
    wall_s: float
    sim_time_s: float

    @property
    def lesion_cm2(self) -> float:
        px = self.anatomy.dx_mm ** 2
        return float((self.cell_death >= DEAD_CONTOUR).sum()) * px / 100.0


def _extrude(field2d: np.ndarray, nz: int) -> np.ndarray:
    """(NY, NX) -> flat (nz*NY*NX,) in UniPhys's x-fastest order.

    UniPhys indexes `i = x + nx*(y + ny*z)`, so a C-order (nz, ny, nx) array
    ravels to exactly that. Getting this backwards silently transposes the
    anatomy, which looks plausible and is wrong.
    """
    return np.repeat(field2d[None, :, :], nz, axis=0).ravel().copy()


def simulate(anat: Anatomy, plan: Plan, *, device: Device = EMPRINT_HP,
             nz: int = NZ, cooldown_s: float = COOLDOWN_S,
             return_volume: bool = False) -> Sample:
    """Solve one plan and return the mid-plane necrosis field."""
    ny, nx = anat.shape
    dx = anat.dx_mm
    cz = nz // 2

    up = require_uniphys()
    phys = anat.physical_channels()
    sim = up.Simulation(up.GridDef(nx, ny, nz, dx), "fdm")
    up._from_channels(
        sim,
        _extrude(phys["material_id"], nz).astype(np.uint8),
        _extrude(phys["rho"], nz), _extrude(phys["c"], nz),
        _extrude(phys["k"], nz), _extrude(phys["sigma"], nz),
        _extrude(phys["perfusion"], nz),
        nx, ny, nz, dx,
        _extrude((anat.vessel_radius_mm > 0).astype(np.uint8), nz).astype(np.uint8),
    )
    # THE RADIUS FIELD IS THE AUTHORITY for the vessel sink, not the VESSEL
    # flag (uniphys src/fdm_cpu.cpp:387). `_from_channels` sets only the flag,
    # so without this line every vessel in the corpus would be thermally inert
    # and the heat-sink lesson would be a lesson about nothing.
    #
    # The floor is applied HERE as well as in the channel, and the two must stay
    # identical — see device.VESSEL_SINK_MIN_RADIUS_MM.
    sink_radius = np.where(anat.vessel_radius_mm >= VESSEL_SINK_MIN_RADIUS_MM,
                           anat.vessel_radius_mm, 0.0).astype(np.float32)
    up._set_vessel_radius(sim, _extrude(sink_radius, nz))

    device.apply(sim)

    events = []
    for n in plan.needles:
        e = up.SimulationEvent()
        e.applicator_type = up.ApplicatorType.MW
        e.energy = n.deposited_w(device)     # WATTS INTO TISSUE, not the dial
        e.duration = n.duration_s
        e.start_time = 0.0
        # p1 is the TIP and p2 is the BASE (uniphys types.h). The radiating
        # slot is measured back from p1, so swapping them fires the antenna out
        # of the patient.
        e.p1 = up.Vec3(float(n.tip_x_mm), float(n.tip_y_mm), (cz + 0.5) * dx)
        e.p2 = up.Vec3(float(n.base_x_mm), float(n.base_y_mm), (cz + 0.5) * dx)
        e.frequency = device.frequency_hz
        e.diameter = device.diameter_mm
        events.append(e)
    sim.set_schedule_mode(True)              # every antenna fires from t=0
    sim.set_events(events)
    sim.set_dynamic_stop(True, 0.0, 0)

    horizon = max((n.duration_s for n in plan.needles), default=0.0) + cooldown_s
    t0 = time.time()
    sim.run(horizon)
    wall = time.time() - t0

    frame = sim.capture()
    dead = np.asarray(frame.dead, np.float32).reshape(nz, ny, nx)
    mid = dead[cz].copy()

    s = Sample(inputs=build_inputs(anat, plan, device=device),
               cell_death=mid, anatomy=anat, plan=plan,
               wall_s=wall, sim_time_s=float(frame.sim_time))
    if return_volume:
        s.volume = dead                       # type: ignore[attr-defined]
    return s
