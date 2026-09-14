"""A 2-D axial slice: liver, hepatic vessels, connective tissue everywhere else.

Deliberately the simplest anatomy that still carries the physics the model has
to learn:

* **liver** — perfused, and the only place a needle is allowed to end up;
* **vessels** — the heat sink, with a CALIBRE that varies, because a sink that
  is the same in a 2 mm and a 9 mm vein makes the label stop being a function of
  the inputs (see `device.vessel_sink_rate`);
* **connective tissue** — everything outside the liver. Not air: an air
  background is not biological, so UniPhys switches perfusion and the vessel
  sink off there and the lesion would run away at the liver edge.

Anatomy is generated in 2-D and EXTRUDED along z by `physics.py`. That is the
one modelling assumption of the whole workshop, and it is what makes the problem
well posed: the 3-D solve's mid-plane is then a deterministic function of the
2-D picture the network is shown. Say it out loud in the session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .device import HEAT_SINK_DIAMETER_MM, TISSUE_OF, tissue_properties

# Grid the whole workshop uses. 128 x 128 at 2 mm is a 256 mm field of view —
# an abdomen is ~350 mm across, a liver ~200 mm, so this frames the liver and
# its surroundings without wasting cells on the scanner table.
NX = NY = 128
DX_MM = 2.0
FOV_MM = NX * DX_MM

LBL_BACKGROUND, LBL_LIVER, LBL_VESSEL = 0, 1, 2


@dataclass
class Anatomy:
    """One 2-D slice. Arrays are (NY, NX), indexed [y, x]."""

    label: np.ndarray           # uint8, LBL_*
    vessel_radius_mm: np.ndarray  # float32, 0 outside vessels
    dx_mm: float = DX_MM
    source: str = "procedural"
    calibres_mm: tuple = ()     # the calibre each vessel was drawn at
    hounsfield: np.ndarray | None = None   # float32 HU, the picture a radiologist has
    #: uniphys::Material ordinal per cell. Present on REAL patient slices, where
    #: the frame holds bone, muscle, fat, lung and air as well as liver — and
    #: absent on the procedural ones, which know only three tissues.
    material_id: np.ndarray | None = None

    def __post_init__(self):
        # A v2 Anatomy without a CT is incomplete, and the controlled
        # experiments (the heat-sink slab, the tests) build label maps directly
        # and do not care what the CT looks like — only that there is one.
        # Synthesised from a FIXED seed so those experiments stay reproducible;
        # `sample_anatomy` and `from_segmentations` always supply their own.
        if self.hounsfield is None:
            self.hounsfield = synth_hounsfield(self.label,
                                               np.random.default_rng(0))

    @property
    def shape(self) -> tuple[int, int]:
        return self.label.shape

    @property
    def liver_mask(self) -> np.ndarray:
        return self.label == LBL_LIVER

    @property
    def vessel_mask(self) -> np.ndarray:
        return self.label == LBL_VESSEL

    def physical_channels(self) -> dict[str, np.ndarray]:
        """The per-voxel material fields, in UniPhys units.

        These are what the solver integrates, so they are built once, here.

        On a REAL slice the per-cell `material_id` from the corpus is the
        authority and the properties are read from UniPhys's own table — the
        same table the solver uses, so a bone or lung cell behaves like bone or
        lung rather than like whichever of three tissues we would have guessed.
        The corpus's own property fields are not stored because they are exactly
        constant per id (measured spread 1e-8), so this reproduces them.
        """
        h, w = self.shape
        if self.material_id is not None:
            return _channels_from_material_ids(self.material_id)

        props = tissue_properties()
        out = {k: np.zeros((h, w), np.float32)
               for k in ("rho", "c", "k", "sigma", "perfusion")}
        mid = np.zeros((h, w), np.uint8)
        for label_name, lbl in (("background", LBL_BACKGROUND),
                                ("liver", LBL_LIVER),
                                ("vessel", LBL_VESSEL)):
            m = self.label == lbl
            if not m.any():
                continue
            p = props[label_name]
            mid[m] = p["material_id"]
            for k in out:
                out[k][m] = p[k]
        out["material_id"] = mid
        return out

    @property
    def vessel_fraction(self) -> float:
        """Vessel area as a fraction of liver area — the number to sanity-check."""
        liv = float(self.liver_mask.sum()) + float(self.vessel_mask.sum())
        return float(self.vessel_mask.sum()) / max(liv, 1.0)

    def summary(self) -> dict:
        px_mm2 = self.dx_mm ** 2
        return {
            "source": self.source,
            "liver_area_cm2": round((float(self.liver_mask.sum())
                                     + float(self.vessel_mask.sum())) * px_mm2 / 100.0, 1),
            "vessel_pct_of_liver": round(100.0 * self.vessel_fraction, 1),
            "calibres_mm": [round(c, 1) for c in self.calibres_mm],
            "heat_sink_px": int((self.vessel_radius_mm * 2 >= HEAT_SINK_DIAMETER_MM).sum()),
        }


#: HU statistics, MEASURED on case-1001 axial 154 (portal-venous phase) rather
#: than taken from a textbook, because the demo runs on that scan and a model
#: trained on cleanly-separable synthetic HU would fall over the moment it met
#: a real one.
#:
#: The headline number: liver 127 +/- 27, vessel 136 +/- 48. That is **d' = 0.23**
#: — the two are very nearly the same distribution, and the best single HU
#: threshold over the real-CT test split scores **F1 0.245**. Per voxel, a CT barely knows
#: where the vessels are. What separates them is SHAPE: vessels are bright,
#: round-or-tubular and connected, and that is a question a local rule iterated
#: 15 times can ask and a threshold cannot.
HU_STATS = {
    "background": (40.0, 20.0),    # soft/connective tissue
    "liver": (127.0, 27.0),
    "vessel": (136.0, 48.0),
}

#: The corpus's encoding, from ThermoNavServer normalization_constants.h.
HU_LO, HU_HI = -1000.0, 1000.0


#: How far the vessel mean sits above liver, in HU. **Sampled per case**, not
#: fixed, because contrast quality varies between scans and the demo slice is at
#: the hard end of it: measured d' = 0.23 at 1 mm and **0.17 at 2 mm**, against
#: 0.37 for a fixed +9 HU offset. Training only on the easy end would produce a
#: vessel head that works in the notebook and fails on the one real image in the
#: session. The range below puts the real slice comfortably inside the corpus
#: rather than at its edge.
VESSEL_CONTRAST_HU = (1.0, 28.0)


def synth_hounsfield(label: np.ndarray, rng: np.random.Generator,
                     correlated: float = 1.2,
                     vessel_contrast_hu: float | None = None) -> np.ndarray:
    """A plausible CT slice for a label map.

    Two noise terms, because CT has two. The per-voxel term is what a histogram
    sees; the SPATIALLY CORRELATED term is what makes parenchyma look mottled
    rather than sandy, and it is the one that decides whether a shape-based
    segmenter has anything to grip. Pure white noise would make the task easier
    than the real scan, not harder.
    """
    hu = np.zeros(label.shape, np.float32)
    if vessel_contrast_hu is None:
        vessel_contrast_hu = float(rng.uniform(*VESSEL_CONTRAST_HU))
    for name, lbl in (("background", LBL_BACKGROUND), ("liver", LBL_LIVER),
                      ("vessel", LBL_VESSEL)):
        m = label == lbl
        if not m.any():
            continue
        mu, sd = HU_STATS[name]
        if name == "vessel":
            mu = HU_STATS["liver"][0] + vessel_contrast_hu
        hu[m] = mu + rng.normal(0.0, sd * 0.75, int(m.sum())).astype(np.float32)
    if correlated > 0:
        f = _smooth(rng.normal(0.0, 1.0, label.shape).astype(np.float32), correlated)
        f /= max(float(f.std()), 1e-6)
        hu += f * 18.0
    return hu


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma)
    except ImportError:                       # a tiny separable box blur
        k = max(1, int(round(sigma)))
        out = a.copy()
        for _ in range(2):
            out = np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(2 * k + 1) / (2 * k + 1), "same"),
                0, out)
            out = np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(2 * k + 1) / (2 * k + 1), "same"),
                1, out)
        return out


def normalise_hu(hu: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(hu, np.float32) - HU_LO) / (HU_HI - HU_LO),
                   0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# procedural generation
# --------------------------------------------------------------------------- #
def _smooth_blob(rng, cx, cy, r_mean_px, n_harm=6, wobble=0.22, shape=(NY, NX)):
    """A closed star-shaped region: a circle whose radius is a smooth random
    function of angle. Six harmonics is enough to look organ-like and few enough
    that the boundary never self-intersects."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(yy - cy, xx - cx)
    amp = rng.normal(0.0, 1.0, n_harm) * wobble / np.arange(1, n_harm + 1)
    pha = rng.uniform(0, 2 * np.pi, n_harm)
    r_theta = np.full_like(ang, float(r_mean_px))
    for i in range(n_harm):
        r_theta *= 1.0 + amp[i] * np.cos((i + 1) * ang + pha[i])
    return rad < r_theta


def _tube(points_px, radius_px, shape=(NY, NX)):
    """Distance-to-polyline <= radius. Returns (mask, distance_px).

    Painting the tube as a distance field rather than stamping discs is what
    makes the RADIUS field exact: for a tube, the inscribed radius at a voxel is
    the tube radius, which is the quantity the sink term wants.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    best = np.full((h, w), np.inf, np.float32)
    pts = np.asarray(points_px, np.float32)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            continue
        t = ((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / L2
        t = np.clip(t, 0.0, 1.0)
        d = np.hypot(xx - (a[0] + t * ab[0]), yy - (a[1] + t * ab[1]))
        np.minimum(best, d, out=best)
    return best <= radius_px, best


def _wander(rng, start, n_steps, step_px, turn_sigma, shape=(NY, NX)):
    """A gently curving centreline. Vessels bend; they do not zig-zag."""
    h, w = shape
    p = np.asarray(start, np.float32)
    theta = rng.uniform(0, 2 * np.pi)
    pts = [p.copy()]
    for _ in range(n_steps):
        theta += rng.normal(0.0, turn_sigma)
        p = p + step_px * np.array([np.cos(theta), np.sin(theta)], np.float32)
        p[0] = np.clip(p[0], 1, w - 2)
        p[1] = np.clip(p[1], 1, h - 2)
        pts.append(p.copy())
    return np.stack(pts)


def sample_anatomy(rng: np.random.Generator, *, nx: int = NX, ny: int = NY,
                   dx_mm: float = DX_MM) -> Anatomy:
    """One random liver slice.

    Marginals chosen so the corpus spans what an axial slice through a liver
    actually looks like rather than one shape with the vessels moved:

    * liver area 90-250 cm^2 (a mid-liver axial slice is ~150 cm^2; the real
      demo slice is 204 cm^2, which is why the ceiling is not 220)
    * 2-6 vessels: at most one trunk at 8-20 mm (portal / hepatic vein, up to
      the calibre of an IVC) and the rest branches at 2.5-7 mm, straddling the
      3 mm clinical heat-sink threshold in both directions on purpose — the
      model has to learn WHERE the sink matters, not that vessels are cold.
      The trunk ceiling is set by the demo slice too: its vessel tree runs to
      26.8 mm inscribed calibre, and at 14 mm the corpus would have left 12 %
      of the demo's vessel area outside anything it had ever seen
    * total vessel area 3-20 % of the organ, rejected and redrawn outside that
      band. The band is not a guess: `seg_vessel` on the real demo slice
      (case-1001, z=154) is 13.4 % of the liver region, and an earlier draft
      capped at 12 % would have put the demo OUTSIDE its own training
      distribution. An even earlier one drew every calibre uniformly from
      2-12 mm and produced livers that were 34 % vein, which is not a liver
      either.

    NOTE the denominator. `seg_liver` and `seg_vessel` are DISJOINT in the
    corpus — the vessel label is carved out of the parenchyma, not painted over
    it — so "the organ" is their union and that is what `vessel_fraction`
    divides by.
    """
    shape = (ny, nx)
    px = dx_mm

    # -- liver ---------------------------------------------------------------
    for _ in range(40):
        r_mean_px = rng.uniform(58.0, 95.0) / px      # 58-95 mm mean radius
        cx = rng.uniform(0.36, 0.64) * nx
        cy = rng.uniform(0.36, 0.64) * ny
        liver = _smooth_blob(rng, cx, cy, r_mean_px, shape=shape)
        # Keep the organ off the frame edge: a liver flush against the boundary
        # puts the adiabatic wall inside the ablation zone.
        liver[:2] = liver[-2:] = False
        liver[:, :2] = liver[:, -2:] = False
        area_cm2 = liver.sum() * px * px / 100.0
        if 90.0 <= area_cm2 <= 250.0:
            break

    liver_px = int(liver.sum())
    ys, xs = np.nonzero(liver)

    # -- vessels -------------------------------------------------------------
    for _attempt in range(12):
        label = np.full(shape, LBL_BACKGROUND, np.uint8)
        label[liver] = LBL_LIVER
        radius = np.zeros(shape, np.float32)
        calibres: list[float] = []
        n_ves = int(rng.integers(2, 7))
        has_trunk = rng.random() < 0.75
        for v in range(n_ves):
            j = rng.integers(len(xs))
            start = (float(xs[j]), float(ys[j]))
            trunk = has_trunk and v == 0
            calibre_mm = float(rng.uniform(8.0, 20.0) if trunk
                               else rng.uniform(2.5, 7.0))
            line = _wander(rng, start,
                           n_steps=int(rng.integers(6, 13) if trunk
                                       else rng.integers(4, 10)),
                           step_px=rng.uniform(6.0, 12.0),
                           turn_sigma=rng.uniform(0.25, 0.7), shape=shape)
            tube, _ = _tube(line, 0.5 * calibre_mm / px, shape=shape)
            # A vessel exists only inside the organ. Clipping to the liver also
            # keeps the sink off the background, where it would be a fiction.
            tube &= liver
            if not tube.any():
                continue
            label[tube] = LBL_VESSEL
            # Overlapping branches: the larger calibre wins, which is what an
            # inscribed-radius measurement of the union would return.
            radius[tube] = np.maximum(radius[tube], 0.5 * calibre_mm)
            calibres.append(calibre_mm)
        frac = float((label == LBL_VESSEL).sum()) / max(liver_px, 1)
        if 0.03 <= frac <= 0.20:
            break

    return Anatomy(label=label, vessel_radius_mm=radius, dx_mm=dx_mm,
                   source="procedural", calibres_mm=tuple(sorted(calibres)),
                   hounsfield=synth_hounsfield(label, rng))


def _material_table(ids) -> dict[int, dict]:
    """`{material_id: properties}` straight from UniPhys, cached per id set."""
    from .device import material_properties_by_id
    return {int(i): material_properties_by_id(int(i)) for i in ids}


def _channels_from_material_ids(mid: np.ndarray) -> dict[str, np.ndarray]:
    mid = np.asarray(mid, np.uint8)
    out = {k: np.zeros(mid.shape, np.float32)
           for k in ("rho", "c", "k", "sigma", "perfusion")}
    for i, p in _material_table(np.unique(mid)).items():
        m = mid == i
        for k in out:
            out[k][m] = p[k]
    out["material_id"] = mid
    return out


def from_patient_slice(hounsfield, material_id, seg_liver, seg_vessel,
                       dx_mm: float = DX_MM, source: str = "patient") -> "Anatomy":
    """One real axial slice: real CT, real segmentation, real material map."""
    sv = np.asarray(seg_vessel, bool)
    sl = np.asarray(seg_liver, bool) | sv
    label = np.full(sl.shape, LBL_BACKGROUND, np.uint8)
    label[sl] = LBL_LIVER
    label[sv] = LBL_VESSEL
    radius = inscribed_radius_mm(sv, dx_mm)
    cal = 2 * radius[sv]
    return Anatomy(label=label, vessel_radius_mm=radius, dx_mm=dx_mm,
                   source=source,
                   hounsfield=np.asarray(hounsfield, np.float32),
                   material_id=np.asarray(material_id, np.uint8),
                   calibres_mm=tuple(np.round(np.percentile(
                       cal, [0, 50, 100]), 1)) if cal.size else ())


# --------------------------------------------------------------------------- #
# a real slice, for the demo
# --------------------------------------------------------------------------- #
def from_segmentations(seg_liver: np.ndarray, seg_vessel: np.ndarray,
                       dx_mm: float = DX_MM, source: str = "patient",
                       hounsfield: np.ndarray | None = None,
                       rng: np.random.Generator | None = None) -> Anatomy:
    """Build an Anatomy from two 2-D binary masks already on the workshop grid.

    **The organ is their UNION**, and it matters which corpus you are holding.

    In the 1 mm source corpus the two masks are DISJOINT — `seg_vessel` is carved
    out of the parenchyma rather than painted over it. Intersecting them there
    (the natural-looking thing to write) leaves only the boundary cells that
    survive downsampling, and turned a real 13.4 %-vessel slice into a 3.2 % one.

    In `data_real/`, the corpus this project actually ships, they are NOT
    disjoint: `scripts/extract_real_corpus.py` writes `sl = block_max(liver) | sv`,
    so the stored liver mask already contains the vessels and `sv` is a subset of
    `sl` (measured: 0.00 % of vessel voxels lie outside). The union is therefore
    the right answer in both, which is why this reads the union rather than
    branching on which corpus it was handed.

    The subset property is also what makes the liver mask safe as a model input:
    every vessel is interior, so the mask's BOUNDARY carries no vessel
    information — measured, the organ mask has the same number of connected
    components as the liver alone (2.97 both). Were a vessel to sit outside the
    parenchyma, the mask would draw it as an island and hand the vessel head its
    own answer.

    The vessel radius is measured, not assumed: the inscribed radius of the
    vessel mask, which is what the corpus's `calibre` means.
    """
    seg_liver = np.asarray(seg_liver, bool)
    seg_vessel = np.asarray(seg_vessel, bool)
    organ = seg_liver | seg_vessel
    label = np.full(organ.shape, LBL_BACKGROUND, np.uint8)
    label[organ] = LBL_LIVER
    label[seg_vessel] = LBL_VESSEL
    radius = inscribed_radius_mm(seg_vessel, dx_mm)
    cal = 2 * radius[seg_vessel]
    # A REAL CT when one is supplied; a synthesised one otherwise, so a bare
    # pair of masks still produces something v2 can be run on.
    hu = (np.asarray(hounsfield, np.float32) if hounsfield is not None
          else synth_hounsfield(label, rng or np.random.default_rng(0)))
    return Anatomy(label=label, vessel_radius_mm=radius, dx_mm=dx_mm,
                   source=source, hounsfield=hu,
                   calibres_mm=tuple(np.round(np.percentile(
                       cal, [0, 50, 100]), 1)) if cal.size else ())


def inscribed_radius_mm(mask: np.ndarray, dx_mm: float = DX_MM) -> np.ndarray:
    """Distance to the nearest non-mask voxel, in mm — the local inscribed radius.

    Uses scipy when it is there and falls back to an exact two-pass chamfer-free
    brute force on the (small) boundary otherwise, so the notebook does not gain
    a dependency just for one figure.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    try:
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(mask)
    except ImportError:
        pad = np.pad(mask, 1, constant_values=False)
        by, bx = np.nonzero(pad[1:-1, 1:-1] & ~(
            pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]))
        yy, xx = np.nonzero(mask)
        d = np.zeros(mask.shape, np.float32)
        if len(by):
            dd = np.hypot(yy[:, None] - by[None, :], xx[:, None] - bx[None, :])
            d[yy, xx] = dd.min(axis=1) + 0.5
    return (d * dx_mm).astype(np.float32)
