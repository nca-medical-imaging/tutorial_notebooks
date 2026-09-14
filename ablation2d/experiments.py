"""The three set pieces of the hands-on session.

Each one is a controlled comparison — one thing changes and everything else is
held fixed — because the honest version of "look, it learned the physics" is a
pair of pictures that differ in exactly one way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from . import viz
from .anatomy import (LBL_LIVER, LBL_VESSEL, Anatomy, sample_anatomy)
from .channels import build_inputs
from .plans import Needle, Plan


def _predict(model, anat, plan, steps, device, head: int = 0,
             mask_ct: bool = False):
    """`head` 0 is the necrosis field, 1 is the vessel segmentation.
    `mask_ct` must match how the model was trained (see `build_inputs`)."""
    x = torch.from_numpy(build_inputs(anat, plan, mask_ct=mask_ct)).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x, steps=steps)[0, head].float().cpu().numpy()


def _needle_through(anat: Anatomy, tip_xy_mm, angle_deg, dial_w=90.0, minutes=5.0):
    th = np.deg2rad(angle_deg)
    u = np.array([np.cos(th), np.sin(th)], np.float32)
    tip = np.asarray(tip_xy_mm, np.float32)
    h, w = anat.shape
    t = 120.0
    for o, d, hi in ((tip[0], u[0], w * anat.dx_mm), (tip[1], u[1], h * anat.dx_mm)):
        if abs(d) > 1e-9:
            t = min(t, ((hi if d > 0 else 0.0) - o) / d)
    base = tip + u * max(t, 30.0)
    return Needle(float(tip[0]), float(tip[1]), float(base[0]), float(base[1]),
                  float(dial_w), float(minutes * 60.0))


# --------------------------------------------------------------------------- #
# 1 · the heat sink
# --------------------------------------------------------------------------- #
def _anatomy_source(seed, real_slices, min_calibre_mm):
    """Yield candidate anatomies — REAL patient slices when the corpus is there.

    This used to call `sample_anatomy` unconditionally. That was fine when the
    model was trained on procedural blobs; against a model trained on real CT it
    feeds the network out-of-distribution input and the figure stops meaning
    anything. Procedural stays as the fallback for a checkout without the
    corpus, and the caption says which was used.
    """
    rng = np.random.default_rng(seed)
    src = Path(real_slices) if real_slices else None
    f = (src / "test_slices.npz") if src else None
    if f is not None and f.exists():
        from .anatomy import from_patient_slice
        d = np.load(f)
        order = rng.permutation(len(d["z"]))
        for i in order:
            a = from_patient_slice(
                hounsfield=d["hounsfield"][i].astype(np.float32),
                material_id=d["material_id"][i],
                seg_liver=d["seg_liver"][i].astype(bool),
                seg_vessel=d["seg_vessel"][i].astype(bool),
                source=f'{d["case"][i]}@z{int(d["z"][i])}')
            if a.calibres_mm and max(a.calibres_mm) >= min_calibre_mm:
                yield a, True
    else:
        for _ in range(200):
            a = sample_anatomy(rng)
            if a.calibres_mm and max(a.calibres_mm) >= min_calibre_mm:
                yield a, False


def heat_sink_pair(model, *, steps=15, device="cpu", seed=11, oracle=None,
                   figsize=(12.5, 4.4), real_slices="data_real",
                   min_calibre_mm=8.0, anat=None, mask_ct=False):
    """The same plan on a slice WITH a large vessel and on the same slice without it.

    The vessel is REMOVED, not moved: the label becomes plain parenchyma, the
    radius field goes to zero, and nothing else about the anatomy or the plan
    changes. Any difference between the two panels is the sink and only the sink.

    Drawn from the REAL slice corpus by default — the model is trained on real
    CT, and showing it a procedural blob measures how it handles anatomy it has
    never seen rather than how well it knows the heat sink.
    """
    if real_slices and not Path(real_slices).is_absolute():
        real_slices = Path(__file__).resolve().parent.parent / real_slices
    was_real = False
    # An explicit `anat` wins: the notebook passes the shipped demo slice, which
    # exists on every checkout, whereas `data_real/` is reproduction input that
    # is not distributed — and without it the search falls back to procedural
    # anatomy a real-CT model has never seen.
    source = ([(anat, True)] if anat is not None
              else _anatomy_source(seed, real_slices, min_calibre_mm))
    for anat, was_real in source:
        # Aim a needle just off the centreline of the FATTEST vessel: close
        # enough that the zone reaches it, far enough that the tip is not in it.
        #
        # Take the single thickest point, by argmax of the inscribed-radius
        # field. The old rule ("every pixel within 0.01 mm of the maximum") was
        # written against procedural vessels, which are uniform tubes where
        # hundreds of pixels share the maximum. On a real distance transform the
        # maximum is attained at ONE point — measured on the test corpus, the
        # old mask held 1-2 pixels against a required 20, so every real slice
        # was rejected and the function raised "no suitable placement" on a
        # corpus whose median largest vessel is 25 mm across.
        r = anat.vessel_radius_mm
        if not r.any():
            continue
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        cy, cx = float(iy), float(ix)
        # THE CT MUST BE REMOVED TOO. v2 gives the model a CT and nothing else,
        # so building `bare` from the labels alone leaves `hounsfield=None` and
        # `Anatomy` SYNTHESISES a fresh one — the second panel then differs from
        # the first in the entire image, and the measured difference is not the
        # sink at all. It is the only input the model reads.
        #
        # Removing a vessel from a CT means inventing parenchyma where it was.
        # We draw from this slice's own liver HU distribution: the texture is
        # right by construction, and a constant fill would leave a flat patch
        # that is itself a feature. That inpainting IS an assumption, and it is
        # the reason this comparison is a demonstration rather than a
        # measurement — `fig_heat_sink` (the solver, uniform liver, one straight
        # vein of controlled calibre) is the measurement.
        ves = anat.vessel_mask
        bare_hu = None
        if anat.hounsfield is not None:
            lv = anat.hounsfield[anat.liver_mask & ~ves]
            bare_hu = anat.hounsfield.copy()
            if lv.size:
                bare_hu[ves] = np.random.default_rng(seed).normal(
                    float(lv.mean()), float(lv.std()) + 1e-6, int(ves.sum())
                ).astype(anat.hounsfield.dtype)
        bare = Anatomy(
            label=np.where(anat.label == LBL_VESSEL, LBL_LIVER, anat.label),
            vessel_radius_mm=np.zeros_like(anat.vessel_radius_mm),
            hounsfield=bare_hu,
            dx_mm=anat.dx_mm, source="vessel removed")

        # SEARCH the placement rather than taking the first feasible angle.
        # The effect is only visible when the vein actually crosses the zone,
        # and the first angle that happens to land in parenchyma usually puts it
        # tangentially at the edge. Candidates are scored by how much the
        # prediction MOVES when the vessel is removed — which is the quantity
        # the figure is about, so this is choosing an illustrative case, not a
        # flattering one. The caption says the placement was chosen.
        best = None
        # The ring is RELATIVE TO THE VESSEL'S OWN RADIUS. A fixed 5 cells was
        # written against procedural vessels of at most ~13 mm; the demo slice's
        # thickest vessel is 26.8 mm across, so 5 cells from its centreline is
        # still inside it and every candidate tip was rejected. Start just
        # outside the wall and step outward, keeping the NEAREST ring that puts
        # a tip in parenchyma — nearest the vessel is where the sink shows.
        r_cells = float(r[int(cy), int(cx)]) / anat.dx_mm
        for off in (r_cells + 2.0, r_cells + 4.0, r_cells + 7.0, r_cells + 11.0):
            for ang in range(0, 360, 10):
                th = np.deg2rad(ang)
                ty, tx = cy + off * np.sin(th), cx + off * np.cos(th)
                iy, ix = int(round(ty)), int(round(tx))
                if not (0 <= iy < anat.shape[0] and 0 <= ix < anat.shape[1]):
                    continue
                if anat.label[iy, ix] != LBL_LIVER:
                    continue
                cand = Plan([_needle_through(
                    anat, (tx * anat.dx_mm, ty * anat.dx_mm), ang + 90, 100.0, 6.0)])
                # COMMON RANDOM NUMBERS. The model fires stochastically, so two
                # rollouts differ even on identical inputs, and searching 36
                # angles for the biggest difference between independent
                # rollouts selects the largest NOISE fluctuation, of either
                # sign. Re-seeding before each gives both rollouts the same
                # update masks: the difference is the vessel and only the vessel.
                torch.manual_seed(ang)
                a = _predict(model, anat, cand, steps, device, mask_ct=mask_ct)
                torch.manual_seed(ang)
                b = _predict(model, bare, cand, steps, device, mask_ct=mask_ct)
                moved = float(np.abs(b - a).sum())
                if best is None or moved > best[0]:
                    best = (moved, cand, a, b)
            if best is not None:
                break
        if best is None or best[0] <= 0:
            continue
        _, plan, p_with, p_without = best
        break
    else:
        raise RuntimeError("no suitable vessel-adjacent placement found")

    px_cm2 = anat.dx_mm ** 2 / 100.0

    n = 3 if oracle is None else 4
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=120)
    a_with = float((p_with > .5).sum()) * px_cm2
    a_without = float((p_without > .5).sum()) * px_cm2
    viz.show_necrosis(p_with, ax[0], anat, plan,
                      f"with vessel — {a_with:.1f} cm²", on_ct=True)
    viz.show_necrosis(p_without, ax[1], bare, plan,
                      f"vessel removed — {a_without:.1f} cm²", on_ct=True)
    d = p_without - p_with
    viz.show_ct(anat, ax[2])
    ax[2].imshow(d, cmap="PuOr_r", vmin=-1, vmax=1, origin="lower", zorder=2, alpha=0.92)
    ax[2].contour(anat.vessel_mask.astype(float), levels=[0.5], colors=["#0ff"],
                  linewidths=1.0, zorder=3)
    # The title states the SIGN it measured. The sink can only take tissue away,
    # so if the model predicts a larger lesion with the vessel present it has
    # got the physics backwards, and the figure must say that rather than
    # captioning it "saved" regardless.
    delta = a_without - a_with
    ax[2].set_title("tissue the sink SAVED" if delta > 0.05 else
                    ("the model predicts a LARGER lesion with the vessel — "
                     "backwards" if delta < -0.05 else
                     "no measurable difference"),
                    fontsize=9, fontweight="bold",
                    color="#111" if delta > 0.05 else "#c0392b")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    if oracle is not None:
        viz.show_necrosis(oracle(anat, plan), ax[3], anat, plan,
                          "UniPhys, with vessel", on_ct=True)
    fig.suptitle(viz.vessel_legend_text(anat) + "  ·  100 W, 6 min  ·  "
                 + ("REAL slice " + str(anat.source) if was_real
                    else "PROCEDURAL anatomy (the real corpus was not found — "
                         "the model is trained on real CT, so read this with "
                         "care)")
                 + "  ·  tip just outside the thickest vessel, angle chosen from 36 to make the effect "
                   f"visible  ·  Δ = {delta:+.2f} cm²",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --------------------------------------------------------------------------- #
# 2 · extrapolation in power
# --------------------------------------------------------------------------- #
def power_sweep(model, anat: Anatomy, *, steps=15, device="cpu",
                dial_w=(0, 40, 75, 110, 150, 250), minutes=5.0,
                tip_xy_mm=None, angle_deg=90.0, figsize=(15, 3.0),
                mask_ct=False):
    """One needle, one anatomy, the dial swept — including past the corpus.

    The training corpus samples the dial in [30, 150] W with 5 % of needles at
    exactly 0. 250 W is off the end of it. An NCA does not know that, and what
    it produces there is worth SEEING rather than being told about.
    """
    if tip_xy_mm is None:
        ys, xs = np.nonzero(anat.liver_mask)
        k = len(xs) // 2
        tip_xy_mm = (float(xs[k]) * anat.dx_mm, float(ys[k]) * anat.dx_mm)
    px_cm2 = anat.dx_mm ** 2 / 100.0
    fig, axes = plt.subplots(1, len(dial_w), figsize=figsize, dpi=115)
    for ax, w in zip(np.atleast_1d(axes), dial_w):
        plan = Plan([_needle_through(anat, tip_xy_mm, angle_deg, w, minutes)])
        p = _predict(model, anat, plan, steps, device, mask_ct=mask_ct)
        viz.show_necrosis(p, ax, anat, plan, None)
        seen = "" if 0 <= w <= 150 else "  ⚠ OUT OF CORPUS"
        ax.set_title(f"{w:.0f} W · {float((p > .5).sum()) * px_cm2:.1f} cm²{seen}",
                     fontsize=8, fontweight="bold",
                     color="#111" if 0 <= w <= 150 else "#b02020")
    fig.suptitle(f"dial swept at {minutes:.0f} min — corpus covers 30-150 W "
                 f"(plus 5 % at exactly 0)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# --------------------------------------------------------------------------- #
# 3 · out of distribution in space
# --------------------------------------------------------------------------- #
def outside_the_liver(model, anat: Anatomy, *, steps=15, device="cpu",
                      dial_w=100.0, minutes=5.0, figsize=(12.5, 3.7),
                      mask_ct=False):
    """Walk the tip from deep inside the organ out into connective tissue.

    Every training plan put the tip — and the whole radiating slot — inside the
    liver. Connective tissue is a real material with real properties, and the
    model has all of them on its input channels, so a physically sensible answer
    exists: a SMALLER zone, because connective tissue conducts and absorbs less
    than liver. Whether the model gives that answer, or merely a plausible one,
    is the question.
    """
    liver = anat.liver_mask | anat.vessel_mask
    ys, xs = np.nonzero(liver)
    cy, cx = ys.mean(), xs.mean()
    # A ray from the centroid outward; sample it at increasing radius.
    ang = np.deg2rad(20.0)
    u = np.array([np.cos(ang), np.sin(ang)])
    picks, labels = [], []
    for frac in (0.0, 0.45, 0.85, 1.05, 1.35):
        rmax = 0.0
        for r in np.arange(0, 90, 1.0):
            iy, ix = int(round(cy + r * u[1])), int(round(cx + r * u[0]))
            if not (0 <= iy < anat.shape[0] and 0 <= ix < anat.shape[1]):
                break
            if liver[iy, ix]:
                rmax = r
        r = frac * rmax
        picks.append(((cx + r * u[0]) * anat.dx_mm, (cy + r * u[1]) * anat.dx_mm))
        iy, ix = int(round(cy + r * u[1])), int(round(cx + r * u[0]))
        inside = (0 <= iy < anat.shape[0] and 0 <= ix < anat.shape[1]
                  and liver[iy, ix])
        labels.append("in liver" if inside else "in connective tissue")

    px_cm2 = anat.dx_mm ** 2 / 100.0
    fig, axes = plt.subplots(1, len(picks), figsize=figsize, dpi=115)
    for ax, tip, lab in zip(np.atleast_1d(axes), picks, labels):
        plan = Plan([_needle_through(anat, tip, 200.0, dial_w, minutes)])
        p = _predict(model, anat, plan, steps, device, mask_ct=mask_ct)
        viz.show_necrosis(p, ax, anat, plan, None)
        ax.set_title(f"{lab}\n{float((p > .5).sum()) * px_cm2:.1f} cm²", fontsize=8,
                     fontweight="bold",
                     color="#111" if lab == "in liver" else "#b02020")
    fig.suptitle(f"tip walked out of the organ — {dial_w:.0f} W, {minutes:.0f} min. "
                 f"Every training plan kept the slot inside the liver.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig
