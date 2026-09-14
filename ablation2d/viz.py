"""Figures. One colour language for the whole session.

Anatomy is greyscale, the needle is drawn in the foreground, and necrosis is a
single warm ramp that goes transparent at zero so the anatomy shows through.
Ground truth and prediction always use the SAME ramp — if the two panels are
coloured differently the eye reads a difference that is not there.
"""
from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from .anatomy import LBL_BACKGROUND, LBL_LIVER, LBL_VESSEL, Anatomy
from .channels import CHANNEL_NAMES
from .device import EMPRINT_HP, HEAT_SINK_DIAMETER_MM

#: Necrosis: transparent -> amber -> deep red.
#:
#: The alpha ramp starts at 0.10, not 0.0, and that is a deliberate display
#: choice with a measured reason. A trained model does not output zero on
#: background: measured on the shipped checkpoint over 24 test cases, the
#: prediction where the truth is background has median **0.045** and 99th
#: percentile 0.147 — while only **0.011 %** of those cells exceed 0.5. Ramping
#: alpha from zero renders that floor as a haze of speckle across the whole
#: organ, which reads as noise in the prediction and is really a rendering
#: choice. Everything below 0.10 is therefore transparent, and the d >= 0.99
#: contour is drawn on top so the number that matters is never a matter of
#: colour perception.
NECROSIS = mcolors.LinearSegmentedColormap.from_list("necrosis", [
    (0.00, (0.99, 0.85, 0.30, 0.00)),
    (0.10, (0.99, 0.85, 0.30, 0.00)),
    (0.22, (0.99, 0.75, 0.20, 0.55)),
    (0.55, (0.94, 0.42, 0.08, 0.85)),
    (1.00, (0.65, 0.06, 0.10, 0.95)),
])

#: Anatomy: connective tissue, liver parenchyma, vessel lumen.
TISSUE_COLOURS = {LBL_BACKGROUND: "#2a2a30", LBL_LIVER: "#7d6a5a", LBL_VESSEL: "#3f6fa8"}
ANATOMY_CMAP = mcolors.ListedColormap([TISSUE_COLOURS[k] for k in (0, 1, 2)])

NEEDLE_COLOUR = "#e8e8ee"
SLOT_COLOUR = "#39d98a"


def anatomy_from_inputs(inputs, dx_mm=None) -> Anatomy:
    """Recover the label map from stored channels, for figures.

    The corpus stores CHANNELS, not the anatomy object, so any figure drawn from
    a `.npz` has no organ to draw underneath unless it can invert them.

    In v1 that inversion was clean: the perfusion channel took three values, one
    per tissue, orders of magnitude apart. **In v2 it is a THRESHOLD ON REAL
    HOUNSFIELD and it is noisy** — real parenchyma, muscle and bowel overlap, so
    the recovered label map speckles. It is good enough to sketch the organ
    outline and no good at all as a background for a figure someone will look
    at closely: pass `on_ct=True` and draw on the CT itself, which is what the
    model was given anyway.

    The vessel RADIUS is not recoverable, so this returns geometry for drawing
    and must not be fed back into the solver.
    """
    from .anatomy import DX_MM, LBL_BACKGROUND, LBL_LIVER, LBL_VESSEL
    from .channels import CHANNEL_NAMES
    x = np.asarray(inputs.detach().cpu() if hasattr(inputs, "detach") else inputs)
    if x.ndim == 4:
        x = x[0]
    hu_n = x[CHANNEL_NAMES.index("hounsfield")]
    hu = hu_n * 2000.0 - 1000.0
    # Organ vs surroundings is easy in HU (liver ~127 against connective ~40);
    # VESSEL vs liver is not — that is the whole point of v2, d' = 0.17 on the
    # real slice. So this returns the ORGAN only and leaves the vessels to the
    # model. A figure that drew them from a threshold would be drawing the
    # answer the model is being asked for.
    label = np.full(hu.shape, LBL_BACKGROUND, np.uint8)
    label[hu > 85.0] = LBL_LIVER
    return Anatomy(label=label,
                   vessel_radius_mm=np.zeros(hu.shape, np.float32),
                   hounsfield=hu.astype(np.float32),
                   dx_mm=dx_mm or DX_MM, source="recovered from channels")


def show_anatomy(anat: Anatomy, ax=None, title=None):
    ax = ax or plt.gca()
    ax.imshow(anat.label, cmap=ANATOMY_CMAP, vmin=0, vmax=2,
              origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return ax


def draw_plan(plan, anat: Anatomy, ax=None, device=EMPRINT_HP, label_power=True):
    """Shaft in white, radiating slot in green, tip as a dot.

    The slot is drawn separately because it is the part that matters and it is
    NOT the whole needle: on the Emprint HP it is the 11.8 mm span sitting
    7.25 mm back from the tip, which is a third of what people assume when they
    see a 20 cm antenna.
    """
    ax = ax or plt.gca()
    dx = anat.dx_mm
    for i, n in enumerate(plan.needles):
        tp, bp = n.tip / dx - 0.5, n.base / dx - 0.5
        ax.plot([tp[0], bp[0]], [tp[1], bp[1]], color=NEEDLE_COLOUR,
                lw=1.6, solid_capstyle="round", zorder=4)
        s0, s1 = n.slot_endpoints(device)
        s0, s1 = s0 / dx - 0.5, s1 / dx - 0.5
        on = n.dial_power_w > 0
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]],
                color=SLOT_COLOUR if on else "#888894", lw=4.0,
                solid_capstyle="butt", zorder=5, alpha=0.95)
        ax.plot(tp[0], tp[1], "o", ms=3.2, color=NEEDLE_COLOUR, zorder=6)
        if label_power:
            ax.annotate(f"{n.dial_power_w:.0f} W · {n.duration_s / 60:.1f} min"
                        if on else "off",
                        xy=(tp[0], tp[1]), xytext=(4, -9), textcoords="offset points",
                        fontsize=6.5, color="w", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="#00000099", ec="none"))
    return ax


def show_ct(anat_or_hu, ax=None, title=None, window=(120.0, 200.0)):
    """The CT, in a liver window — what the model is actually given.

    Default W/L 200/120 rather than a soft-tissue window, because that is the
    window a radiologist reads a portal-venous liver in and the one in which the
    vessels are visible at all.
    """
    ax = ax or plt.gca()
    hu = getattr(anat_or_hu, "hounsfield", None)
    hu = np.asarray(anat_or_hu if hu is None else hu, np.float32)
    lvl, wid = window
    ax.imshow(hu, cmap="gray", vmin=lvl - wid / 2, vmax=lvl + wid / 2,
              origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return ax


def show_necrosis(field, ax=None, anat=None, plan=None, title=None,
                  contour=0.99, contour_colour="#ffffff", on_ct=False):
    """Anatomy underneath, damage field on top, contour where the tissue is dead."""
    ax = ax or plt.gca()
    if anat is not None:
        if on_ct and getattr(anat, "hounsfield", None) is not None:
            show_ct(anat, ax)
        else:
            show_anatomy(anat, ax)
    im = ax.imshow(np.asarray(field), cmap=NECROSIS, vmin=0, vmax=1,
                   origin="lower", interpolation="nearest", zorder=2)
    f = np.asarray(field)
    if contour is not None and (f >= contour).any():
        ax.contour(f, levels=[contour], colors=[contour_colour],
                   linewidths=0.9, zorder=3)
    if plan is not None and anat is not None:
        draw_plan(plan, anat, ax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")
    return im


def compare(truth, pred, anat=None, plan=None, contour=0.99, figsize=(11, 3.6),
            titles=("UniPhys ground truth", "C-NCA prediction", "disagreement")):
    """Truth | prediction | signed difference. The third panel is the one to read.

    Blue is tissue the model said would survive and the physics killed
    (under-treatment — the dangerous direction). Red is the opposite.
    """
    truth = np.asarray(truth); pred = np.asarray(pred)
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=120)
    show_necrosis(truth, axes[0], anat, plan, titles[0], contour)
    show_necrosis(pred, axes[1], anat, plan, titles[1], contour)
    if anat is not None:
        show_anatomy(anat, axes[2])
    d = pred - truth
    axes[2].imshow(d, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   interpolation="nearest", alpha=0.9, zorder=2)
    t = (truth >= contour); p = (pred >= 0.5)
    inter = (t & p).sum(); dsc = 2 * inter / max(t.sum() + p.sum(), 1)
    axes[2].contour(truth, levels=[contour], colors=["#111"], linewidths=0.8, zorder=3)
    axes[2].set_title(f"{titles[2]} — DSC {dsc:.3f}", fontsize=9, fontweight="bold")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout()
    return fig, axes


def show_channels(inputs, anat=None, figsize=(10, 3.9), per_panel_scale=True):
    """All eight input channels. Worth showing once: it is the whole model input.

    **Each panel is scaled to its own maximum**, and that maximum is printed.
    On a shared 0-1 scale the two applicator channels are a handful of cells at
    a fraction of full dial and the vessel sink peaks around 0.37 — all three
    render as very nearly black, which makes the figure argue the opposite of
    its caption. The true range is in the subtitle so nothing is hidden by the
    stretch; pass `per_panel_scale=False` to see them on one scale and
    understand why this exists.
    """
    inputs = np.asarray(inputs)
    ncol = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(1, ncol, figsize=figsize, dpi=110)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ch = inputs[i]
        hi = float(ch.max())
        vmax = max(hi, 1e-6) if per_panel_scale else 1.0
        ax.imshow(ch, cmap="viridis", vmin=0, vmax=vmax,
                  origin="lower", interpolation="nearest")
        nz = int((ch > 0).sum())
        ax.set_title(f"{i}  {CHANNEL_NAMES[i]}", fontsize=8.5, fontweight="bold")
        ax.set_xlabel(f"max {hi:.3f} · {nz} cells > 0"
                      + ("" if per_panel_scale else "  (shared scale)"),
                      fontsize=6.5, color="#555", labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("What the network sees — a CT slice, a needle, and a power",
                 fontsize=11, fontweight="bold")
    if per_panel_scale:
        fig.text(0.5, -0.02, "each panel scaled to its own maximum, printed "
                             "beneath it — the applicator channel is a few dozen "
                             "cells and would otherwise be invisible",
                 ha="center", fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    # The per-panel captions live in the xlabel, so the rows need room or the
    # top row's caption collides with the bottom row's title.
    fig.subplots_adjust(hspace=0.28)
    return fig, axes


def show_rollout(trace, every=1, max_frames=8, anat=None, figsize=(14, 2.4)):
    """The field emerging, step by step. The picture that explains what an NCA is."""
    idx = list(range(0, len(trace), every))[:max_frames]
    fig, axes = plt.subplots(1, len(idx), figsize=figsize, dpi=110)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, idx):
        f = trace[k]
        f = f.detach().cpu().numpy()[0, 0] if hasattr(f, "detach") else np.asarray(f)
        if anat is not None:
            show_anatomy(anat, ax)
        ax.imshow(f, cmap=NECROSIS, vmin=0, vmax=1, origin="lower",
                  interpolation="nearest", zorder=2)
        ax.set_title(f"step {k + 1}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig, axes


def vessel_legend_text(anat: Anatomy) -> str:
    r = anat.vessel_radius_mm[anat.vessel_mask]
    if not r.size:
        return "no vessels"
    cal = 2 * r
    n_sink = int((cal >= HEAT_SINK_DIAMETER_MM).sum())
    return (f"vessels {100 * anat.vessel_fraction:.0f} % of organ · "
            f"calibre {cal.min():.1f}-{cal.max():.1f} mm · "
            f"{100 * n_sink / len(cal):.0f} % of vessel area is heat-sink "
            f"capable (>= {HEAT_SINK_DIAMETER_MM:.0f} mm)")
