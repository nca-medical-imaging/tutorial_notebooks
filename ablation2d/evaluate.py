"""Scoring, and the baseline that makes the score mean something.

A surrogate that beats "predict zero" has proved nothing — 96-99 % of every
slice is background. The baseline that is actually hard to beat, and the one a
clinician would defend, is the **device chart**: an ablation zone whose size
comes from power and time and whose shape comes from the manufacturer's
brochure, with the anatomy ignored entirely.

If the NCA does not beat that, it has not learned about tissue. If it does, the
margin is exactly the value of knowing where the vessels are.
"""
from __future__ import annotations

import numpy as np
import torch

from .channels import CHANNEL_NAMES, FIXED_DURATION_S
from .device import EMPRINT_HP
from .physics import DEAD_CONTOUR

ACT = CHANNEL_NAMES.index("applicator_activation")
DX_MM = 2.0
PX_CM2 = (DX_MM ** 2) / 100.0

#: r_mm = C * (P_W ** ALPHA) * (t_s ** BETA). There is deliberately NO fitted
#: default here: a baseline nobody fitted is a straw man, and one that silently
#: uses dimensional-analysis guesses would make the model look good for free.
#: Call `fit_sphere_baseline(train_split)` and pass the result.
SPHERE_PARAMS = None


# --------------------------------------------------------------------------- #
# the device-chart baseline
# --------------------------------------------------------------------------- #
def _slot_groups(x: np.ndarray):
    """Yield `(mask, dial_W, duration_s)` for each distinct applicator setting.

    Grouping by VALUE rather than by connected component is exact here and much
    simpler: two antennas at the same power and time produce the same disc
    radius, so they can share a distance transform, and two at different
    settings are separated by construction.
    """
    act = x[ACT]
    on = act > 0
    if not on.any():
        return
    # v2 fixes the burn time, so a setting is a power and nothing else.
    for a in np.unique(act[on]):
        m = on & np.isclose(act, a)
        yield m, float(a) * EMPRINT_HP.max_power_w, FIXED_DURATION_S


def _distance_to(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask)
    except ImportError:
        ys, xs = np.nonzero(mask)
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        return np.min(np.hypot(yy[..., None] - ys, xx[..., None] - xs), axis=-1)


def sphere_baseline(inputs, params: dict | None = None) -> torch.Tensor:
    """(B, 8, H, W) -> (B, 1, H, W). Discs from the chart, anatomy ignored."""
    p = params or SPHERE_PARAMS
    if p is None:
        raise ValueError(
            "the sphere baseline has no fitted parameters. Call "
            "fit_sphere_baseline(train_split) and pass the result as `params` — "
            "an unfitted baseline is not a baseline.")
    arr = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)
    out = np.zeros((arr.shape[0], 1, *arr.shape[2:]), np.float32)
    for b in range(arr.shape[0]):
        for mask, dial_w, dur_s in _slot_groups(arr[b]):
            if dial_w <= 0:
                continue
            r_mm = p["C"] * (dial_w ** p["ALPHA"]) * (dur_s ** p["BETA"])
            out[b, 0] = np.maximum(out[b, 0],
                                   (_distance_to(mask) * DX_MM <= r_mm).astype(np.float32))
    t = torch.from_numpy(out)
    return t.to(inputs.device) if torch.is_tensor(inputs) else t


def fit_sphere_baseline(ds, max_samples: int = 600) -> dict:
    """Least squares for `log r = log C + a log P + b log t` on SINGLE-setting cases.

    Only cases with one distinct applicator setting are used, so the radius that
    is fitted is a single lesion's and not a union's. The law is then applied to
    every case, unions included — which is the baseline being used exactly as a
    planner would use a brochure.
    """
    X, Y = [], []
    n = min(len(ds), max_samples)
    for i in range(n):
        xi, yi = ds.sample(i)
        x = xi[0].cpu().numpy()
        groups = list(_slot_groups(x))
        if len(groups) != 1:
            continue
        _, dial_w, dur_s = groups[0]
        if dial_w <= 0:
            continue
        area_px = float((yi[0, 0].cpu().numpy() >= DEAD_CONTOUR).sum())
        if area_px < 4:
            continue
        r_mm = np.sqrt(area_px * DX_MM * DX_MM / np.pi)
        X.append([1.0, np.log(dial_w), np.log(dur_s)])
        Y.append(np.log(r_mm))
    if len(X) < 20:
        raise RuntimeError(f"only {len(X)} usable single-setting cases to fit on")
    X, Y = np.asarray(X), np.asarray(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    return {"C": float(np.exp(coef[0])), "ALPHA": float(coef[1]),
            "BETA": float(coef[2]),
            "_fit": f"n={len(X)} single-setting cases, "
                    f"log-radius RMSE {float(np.sqrt(((pred - Y) ** 2).mean())):.4f}, "
                    f"R2 {float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()):.3f}"}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _death(t: torch.Tensor) -> torch.Tensor:
    """The necrosis head of a (B, C, H, W) tensor, whichever C it has."""
    return t[:, 0:1]


@torch.no_grad()
def score(pred: torch.Tensor, truth: torch.Tensor, threshold: float = 0.5) -> dict:
    """DSC, area error, recall and precision — the last two SPLIT on purpose.

    Under-treatment and over-treatment are not the same mistake, and a single
    DSC averages them into a number that hides which one the model makes.
    """
    p = (_death(pred) > threshold).flatten(1).float()
    t = (_death(truth) >= DEAD_CONTOUR).flatten(1).float()
    tp = (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    d = torch.where(den > 0, 2 * tp / den, torch.ones_like(den))
    area_err = (p.sum(1) - t.sum(1)) * PX_CM2
    has = t.sum(1) > 0
    return {
        "dice": float(d.mean()),
        "dice_p10": float(d.quantile(0.10)),
        "area_err_cm2": float(area_err.mean()),
        "area_abs_err_cm2": float(area_err.abs().mean()),
        "recall": float((tp[has] / t.sum(1)[has].clamp(min=1)).mean()),
        "precision": float((tp[has] / p.sum(1)[has].clamp(min=1)).mean()),
        "n": int(len(d)),
    }


@torch.no_grad()
def predict_all(model, ds, steps: int, batch_size: int = 16) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x, steps=steps) for x, _ in ds.batches(batch_size, shuffle=False)])


@torch.no_grad()
def mask_to_organ(pred: torch.Tensor, organ: torch.Tensor,
                  channels=(1,)) -> torch.Tensor:
    """Zero the named prediction channels outside the organ.

    **This uses information the model was never given.** The model sees a CT, a
    needle and a power; a liver contour is extra. It is legitimate — an ablation
    planning system has one — but a score computed with it must be reported as
    "C-NCA + liver mask" and never folded into the model's own number.

    For the VESSEL channel it is close to free. Measured on the control, 300
    test cases: 33.6 % of its false positives lie outside the liver and 0.0 % of
    its misses do, because every true vessel voxel in this corpus is inside the
    organ by construction. So deleting outside-organ predictions costs no recall
    at all and buys precision — vessel F1 0.6146 -> 0.6908.

    Defaults to the vessel channel only. The NECROSIS channel is deliberately
    not masked: 11.7 % of the necrosis area lies outside the liver on average
    and 32.6 % of cases put more than 15 % of it there, so masking that head
    would delete correct predictions.
    """
    out = pred.clone()
    m = organ.to(pred.device)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    for c in channels:
        out[:, c:c + 1] = out[:, c:c + 1] * m
    return out


def report(model, ds, steps: int, baselines: dict | None = None,
           batch_size: int = 16, vessel_threshold: float | None = None,
           tta: bool = False, average: str = "none") -> dict:
    """`{name: metrics}` for the model and every baseline, on the same split.

    `vessel_threshold` should be the one chosen on VALIDATION. Leaving it None
    scores the vessel head at 0.5, which measures the head's calibration under
    3 % class prevalence rather than its segmentation — every model this project
    has trained prefers a threshold well above 0.5.

    `average` selects the test-time reduction:

      * `"none"`     one stochastic rollout — a SINGLE SAMPLE of a random
                     variable, because `fire_rate < 1` draws a fresh mask every
                     step. It understates the model's mean by ~0.09 DSC.
      * `"repeats"`  8 rollouts, same orientation. The control for TTA.
      * `"d4"`       8 rollouts, one per orientation of D4.

    Report `"repeats"` next to `"d4"` or the group gets credit for variance
    reduction it did not do — measured, two thirds of the DSC gain.
    """
    from .train import predict_d4_tta, predict_repeats

    if tta:                                   # back-compat for older callers
        average = "d4"
    truth = ds.targets if hasattr(ds, "targets") else ds.cell_death
    if average == "d4":
        pred = torch.cat([predict_d4_tta(model, x, steps)
                          for x, _ in ds.batches(batch_size, shuffle=False)])
    elif average == "repeats":
        pred = torch.cat([predict_repeats(model, x, steps)
                          for x, _ in ds.batches(batch_size, shuffle=False)])
    else:
        pred = predict_all(model, ds, steps, batch_size)
    out = {"C-NCA": score(pred, truth)}
    for name, fn in (baselines or {}).items():
        preds = torch.cat([fn(x) for x, _ in ds.batches(batch_size, shuffle=False)])
        out[name] = score(preds.to(truth.device), truth)
    out["predict zero"] = score(torch.zeros_like(truth), truth)

    # The vessel head, against the number a threshold on HU alone can reach.
    if truth.shape[1] > 1:
        from .model import vessel_f1
        thr = 0.5 if vessel_threshold is None else float(vessel_threshold)
        f1 = vessel_f1(pred, truth, thr)
        out["C-NCA"]["vessel_f1"] = float(f1.mean())
        out["C-NCA"]["vessel_f1_p10"] = float(f1.quantile(0.10))
        out["C-NCA"]["vessel_threshold"] = thr
        out["hu threshold (vessel)"] = {"vessel_f1": hu_threshold_vessel_f1(ds)}
    return out


@torch.no_grad()
def hu_threshold_vessel_f1(ds, hu_channel: int = 0) -> float:
    """The best F1 any single Hounsfield threshold can reach on this split.

    The vessel head's real competition: **0.245** on the real-CT test split,
    0.226 on val. A segmenter that does not beat this has learned nothing a `>`
    could not do.
    The threshold is fitted ON THE SPLIT IT IS SCORED ON, which flatters the
    baseline — deliberately, because a baseline you have to help is not a
    baseline.
    """
    x = ds.inputs[:, hu_channel].flatten()
    t = (ds.targets[:, 1] > 0.5).flatten()
    best = 0.0
    for q in np.linspace(50, 99.5, 60):
        thr = float(torch.quantile(x[:200000], q / 100.0))
        p = x > thr
        tp = float((p & t).sum())
        den = float(p.sum() + t.sum())
        if den > 0:
            best = max(best, 2 * tp / den)
    return round(best, 4)
