"""Training, with the tricks that an NCA needs and a CNN does not.

Every trick below is one of the ten minutes of lecture that precede the hands-on
session, so the code is written to be read next to the slides rather than to be
short.

1. **Sample the rollout length each batch.** Fixing K teaches a choreography
   rather than a rule: the model learns what to do on step 7 instead of what to
   do. Production draws K from an adaptive window; here it is a plain uniform
   [3, 18] at 2 mm, which is the same range.

2. **Train from states the model actually reaches.** With probability
   `further_steps_rate` the state is rolled forward under `no_grad` first and
   the gradient step is taken from THERE. A blank grid is not representative of
   step 30 of a rollout, and a model trained only from blank grids falls apart
   when you ask for more steps than you trained on.

3. **Teach it to HOLD the answer.** With probability `target_to_seed_rate` the
   ground-truth necrosis field is written into the readout channel of the seed.
   The correct behaviour is then to leave it alone. Without this the model
   learns a trajectory with no fixed point and the field decays if you keep
   iterating — which is exactly what the interactive demo does.

4. **Per-parameter gradient normalisation.** Not global clipping: the two
   convolutions in a sub-model see wildly different gradient scales, and one
   global norm lets the larger one set the step size for both.

5. **Persistence.** Sampling the rollout length alone buys stability out to
   roughly twice the sampled range; past that the state drifts. Continuing the
   SAME rollout for a second, longer stretch and supervising again teaches the
   grid to SIT STILL once it has answered. The long stretch runs under
   `no_grad` and only the last `persist_bptt` steps carry a gradient —
   truncated BPTT, so memory stays flat however long the stretch is. This is
   the trick the interactive demo depends on: the step slider goes to 40, and
   a model trained only at K <= 18 turns to mush somewhere around 25.

6. **The inference rollout length is a free hyperparameter.** It is chosen on
   validation AFTER training, jointly with nothing else, and the cheapest K
   within 1 % of the best is taken. It costs nothing and almost nobody does it.
"""
from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from .data import AblationDataset
from .model import (AblationCNCA, best_vessel_threshold, dice,  # noqa: F401
                    multitask_loss, vessel_f1)


@dataclass
class TrainConfig:
    # architecture
    #: 2 readouts + 3 conditioning + **11 scratch**. Chosen by measurement, not
    #: by counting inputs — `scripts/capacity_ab.py` on the real-CT corpus,
    #: 60 epochs per arm, scored against a best-HU-threshold vessel F1 of 0.226:
    #:
    #:    6 ch (hidden  1)    2 616 p   DSC 0.302   vessel F1 0.322   11 min
    #:   12 ch (hidden  7)   10 416 p   DSC 0.658   vessel F1 0.545   21 min
    #:   16 ch (hidden 11)   41 616 p   DSC 0.706   vessel F1 0.656   52 min
    #:   24 ch (hidden 19)   93 528 p   DSC 0.734   vessel F1 0.652   94 min
    #:
    #: BOTH heads improve together with width, which is what says they were
    #: competing for scratch space rather than trading off. 24 buys +0.03 DSC,
    #: loses 0.004 vessel F1 and costs 1.8x the time, so 16 is the knee.
    #:
    #: An earlier draft cut this to 8 "because the conv cost goes as C^2 and we
    #: are launch-bound so it is nearly free". Both halves were wrong: the model
    #: starved at 8, and widening is NOT free — 6 to 16 channels measured 4.5x
    #: slower (some of which is this laptop's thermal throttling).
    channels: int = 16
    hidden_mult: int = 3
    n_sub_models: int = 3
    fire_rate: float = 0.5
    restore_conditioning: bool = True

    # rollout
    step_min: int = 3
    step_max: int = 18
    infer_steps: int = 15

    # optimisation
    epochs: int = 300
    batch_size: int = 16
    lr: float = 2e-3
    lr_min: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 0.0

    # NCA robustness. These three defaults are MEASURED, not inherited — see
    # scripts/run_persistence_study.py and docs/RESULTS.md. Six arms, one seed
    # each, scored by how much validation DSC survives from K=15 out to K=400:
    #
    #   neither trick                     0.54 retained
    #   target-to-seed only               0.75
    #   persistence, range (8, 24)        0.94
    #   persistence, range (40, 200)      1.00   <- flat from K=20 to K=400
    #   both, short range                 0.98
    #   both, long range                  0.99
    #
    # Two things fall out. **The range has to cover the horizon you care
    # about** — (8, 24) was copied from a config tuned for a shorter test and
    # nothing in that objective ever mentions step 200. And **target-to-seed is
    # redundant once persistence is long**: it rescues a model that has no
    # persistence (0.54 -> 0.75) and slightly hurts one that does (0.920 ->
    # 0.881 at K=200), so it is off by default and kept as a lesson.
    #
    # CAVEAT, because it matters: n=1 per arm, and long-horizon stability is
    # high-variance run to run — the same config measured 0.469 and 0.876 at
    # K=200 in two runs differing only in epoch count. The ORDERING across six
    # arms is what to trust here, not any single cell.
    further_steps_rate: float = 0.07
    target_to_seed_rate: float = 0.0
    persist_range: tuple = (40, 200)  # extra steps supervised a second time
    persist_bptt: int = 8             # of those, how many carry a gradient
    #: 0.1, not 0.5, and the RATE is the knob rather than the range. Measured
    #: cost against no persistence at all, on carnation, 512 samples:
    #:
    #:   (40,200) @ 0.10   1.16x      (20, 80) @ 0.50   1.73x
    #:   ( 8, 24) @ 0.50   1.40x      (40,200) @ 0.50   2.45x
    #:   (40,200) @ 0.20   1.47x
    #:
    #: A LONG range sampled RARELY is cheaper than a short range sampled often,
    #: and the retention study says the long range is the one that works. The
    #: reason both facts point the same way is that this model is launch-bound:
    #: a no_grad rollout costs nearly as much as a gradient one, so what you pay
    #: for is the NUMBER of extra steps, and 0.1 x 120 beats 0.5 x 16.
    persist_rate: float = 0.1

    # loss
    foreground_weight: float = 100.0
    #: Multiplier on the vessel loss term. **This is not a free knob — it decides
    #: how much gradient the NECROSIS head gets**, and the two terms are on wildly
    #: different scales. Measured on the first 16-channel run, one validation
    #: batch at its own K:
    #:
    #:     necrosis (weighted MSE)   0.0744          3 % of the loss
    #:     vessel (Dice + wBCE)      0.8025 x 3.0   97 % of the loss
    #:
    #: 3.0 was set while fighting to get the vessel head off F1 0.18 and left
    #: there after that problem was solved. The cost was large and invisible: the
    #: necrosis head got 3 % of the loss for every run that followed. At 0.3 it
    #: gets roughly a quarter, and changing only that number moves val DSC at
    #: epoch 10 from 0.462 to 0.787 (checkpoints_w03_full vs the control).
    #:
    #: Check the SHARE, not the weight, whenever either loss changes form.
    #:
    #: **0.3 is PROVISIONAL.** That 3.0 is wrong is settled by the share above and
    #: does not depend on any run finishing. That 0.3 is the right replacement is
    #: not: the arm testing it is still training, its vessel head is tracking
    #: slightly BELOW the 3.0 arm (F1 0.480 vs 0.497 at epoch 20), and the value
    #: was picked to put necrosis at roughly a quarter of the loss rather than
    #: measured against 1.0 or 0.1. Revisit when the arm lands.
    vessel_weight: float = 0.3          # ignored when uncertainty_weighting is on

    #: D4 augmentation — 4 rotations x 2 mirrors, uniform, per sample. Free
    #: 8x on the effective corpus for a PDE that has no preferred axis.
    augment_d4: bool = True

    #: VESSEL PRETRAINING. Epochs of vessel-only training on `pretrain_data`
    #: before the dual task starts, 0 to skip. That corpus is CT + vessel mask
    #: with no needle and no power, costs no solver time, and carries every
    #: slice — including the ones no feasible plan could be sampled in.
    pretrain_epochs: int = 60
    #: "data_vessel" is derived from the ablation corpus (1400 train slices, the
    #: ones that admit a feasible needle path). "data_vessel_wide" is extracted
    #: straight from the patient volumes with the plan constraint dropped and no
    #: per-patient cap: 4589 train slices, same 35 patients, no solver cost.
    #: Swapping it changes the number of GRADIENT STEPS per epoch by 3.3x, so
    #: scale `pretrain_epochs` down if you want the budgets comparable.
    pretrain_data: str = "data_vessel"
    #: Weight on the necrosis head during pretraining. The necrosis target there
    #: is exactly zero (the antenna is off), which is TRUE but degenerate: train
    #: on it hard and the first head learns "predict nothing" and has to unlearn
    #: it.
    #:
    #: 0 DROPS the necrosis term outright — see `multitask_loss`, where this is
    #: special-cased. It has to be: `necrosis_loss` weights background by a fixed
    #: 1.0 regardless, so "weight 0" on an all-zero target is a full-strength
    #: gradient telling the head to output nothing. Anything > 0 keeps the term
    #: and teaches "no power, no lesion", which is real supervision — just not
    #: what this phase is for.
    pretrain_death_weight: float = 0.0
    vessel_pos_weight: float = 20.0
    #: Learn the multi-task balance instead of setting it. Two extra scalars
    #: (Kendall & Gal homoscedastic uncertainty); the values they settle at are
    #: themselves a result — a vessel log-variance that climbs is the model
    #: saying the second task is not paying for itself.
    #: OFF. Kendall & Gal balances tasks by their homoscedastic uncertainty,
    #: which systematically downweights the harder, noisier one — measured here
    #: at a 27:1 log-variance split against the vessel head, whose F1 then FELL
    #: as the necrosis head improved. Right for an auxiliary head; wrong when
    #: the second head is a deliverable.
    uncertainty_weighting: bool = False

    # THE GROWING WINDOW (see ablation2d/window.py). `step_min`/`step_max`
    # become the FLOOR and the CEILING of a window that starts
    # `window_init_width` wide and grows toward `step_max` only while a
    # measured, paired validation probe says longer rollouts are paying
    # (`best beyond - best inside >= window_margin`, `window_streak` probes in
    # a row). OFF by default: the fixed range is what every number in
    # docs/RESULTS.md was measured with, and the A/B that justifies flipping
    # the default is `scripts/run_study.py --window`.
    window_grow: bool = False
    window_init_width: int = 4
    window_period: int = 5
    window_step: int = 2
    window_margin: float = 0.003
    window_streak: int = 2
    window_grow_after: int = 5
    window_explore_rate: float = 0.1

    # runtime
    device: str = "cuda"
    amp_dtype: str = "bf16"      # "bf16" | "fp32"
    seed: int = 0
    validate_every: int = 5
    grad_norm_eps: float = 1e-8
    #: Wall-clock cap in seconds; stop starting new epochs once it is passed.
    #: A live session has a TIME budget, not an epoch budget (backported for
    #: the notebook's `time_budget_s=8*60`; None = no cap).
    time_budget_s: float | None = None


#: The short version, for a laptop or a Colab T4. Smaller state, narrower
#: sub-models, shorter chain, shorter rollout, fewer epochs.
#:
#: **12 channels, not 6.** Measured on the real-CT corpus at a matched 80-epoch
#: budget, per the capacity sweep in `results/capacity_ab.json`:
#:
#:      6 ch (1 free)   necrosis DSC 0.302   vessel F1 0.322
#:     12 ch (7 free)   necrosis DSC 0.658   vessel F1 0.545
#:
#: At 6 channels the room trains a model that loses to a single Hounsfield
#: threshold on the vessel head and predicts the lesion badly enough to be
#: unconvincing. A demo that does not work teaches the wrong lesson about NCAs.
#: 12 is the cheapest width where both heads visibly function.
#:
#: `epochs` is NOT a measured 6 minutes any more and the name has been dropped
#: accordingly — the timing depends entirely on the room's hardware, and the one
#: machine available here holds SW Thermal Slowdown throughout (see README).
#: Time one epoch on the actual machine and scale; `train.py` stamps the GPU
#: clock on every validated epoch so the number carries its own provenance.
FAST = TrainConfig(channels=12, hidden_mult=2, n_sub_models=2,
                   step_min=3, step_max=10, infer_steps=8,
                   epochs=40, batch_size=16, lr=3e-3,
                   pretrain_epochs=0,
                   persist_range=(40, 200), persist_bptt=6)

#: The checkpoint the interactive demo loads.
FULL = TrainConfig()


#: The dihedral group of the square: 4 rotations x {identity, mirror} = 8
#: elements, every one reachable exactly once. Composing further mirrors adds
#: nothing — a mirror after a rotation IS one of these eight.
N_D4 = 8


def d4(t: torch.Tensor, g: int) -> torch.Tensor:
    """Element `g` of D4 applied to the trailing two axes of `t`.

    `g % 4` quarter-turns, then a mirror when `g >= 4`. Leading axes (batch,
    channel) are untouched: addressing from the END is what stops a target
    shaped (B,T,H,W) from being rotated against its channel axis.
    """
    out = torch.rot90(t, g % 4, dims=(-2, -1))
    if g >= 4:
        out = torch.flip(out, dims=(-1,))
    # rot90 and flip return VIEWS with scrambled strides. Handing one to cuDNN
    # makes it copy into its own layout on every convolution of every one of K
    # steps — and this model runs K sub-model applications per step. Materialise
    # once, here, in the layout the weights already use.
    return out.contiguous()


def d4_inv(t: torch.Tensor, g: int) -> torch.Tensor:
    """The inverse of `d4(., g)`.

    A mirror is an involution, so for `g >= 4` the inverse is "mirror first,
    then rotate back" — NOT "rotate back, then mirror", which is a different
    element and silently mis-registers every test-time-augmented prediction.
    """
    if g >= 4:
        return torch.rot90(torch.flip(t, dims=(-1,)), -(g % 4),
                           dims=(-2, -1)).contiguous()
    return torch.rot90(t, -(g % 4), dims=(-2, -1)).contiguous()


@torch.no_grad()
def predict_repeats(model, inputs: torch.Tensor, steps: int,
                    n: int = 8) -> torch.Tensor:
    """Average `n` rollouts in the SAME orientation.

    The control for `predict_d4_tta`, and most of its effect. With
    `fire_rate < 1` every rollout draws a fresh update mask, so a single
    prediction is one sample of a random variable — measured on arm 2, a
    per-case spread of 0.004 DSC and 0.010 vessel F1, and a MEAN that a single
    sample understates by ~0.09 DSC. Averaging repeats costs the same 8 rollouts
    as TTA and buys two thirds of what TTA buys, with no augmentation involved.
    """
    return sum(model(inputs, steps=steps) for _ in range(n)) / n


@torch.no_grad()
def predict_d4_tta(model, inputs: torch.Tensor, steps: int, *,
                   reduce: str = "mean") -> torch.Tensor:
    """Average the prediction over all eight orientations.

    Each orientation is predicted in ITS OWN frame and mapped back before the
    average, so the reduction happens in the original frame. Averaging first and
    un-rotating afterwards would be averaging eight differently-oriented images.

    **Do not attribute the whole gain to the group.** These are eight STOCHASTIC
    rollouts as well as eight orientations, so this averages two variances at
    once. Measured on arm 2 (150 test cases, K=14):

        1 rollout                          DSC 0.6931   vessel F1 0.5830
        8 rollouts, same orientation       DSC 0.7833   vessel F1 0.6354
        8 rollouts, D4 orientations        DSC 0.8276   vessel F1 0.6459

    Two thirds of the DSC gain (+0.090 of +0.134) is variance reduction over the
    fire mask and would happen on any model with `fire_rate < 1`, augmented or
    not. D4 adds the remaining third. On the vessel head the split is starker:
    +0.052 of +0.063 is noise. `predict_repeats` is the control that separates
    them, and it should be reported alongside whenever this is.
    """
    acc = None
    for g in range(N_D4):
        p = model(d4(inputs, g) if g else inputs, steps=steps)
        p = d4_inv(p, g) if g else p
        acc = p if acc is None else acc + p
    return acc / N_D4 if reduce == "mean" else acc


def augment_d4(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator):
    """Re-orient each sample of a batch independently, uniformly over D4.

    PER SAMPLE, not per batch: with 8 groups and a batch of 16, a per-batch draw
    would give one orientation to the whole step and the effective corpus grows
    by 8 only across epochs. Grouping costs at most 8 slice-copies.

    WHY IT IS SOUND HERE. Every input is a scalar field — a CT slice, a needle
    mask, a deposited-power map — and so is every target. There is no vector to
    re-orient, no chirality in the bioheat PDE, and the grid is square and
    voxel-aligned, so a quarter-turn is EXACT: no interpolation, no resampling,
    and the targets turn with the inputs. The one thing it does change is the
    marginal over needle direction, which the physics has no opinion about.

    WHY IT TAKES ITS OWN GENERATOR. The rollout length K, the further-steps coin
    and the persistence coin are all drawn from torch's GLOBAL stream inside the
    training loop. Drawing the orientation from that stream too would mean
    turning augmentation on ALSO changes every K in the run, and the A/B would
    measure the two together. (nca-forge hit exactly this and calls it out in
    `surrogate/config.py`.)

    WHY IT IS APPLIED HERE and not in the dataset: `AblationDataset` holds the
    corpus resident and hands out views of it. Augmenting at load would freeze
    ONE orientation per sample for the whole run — 8x fewer distinct samples
    than intended, silently.
    """
    if x.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"D4 augmentation needs a square grid; got {tuple(x.shape[-2:])}. A "
            f"quarter-turn transposes the two spatial axes, which changes the "
            f"shape unless they are equal.")
    g = torch.randint(0, N_D4, (x.shape[0],), generator=gen, device=x.device)
    xo, yo = torch.empty_like(x), torch.empty_like(y)
    for k in range(N_D4):
        m = g == k
        if not bool(m.any()):
            continue
        if k == 0:
            xo[m], yo[m] = x[m], y[m]
        else:
            xo[m], yo[m] = d4(x[m], k), d4(y[m], k)
    return xo, yo


_GPU_PROBE_OK = True


def _gpu_state() -> str:
    """SM clock, ceiling, temperature and throttle flags, sampled per validated
    epoch.

    Every wall-clock number this project has reported wrong traced back to a GPU
    that was quietly clocked down: a laptop card holding 1800-2070 MHz against a
    3105 MHz ceiling at 82 C reports a training time that says more about its
    cooling than about the model. A duration without the clock it was measured
    at is not a measurement, so the log now carries its own provenance.

    Returns "" when there is no NVIDIA GPU or nvidia-smi is missing — the probe
    is disabled permanently after one failure, never retried per epoch.
    """
    global _GPU_PROBE_OK
    if not _GPU_PROBE_OK or not torch.cuda.is_available():
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
             "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        sm, mx, tc, reasons = [f.strip() for f in out.splitlines()[0].split(",")]
        flags = int(reasons, 16)
        # 0x4 SwPowerCap, 0x8 HwSlowdown, 0x20 SwThermal, 0x40 HwThermal,
        # 0x80 HwPowerBrake. 0x1 (idle) and 0x2 (app clocks) are not throttles.
        names = [n for bit, n in ((0x4, "PWRCAP"), (0x8, "HWSLOW"),
                                  (0x20, "THERMAL"), (0x40, "HWTHERMAL"),
                                  (0x80, "PWRBRAKE")) if flags & bit]
        tag = "/".join(names) if names else "clear"
        return f"  [{sm}/{mx}MHz {tc}C {tag}]"
    except Exception:
        _GPU_PROBE_OK = False
        return ""


def _amp(cfg):
    if cfg.device.startswith("cuda") and cfg.amp_dtype == "bf16" \
            and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return False, torch.float32


def normalise_grads(model, eps: float):
    """TRICK 4 — divide each parameter's gradient by its own norm."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(p.grad.norm() + eps)


@torch.no_grad()
def evaluate(model, ds: AblationDataset, steps: int, batch_size: int = 16,
             foreground_weight: float = 100.0, vessel_weight: float = 1.0):
    """Returns (loss, necrosis DSC). `evaluate_full` also gives the vessel F1."""
    return evaluate_full(model, ds, steps, batch_size,
                         foreground_weight, vessel_weight)[:2]


@torch.no_grad()
def evaluate_full(model, ds: AblationDataset, steps: int, batch_size: int = 16,
                  foreground_weight: float = 100.0, vessel_weight: float = 1.0,
                  tune_threshold: bool = True):
    """(loss, necrosis DSC, vessel F1). Both heads, always — a multi-task model
    reported on one head is a model whose other head nobody is watching.

    The vessel F1 is reported at the threshold chosen ON THIS SPLIT, for the
    same reason the rollout length is: a head trained under 40:1 imbalance is
    systematically under-confident, and 0.5 measures its calibration rather than
    its segmentation. `vessel_threshold` comes back so the caller can quote it.
    """
    model.eval()
    preds, tgts = [], []
    tot_loss = tot_dice = 0.0
    n = 0
    for x, y in ds.batches(batch_size, shuffle=False):
        p = model(x, steps=steps)
        tot_loss += float(multitask_loss(p, y, foreground_weight,
                                         vessel_weight)) * x.shape[0]
        tot_dice += float(dice(p, y).sum())
        n += x.shape[0]
        if y.shape[1] > 1:
            preds.append(p.detach()); tgts.append(y)
    model.train()
    f1 = thr = 0.0
    if preds:
        P, T = torch.cat(preds), torch.cat(tgts)
        if tune_threshold:
            thr, f1 = best_vessel_threshold(P, T)
        else:
            thr, f1 = 0.5, float(vessel_f1(P, T).mean())
    evaluate_full.vessel_threshold = thr
    return tot_loss / n, tot_dice / n, f1


@torch.no_grad()
def choose_inference_steps(model, ds: AblationDataset, candidates=range(4, 41, 2),
                           batch_size: int = 16, tolerance: float = 0.01,
                           on: str = "dice", return_vessel: bool = False):
    """TRICK 6 — pick K on validation, then take the CHEAPEST within 1 %.

    `on="dice"` selects on the necrosis head, which is the product. But the two
    heads need not want the same K: the vessel head reads a static property of
    the CT and could be done in three steps while the necrosis front is still
    propagating. Selecting on one and never LOOKING at the other is how a
    second head silently gets served a rollout length that is wrong for it, so
    `return_vessel=True` returns its curve too and the caller can print both.
    """
    scores, vscores = {}, {}
    have_vessel = getattr(ds, "targets", None) is not None \
        and ds.targets.shape[1] > 1
    for k in candidates:
        k = int(k)
        _, d = evaluate(model, ds, steps=k, batch_size=batch_size)
        scores[k] = d
        if have_vessel and (return_vessel or on == "vessel_f1"):
            from .model import best_vessel_threshold
            with torch.no_grad():
                pv = torch.cat([model(x, steps=k)
                                for x, _ in ds.batches(batch_size, shuffle=False)])
            vscores[k] = best_vessel_threshold(pv, ds.targets)[1]
    pick = vscores if on == "vessel_f1" and vscores else scores
    best = max(pick.values())
    cheapest = min(k for k, v in pick.items() if v >= best * (1.0 - tolerance))
    if return_vessel:
        return cheapest, scores, vscores
    return cheapest, scores


def train(cfg: TrainConfig, train_ds: AblationDataset, val_ds: AblationDataset,
          *, model=None, out_dir: Path | None = None, on_epoch=None, log=print,
          select: str = "dice"):
    """Returns (model, history). `on_epoch(epoch, record, model)` drives the
    live dashboard in the notebook.

    Pass `model=` to train the class YOU wrote — that is what the notebook does,
    so the thing the room typed out is the thing that gets trained. With it
    omitted, the packaged `AblationCNCA` is built from `cfg`.
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    if model is None:
        model = AblationCNCA(channels=cfg.channels, hidden_mult=cfg.hidden_mult,
                             n_sub_models=cfg.n_sub_models, fire_rate=cfg.fire_rate,
                             restore_conditioning=cfg.restore_conditioning)
    model = model.to(dev)
    # The two log-variances are PARAMETERS: they are optimised alongside the
    # weights, which is the whole point of learning the balance.
    log_vars = (torch.zeros(2, device=dev, requires_grad=True)
                if cfg.uncertainty_weighting else None)
    params = list(model.parameters()) + ([log_vars] if log_vars is not None else [])
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp, amp_dtype = _amp(cfg)
    gen = torch.Generator(device=train_ds.torch_device).manual_seed(cfg.seed)
    #: A SEPARATE stream — see augment_d4 for why it must not share with K.
    aug_gen = (torch.Generator(device=train_ds.torch_device)
               .manual_seed(cfg.seed ^ 0x5F11) if cfg.augment_d4 else None)

    history: list[dict] = []
    best = {"dice": -1.0, "epoch": -1}
    t0 = time.time()
    steps_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    from ablation2d.window import GrowingWindow
    window = GrowingWindow.from_config(cfg)

    for epoch in range(cfg.epochs):
        if cfg.time_budget_s and epoch and time.time() - t0 > cfg.time_budget_s:
            log(f"time budget reached at epoch {epoch} "
                f"({time.time() - t0:.0f} s > {cfg.time_budget_s:.0f} s)")
            break
        # Warmup then cosine. An NCA's first epochs are where a too-large step
        # can push the state into the tanh saturation it never comes back from.
        if epoch < cfg.warmup_epochs:
            lr = cfg.lr * (epoch + 1) / cfg.warmup_epochs
        else:
            t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * t))
        for g in opt.param_groups:
            g["lr"] = lr

        run_loss = 0.0
        for x, y in train_ds.batches(cfg.batch_size, generator=gen):
            if aug_gen is not None:
                x, y = augment_d4(x, y, aug_gen)
            state = model.seed(x)

            # TRICK 2 — start from a state the model actually reaches.
            if cfg.further_steps_rate > 0 and torch.rand(1).item() < cfg.further_steps_rate:
                pre = (window.sample() if window is not None else
                       int(torch.randint(cfg.step_min, cfg.step_max + 1, (1,)).item()))
                with torch.no_grad():
                    _, state = model(x, steps=pre, state=state, return_state=True)
                state = state.detach()

            # TRICK 3 — sometimes hand it the answer and ask it to keep it.
            if cfg.target_to_seed_rate > 0 and torch.rand(1).item() < cfg.target_to_seed_rate:
                state = state.clone()
                # The readouts are sigmoids, so seed the PRE-activation — BOTH
                # of them, or the vessel head never learns it has a fixed point.
                state[:, :y.shape[1]] = torch.logit(y.clamp(1e-4, 1 - 1e-4)) * 0.25

            # TRICK 1 — sample the rollout length. With the growing window on,
            # this draw comes from [lo, hi] (plus a sliver of exploration
            # beyond) instead of the full fixed range; the window only widens
            # on the measured evidence its decisions log.
            k = (window.sample() if window is not None else
                 int(torch.randint(cfg.step_min, cfg.step_max + 1, (1,)).item()))

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred, rolled = model(x, steps=k, state=state, return_state=True)
            loss = multitask_loss(pred.float(), y, cfg.foreground_weight,
                                  cfg.vessel_weight, cfg.vessel_pos_weight,
                                  log_vars)

            # TRICK 5 — persistence, with truncated BPTT.
            if cfg.persist_rate > 0 and torch.rand(1).item() < cfg.persist_rate:
                extra = int(torch.randint(*cfg.persist_range, (1,)).item())
                free = max(0, extra - cfg.persist_bptt)
                s2 = rolled.detach()
                if free:
                    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype,
                                                         enabled=use_amp):
                        _, s2 = model(x, steps=free, state=s2, return_state=True)
                    s2 = s2.detach()
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred2 = model(x, steps=min(extra, cfg.persist_bptt), state=s2)
                loss = loss + multitask_loss(pred2.float(), y,
                                             cfg.foreground_weight,
                                             cfg.vessel_weight,
                                             cfg.vessel_pos_weight, log_vars)

            loss.backward()
            normalise_grads(model, cfg.grad_norm_eps)          # TRICK 4
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # second net
            opt.step()
            run_loss += float(loss.detach())

        rec = {"epoch": epoch, "lr": lr, "train_loss": run_loss / steps_per_epoch,
               "wall_s": round(time.time() - t0, 1)}
        if epoch % cfg.validate_every == 0 or epoch == cfg.epochs - 1:
            rec["gpu"] = _gpu_state()
        if epoch % cfg.validate_every == 0 or epoch == cfg.epochs - 1:
            vl, vd, vf = evaluate_full(model, val_ds, cfg.infer_steps,
                                       cfg.batch_size, cfg.foreground_weight,
                                       cfg.vessel_weight)
            rec["val_loss"], rec["val_dice"], rec["val_vessel_f1"] = vl, vd, vf
            rec["vessel_threshold"] = getattr(evaluate_full, "vessel_threshold", 0.5)
            if log_vars is not None:
                rec["log_vars"] = [round(float(v), 3) for v in log_vars]
            # Which number picks the kept weights. During vessel pretraining
            # the necrosis head is not being trained at all, so its DSC is
            # noise and selecting on it would keep an arbitrary epoch.
            #
            # KNOWN ISSUE, not fixed here because changing it mid-campaign would
            # make the arms incomparable. Validation DSC swings ~±0.04 between
            # validations while validation LOSS falls monotonically — measured on
            # the rebalanced arm: DSC 0.764/0.794/0.793/0.803/0.785/0.863/0.851/
            # 0.792 against loss 0.455/0.452/0.412/0.391/0.400/0.375/0.373/0.360.
            # Selecting the best DSC therefore keeps whichever epoch got the
            # luckiest validation pass, and biases the reported score optimistic
            # by about the size of that swing. It is NOT inference noise: that
            # is ±0.004 per case, ~0.0002 over 400 validation samples. Select on
            # loss, or on a running mean of DSC, next time.
            score = vf if select == "vessel_f1" else vd
            if score > best["dice"]:
                best = {"dice": score, "epoch": epoch,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
        if window is not None:
            rec["window"] = [window.lo, window.hi]
            if window.due(epoch):
                # Probe under FORKED RNG: each K gets the same fire-mask stream
                # prefix (common random numbers), and the training stream is
                # left exactly where it was — the probes must not perturb the
                # run they are measuring.
                curve, vcurve = {}, {}
                for kk in window.probe_ks():
                    with torch.random.fork_rng():
                        torch.manual_seed(0xC0FFEE)
                        with torch.no_grad():
                            _, d, f1 = evaluate_full(model, val_ds, kk,
                                                     cfg.batch_size,
                                                     tune_threshold=False)
                        curve[kk] = float(d)
                        vcurve[kk] = float(f1)
                dec = window.decide(epoch, curve, vcurve)
                rec["window_decision"] = dec.to_row()
                log(f"  window {dec.window[0]}-{dec.window[1]} -> "
                    f"{window.lo}-{window.hi}  {dec.action}: {dec.reason}")
        history.append(rec)
        if on_epoch is not None:
            on_epoch(epoch, rec, model)
        elif "val_dice" in rec:
            log(f"  epoch {epoch:4d}  loss {rec['train_loss']:.5f}  "
                f"val {rec['val_loss']:.5f}  DSC {rec['val_dice']:.4f}  "
                f"vesselF1 {rec['val_vessel_f1']:.4f}"
                f"@{rec.get('vessel_threshold', 0.5):.2f}"
                + (f"  logvar {rec['log_vars'][0]:+.2f}/{rec['log_vars'][1]:+.2f}"
                   if 'log_vars' in rec else "")
                + f"  {rec['wall_s']:.0f}s" + rec.get("gpu", ""))

    # Best-by-validation weights are what gets kept, so a longer run can only
    # cost compute — never a worse result.
    if best["epoch"] >= 0:
        model.load_state_dict(best["state"])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                    "best_epoch": best["epoch"], "best_val_dice": best["dice"],
                    "vessel_threshold": getattr(evaluate_full,
                                                "vessel_threshold", 0.5),
                    "log_vars": ([float(v) for v in log_vars]
                                 if log_vars is not None else None)},
                   out_dir / "model.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return model, history


# --------------------------------------------------------------------------- #
# two-stage training
# --------------------------------------------------------------------------- #
def pretrain_vessels(cfg: TrainConfig, root: Path | str, *, model=None,
                     log=print):
    """Vessel-only warm start, then hand the weights to the dual task.

    Returns `(model, history)`; `history` is empty and the model untouched when
    `cfg.pretrain_epochs == 0`.

    The corpus is CT-in, vessel-out, with the needle and power channels at zero
    — see `scripts/make_vessel_corpus.py`. Two things make this worth doing
    rather than just training the dual task for longer:

    * **it is more data, not the same data twice.** Every slice appears, not
      only the ones a feasible plan could be sampled in, and each appears ONCE
      rather than once per plan — so the vessel signal is not diluted by
      near-duplicate rows that differ only in where a needle went.
    * **it is the stationary half of the problem.** Vessel morphology depends on
      the CT alone; the necrosis field moves with every plan. Learning the fixed
      part with the necrosis gradient switched off means the dual phase starts
      from a state that already encodes the anatomy.

    Selection is on vessel F1, not DSC: with `pretrain_death_weight = 0` the
    necrosis head is not being trained here and its DSC is noise.
    """
    if cfg.pretrain_epochs <= 0:
        return model, []
    root = Path(root)
    tr, va = root / "train.npz", root / "val.npz"
    if not (tr.exists() and va.exists()):
        raise FileNotFoundError(
            f"no vessel corpus at {root}. Build it with "
            f"`python3 scripts/make_vessel_corpus.py` — it needs no solver and "
            f"takes seconds.")
    pre_ds = AblationDataset(tr, device=cfg.device)
    pre_val = AblationDataset(va, device=cfg.device)
    pcfg = dataclasses.replace(
        cfg, epochs=cfg.pretrain_epochs,
        foreground_weight=cfg.pretrain_death_weight,
        vessel_weight=1.0,
        uncertainty_weighting=False,      # nothing to balance with one task
        persist_rate=0.0,                 # retention is the dual task's problem
        target_to_seed_rate=0.0)
    log(f"[pre ] vessel-only warm start: {len(pre_ds)} slices, "
        f"{cfg.pretrain_epochs} epochs, death weight "
        f"{cfg.pretrain_death_weight}")
    return train(pcfg, pre_ds, pre_val, model=model, log=log,
                 select="vessel_f1")


def train_staged(cfg: TrainConfig, train_ds, val_ds, *, model=None,
                 stage1_frac: float = 0.7, factor: int = 2,
                 mode: str = "coarse", policy: str = "sparse+vessel",
                 patch: int = 64, out_dir: Path | None = None,
                 on_epoch=None, log=print):
    """Most of the epochs cheap, the last few at full resolution.

    The weights are a 3x3 rule and carry over unchanged — there is nothing to
    convert between the stages, which is the whole reason this works on an NCA
    and not on a U-Net.

    `mode="coarse"` halves the resolution (and, with it, the rollout length —
    see `multires.scale_steps`). `mode="patch"` keeps the resolution and crops.
    `mode="none"` is the single-stage control, so the three can be compared
    under one code path rather than three.

    **Stage 2 always runs at full resolution and full rollout**, and its
    validation is the only score that may be quoted: a coarse stage's DSC is
    computed against a mean-pooled target and is not the same quantity.
    """
    from .multires import CoarseView, PatchView, scale_steps

    e1 = int(round(cfg.epochs * stage1_frac))
    e2 = cfg.epochs - e1
    hist: list[dict] = []

    if mode == "none" or e1 == 0:
        return train(cfg, train_ds, val_ds, model=model, out_dir=out_dir,
                     on_epoch=on_epoch, log=log)

    if mode == "coarse":
        tr1, va1 = CoarseView(train_ds, factor, policy), CoarseView(val_ds, factor, policy)
        c1 = dataclasses.replace(
            cfg, epochs=e1,
            step_min=scale_steps(cfg.step_min, factor),
            step_max=scale_steps(cfg.step_max, factor),
            infer_steps=scale_steps(cfg.infer_steps, factor),
            persist_range=(scale_steps(cfg.persist_range[0], factor),
                           scale_steps(cfg.persist_range[1], factor)),
            persist_bptt=scale_steps(cfg.persist_bptt, factor, floor=2))
    elif mode == "patch":
        tr1, va1 = PatchView(train_ds, patch, cfg.seed), val_ds
        c1 = dataclasses.replace(cfg, epochs=e1)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    log(f"[stage 1] {mode}  {e1} epochs  grid {tr1.shape}  "
        f"K {c1.step_min}-{c1.step_max}")
    model, h1 = train(c1, tr1, va1, model=model, on_epoch=on_epoch, log=log)
    for r in h1:
        r["stage"] = 1
    hist += h1

    # Stage 2 restarts the LR schedule at a fraction of the peak: the coarse
    # weights are a good rule at the wrong scale, not a bad rule, and hitting
    # them with the full warmup peak throws away most of what stage 1 bought.
    c2 = dataclasses.replace(cfg, epochs=e2, lr=cfg.lr * 0.3, warmup_epochs=2)
    log(f"[stage 2] full  {e2} epochs  grid {train_ds.shape}  "
        f"K {c2.step_min}-{c2.step_max}  lr {c2.lr:.2e}")
    model, h2 = train(c2, train_ds, val_ds, model=model, out_dir=out_dir,
                      on_epoch=on_epoch, log=log)
    for r in h2:
        r["stage"] = 2
        r["epoch"] += e1
    hist += h2
    return model, hist


def load_checkpoint(path, device="cpu"):
    """(model, cfg, checkpoint). `ck["config_defaulted"]` lists the fields the
    stored config did NOT contain.

    A checkpoint outlives its own `TrainConfig`. Fields added since it was
    written silently take today's DEFAULT, so a run that had no augmentation
    comes back claiming `augment_d4=True` — and an A/B between an old arm and a
    new one would then be reported with both arms looking identically
    configured. The missing keys are recorded so the lie is visible.
    """
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    stored = dict(ck["config"])
    ck["config_defaulted"] = sorted(known - set(stored))
    ck["config_unknown"] = sorted(set(stored) - known)
    cfg = TrainConfig(**{k: v for k, v in stored.items() if k in known})
    model = AblationCNCA(channels=cfg.channels, hidden_mult=cfg.hidden_mult,
                         n_sub_models=cfg.n_sub_models, fire_rate=cfg.fire_rate,
                         restore_conditioning=cfg.restore_conditioning)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, cfg, ck
