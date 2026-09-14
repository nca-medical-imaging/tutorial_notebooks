"""A 2-D chained NCA for microwave ablation.

Deliberately small and readable — this is teaching code, not a library. It is
the 2-D twin of the C-NCA in `nca-forge`, with the same three structural
choices, so a number measured here means something next to a number measured
there.

**1. The conditioning lives IN the state.** There is no separate encoder. The
grid is one tensor whose first channels are the answer and the plan:

    [0]      cell_death        <- what we read out
    [1:9]    the 8 input channels (see channels.py)
    [9:C]    hidden / communication

**2. Chained sub-models.** One CA step is not one convolution block but N of
them applied in sequence, each adding a residual:

    for sub in sub_models:  grid = grid + sub(grid)

That is the "chained" in C-NCA. It buys receptive field per step — each
sub-model is two 3x3 convolutions, so one step reaches 2N cells — without
adding a second set of weights per step the way a deeper single block would.

**3. No normalisation anywhere.** Not an oversight. A normalisation layer reads
statistics over the whole grid, and a cell that can see the whole grid is not a
cellular automaton any more. Everything that keeps the state bounded here is
local: a soft clamp, a small init, and a zero-init output layer.

Where this DIFFERS from `nca.py` (the detection model in the same repo), the
difference is the task:

* the readout is a **rate**, not a probability, so the output is read straight
  off the channel through a sigmoid at the end rather than being a logit map;
* the environment channels are re-written every step by default
  (`restore_conditioning=True`). The 3-D production model runs with this OFF and
  lets the plan channels drift; that is one of the ablations in the notebook,
  not a settled question.

**With `restore_conditioning=False` the automaton may rewrite every channel it
has, the CT and the plan included.** That is the more general model — a rule
that transforms its whole state rather than one held in place by an external
hand — and it is what a C-NCA is in the literature. It also has a measured
failure mode. nca-forge instrumented exactly this on its 3-D surrogate
(`configs/surrogate/experiments/conditioning_restore_2mm.yaml`) and found, at
the model's own selected depth of 9 steps:

    ch  name                    input mean   mean|drift|    ratio
     1  applicator_activation     0.0001       0.8888      7630x
     2  applicator_duration       0.0004       0.6071      1730x
     6  needle                    0.0045       0.3391        75x
     4  perfusion                 0.0890       0.2457       2.8x
     3  density                   0.5190       0.4139       0.8x
     5  specific_heat             0.7906       0.5322       0.7x

The dense tissue channels are merely corrupted; the SPARSE ones — the treatment
plan — are obliterated, by three to four orders of magnitude more than the
signal they encode. Whatever the network learns about the plan it must extract
in the first few steps, because after that the plan is gone. That is a coherent
explanation for why their model selection settles on nine iterations while ours,
which restores, is still flat at K = 32.

So: freedom to transform everything is the more expressive model AND the one
that can eat its own inputs. Both arms are worth running, which is why this is a
flag and not a constant.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channels import (CHANNEL_NAMES, N_INPUT_CHANNELS,
                       N_TARGET_CHANNELS, TARGET_NAMES)

#: The first channels of the state ARE the answers, in `channels.TARGET_NAMES`
#: order: [0] cell_death, [1] vessel.
DEATH_CH, VESSEL_CH = 0, 1


def channel_roles(n_channels: int, restored: bool = True) -> list[str]:
    """A name for every slot of the state, in order.

    The state is not a bag of equivalent features: the first two are the
    readouts, the next three are the environment the automaton is forbidden to
    edit, and everything after is scratch. Anything that plots a state channel
    should say which kind it is looking at.

    `restored` says what the middle group MEANS. With restoration on they are an
    environment the automaton may read and not write, and any drift is a bug.
    With it off they are ordinary state that merely STARTS as the inputs — drift
    is then expected, and the thing to watch is whether the plan survives long
    enough to be used (nca-forge measured 7630x drift on its sparse power
    channel within nine steps).
    """
    kind = "cond" if restored else "was"
    names = list(TARGET_NAMES) + [f"{kind}: {c}" for c in CHANNEL_NAMES]
    return names + [f"hidden {i}" for i in range(n_channels - len(names))]
OUT_SLICE = slice(0, N_TARGET_CHANNELS)
#: then the conditioning, then whatever is left over as scratch.
COND_SLICE = slice(N_TARGET_CHANNELS, N_TARGET_CHANNELS + N_INPUT_CHANNELS)

#: TRICK — bound the state SOFTLY. `x = x + dx` iterated 15-40 times is an
#: unbounded random walk; with a residual readout the loss stays finite and just
#: stops improving, which is the worst kind of failure. A hard clamp fixes the
#: explosion and introduces a worse bug (zero gradient outside the bounds, so
#: training stops dead once the state parks on the boundary). tanh saturates
#: just as firmly, stays differentiable, and is the identity where the state
#: actually lives.
STATE_CLAMP = 4.0


class AblationCNCA(nn.Module):
    """Chained NCA surrogate for the necrosis field and the vessel segmentation.

    Args:
        channels: total state width. 2 readouts (necrosis, vessels) + 3
            conditioning (CT, needle, power) + hidden scratch.
        hidden_mult: sub-model width, as a multiple of `channels`.
        n_sub_models: chained blocks per CA step.
        fire_rate: probability a cell updates on a given sub-step. < 1 breaks
            global synchrony, which is what stops the grid from learning a
            choreography instead of a rule.
        restore_conditioning: re-write the plan/tissue channels every step.
            False lets the automaton transform ITS WHOLE STATE, the CT and the
            plan included — the more general rule, and the one that can destroy
            its own inputs. See the module docstring for the measurement.
    """

    def __init__(self, channels: int = 16, hidden_mult: int = 3,
                 n_sub_models: int = 3, kernel_size: int = 3,
                 fire_rate: float = 0.5, restore_conditioning: bool = True):
        super().__init__()
        if channels < N_TARGET_CHANNELS + N_INPUT_CHANNELS:
            raise ValueError(
                f"channels={channels} cannot hold {N_TARGET_CHANNELS} readouts + "
                f"{N_INPUT_CHANNELS} conditioning channels")
        self.channels = channels
        self.n_sub_models = n_sub_models
        self.fire_rate = fire_rate
        self.restore_conditioning = restore_conditioning
        pad = kernel_size // 2
        hidden = channels * hidden_mult

        self.sub_models = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size, padding=pad),
                nn.ReLU(),
                nn.Conv2d(hidden, channels, kernel_size, padding=pad, bias=False),
            ) for _ in range(n_sub_models)
        ])
        self._init_weights()

    def _init_weights(self):
        for sub in self.sub_models:
            first, last = sub[0], sub[2]
            # He-ish, scaled down 10x: an NCA composes its own update tens of
            # times, so a "correct" single-layer init is an order of magnitude
            # too hot by step 20.
            fan_in = first.weight.shape[1] * first.weight.shape[2] * first.weight.shape[3]
            nn.init.normal_(first.weight, 0.0, (2.0 / fan_in) ** 0.5 * 0.1)
            nn.init.zeros_(first.bias)
            # ZERO-INIT THE OUTPUT LAYER. Every sub-model starts as an exact
            # no-op, so an untrained model predicts a flat field — which is a
            # free sanity check you can run before training a single step, and
            # a guarantee that the first gradients are not fighting noise the
            # init created.
            nn.init.zeros_(last.weight)

    # -- one update ---------------------------------------------------------
    def step(self, x: torch.Tensor) -> torch.Tensor:
        for sub in self.sub_models:
            dx = sub(x)
            if self.fire_rate < 1.0:
                # PER CELL, not per channel: a cell either takes its update or
                # does not. Masking channels independently would let a cell
                # advance half of its state, which is not asynchrony, it is
                # noise.
                mask = (torch.rand_like(x[:, :1]) <= self.fire_rate).to(x.dtype)
                dx = dx * mask
            x = STATE_CLAMP * torch.tanh((x + dx) / STATE_CLAMP)
        return x

    # -- rollout ------------------------------------------------------------
    def seed(self, inputs: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) inputs -> (B, C, H, W) initial state.

        Both readouts start at zero: nothing is dead before the applicator
        fires, and the automaton has not found a vessel yet either. It grows
        both answers rather than carving them out of a guess.
        """
        b, _, h, w = inputs.shape
        x = inputs.new_zeros(b, self.channels, h, w)
        x[:, COND_SLICE] = inputs
        return x

    def forward(self, inputs: torch.Tensor, steps: int = 15, *,
                state: torch.Tensor | None = None,
                return_trace: bool = False, return_state: bool = False,
                trace_full: bool = False):
        """Returns (B, 2, H, W) in [0, 1] — necrosis, then vessel.

        `state` continues a previous rollout instead of starting from the seed —
        that is what makes persistence training and the "watch it think"
        animation possible.

        `return_trace` records one frame per step. By default a frame is the two
        READOUTS, squashed — what the model currently believes. `trace_full=True`
        records the WHOLE state instead, all `channels` of it, RAW:

          * the readouts are left as logits, not sigmoids, because the point of
            looking at them per-step is to see saturation, and a sigmoid hides
            exactly that;
          * the conditioning channels are included even though they are rewritten
            every step — seeing them sit still is how you confirm
            `restore_conditioning` is doing what it claims;
          * the rest are the automaton's scratch space, which has no units and no
            preferred sign. `CHANNEL_ROLES` names each slot.

        It costs `steps x channels x H x W` floats, so it is opt-in.
        """
        x = self.seed(inputs) if state is None else state
        cond = inputs
        trace = []
        for _ in range(steps):
            x = self.step(x)
            if self.restore_conditioning:
                # The CT and the plan are the ENVIRONMENT, not something the
                # automaton is allowed to edit. Re-writing them every step is
                # also what lets you move a needle mid-rollout and watch the
                # field re-converge, which is the interactive demo.
                x = torch.cat([x[:, OUT_SLICE], cond,
                               x[:, COND_SLICE.stop:]], dim=1)
            if return_trace:
                trace.append((x if trace_full
                              else torch.sigmoid(x[:, OUT_SLICE])).detach().clone())
        out = torch.sigmoid(x[:, OUT_SLICE])
        if return_trace:
            return (out, trace, x) if return_state else (out, trace)
        return (out, x) if return_state else out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def reach_cells(self) -> int:
        """Cells of receptive field per CA step: 2 per 3x3 conv, N sub-models."""
        return 2 * self.n_sub_models


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
    """1 - soft Dice, per sample, meaned.

    The right loss for a thin binary structure and the reason the first version
    of this model could not segment. Weighted MSE on a 2.5 %-positive mask is
    minimised by predicting a low, blurry, *confident-nowhere* field: the
    gradient from a background cell at 0.05 is tiny, and there are forty times
    more of them. Dice is scale-free in the positive class — it cares about
    OVERLAP, so it pushes the prediction across 0.5 where the vessel is instead
    of merely nudging it upward.
    """
    p = pred.flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1.0 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def masked_bce(pred: torch.Tensor, target: torch.Tensor,
               pos_weight: float = 20.0, eps: float = 1e-6) -> torch.Tensor:
    """Positive-weighted BCE on a probability (the readout is already sigmoid).

    Paired with Dice because the two fail differently: Dice is indifferent to
    calibration away from the boundary and can sit happily at a degenerate
    all-or-nothing solution early in training, while BCE gives every cell a
    gradient from step one. Dice fixes what BCE gets wrong under imbalance, and
    BCE fixes what Dice gets wrong at initialisation.
    """
    p = pred.clamp(eps, 1.0 - eps)
    loss = -(pos_weight * target * torch.log(p) + (1 - target) * torch.log(1 - p))
    return loss.mean()


def multitask_loss(pred: torch.Tensor, target: torch.Tensor,
                   death_weight: float = 100.0, vessel_weight: float = 1.0,
                   vessel_pos_weight: float = 20.0,
                   log_vars: torch.Tensor | None = None) -> torch.Tensor:
    """Necrosis + vessel segmentation, each with the loss its target deserves.

    * **necrosis** is a continuous damage field in [0,1] with a meaningful
      interior, so foreground-weighted MSE is right: getting 0.8 where the truth
      is 0.9 should cost something.
    * **vessel** is a thin binary mask covering ~2.5 % of the frame, so it gets
      **Dice + weighted BCE**. The first version used weighted MSE here too and
      the head scored F1 0.17-0.22 against a 0.272 threshold baseline — worse
      than a `>` on the raw Hounsfield channel.

    `log_vars`, when supplied, is a learnable 2-vector of log-variances and the
    two terms are combined as Kendall & Gal's homoscedastic uncertainty
    weighting, `sum_i exp(-s_i) L_i + s_i`. That removes `vessel_weight` as a
    hand-set number: the model learns the balance, and the learned values are
    themselves informative — if it drives the vessel variance up, it is telling
    you the task is not helping.
    """
    # death_weight == 0 DROPS the necrosis term. It does not merely un-weight
    # it: `necrosis_loss` weights foreground by `death_weight` and background by
    # a fixed 1.0, so with a target that is all zeros — the vessel-only warm
    # start — every pixel is background and a "disabled" head is in fact trained
    # hard to output zero everywhere, on a full-weight gradient, competing with
    # the task the phase exists for.
    v = (soft_dice_loss(pred[:, 1:2], target[:, 1:2])
         + masked_bce(pred[:, 1:2], target[:, 1:2], vessel_pos_weight))
    if death_weight == 0 and log_vars is None:
        return vessel_weight * v
    d = necrosis_loss(pred[:, 0:1], target[:, 0:1], death_weight)
    if log_vars is not None:
        return (torch.exp(-log_vars[0]) * d + log_vars[0]
                + torch.exp(-log_vars[1]) * v + log_vars[1])
    return d + vessel_weight * v


def necrosis_loss(pred: torch.Tensor, target: torch.Tensor,
                  foreground_weight: float = 100.0,
                  background_weight: float = 1.0,
                  threshold: float = 0.01) -> torch.Tensor:
    """Foreground-weighted MSE — the production C-NCA's `mse_uc`.

    A lesion covers 1-4 % of the slice, so an unweighted MSE is minimised by
    predicting zero everywhere and scores well doing it. 100:1 says a missed
    necrotic cell costs a hundred invented ones.

    That ratio is a REAL hyperparameter and it has a measured cost: the 3-D
    models trained at 100:1 come out at recall 0.935 against precision 0.880,
    i.e. deliberately 4.6 % over-inclusive. For an ablation margin that is the
    right direction to be wrong in, but say so rather than letting the number
    look neutral.
    """
    w = torch.where(target > threshold,
                    torch.as_tensor(foreground_weight, device=pred.device, dtype=pred.dtype),
                    torch.as_tensor(background_weight, device=pred.device, dtype=pred.dtype))
    return (w * (pred - target) ** 2).sum() / w.sum()


@torch.no_grad()
def vessel_f1(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5) -> torch.Tensor:
    """F1 of the vessel head. The number to beat is the best single HU threshold
    on THE SAME INPUT the model sees, fitted on the split it is scored on so it
    is not handicapped: **0.245** on the full real-CT test split (0.226 val),
    **0.439** once the CT is masked to the liver (0.432 val). Masking helps the
    threshold more than the model, so a full-CT baseline flatters a masked model. The procedural
    corpus gave 0.272, and on it the head never got past 0.18: synthetic vessels
    are bright blobs, so `hu > t` is the whole rule and there is no shape left
    to learn."""
    p = (pred[:, 1] > threshold).flatten(1).float()
    t = (target[:, 1] > 0.5).flatten(1).float()
    tp = (p * t).sum(1)
    den = p.sum(1) + t.sum(1)
    return torch.where(den > 0, 2 * tp / den, torch.ones_like(den))


@torch.no_grad()
def best_vessel_threshold(pred: torch.Tensor, target: torch.Tensor,
                          candidates=None):
    """The decision threshold, chosen on VALIDATION — like the rollout length.

    0.5 is not a law, it is the point where a sigmoid crosses a half. A head
    trained under class imbalance is systematically under-confident, and
    reporting its F1 at 0.5 measures the calibration rather than the
    segmentation. Returns `(threshold, f1)`.
    """
    cands = candidates if candidates is not None else \
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    best = (0.5, -1.0)
    for t in cands:
        f = float(vessel_f1(pred, target, t).mean())
        if f > best[1]:
            best = (float(t), f)
    return best


@torch.no_grad()
def dice(pred: torch.Tensor, target: torch.Tensor, contour: float = 0.99,
         pred_threshold: float = 0.5) -> torch.Tensor:
    """DSC of the necrosis contour, per sample.

    The target is thresholded at the corpus's own d >= 0.99 contour, and the
    PREDICTION at 0.5 — they are not the same number and should not be: 0.99 is
    where the physics says the tissue is dead, 0.5 is where the model is more
    sure than not that it is.
    """
    # The DEATH head only — channel 0. `vessel_f1` scores the other one.
    if pred.shape[1] > 1:
        pred, target = pred[:, 0:1], target[:, 0:1]
    p = (pred > pred_threshold).flatten(1).float()
    t = (target >= contour).flatten(1).float()
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    return torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
