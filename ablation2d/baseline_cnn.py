"""A parameter-matched dilated CNN — the control for "what does iterating buy?".

The comparison is only worth making if the CNN is given every advantage that is
not *iteration*: the same inputs, the same corpus, the same loss, the same
split, the same schedule, and a receptive field at least as large as the NCA's.

* **Same parameter budget.** Width is chosen to land within a few percent of the
  NCA's count, so the answer cannot be "the NCA had more capacity".
* **At least the same reach.** The NCA sees `2 x n_sub_models x steps` cells —
  90 at the shipped setting. Dilations 1,2,4,8,16,32 on 3x3 kernels give
  1 + 2(1+2+4+8+16+32) = **127**. The CNN sees further, on purpose: if it still
  loses, reach is not the explanation.
* **No downsampling.** A U-Net would win on reach and lose the comparison's
  meaning — the question is what the *recurrence* buys, not what pooling buys.

What it cannot have is the NCA's one structural gift: a rule that is applied
over and over, so that information propagates and cells settle against their
neighbours. That is the whole variable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .channels import N_INPUT_CHANNELS, N_TARGET_CHANNELS

DILATIONS = (1, 2, 4, 8, 16, 32)


class DilatedCNN(nn.Module):
    def __init__(self, width: int = 28, dilations=DILATIONS,
                 in_channels: int = N_INPUT_CHANNELS, out_channels: int = 2):
        super().__init__()
        self.dilations = tuple(dilations)
        layers = [nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU()]
        for d in self.dilations:
            layers += [nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.ReLU()]
        layers += [nn.Conv2d(width, out_channels, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def seed(self, inputs):
        """A CNN has no state to seed. It is returned unchanged so the SAME
        training loop drives both models — the comparison is worthless if the
        CNN gets its own loop with its own bugs."""
        return inputs

    def forward(self, inputs, steps=None, state=None,
                return_trace=False, return_state=False):
        """`steps` is accepted and IGNORED, so the CNN is a drop-in for the NCA
        everywhere the evaluation code takes a rollout length. A CNN has no
        rollout; pretending otherwise in the call signature is what lets the two
        share one `report()`."""
        out = torch.sigmoid(self.net(inputs))
        if return_trace:
            return (out, [out], inputs) if return_state else (out, [out])
        return (out, inputs) if return_state else out

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def reach_cells(self) -> int:
        return 1 + 2 * sum(self.dilations)


def match_params(target: int, dilations=DILATIONS, tol: float = 0.10) -> int:
    """The width whose parameter count is closest to `target`.

    The tolerance is loose (10 %) because width is an integer and the count goes
    as w^2: at v2's ~10 k budget the neighbouring widths are 9 596 and 11 090,
    so nothing lands closer than 6 %. `run_study` prints BOTH counts rather than
    claiming a match it does not have — a comparison whose parameter budgets
    differ by 6 % in the CNN's favour is still a comparison the NCA has to win.
    """
    best, best_err = None, float("inf")
    for w in range(4, 129):
        n = DilatedCNN(w, dilations).n_params
        err = abs(n - target) / target
        if err < best_err:
            best, best_err = w, err
    if best_err > tol:
        raise RuntimeError(f"closest width {best} is {best_err:.1%} off {target}")
    return best
