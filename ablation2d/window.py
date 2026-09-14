"""The growing rollout window — nca-forge's adaptive iteration window, scaled
to this repo.

nca-forge trains surrogates whose rollout range is not a fixed ``[k_min, k_max]``
but a WINDOW that starts narrow and grows **only on measured benefit** (its
`IterationWindow`: growth is irreversible, evidence-driven, and every decision
is logged with the curve it read). The same principle lands here on the
sampled-length trick:

* ``step_range`` stays the ceiling. The window ``[lo, lo + width]`` is where
  training actually samples K from, starting narrow.
* Every ``period`` epochs the trainer probes validation DSC at a few K inside
  the window and a few beyond it — same split, same weights, RNG state forked
  per probe, so the Ks are compared under common random numbers and the
  comparison measures K, not dice luck.
* The ceiling rises by ``step`` only when the best K beyond the window beats
  the best inside by at least ``margin``, ``streak`` probes in a row. A hand
  grown curriculum cannot know when longer rollouts start paying; this does,
  and if they never do the window never grows and the compute was never spent.
* A small ``explore_rate`` fraction of batches trains slightly beyond the
  ceiling even while the window is narrow — otherwise the probe beyond it
  would measure a K the model has literally never seen (the chicken-and-egg
  nca-forge solves the same way).

Why bother, when sampled lengths already work (the deck's trick 1)? Two
measured reasons to expect a win, and one honest unknown:

1. **Early epochs are spent where the gradient is.** A fresh model is wrong
   everywhere at K = 18, so an 18-step rollout supervises mostly noise; the
   same batch at K = 5 supervises a lesion boundary it can actually reach.
   Narrow windows front-load the informative gradients.
2. **Cost scales with K.** Epochs whose Ks are drawn from [3, 7] are ~half the
   rollouts of [3, 18] — the early curriculum is also the cheap curriculum.
3. The unknown — whether the saved epochs buy a better *final* model or only a
   faster early one — is exactly what the A/B in `scripts/run_study.py
   --window` measures before this goes anywhere near a default.
"""
from dataclasses import dataclass, field


@dataclass
class WindowDecision:
    """What the controller concluded, and the evidence it concluded it from.

    Logged verbatim into the epoch's history entry. Every field is evidence,
    not commentary: months later the row must be enough to re-derive the
    decision without the model or the run.
    """
    epoch: int
    window: tuple[int, int]
    action: str                      # "grow" | "hold"
    reason: str
    curve: dict[int, float]          # K -> val DSC as probed (common random numbers)
    best_inside: float
    best_beyond: float
    margin: float
    streak: int
    new_window: tuple[int, int] | None = None
    vessel_inside: float | None = None
    vessel_beyond: float | None = None
    vessel_curve: dict | None = None

    def to_row(self) -> dict:
        row = {"epoch": self.epoch, "window": list(self.window),
               "action": self.action, "reason": self.reason,
               "curve": {str(k): round(v, 4) for k, v in self.curve.items()},
               "best_inside": round(self.best_inside, 4),
               "best_beyond": round(self.best_beyond, 4),
               "margin": self.margin, "streak": self.streak}
        if self.new_window is not None:
            row["new_window"] = list(self.new_window)
        if self.vessel_curve is not None:
            row["vessel_curve"] = self.vessel_curve
            row["vessel_best_inside"] = (None if self.vessel_inside is None
                                         else round(self.vessel_inside, 4))
            row["vessel_best_beyond"] = (None if self.vessel_beyond is None
                                         else round(self.vessel_beyond, 4))
        return row


class GrowingWindow:
    """Sample K from a window that only ever grows, and only on evidence.

    Construct via :meth:`from_config` from a ``TrainConfig``; drive it from the
    training loop as::

        window = GrowingWindow.from_config(cfg)      # None unless cfg.window_grow
        k = window.sample()                          # int in [lo, hi], or beyond
                                                    # at explore_rate while armed
        ...
        if window.due(epoch):
            with torch.random.fork_rng():
                curve = {kk: probe(kk) for kk in window.probe_ks()}
            record["window_decision"] = window.decide(epoch, curve).to_row()

    ``sample()`` draws from the global torch RNG — the same stream the fixed
    range sampled from — so a run stays reproducible seed-for-seed.
    """

    def __init__(self, lo: int, hi: int, k_max: int, *, period: int = 5,
                 margin: float = 0.003, step: int = 2, streak_needed: int = 2,
                 grow_after: int = 5, explore_rate: float = 0.1):
        if not lo <= hi <= k_max:
            raise ValueError(f"need lo <= hi <= k_max, got {lo}, {hi}, {k_max}")
        self.lo, self.hi, self.k_max = int(lo), int(hi), int(k_max)
        self.period = int(period)
        self.margin = float(margin)
        self.step = int(step)
        self.streak_needed = int(streak_needed)
        self.grow_after = int(grow_after)
        self.explore_rate = float(explore_rate)
        self.streak = 0
        self.explore_draws = 0
        self.decisions: list[WindowDecision] = []

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "GrowingWindow | None":
        """None unless the config asks for it — the fixed range is the default."""
        if not getattr(cfg, "window_grow", False):
            return None
        return cls(cfg.step_min, cfg.step_min + cfg.window_init_width,
                   cfg.step_max, period=cfg.window_period,
                   margin=cfg.window_margin, step=cfg.window_step,
                   streak_needed=cfg.window_streak,
                   grow_after=cfg.window_grow_after,
                   explore_rate=cfg.window_explore_rate)

    # -- state -------------------------------------------------------------

    @property
    def ceiling(self) -> int:
        """The highest `hi` may ever reach — the config's step_max."""
        return self.k_max

    @property
    def can_grow(self) -> bool:
        return self.hi < self.ceiling

    def due(self, epoch: int) -> bool:
        """Is a decision owed at the END of this epoch? Never during warmup."""
        return (epoch >= self.grow_after
                and (epoch + 1) % self.period == 0 and self.can_grow)

    # -- training side -----------------------------------------------------

    def sample(self) -> int:
        """One rollout length: uniform in [lo, hi], or beyond it while armed.

        The exploration band is [hi + 1, min(hi + step, ceiling)] — one growth
        step ahead, never more, so the beyond-the-window probe reads a K the
        model has at least occasionally trained at.
        """
        import torch
        if self.can_grow and self.explore_rate > 0 and \
                float(torch.rand(1).item()) < self.explore_rate:
            top = min(self.hi + self.step, self.ceiling)
            if top > self.hi:
                self.explore_draws += 1
                return int(torch.randint(self.hi + 1, top + 1, (1,)).item())
        return int(torch.randint(self.lo, self.hi + 1, (1,)).item())

    # -- decision side -----------------------------------------------------

    def probe_ks(self) -> list[int]:
        """The Ks a decision is read from: two inside the window, two beyond.

        Inside probes the TOP of the window (the model's current comfort
        edge), not its floor — growth must beat the best K it already owns.
        """
        inside = sorted({max(self.lo, self.hi - 1), self.hi})
        beyond = sorted({k for k in (self.hi + self.step,
                                     min(self.hi + 2 * self.step, self.ceiling))
                         if k > self.hi})
        return inside + beyond

    def decide(self, epoch: int, curve: dict[int, float],
               vessel_curve: dict[int, float] | None = None) -> WindowDecision:
        """Grow on proven benefit, or hold — and log which and why.

        `curve` is K -> validation DSC, `vessel_curve` optionally K -> vessel
        F1, both probed under common random numbers. This model is a
        multi-task one and the window serves BOTH heads, so evidence may come
        from either: growth fires when the best beyond-window K beats the
        best inside one by `margin` on EITHER metric, `streak_needed`
        decisions in a row. Measured the hard way — with the necrosis curve
        alone, this task's rollout plateau is so flat that growth never fired,
        the window sat at [3, 7], the necrosis head matched the wide range at
        two thirds the cost, and the vessel head — which needs the longer
        rollouts the window never bought — fell from F1 0.72 to 0.32 while
        the controller logged "hold" nine times. A probe that ignores a head
        is a decision that head was not invited to.
        """
        def best(d):
            inside = {k: v for k, v in d.items() if k <= self.hi}
            return max(inside.values(), default=float("-inf"))

        def best_beyond(d):
            beyond = {k: v for k, v in d.items() if k > self.hi}
            return max(beyond.values(), default=float("-inf"))

        best_in, best_bey = best(curve), best_beyond(curve)
        holds = best_bey - best_in >= self.margin
        vin = vbey = None
        vholds = False
        if vessel_curve:
            vin, vbey = best(vessel_curve), best_beyond(vessel_curve)
            vholds = vbey - vin >= self.margin
            holds = holds or vholds
        self.streak = self.streak + 1 if holds else 0

        win = (self.lo, self.hi)
        if holds and self.streak >= self.streak_needed and self.can_grow:
            new_hi = min(self.hi + self.step, self.ceiling)
            why = (f"vessel F1 {vbey:.4f} beats {vin:.4f} beyond the window"
                   if (vessel_curve and vholds and not
                       (best_bey - best_in >= self.margin))
                   else f"best beyond {best_bey:.4f} beats best inside "
                        f"{best_in:.4f} by {best_bey - best_in:+.4f}")
            d = WindowDecision(epoch, win, "grow",
                               f"{why} (>= {self.margin}) for {self.streak} probes",
                               curve, best_in, best_bey, self.margin, self.streak,
                               (self.lo, new_hi))
            if vessel_curve:
                d.vessel_inside, d.vessel_beyond = vin, vbey
            self.hi = new_hi
        elif not self.can_grow:
            d = WindowDecision(epoch, win, "hold",
                               f"ceiling {self.ceiling} reached — window is the "
                               f"full range", curve, best_in, best_bey,
                               self.margin, self.streak)
        elif holds:
            d = WindowDecision(epoch, win, "hold",
                               f"evidence holds but streak {self.streak} < "
                               f"{self.streak_needed}", curve, best_in, best_bey,
                               self.margin, self.streak)
        else:
            beyond_txt = f", vessel beyond peaked at {vbey:.4f}" if vessel_curve else ""
            d = WindowDecision(epoch, win, "hold",
                               f"best inside {best_in:.4f} not beaten by "
                               f"{self.margin}: beyond peaked at {best_bey:.4f}"
                               + beyond_txt, curve, best_in, best_bey,
                               self.margin, self.streak)
        if vessel_curve:
            d.vessel_curve = {k: round(v, 4) for k, v in vessel_curve.items()}
        self.decisions.append(d)
        return d
