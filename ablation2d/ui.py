"""The interactive planner: move a needle, watch the lesion move.

Built on plain `ipywidgets`, which is the only interactive stack that behaves
the same in Colab, JupyterLab, VS Code and nbclassic. `ipympl` would give
continuous mouse dragging and does not survive Colab, and a workshop demo that
fails in the room is worth less than one that is a slider.

The frame is rendered to a PNG and pushed into an `ipywidgets.Image`, rather
than re-`display`ing a figure into an `Output`. Same picture, no flicker, and it
does not grow the notebook's output every time a slider moves — which is what
makes a 40-minute session end with a 200 MB .ipynb.
"""
from __future__ import annotations

import io
import time

import numpy as np

from .anatomy import Anatomy
from .channels import build_inputs
from .device import EMPRINT_HP, Device
from .plans import DIAL_MAX_W, FIXED_DURATION_S, Needle, Plan


def _best_hu_f1(anat: Anatomy) -> float:
    """The best F1 any single Hounsfield threshold reaches on this slice.

    Computed, not quoted: it is a property of the slice on screen, and the demo
    used to print a constant measured on a different one."""
    hu = anat.hounsfield
    if hu is None:
        return float("nan")
    organ = anat.liver_mask | anat.vessel_mask
    t = anat.vessel_mask
    best = 0.0
    for q in np.linspace(50, 99.7, 60):
        thr = float(np.percentile(hu[organ], q))
        p = (hu > thr) & organ
        den = p.sum() + t.sum()
        if den:
            best = max(best, 2 * float((p & t).sum()) / float(den))
    return best


class AblationStudio:
    """Place needles on a slice and read the predicted necrosis, live.

    Args:
        model: a trained `AblationCNCA` (or anything with the same call
            signature). Put it on the device you want inference on BEFORE
            handing it over.
        anat: the slice to plan on.
        steps: rollout length at inference. Tune it on validation first —
            `train.choose_inference_steps` does exactly that.
        oracle: optional callable `(anat, plan) -> 2-D necrosis field`, wired to
            the "check against physics" button. `physics.simulate` fits, but it
            needs UniPhys, so it is optional and the button hides without it.
    """

    def __init__(self, model, anat: Anatomy, *, steps: int = 15,
                 device: Device = EMPRINT_HP, torch_device: str = "cpu",
                 oracle=None, max_needles: int = 4, tta: bool = False,
                 average: str = "repeats", mask_vessels: bool = True,
                 mask_ct: bool = False, build: bool = True):
        import torch
        self.torch = torch
        self.model = model.eval()
        #: How to reduce the prediction. THE MODEL IS STOCHASTIC — `fire_rate`
        #: below 1 draws a fresh update mask every step — so one rollout is one
        #: sample of a random variable, not "the" prediction. Measured on the
        #: shipped checkpoint, 300 test cases:
        #:
        #:     1 rollout    DSC 0.786   vessel F1 0.615
        #:     8 repeats    DSC 0.831   vessel F1 0.648
        #:
        #: "repeats" is therefore the honest default, not a trick. "d4" also
        #: rotates, which helps ONLY a model trained on the group: on this one
        #: it drops the vessel head to 0.503 because averaging rotated
        #: predictions smears thin branching structure a non-equivariant model
        #: disagrees on. "none" is the single sample.
        self.average = "d4" if tta else average
        #: Delete vessel predictions outside the liver. Uses the organ contour,
        #: which the model was never given but a planning system has — so a
        #: score with this on is "C-NCA + liver mask", never the model's own.
        #: 34 % of the vessel head's false positives are outside the organ and
        #: 0 % of its misses are: F1 0.648 -> 0.709, with no recall cost.
        self.mask_vessels = bool(mask_vessels)
        #: Zero the CT outside the organ BEFORE the model sees it. Must match how
        #: the model was trained: a model trained on `data_liver/` fed a full
        #: abdominal CT is being tested on a distribution it never saw.
        self.mask_ct = bool(mask_ct)
        self.anat = anat
        self.steps = steps
        self.device = device
        self.torch_device = torch_device
        self.oracle = oracle
        self.max_needles = max_needles
        self._truth = None
        self._last_ms = 0.0
        # `build=False` constructs the inference half without ipywidgets, so the
        # prediction path — which is what the workshop actually depends on — can
        # be tested on a machine that has no notebook stack.
        if build:
            self._build()

    # -- inference ----------------------------------------------------------
    def plan(self) -> Plan:
        return Plan([g.needle(self.anat) for g in self._groups if g.enabled])

    def predict(self, plan: Plan):
        """(necrosis, vessel) — both heads, every time. The vessel map is what
        the model believes the anatomy is, and showing the burn without it hides
        the half of the answer that explains the other half."""
        x = build_inputs(self.anat, plan, device=self.device,
                         mask_ct=self.mask_ct)
        t = self.torch.from_numpy(x).unsqueeze(0).to(self.torch_device)
        t0 = time.perf_counter()
        from .train import predict_d4_tta, predict_repeats
        if self.average == "d4":
            p = predict_d4_tta(self.model, t, self.steps)
        elif self.average == "repeats":
            p = predict_repeats(self.model, t, self.steps)
        else:
            with self.torch.no_grad():
                p = self.model(t, steps=self.steps)
        if self.mask_vessels and p.shape[1] > 1:
            from .evaluate import mask_to_organ
            organ = self.torch.from_numpy(
                (self.anat.liver_mask | self.anat.vessel_mask).astype("float32")
            ).unsqueeze(0)
            p = mask_to_organ(p, organ)
        self._last_ms = (time.perf_counter() - t0) * 1e3
        p = p[0].float().cpu().numpy()
        return p[0], (p[1] if p.shape[0] > 1 else None)

    # -- widgets ------------------------------------------------------------
    def _build(self):
        import ipywidgets as W

        h, w = self.anat.shape
        dx = self.anat.dx_mm
        self._groups = [_NeedleControls(i, w * dx, h * dx, self._refresh,
                                        enabled=(i == 0))
                        for i in range(self.max_needles)]

        self.image = W.Image(format="png",
                             layout=W.Layout(width="560px", height="560px"))
        self.readout = W.HTML()
        self.steps_slider = W.IntSlider(value=self.steps, min=2, max=40, step=1,
                                        description="NCA steps",
                                        continuous_update=False,
                                        style={"description_width": "80px"})
        self.steps_slider.observe(self._on_steps, names="value")

        self.check_btn = W.Button(description="check against physics",
                                  icon="flask", button_style="warning",
                                  layout=W.Layout(width="220px"))
        self.check_btn.on_click(self._on_check)
        self.clear_btn = W.Button(description="clear check",
                                  layout=W.Layout(width="130px"))
        self.clear_btn.on_click(self._on_clear)
        buttons = [self.check_btn, self.clear_btn] if self.oracle else []

        controls = W.VBox([g.box for g in self._groups]
                          + [W.HTML("<hr style='margin:6px 0'>"), self.steps_slider,
                             W.HBox(buttons), self.readout],
                          layout=W.Layout(width="430px"))
        self.widget = W.HBox([self.image, controls])

    def _on_steps(self, change):
        self.steps = int(change["new"])
        self._refresh()

    def _on_check(self, _btn):
        self.check_btn.description = "solving…"
        self.check_btn.disabled = True
        try:
            self._truth = self.oracle(self.anat, self.plan())
        finally:
            self.check_btn.description = "check against physics"
            self.check_btn.disabled = False
        self._refresh()

    def _on_clear(self, _btn):
        self._truth = None
        self._refresh()

    # -- drawing ------------------------------------------------------------
    def _refresh(self, *_):
        import matplotlib
        from matplotlib import pyplot as plt
        from . import viz

        plan = self.plan()
        if not plan.needles:
            self.readout.value = "<b>no needles</b> — enable one on the right"
            return
        pred, ves = self.predict(plan)

        with matplotlib.rc_context({"figure.dpi": 110}):
            ncol = 2 + (self._truth is not None)
            fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.8),
                                     squeeze=False)
            # The CT underneath, because that is all the model was given.
            viz.show_necrosis(pred, axes[0][0], self.anat, plan,
                              "predicted necrosis", on_ct=True)
            a1 = axes[0][1]
            viz.show_ct(self.anat, a1, "predicted vessels")
            if ves is not None:
                a1.contour(ves, levels=[0.5], colors=["#00e5ff"], linewidths=1.0)
                a1.contour(self.anat.vessel_mask.astype(float), levels=[0.5],
                           colors=["#ffb300"], linewidths=0.8, linestyles=":")
            if self._truth is not None:
                viz.show_necrosis(self._truth, axes[0][2], self.anat, plan,
                                  "UniPhys ground truth", on_ct=True)
            fig.tight_layout(pad=0.4)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
        self.image.value = buf.getvalue()

        px_cm2 = (self.anat.dx_mm ** 2) / 100.0
        area = float((pred > 0.5).sum()) * px_cm2
        line = (f"<b>{len(plan.needles)} needle(s)</b> · predicted necrosis "
                f"<b>{area:.1f} cm²</b> · {self._last_ms:.0f} ms · "
                f"{self.steps} steps"
                + {"d4": " · <b>8 D4 orientations</b> averaged",
                   "repeats": " · <b>8 rollouts</b> averaged",
                   "none": " · <b>1 rollout</b> (stochastic — a single sample)"}
                  .get(self.average, "")
                + (" · vessels <b>masked to the liver</b>"
                   if self.mask_vessels else ""))
        if ves is not None:
            t = self.anat.vessel_mask
            p = ves > 0.5
            f1 = 2 * (p & t).sum() / max(p.sum() + t.sum(), 1)
            base = _best_hu_f1(self.anat)
            line += (f"<br>vessel head F1 <b>{f1:.3f}</b> "
                     f"(best HU threshold on THIS slice: {base:.3f})")
        if self._truth is not None:
            t = self._truth >= 0.99
            p = pred > 0.5
            dsc = 2 * (t & p).sum() / max(t.sum() + p.sum(), 1)
            line += (f"<br>ground truth <b>{t.sum() * px_cm2:.1f} cm²</b> · "
                     f"DSC <b>{dsc:.3f}</b>")
        self.readout.value = line

    def scope(self, steps: int | None = None):
        """A `ChannelScope` on the plan currently set in this studio — the same
        rollout you are looking at, opened up channel by channel."""
        return ChannelScope(self.model, self.anat, self.plan(),
                            steps=steps or self.steps, device=self.device,
                            torch_device=self.torch_device,
                            mask_ct=self.mask_ct)

    def show(self):
        from IPython.display import display
        self._refresh()
        display(self.widget)
        return self.widget


class _NeedleControls:
    """One needle's five degrees of freedom: where, which way, how hard, how long."""

    def __init__(self, index, fov_x_mm, fov_y_mm, on_change, enabled=False):
        import ipywidgets as W

        self.index = index
        self._on_change = on_change
        style = {"description_width": "58px"}
        lay = W.Layout(width="360px")
        kw = dict(continuous_update=False, style=style, layout=lay)

        self.on = W.Checkbox(value=enabled, description=f"needle {index + 1}",
                             indent=False)
        self.x = W.FloatSlider(value=fov_x_mm * (0.42 + 0.06 * index), min=8,
                               max=fov_x_mm - 8, step=1, description="tip x mm", **kw)
        self.y = W.FloatSlider(value=fov_y_mm * (0.48 + 0.05 * index), min=8,
                               max=fov_y_mm - 8, step=1, description="tip y mm", **kw)
        self.angle = W.FloatSlider(value=90.0 + 25 * index, min=0, max=360, step=2,
                                   description="angle °", **kw)
        self.power = W.FloatSlider(value=75.0, min=0, max=DIAL_MAX_W, step=5,
                                   description="dial W", **kw)
        # v2 fixes the burn at 5 minutes, so there is no slider for it: a
        # control that changes an input the model does not have would be a lie.
        self.rows = [self.x, self.y, self.angle, self.power]
        for wdg in [self.on] + self.rows:
            wdg.observe(self._changed, names="value")
        self._inner = W.VBox(self.rows)
        self._inner.layout.display = "" if enabled else "none"
        self.box = W.VBox([self.on, self._inner],
                          layout=W.Layout(border="1px solid #ddd",
                                          padding="4px", margin="2px"))

    @property
    def enabled(self) -> bool:
        return bool(self.on.value)

    def _changed(self, _):
        self._inner.layout.display = "" if self.enabled else "none"
        self._on_change()

    def needle(self, anat: Anatomy) -> Needle:
        """Angle is the direction the needle came FROM, so 90° enters from below.

        The base is placed 120 mm back along that direction and then clipped to
        the frame, which keeps the shaft channel inside the picture the network
        is shown.
        """
        th = np.deg2rad(self.angle.value)
        u = np.array([np.cos(th), np.sin(th)], np.float32)
        tip = np.array([self.x.value, self.y.value], np.float32)
        h, w = anat.shape
        fov = np.array([w * anat.dx_mm, h * anat.dx_mm], np.float32)
        t = 120.0
        for o, d, hi in ((tip[0], u[0], fov[0]), (tip[1], u[1], fov[1])):
            if abs(d) > 1e-9:
                t = min(t, ((hi if d > 0 else 0.0) - o) / d)
        base = tip + u * max(t, 25.0)
        return Needle(tip_x_mm=float(tip[0]), tip_y_mm=float(tip[1]),
                      base_x_mm=float(base[0]), base_y_mm=float(base[1]),
                      dial_power_w=float(self.power.value),
                      duration_s=FIXED_DURATION_S)


class ChannelScope:
    """Watch ONE state channel evolve across the rollout.

    The automaton's answer is two channels wide; its *state* is sixteen. The
    other fourteen are where the computation actually happens, and a demo that
    only ever shows the readout shows the conclusion without the argument. This
    widget puts the whole state on a dropdown and steps through it.

    Three things it is built to make visible:

    * **whether conditioning is frozen — and whether it should be.** With
      `restore_conditioning=True` the three input channels are rewritten every
      step and their per-step delta is exactly zero; that flatline is the
      feature working. With it False the automaton owns those channels too, and
      the widget switches to reporting drift AGAINST THE INPUT'S OWN SCALE,
      because that ratio is the question: nca-forge measured a sparse power
      channel drifting 7630x its own mean within nine steps, i.e. erased.
    * **saturation.** Readouts are shown as LOGITS, not probabilities. A head
      that has run to +-4 is pinned against `STATE_CLAMP` and has stopped
      learning; a sigmoid would draw that as a confident, healthy 0.98.
    * **where the front is.** The per-step |delta| panel is the automaton's
      activity: early on it is the whole neighbourhood of the antenna, and a
      converged rollout goes quiet. That is the honest answer to "how many steps
      do I need" — cheaper to look at than a DSC sweep.

    Args:
        model: a trained `AblationCNCA`.
        anat: the slice.
        plan: the plan to roll out. `AblationStudio.scope()` passes its own.
        steps: rollout length.
    """

    def __init__(self, model, anat: Anatomy, plan: Plan, *, steps: int = 24,
                 device: Device = EMPRINT_HP, torch_device: str = "cpu",
                 mask_ct: bool = False, build: bool = True):
        import torch
        from .model import channel_roles

        self.torch = torch
        self.model = model.eval()
        self.anat = anat
        self.plan = plan
        self.steps = steps
        self.device = device
        self.torch_device = torch_device
        self.mask_ct = bool(mask_ct)
        self.restored = bool(getattr(model, "restore_conditioning", True))
        self.roles = channel_roles(int(model.channels), restored=self.restored)
        self._trace = None
        if build:
            self._build()

    # -- inference ----------------------------------------------------------
    def rollout(self):
        """(steps, C, H, W) — the whole state at every step, cached."""
        if self._trace is not None and len(self._trace) == self.steps:
            return self._trace
        x = build_inputs(self.anat, self.plan, device=self.device,
                         mask_ct=self.mask_ct)
        t = self.torch.from_numpy(x).unsqueeze(0).to(self.torch_device)
        with self.torch.no_grad():
            _, tr = self.model(t, steps=self.steps, return_trace=True,
                               trace_full=True)
        self._trace = np.stack([f[0].float().cpu().numpy() for f in tr])
        return self._trace

    # -- widgets ------------------------------------------------------------
    def _build(self):
        import ipywidgets as W

        self.channel = W.Dropdown(
            options=[(f"{i:2d}  {n}", i) for i, n in enumerate(self.roles)],
            value=0, description="channel",
            layout=W.Layout(width="320px"),
            style={"description_width": "70px"})
        self.step = W.IntSlider(value=self.steps - 1, min=0, max=self.steps - 1,
                                description="step", continuous_update=False,
                                style={"description_width": "70px"})
        self.play = W.Play(value=0, min=0, max=self.steps - 1, interval=180)
        W.jslink((self.play, "value"), (self.step, "value"))
        # Shared scale is the default because the question is almost always
        # "how does this channel CHANGE", and a per-step rescale makes a channel
        # that has converged look as busy as one that has not.
        self.shared = W.Checkbox(value=True, description="one colour scale for "
                                                         "the whole rollout",
                                 indent=False)
        self.image = W.Image(format="png",
                             layout=W.Layout(width="900px", height="330px"))
        self.readout = W.HTML()
        for w in (self.channel, self.step, self.shared):
            w.observe(self._refresh, names="value")
        self.widget = W.VBox([
            W.HBox([self.channel, self.play, self.step]),
            self.shared, self.image, self.readout])

    # -- drawing ------------------------------------------------------------
    def _refresh(self, *_):
        png, html = self.render(int(self.channel.value), int(self.step.value),
                                bool(self.shared.value))
        self.image.value = png
        self.readout.value = html

    def render(self, c: int, k: int, shared: bool = True):
        """(png bytes, html caption) for one channel at one step.

        Split out from the widget so the drawing can be tested, and figured,
        without ipywidgets — which is not installed everywhere this package
        runs, and must not be needed to render a picture.
        """
        import matplotlib
        from matplotlib import pyplot as plt

        tr = self.rollout()
        if not 0 <= c < tr.shape[1]:
            raise IndexError(
                f"channel {c} does not exist: this model has {tr.shape[1]} "
                f"state channels ({', '.join(self.roles)})")
        k = max(0, min(int(k), len(tr) - 1))
        f = tr[:, c]
        vmax = float(np.abs(f).max()) if shared else float(np.abs(f[k]).max())
        vmax = max(vmax, 1e-6)

        with matplotlib.rc_context({"figure.dpi": 110}):
            fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))
            ax[0].imshow(f[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         origin="lower", interpolation="nearest")
            ax[0].set_title(f"{self.roles[c]} — step {k}", fontsize=9,
                            fontweight="bold")

            d = np.abs(f[k] - f[k - 1]) if k else np.abs(f[0])
            ax[1].imshow(d, cmap="magma", origin="lower",
                         interpolation="nearest")
            ax[1].set_title("|change| since the step before" if k
                            else "|value| at the first step",
                            fontsize=9, fontweight="bold")

            act = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
            ax[2].plot(range(1, len(f)), act, "-o", ms=2.6, lw=1.3,
                       color="#c0392b")
            ax[2].axvline(k, ls=":", lw=1.0, color="#333")
            # A conditioning channel is rewritten every step, so its activity is
            # identically zero and a log axis has nothing to draw. That flatline
            # IS the result, so say so rather than emitting a warning and an
            # empty panel.
            if (act > 0).any():
                ax[2].set_yscale("log")
            else:
                ax[2].set_ylim(-1, 1)
                ax[2].text(0.5, 0.5, "exactly 0 at every step\n"
                                     "(rewritten from the inputs)",
                           ha="center", va="center", fontsize=8.5,
                           color="#2980b9", transform=ax[2].transAxes)
            ax[2].set_xlabel("step", fontsize=8)
            ax[2].set_ylabel("mean |change|", fontsize=8)
            ax[2].set_title("how busy this channel still is", fontsize=9,
                            fontweight="bold")
            ax[2].grid(alpha=0.25)
            ax[2].tick_params(labelsize=7)
            for a in ax[:2]:
                a.set_xticks([]); a.set_yticks([])
            fig.tight_layout(pad=0.4)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)

        kind = ("readout (logit)" if c < 2 else
                ("conditioning — rewritten every step" if self.restored
                 else "free state — SEEDED from the input, then the automaton's")
                if c < 5 else "scratch")
        last = float(np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))[-1]) \
            if len(f) > 1 else 0.0
        note = ""
        if c < 2 and np.abs(f[k]).max() > 3.9:
            note = ("  ·  <b style='color:#c0392b'>saturated</b> — pinned "
                    "against STATE_CLAMP")
        if 2 <= c < 5:
            drift = float(np.abs(f - f[0]).max())
            if self.restored:
                note = (f"  ·  drift across the rollout <b>{drift:.2e}</b> "
                        "(must be 0)")
            else:
                # Not a bug here — it is the regime. What matters is the SIZE of
                # the drift against the signal it is drifting from: a sparse
                # plan channel whose drift dwarfs its own mean has been erased,
                # and whatever the model knows about the plan it had to read in
                # the first few steps.
                scale = float(np.abs(f[0]).mean()) + 1e-9
                note = (f"  ·  drift <b>{drift:.2e}</b> against input mean "
                        f"{scale:.2e} — <b>{drift / scale:.0f}x</b>"
                        + ("  <b style='color:#c0392b'>input erased</b>"
                           if drift / scale > 50 else ""))
        return buf.getvalue(), (
            f"<b>{self.roles[c]}</b> — {kind}  ·  range "
            f"[{f[k].min():+.2f}, {f[k].max():+.2f}]  ·  activity at the last "
            f"step <b>{last:.2e}</b>{note}")

    def show(self):
        from IPython.display import display
        self._refresh()
        display(self.widget)
        return self.widget
