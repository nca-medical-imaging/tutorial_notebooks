"""A live training dashboard for the notebook.

Loss, validation DSC and the current prediction, redrawn in place while
training runs. In a 30-minute session a six-minute silent cell is dead air; a
plot that moves is the difference between people watching and people opening
their laptops.

Redraws into a single `Output` with `clear_output(wait=True)`, which is the one
pattern that behaves in Colab, JupyterLab and VS Code alike.
"""
from __future__ import annotations

import numpy as np


class LiveDashboard:
    """Pass as `on_epoch=` to `train.train`."""

    def __init__(self, val_ds, every: int = 2, sample_index: int = 0,
                 steps: int | None = None, figsize=(11, 3.3)):
        from IPython.display import display
        import ipywidgets as W

        self.val = val_ds
        self.every = every
        self.i = sample_index
        self.steps = steps
        self.figsize = figsize
        self.hist = {"epoch": [], "train_loss": [], "val_epoch": [], "val_dice": []}
        self.out = W.Output()
        display(self.out)

    def __call__(self, epoch, rec, model):
        import torch
        from IPython.display import clear_output
        from matplotlib import pyplot as plt

        from . import viz

        self.hist["epoch"].append(epoch)
        self.hist["train_loss"].append(rec["train_loss"])
        if "val_dice" in rec:
            self.hist["val_epoch"].append(epoch)
            self.hist["val_dice"].append(rec["val_dice"])
        if epoch % self.every and "val_dice" not in rec:
            return

        steps = self.steps or 15
        # `.sample()` rather than `.inputs[i]`: the corpus is stored quantised,
        # and indexing the property would dequantise the WHOLE split every epoch
        # just to draw one panel.
        xi, yi = self.val.sample(self.i)
        with torch.no_grad():
            was = model.training
            model.eval()
            pred = model(xi, steps=steps)
            model.train(was)
        pred = pred[0, 0].float().cpu().numpy()
        truth = yi[0, 0].cpu().numpy()

        with self.out:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 3, figsize=self.figsize, dpi=110)
            ax[0].plot(self.hist["epoch"], self.hist["train_loss"], lw=1.2)
            ax[0].set_yscale("log")
            ax[0].set_title("train loss", fontsize=9, fontweight="bold")
            ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.25)

            if self.hist["val_dice"]:
                ax[1].plot(self.hist["val_epoch"], self.hist["val_dice"],
                           lw=1.4, color="#c0392b", marker="o", ms=2.5)
                ax[1].set_ylim(0, 1)
                best = max(self.hist["val_dice"])
                ax[1].axhline(best, ls="--", lw=0.8, color="#888")
                ax[1].set_title(f"val DSC — best {best:.3f}", fontsize=9,
                                fontweight="bold")
            ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.25)

            viz.show_necrosis(pred, ax[2], title=f"prediction · epoch {epoch}")
            ax[2].contour(truth, levels=[0.99], colors=["#00e0ff"], linewidths=1.0)
            fig.suptitle(f"epoch {epoch}  ·  loss {rec['train_loss']:.5f}  ·  "
                         f"{rec['wall_s']:.0f} s"
                         + (f"  ·  val DSC {rec['val_dice']:.4f}"
                            if "val_dice" in rec else ""),
                         fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            plt.show()

    def summary(self) -> dict:
        return {"best_val_dice": max(self.hist["val_dice"], default=float("nan")),
                "final_train_loss": self.hist["train_loss"][-1]
                if self.hist["train_loss"] else float("nan")}
