# Ablation2d — MICCAI 2026 hands-on 3 (Colab test)

Branch deployment of the *Ablation planning with Neural Cellular Automata*
hands-on: a chained NCA (~10k parameters) learns to replace a Pennes
bioheat + SAR + Arrhenius solve. Give it a CT slice, needle positions and a
power; it returns the necrosis field and the vessels it had to find — in
milliseconds.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nca-medical-imaging/tutorial_notebooks/blob/ablation2d/ablation2d_colab.ipynb)

> **Private repo.** Colab needs two one-time things: connect your GitHub
> account in Colab (Settings → GitHub) to open the notebook, and store a
> GitHub token as the `GITHUB_TOKEN` secret (sidebar key icon) so the setup
> cell can clone the branch.

## Run it

1. Open `ablation2d_colab.ipynb` in Colab with the badge above (branch `ablation2d`).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**. The first cell clones this branch (package,
   corpus and checkpoint ship together — nothing else downloads).

## Run it locally

From the repo root, exactly what Colab runs:

```bash
NCA_NOTEBOOK_QUICK=1 jupyter nbconvert --to notebook --execute \
    --output /tmp/ablation2d_executed.ipynb ablation2d_colab.ipynb
```

`NCA_NOTEBOOK_QUICK=1` is the test-harness switch: 48 cases per split,
2 training epochs, full pipeline otherwise. Drop it for the real 24-epoch
session run (or set `LOAD_PRETRAINED = True` in §3 to load the shipped
checkpoint instead of training).

## What is on this branch

| path | what |
|---|---|
| `ablation2d_colab.ipynb` | the runnable notebook (SOLUTION variant, 0 TODOs) |
| `ablation2d/` | the plumbing package: data, model registry, training loop, viz, UI |
| `data_liver/` | the corpus the notebook trains on — real patient CT masked to the liver |
| `data/test.npz`, `data/real_slice.npz` | unmasked comparison slice + demo slice |
| `checkpoints/ablation_cnca_liver.pt` | pretrained fallback (best measured model) |

## Before this branch goes anywhere

- **Patient data.** `data_liver/` and `data/` are de-identified real patient CT
  (see `data_liver/DATASHEET.md`). Redistribution outside a governed context is
  **not settled** — treat this branch as internal until it is.
- **Upstream.** The notebook is authored in `miccai2026-nca-tutorial` via
  `build_notebooks.py` (TUTORIAL + SOLUTION from one source). Only the setup
  cell differs here, so it points at this repo/branch. Regenerate from upstream
  before the session; do not hand-edit further.

The TUTORIAL variant (6 TODOs across 4 cells, for participants) is generated
from the same source and can be added to this branch once the deployment is
validated.
