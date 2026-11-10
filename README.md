# From Cellular Automata to Neural Cellular Automata

Hands-on material for the MICCAI 2026 tutorial **Learning to Self-Organize: Neural Cellular Automata for Medical Imaging**.

This repository contains an introductory Google Colab notebook for participants with deep-learning experience but little or no prior knowledge of Neural Cellular Automata (NCA).

## Learning objectives

Participants will:

- explore how local perception and update rules produce global behavior;
- transition from handcrafted Cellular Automata rules to a learned neural update rule;
- train a minimal NCA that grows a color image from a single seed;
- see how alive masking, random updates, sample-pool training, and damage make an NCA stable and able to repair itself;
- see how changing the loss turns the same architecture into a texture generator;
- inspect visible states, hidden channels, alive cells, and update activity interactively; and
- experiment with model, training, perception, and initial-state choices.

## Notebook

[`from_ca_to_nca.ipynb`](from_ca_to_nca.ipynb) contains the complete session:

1. an interactive Cellular Automata playground implemented as an embedded HTML/JavaScript app;
2. configurable NCA model and training settings;
3. a minimal PyTorch NCA that learns to grow a lizard emoji (RGBA) from a single seed;
4. a comparison of five pretrained NCAs that each add one training ingredient;
5. a pretrained texture NCA that keeps generating a spot pattern on a wrap-around grid of any size; and
6. an interactive browser app to run all models, damage them while they run, rotate their perception, and inspect their internal channels.

The target image and pretrained weights are stored in `data/` and `models/`. When the notebook runs inside a clone of this repository, it reads them from disk. Otherwise, for example in Colab, it downloads them once from GitHub (about 200 KB) into `nca_data/`. The branch it downloads from is set by `BRANCH` in the first code cell of Section 2; switch it to `main` after merging.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `from_ca_to_nca.ipynb`.
4. Run the cells from top to bottom.

Alternatively, open it directly from GitHub: `https://colab.research.google.com/github/nca-medical-imaging/tutorial_notebooks/blob/<branch>/from_ca_to_nca.ipynb`.

A GPU is optional. The default experiment also runs on CPU.

## Run locally

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
jupyter notebook from_ca_to_nca.ipynb
```

## Reproducing the pretrained models

Section 3 compares five pretrained NCAs with the same architecture (16 channels, hidden size 64). Each one adds one training ingredient to the previous one: `baseline`, `alive` (alive mask), `stochastic` (random cell updates), `pool` (sample-pool training), and `regenerate` (damage during training). To reproduce them:

```bash
python scripts/train_nca_variants.py --iterations 2000
```

Training all five variants took about 30 minutes on an Apple M3 (`--device mps`, or `cpu`/`cuda`). Checkpoints are written to `models/nca_<variant>.pt`; configuration, evaluation history, and the selected iteration are stored in `models/nca_variants_metadata.json`. The best checkpoint per variant is selected by the loss at 64 and 96 steps.

Section 4 uses a texture NCA (12 channels, hidden size 96, wrap-around grid). It is trained to match the VGG16 Gram-matrix statistics of a reaction-diffusion spot pattern that the script generates itself. Training needs `torchvision` and downloads the VGG16 ImageNet weights (~528 MB) on first use; the notebook does not need either:

```bash
python -m pip install "torchvision==0.23.*"
cd scripts && python train_texture_nca.py --iterations 3000 --output-dir ../models
```

This writes `models/nca_texture.pt`, `models/texture_target.png`, and `models/nca_texture_metadata.json`.

The target is the lizard emoji (U+1F98E) from [Noto Emoji](https://github.com/googlefonts/noto-emoji) (Apache License 2.0), file `2D/png/128/emoji_u1f98e.png`, stored as `data/emoji_u1f98e.png`. Commit the updated files in `models/` after retraining so the notebook can download them.
