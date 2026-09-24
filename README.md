# From Cellular Automata to Neural Cellular Automata

Hands-on material for the MICCAI 2026 tutorial **Learning to Self-Organize: Neural Cellular Automata for Medical Imaging**.

This repository contains an introductory Google Colab notebook for participants with deep-learning experience but little or no prior knowledge of Neural Cellular Automata (NCA).

## Learning objectives

Participants will:

- explore how local perception and update rules produce global behavior;
- transition from handcrafted Cellular Automata rules to a learned neural update rule;
- train a minimal NCA from a single seed;
- inspect visible states, hidden channels, and update activity interactively; and
- experiment with model, training, rollout, and initial-state choices.

## Notebook

[`from_ca_to_nca.ipynb`](from_ca_to_nca.ipynb) contains the complete session:

1. an interactive Cellular Automata playground implemented as an embedded HTML/JavaScript app;
2. configurable NCA model and training settings;
3. a minimal PyTorch NCA training example; and
4. a comparison with an embedded, longer-trained checkpoint; and
5. an interactive browser app for comparing both models and their internal channels.

The notebook is self-contained. The example image is embedded so that no additional download is required in Colab.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `from_ca_to_nca.ipynb`.
4. Run the cells from top to bottom.

A GPU is optional. The default experiment also runs on CPU.

## Run locally

Create a Python environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
jupyter notebook from_ca_to_nca.ipynb
```

## Reproducing the pretrained model

The embedded checkpoint uses the same minimal architecture and target as the live example. To reproduce it:

```bash
python scripts/train_pretrained_nca.py --iterations 3000 --output-dir models
```

The script samples rollout lengths between 24 and 40 steps and selects the best checkpoint using validation losses at 24, 32, 40, and 64 steps. Training configuration and evaluation history are stored in `models/pretrained_nca_metadata.json`.
