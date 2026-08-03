# Growing NCA Colab prototype

This repository contains a deliberately small Google Colab experiment for a future MICCAI 2026 tutorial. A neural cellular automaton learns to grow the MICCAI France logo from one living cell.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoKalk/colab_test_miccai2026/blob/main/growing_nca_colab.ipynb)

## Run it

1. Open `growing_nca_colab.ipynb` in Colab using the badge above.
2. Select **Runtime → Change runtime type → GPU**.
3. Select **Runtime → Run all**.
4. Watch the target, current NCA output, and loss curve refresh during training.
5. Inspect the growth snapshots and animation at the end.

The default experiment uses a small 32×32 target, a 12-channel cellular state, and 500 optimizer updates. This is intended as a quick Colab simplicity test. Increase `training_steps` to around 1200 for a cleaner result.

## Included controls

- Target resolution
- Number of optimizer updates
- Batch size and learning rate
- Minimum and maximum growth duration
- Live preview frequency

No package installation or dataset download is required beyond loading the included target image.
