"""Train the minimal tutorial NCA and save a reproducible checkpoint.

The script intentionally mirrors the architecture in ``from_ca_to_nca.ipynb``.
It reads the portrait embedded in the notebook so that training and tutorial use
exactly the same target without requiring the untracked source asset.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainingConfig:
    grid_size: int = 48
    state_channels: int = 8
    hidden_size: int = 32
    iterations: int = 3000
    rollout_min: int = 24
    rollout_max: int = 40
    learning_rate: float = 2e-3
    seed: int = 7
    evaluation_interval: int = 100


class MinimalNCA(nn.Module):
    def __init__(self, state_channels: int, hidden_size: int):
        super().__init__()
        self.state_channels = state_channels

        identity = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32
        )
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ) / 8
        sobel_y = sobel_x.T
        laplacian = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ) / 4
        kernels = torch.stack([identity, sobel_x, sobel_y, laplacian])[:, None]
        self.register_buffer(
            "perception_kernels", kernels.repeat(state_channels, 1, 1, 1)
        )

        self.update_net = nn.Sequential(
            nn.Conv2d(4 * state_channels, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, state_channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            state,
            self.perception_kernels,
            padding=1,
            groups=self.state_channels,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.update_net(self.perceive(state))


def load_embedded_target(notebook_path: Path, grid_size: int) -> torch.Tensor:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    match = re.search(r"const PORTRAIT='([^']+)'", source)
    if match is None:
        raise RuntimeError("Could not find the embedded portrait in the notebook.")

    pixels = np.frombuffer(base64.b64decode(match.group(1)), dtype=np.uint8)
    if pixels.size % 48:
        raise RuntimeError("Embedded portrait data has an unexpected shape.")
    image = Image.fromarray(pixels.reshape(-1, 48)).resize(
        (grid_size, grid_size), Image.Resampling.LANCZOS
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, None]


def make_seed(config: TrainingConfig, device: torch.device) -> torch.Tensor:
    state = torch.zeros(
        1,
        config.state_channels,
        config.grid_size,
        config.grid_size,
        device=device,
    )
    center = config.grid_size // 2
    state[:, :, center, center] = 1.0
    return state


@torch.no_grad()
def evaluate(
    model: MinimalNCA,
    target: torch.Tensor,
    config: TrainingConfig,
) -> dict[str, float]:
    model.eval()
    state = make_seed(config, target.device)
    losses: dict[str, float] = {}
    evaluation_steps = (24, 32, 40, 64)
    for step in range(1, max(evaluation_steps) + 1):
        state = model(state)
        if step in evaluation_steps:
            losses[str(step)] = F.mse_loss(state[:, :1], target).item()

    values_are_finite = bool(torch.isfinite(state).all())
    losses["finite_at_64"] = values_are_finite
    losses["score"] = (
        np.mean([losses["24"], losses["32"], losses["40"]])
        + 0.25 * losses["64"]
        if values_are_finite
        else float("inf")
    )
    return losses


def save_checkpoint(
    path: Path,
    model: MinimalNCA,
    config: TrainingConfig,
    iteration: int,
    metrics: dict[str, float],
    target_hash: str,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "iteration": iteration,
            "metrics": metrics,
            "target_sha256": target_hash,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        iterations=args.iterations,
        evaluation_interval=args.evaluation_interval,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    notebook_path = Path(args.notebook).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target = load_embedded_target(notebook_path, config.grid_size).to(device)
    target_hash = hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest()
    model = MinimalNCA(config.state_channels, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_score = float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for iteration in range(config.iterations + 1):
        model.train()
        state = make_seed(config, device)
        rollout_steps = random.randint(config.rollout_min, config.rollout_max)
        for _ in range(rollout_steps):
            state = model(state)

        loss = F.mse_loss(state[:, :1], target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if iteration % config.evaluation_interval == 0 or iteration == config.iterations:
            metrics = evaluate(model, target, config)
            metrics["training_loss"] = loss.item()
            metrics["iteration"] = iteration
            history.append(metrics.copy())
            print(
                f"iteration {iteration:4d} | train {loss.item():.5f} | "
                f"eval32 {metrics['32']:.5f} | eval64 {metrics['64']:.5f} | "
                f"score {metrics['score']:.5f}",
                flush=True,
            )

            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best_metrics = metrics.copy()
                save_checkpoint(
                    output_dir / "pretrained_nca.pt",
                    model,
                    config,
                    iteration,
                    metrics,
                    target_hash,
                )

    elapsed = time.perf_counter() - started
    metadata = {
        "architecture": "MinimalNCA",
        "config": asdict(config),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "best_metrics": best_metrics,
        "target_sha256": target_hash,
        "checkpoint": "pretrained_nca.pt",
        "history": history,
    }
    (output_dir / "pretrained_nca_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Saved best checkpoint at iteration {best_metrics['iteration']} "
        f"after {elapsed:.1f} seconds."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--notebook", default="from_ca_to_nca.ipynb")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
