"""Train the pretrained NCA variants used in the tutorial's ablation section.

Each variant adds one training ingredient to the previous one:

    baseline    synchronous updates, no alive mask, always start from the seed
    alive       + alive masking (empty cells stay empty)
    stochastic  + random (asynchronous) cell updates
    pool        + sample-pool training (learn to stay stable)
    regenerate  + damage during training (learn to repair)

All variants share the architecture defined here, which the notebook mirrors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

VISIBLE_CHANNELS = 4  # RGBA; channel 3 (alpha) defines which cells are alive.
ALPHA = 3


@dataclass(frozen=True)
class VariantConfig:
    name: str
    alive_mask: bool = False
    fire_rate: float = 1.0
    use_pool: bool = False
    damage: bool = False


VARIANTS = {
    "baseline": VariantConfig("baseline"),
    "alive": VariantConfig("alive", alive_mask=True),
    "stochastic": VariantConfig("stochastic", alive_mask=True, fire_rate=0.5),
    "pool": VariantConfig("pool", alive_mask=True, fire_rate=0.5, use_pool=True),
    "regenerate": VariantConfig(
        "regenerate", alive_mask=True, fire_rate=0.5, use_pool=True, damage=True
    ),
}


@dataclass(frozen=True)
class TrainingConfig:
    grid_size: int = 48
    emoji_size: int = 40
    state_channels: int = 16
    hidden_size: int = 64
    iterations: int = 2000
    batch_size: int = 8
    pool_size: int = 512
    damaged_per_batch: int = 3
    rollout_min: int = 64
    rollout_max: int = 96
    learning_rate: float = 2e-3
    seed: int = 7
    evaluation_interval: int = 100


class NCA(nn.Module):
    def __init__(
        self,
        state_channels: int,
        hidden_size: int,
        alive_mask: bool = False,
        fire_rate: float = 1.0,
        circular: bool = False,
    ):
        super().__init__()
        self.state_channels = state_channels
        self.alive_mask = alive_mask
        self.fire_rate = fire_rate
        self.circular = circular  # True: the grid wraps around (no edges), used for textures.

        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        sobel_y = sobel_x.T
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32) / 4
        kernels = torch.stack([identity, sobel_x, sobel_y, laplacian])[:, None]
        self.register_buffer("perception_kernels", kernels.repeat(state_channels, 1, 1, 1))

        self.update_net = nn.Sequential(
            nn.Conv2d(4 * state_channels, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, state_channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, state: torch.Tensor) -> torch.Tensor:
        if self.circular:
            state = F.pad(state, (1, 1, 1, 1), mode="circular")
            return F.conv2d(state, self.perception_kernels, groups=self.state_channels)
        return F.conv2d(state, self.perception_kernels, padding=1, groups=self.state_channels)

    def alive(self, state: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(state[:, ALPHA : ALPHA + 1], 3, stride=1, padding=1) > 0.1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        pre_alive = self.alive(state) if self.alive_mask else None
        delta = self.update_net(self.perceive(state))
        if self.fire_rate < 1.0:
            fire = torch.rand_like(state[:, :1]) < self.fire_rate
            delta = delta * fire
        state = state + delta
        if self.alive_mask:
            state = state * (pre_alive & self.alive(state))
        return state


def load_target(path: Path, config: TrainingConfig) -> torch.Tensor:
    """Load an RGBA image, pad it to the grid, and premultiply alpha."""
    image = Image.open(path).convert("RGBA")
    image = image.resize((config.emoji_size, config.emoji_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (config.grid_size, config.grid_size))
    offset = (config.grid_size - config.emoji_size) // 2
    canvas.paste(image, (offset, offset))
    rgba = np.asarray(canvas, dtype=np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:]
    return torch.from_numpy(rgba).permute(2, 0, 1)[None]


def make_seed(batch_size: int, config: TrainingConfig, device: torch.device) -> torch.Tensor:
    state = torch.zeros(
        batch_size, config.state_channels, config.grid_size, config.grid_size, device=device
    )
    center = config.grid_size // 2
    state[:, ALPHA:, center, center] = 1.0
    return state


def damage_mask(batch_size: int, config: TrainingConfig, device: torch.device) -> torch.Tensor:
    """Random circles that are zeroed out (1 = keep, 0 = erase)."""
    coords = torch.linspace(-1, 1, config.grid_size, device=device)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    center = torch.rand(batch_size, 2, 1, 1, device=device) - 0.5
    radius = 0.1 + 0.3 * torch.rand(batch_size, 1, 1, device=device)
    distance = torch.sqrt((x - center[:, 0]) ** 2 + (y - center[:, 1]) ** 2)
    return (distance > radius).float()[:, None]


def visible_loss(state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE on the RGBA channels."""
    return ((state[:, :VISIBLE_CHANNELS] - target) ** 2).mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate(model: NCA, target: torch.Tensor, config: TrainingConfig) -> dict[str, float]:
    """Loss while growing, long after growing, and after damage."""
    model.eval()
    torch.manual_seed(config.seed + 1)
    runs = 4
    device = target.device
    state = make_seed(runs, config, device)
    checkpoints = (48, 64, 96, 200, 500)
    metrics: dict[str, float] = {}
    for step in range(1, max(checkpoints) + 1):
        state = model(state)
        if step in checkpoints:
            loss = visible_loss(state, target).mean()
            metrics[f"loss_{step}"] = float(loss) if torch.isfinite(loss) else float("inf")

    state = make_seed(runs, config, device)
    for _ in range(96):
        state = model(state)
    state = state * damage_mask(runs, config, device)
    for _ in range(96):
        state = model(state)
    loss = visible_loss(state, target).mean()
    metrics["loss_after_damage"] = float(loss) if torch.isfinite(loss) else float("inf")
    metrics["score"] = (metrics["loss_64"] + metrics["loss_96"]) / 2
    return metrics


def train_variant(
    variant: VariantConfig,
    config: TrainingConfig,
    target: torch.Tensor,
    target_hash: str,
    output_dir: Path,
    device: torch.device,
) -> dict:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = NCA(
        config.state_channels, config.hidden_size, variant.alive_mask, variant.fire_rate
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[config.iterations // 2], gamma=0.3
    )
    batch_target = target.expand(config.batch_size, -1, -1, -1)
    pool = make_seed(config.pool_size, config, device) if variant.use_pool else None

    best_score = float("inf")
    best_metrics: dict[str, float] = {}
    history = []
    started = time.perf_counter()
    checkpoint_path = output_dir / f"nca_{variant.name}.pt"

    for iteration in range(config.iterations + 1):
        model.train()
        if pool is not None:
            indices = torch.randperm(config.pool_size, device=device)[: config.batch_size]
            state = pool[indices].clone()
            # Worst sample restarts from the seed so the model keeps learning to grow.
            order = visible_loss(state, batch_target).argsort(descending=True)
            indices, state = indices[order], state[order]
            state[:1] = make_seed(1, config, device)
            if variant.damage:
                count = config.damaged_per_batch
                state[-count:] *= damage_mask(count, config, device)
        else:
            state = make_seed(config.batch_size, config, device)

        steps = random.randint(config.rollout_min, config.rollout_max)
        for _ in range(steps):
            state = model(state)

        loss = visible_loss(state, batch_target).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if pool is not None:
            pool[indices] = state.detach()

        if iteration % config.evaluation_interval == 0 or iteration == config.iterations:
            metrics = evaluate(model, target, config)
            metrics["training_loss"] = loss.item()
            metrics["iteration"] = iteration
            history.append(metrics)
            print(
                f"[{variant.name}] it {iteration:4d} | train {loss.item():.4f} | "
                f"64 {metrics['loss_64']:.4f} | 500 {metrics['loss_500']:.4f} | "
                f"damage {metrics['loss_after_damage']:.4f}",
                flush=True,
            )
            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best_metrics = metrics.copy()
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "variant": asdict(variant),
                        "config": asdict(config),
                        "iteration": iteration,
                        "metrics": metrics,
                        "target_sha256": target_hash,
                    },
                    checkpoint_path,
                )

    elapsed = time.perf_counter() - started
    print(f"[{variant.name}] best iteration {best_metrics['iteration']} in {elapsed:.0f}s")
    return {
        "variant": asdict(variant),
        "checkpoint": checkpoint_path.name,
        "elapsed_seconds": elapsed,
        "best_metrics": best_metrics,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--target", default="data/emoji_u1f98e.png", help="RGBA target image.")
    parser.add_argument("--iterations", type=int, default=TrainingConfig.iterations)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    config = replace(TrainingConfig(), iterations=args.iterations)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target = load_target(Path(args.target), config).to(device)
    target_hash = hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest()

    metadata_path = output_dir / "nca_variants_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {"variants": {}}
    )
    metadata.update(
        {"config": asdict(config), "device": str(device), "target_sha256": target_hash}
    )
    for name in args.variants:
        result = train_variant(VARIANTS[name], config, target, target_hash, output_dir, device)
        metadata["variants"][name] = result
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
