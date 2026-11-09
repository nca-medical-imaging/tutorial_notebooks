"""Train the texture NCA used in the tutorial's texture section.

Instead of matching one image pixel by pixel, a texture NCA matches the
*statistics* of a texture: Gram matrices of VGG16 features (as in neural style
transfer). It starts from an empty grid, uses random updates, a sample pool and
a wrap-around grid, and keeps the texture alive forever.

The target texture is a reaction-diffusion (Gray-Scott) pattern generated here,
so no external image is needed. Requires torchvision (training only).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

from train_nca_variants import NCA


@dataclass(frozen=True)
class TextureConfig:
    pattern_size: int = 128     # Gray-Scott simulation size.
    texture_size: int = 64      # Pattern is shrunk to this size, which sets the spot scale.
    grid_size: int = 64         # NCA training grid; the trained rule works on any size.
    feed: float = 0.0367        # Gray-Scott parameters for a "spots" pattern.
    kill: float = 0.0649
    simulation_steps: int = 8000
    state_channels: int = 12
    hidden_size: int = 96
    fire_rate: float = 0.5
    iterations: int = 3000
    batch_size: int = 4
    pool_size: int = 1024
    rollout_min: int = 32
    rollout_max: int = 64
    learning_rate: float = 2e-3
    overflow_weight: float = 1.0
    seed: int = 7


def gray_scott(config: TextureConfig) -> np.ndarray:
    """Simulate a reaction-diffusion pattern on a wrap-around grid."""
    generator = torch.Generator().manual_seed(config.seed)
    size = config.pattern_size
    u = torch.ones(1, 1, size, size)
    v = torch.zeros(1, 1, size, size)
    for _ in range(12):
        r, c = torch.randint(0, size - 6, (2,), generator=generator).tolist()
        v[..., r : r + 6, c : c + 6] = 0.5
        u[..., r : r + 6, c : c + 6] = 0.25
    laplacian = torch.tensor([[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]])[None, None]
    for _ in range(config.simulation_steps):
        lap_u = F.conv2d(F.pad(u, (1, 1, 1, 1), mode="circular"), laplacian)
        lap_v = F.conv2d(F.pad(v, (1, 1, 1, 1), mode="circular"), laplacian)
        uvv = u * v * v
        u = u + (lap_u - uvv + config.feed * (1 - u))
        v = v + (0.5 * lap_v + uvv - (config.feed + config.kill) * v)
    return v[0, 0].numpy()


def make_texture(config: TextureConfig) -> np.ndarray:
    """Color the pattern and shrink it. Returns RGB in [0, 1], shape (H, W, 3)."""
    v = gray_scott(config)
    v = (v - v.min()) / (v.max() - v.min())
    rgb = matplotlib.colormaps["magma"](v)[..., :3]
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    image = image.resize((config.texture_size, config.texture_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


class StyleLoss(torch.nn.Module):
    """Compare Gram matrices of VGG16 features at several depths."""

    LAYERS = (1, 6, 11, 18, 25)  # ReLU outputs of the first conv in each VGG block.

    def __init__(self, texture: torch.Tensor):
        super().__init__()
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(self.LAYERS) + 1].eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.to(texture.device)
        with torch.no_grad():
            self.target_grams = [g.detach() for g in self.grams(texture)]

    def grams(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = (images - self.mean) / self.std
        grams = []
        for index, layer in enumerate(self.vgg):
            x = layer(x)
            if index in self.LAYERS:
                b, c, h, w = x.shape
                features = x.reshape(b, c, h * w)
                grams.append(features @ features.transpose(1, 2) / (h * w))
        return grams

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Per-sample loss."""
        # Each layer's error is relative to the target's magnitude, so all layers count equally.
        return sum(
            ((g - t) ** 2).mean(dim=(1, 2)) / (t**2).mean()
            for g, t in zip(self.grams(images), self.target_grams)
        )


def to_image(state: torch.Tensor) -> torch.Tensor:
    """Texture NCAs show channels 0-2 shifted by 0.5, so an empty grid is mid-gray."""
    return state[:, :3] + 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=TextureConfig.iterations)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    config = TextureConfig(iterations=args.iterations)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texture_rgb = make_texture(config)
    Image.fromarray((texture_rgb * 255).astype(np.uint8)).save(output_dir / "texture_target.png")
    texture = torch.from_numpy(texture_rgb).permute(2, 0, 1)[None].to(device)
    style_loss = StyleLoss(texture).to(device)

    model = NCA(
        config.state_channels, config.hidden_size, fire_rate=config.fire_rate, circular=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[config.iterations // 2], gamma=0.3
    )
    empty = lambda n: torch.zeros(  # noqa: E731
        n, config.state_channels, config.grid_size, config.grid_size, device=device
    )
    pool = empty(config.pool_size)

    history = []
    started = time.perf_counter()
    for iteration in range(config.iterations + 1):
        indices = torch.randperm(config.pool_size, device=device)[: config.batch_size]
        state = pool[indices].clone()
        if iteration % 8 == 0:
            state[:1] = empty(1)  # Keep learning to start from an empty grid.

        for _ in range(random.randint(config.rollout_min, config.rollout_max)):
            state = model(state)

        texture_loss = style_loss(to_image(state)).mean()
        overflow = (state - state.clamp(-1.0, 1.0)).abs().mean()
        loss = texture_loss + config.overflow_weight * overflow
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():  # Normalized gradients, as in the original texture NCA.
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        optimizer.step()
        scheduler.step()
        pool[indices] = state.detach()

        if iteration % 100 == 0 or iteration == config.iterations:
            history.append(
                {"iteration": iteration, "texture_loss": texture_loss.item(), "overflow": overflow.item()}
            )
            print(
                f"[texture] it {iteration:4d} | texture {texture_loss.item():.4f} | "
                f"overflow {overflow.item():.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(config), "iteration": config.iterations},
        output_dir / "nca_texture.pt",
    )
    (output_dir / "nca_texture_metadata.json").write_text(
        json.dumps(
            {"config": asdict(config), "device": str(device), "elapsed_seconds": elapsed, "history": history},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[texture] saved after {elapsed:.0f}s")


if __name__ == "__main__":
    main()
