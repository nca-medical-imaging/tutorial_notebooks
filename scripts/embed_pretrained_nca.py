"""Embed the target image and the pretrained NCA variants into the notebook.

The notebook stays self-contained in Colab: it needs no downloads or extra files.
Run after ``train_nca_variants.py``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import zlib
from pathlib import Path

import torch

from train_nca_variants import VARIANTS


def replace_line(notebook: dict, prefix: str, new_line: str) -> None:
    hits = 0
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        for index, line in enumerate(source):
            if line.startswith(prefix):
                source[index] = new_line
                hits += 1
    if hits != 1:
        raise RuntimeError(f"Expected one line starting with {prefix!r}, found {hits}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--target", default="assets/emoji_u1f98e.png")
    parser.add_argument("--notebook", default="from_ca_to_nca.ipynb")
    args = parser.parse_args()

    state_dicts = {}
    for name in VARIANTS:
        checkpoint = torch.load(
            Path(args.models_dir) / f"nca_{name}.pt", map_location="cpu", weights_only=False
        )
        state_dicts[name] = checkpoint["state_dict"]
    buffer = io.BytesIO()
    torch.save(state_dicts, buffer)
    weights = base64.b64encode(zlib.compress(buffer.getvalue(), level=9)).decode("ascii")

    notebook_path = Path(args.notebook)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    replace_line(notebook, "PRETRAINED_VARIANTS_B64 = ", f'PRETRAINED_VARIANTS_B64 = "{weights}"\n')

    texture_checkpoint = Path(args.models_dir) / "nca_texture.pt"
    texture_png = Path(args.models_dir) / "texture_target.png"
    if texture_checkpoint.exists():
        checkpoint = torch.load(texture_checkpoint, map_location="cpu", weights_only=False)
        buffer = io.BytesIO()
        torch.save(checkpoint["state_dict"], buffer)
        encoded = base64.b64encode(zlib.compress(buffer.getvalue(), level=9)).decode("ascii")
        replace_line(notebook, "TEXTURE_WEIGHTS_B64 = ", f'TEXTURE_WEIGHTS_B64 = "{encoded}"\n')
        png = base64.b64encode(texture_png.read_bytes()).decode("ascii")
        replace_line(notebook, "TEXTURE_PNG_B64 = ", f'TEXTURE_PNG_B64 = "{png}"\n')
    else:
        print(f"{texture_checkpoint} not found; keeping the texture model already in the notebook.")

    target = Path(args.target)
    if target.exists():
        png = base64.b64encode(target.read_bytes()).decode("ascii")
        replace_line(notebook, "LIZARD_PNG_B64 = ", f'LIZARD_PNG_B64 = "{png}"\n')
    else:
        print(f"{target} not found; keeping the image already embedded in the notebook.")

    notebook_path.write_text(
        json.dumps(notebook, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Embedded {len(state_dicts)} variants into {notebook_path}.")


if __name__ == "__main__":
    main()
