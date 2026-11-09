"""Embed the target images and pretrained NCA weights into the notebook.

The notebook stays self-contained in Colab: it needs no downloads or extra files.
Run after ``train_nca_variants.py`` and ``train_texture_nca.py``.

All data goes into the collapsed "Embedded data" cell. Long base64 strings are
split into short lines, because the Colab editor can freeze on very long lines.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import zlib
from pathlib import Path

import torch

from train_nca_variants import VARIANTS

DATA_CELL_TITLE = "#@title Embedded data"
NAMES = ("LIZARD_PNG_B64", "PRETRAINED_VARIANTS_B64", "TEXTURE_WEIGHTS_B64", "TEXTURE_PNG_B64")
LINE_LENGTH = 100


def encode_state_dict(state_dict) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(zlib.compress(buffer.getvalue(), level=9)).decode("ascii")


def find_data_cell(notebook: dict) -> dict:
    cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(DATA_CELL_TITLE)
    ]
    if len(cells) != 1:
        raise RuntimeError(f"Expected one cell starting with {DATA_CELL_TITLE!r}, found {len(cells)}.")
    return cells[0]


def read_existing(cell: dict) -> dict[str, str]:
    """Read string assignments from the data cell without executing it."""
    values = {}
    for node in ast.parse("".join(cell["source"])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return values


def format_string(name: str, value: str) -> str:
    chunks = [value[i : i + LINE_LENGTH] for i in range(0, len(value), LINE_LENGTH)] or [""]
    body = "\n".join(f'    "{chunk}"' for chunk in chunks)
    return f"{name} = (\n{body}\n)\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--target", default="assets/emoji_u1f98e.png")
    parser.add_argument("--notebook", default="from_ca_to_nca.ipynb")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)

    notebook_path = Path(args.notebook)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = find_data_cell(notebook)
    title = "".join(cell["source"]).splitlines()[0]
    values = read_existing(cell)

    state_dicts = {}
    for name in VARIANTS:
        checkpoint = torch.load(models_dir / f"nca_{name}.pt", map_location="cpu", weights_only=False)
        state_dicts[name] = checkpoint["state_dict"]
    values["PRETRAINED_VARIANTS_B64"] = encode_state_dict(state_dicts)

    texture_checkpoint = models_dir / "nca_texture.pt"
    if texture_checkpoint.exists():
        checkpoint = torch.load(texture_checkpoint, map_location="cpu", weights_only=False)
        values["TEXTURE_WEIGHTS_B64"] = encode_state_dict(checkpoint["state_dict"])
        png = (models_dir / "texture_target.png").read_bytes()
        values["TEXTURE_PNG_B64"] = base64.b64encode(png).decode("ascii")
    else:
        print(f"{texture_checkpoint} not found; keeping the texture model already in the notebook.")

    target = Path(args.target)
    if target.exists():
        values["LIZARD_PNG_B64"] = base64.b64encode(target.read_bytes()).decode("ascii")
    else:
        print(f"{target} not found; keeping the image already embedded in the notebook.")

    missing = [name for name in NAMES if not values.get(name) or values[name].startswith("__")]
    if missing:
        raise RuntimeError(f"No data for {missing}.")

    source = title + "\n" + "".join(format_string(name, values[name]) for name in NAMES)
    source += 'print("Embedded data loaded.")\n'
    cell["source"] = source.splitlines(keepends=True)

    notebook_path.write_text(
        json.dumps(notebook, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Embedded {len(state_dicts)} variants and the texture model into {notebook_path}.")


if __name__ == "__main__":
    main()
