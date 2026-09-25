"""Embed a trained NCA state dict into the self-contained tutorial notebook."""

from __future__ import annotations

import argparse
import base64
import io
import json
import zlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/pretrained_nca.pt")
    parser.add_argument("--notebook", default="from_ca_to_nca.ipynb")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    buffer = io.BytesIO()
    torch.save(checkpoint["state_dict"], buffer)
    encoded = base64.b64encode(zlib.compress(buffer.getvalue(), level=9)).decode("ascii")

    notebook_path = Path(args.notebook)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    matching_cells = []
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        if any(line.startswith("PRETRAINED_WEIGHTS_B64 = ") for line in source):
            matching_cells.append(cell)

    if len(matching_cells) != 1:
        raise RuntimeError(f"Expected one pretrained-weight cell, found {len(matching_cells)}.")

    source = matching_cells[0]["source"]
    for index, line in enumerate(source):
        if line.startswith("PRETRAINED_WEIGHTS_B64 = "):
            source[index] = f'PRETRAINED_WEIGHTS_B64 = "{encoded}"\n'
        elif line.startswith("PRETRAINED_ITERATION = "):
            source[index] = f'PRETRAINED_ITERATION = {checkpoint["iteration"]}\n'

    notebook_path.write_text(
        json.dumps(notebook, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Embedded iteration {checkpoint['iteration']} from {args.checkpoint} "
        f"into {notebook_path}."
    )


if __name__ == "__main__":
    main()
