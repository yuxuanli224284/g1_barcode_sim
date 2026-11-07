"""Make sure every barcode in a layout has its label PNG (generated on demand, decoded once)."""
from __future__ import annotations

import os

from .barcode import decode_image, make_label


def ensure_textures(layout: dict, bcfg: dict, root: str) -> str:
    out = os.path.join(root, bcfg["type"])
    os.makedirs(out, exist_ok=True)
    for bc in layout["barcodes"]:
        path = os.path.join(out, f"{bc['content']}.png")
        if os.path.exists(path):
            continue
        lab = make_label(bc["content"], bcfg)
        dec = decode_image(lab.image)
        if dec != [bc["content"]]:
            raise RuntimeError(f"generated label for {bc['content']!r} decodes as {dec}")
        lab.image.save(path)
    return out
