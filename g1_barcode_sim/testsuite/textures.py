"""Textures for test-suite scenarios (pure Python / Pillow): barcode labels and box decals, cached by a hash of their
parameters under assets/barcodes/cache/.

Labels are drawn at an integer number of pixels per module (factors.yaml barcode.texture_px_per_module), so bars
stay crisp and the drawn symbol has exactly the requested physical size once the quad is scaled to it. With a
white backing the texture covers pattern + border; without it the texture is just the pattern on the box colour.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from ..barcode import _code128_modules, _font, _qr_modules, decode_image
from ..config import PKG_ROOT
from .scenario import label_size, pattern_size

CACHE = os.path.join(PKG_ROOT, "assets", "barcodes", "cache")

# base colours of the box appearances (linear-ish sRGB 0-1); decals carry the rest of the look
APPEARANCE_COLOR = {"kraft": (0.66, 0.50, 0.32), "kraft_tape": (0.66, 0.50, 0.32), "white_box": (0.90, 0.90, 0.88),
                    "printed": (0.70, 0.56, 0.38),
                    # "cardbox": the Isaac warehouse shipping-box asset (factors.yaml appearance.cardbox_usd) scaled to
                    # the box; this colour is only the background of an unbacked label on it (kraft, as the asset)
                    "cardbox": (0.66, 0.50, 0.32)}
TAPE_COLOR = (0.80, 0.68, 0.46)


def _key(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:20]


def label_texture(code: dict, appearance: str, border: float, px_per_module: int) -> str:
    """PNG path of the label texture for a scenario barcode entry (generated on first use, decode-checked)."""
    bg = APPEARANCE_COLOR[appearance] if not code["label_backing"] else (1.0, 1.0, 1.0)
    spec = {"content": code["content"], "type": code["type"], "pattern": pattern_size(code), "backing": code["label_backing"],
            "border": border, "bg": bg, "ppm": px_per_module, "v": 1}
    path = os.path.join(CACHE, f"label_{_key(spec)}.png")
    if os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    pw, ph = pattern_size(code)
    if code["type"] == "code128":
        mods = _code128_modules(code["content"])
        wpx = len(mods) * px_per_module
        hpx = max(1, int(round(wpx * ph / pw)))
        pat = Image.new("L", (wpx, hpx), 255)
        d = ImageDraw.Draw(pat)
        for i, m in enumerate(mods):
            if m == "1":
                d.rectangle([i * px_per_module, 0, (i + 1) * px_per_module - 1, hpx - 1], fill=0)
    else:
        mat = _qr_modules(code["content"])
        n = mat.shape[0]
        wpx = hpx = n * px_per_module
        arr = np.where(np.kron(mat, np.ones((px_per_module, px_per_module), dtype=bool)), 0, 255).astype(np.uint8)
        pat = Image.fromarray(arr, "L")
    ppm = wpx / pw                                              # texture pixels per metre
    bgc = tuple(int(round(255 * c)) for c in bg)
    if code["label_backing"]:
        b = int(round(border * ppm))
        img = Image.new("RGB", (wpx + 2 * b, hpx + 2 * b), (255, 255, 255))
        img.paste(pat.convert("RGB"), (b, b))
    else:
        rgb = np.array(pat.convert("RGB"))
        rgb[np.array(pat) > 127] = bgc                          # light modules = box surface
        img = Image.fromarray(rgb)
    # sanity: the texture itself must decode (pad with a quiet zone of the background colour first)
    q = 12 * px_per_module
    padded = Image.new("RGB", (img.width + 2 * q, img.height + 2 * q), bgc if not code["label_backing"] else (255, 255, 255))
    padded.paste(img, (q, q))
    dec = decode_image(padded)
    if dec != [code["content"]]:
        raise RuntimeError(f"label texture for {code['content']!r} decodes as {dec}")
    img.save(path)
    return path


def printed_decal(seed_text: str, size_m, px_per_m: int = 1500) -> str:
    """Random printed artwork (text lines, logos, arrows) on the printed-box colour, covering a whole face; deterministic
    from seed_text."""
    spec = {"seed": seed_text, "size": [round(x, 4) for x in size_m], "ppm": px_per_m, "v": 1}
    path = os.path.join(CACHE, f"printed_{_key(spec)}.png")
    if os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    rng = np.random.default_rng(int(hashlib.sha1(seed_text.encode()).hexdigest()[:12], 16))
    W, H = max(8, int(size_m[0] * px_per_m)), max(8, int(size_m[1] * px_per_m))
    base = tuple(int(round(255 * c)) for c in APPEARANCE_COLOR["printed"])
    img = Image.new("RGB", (W, H), base)          # opaque: the decal covers the whole face
    d = ImageDraw.Draw(img)
    palette = [(30, 30, 30), (160, 20, 20), (20, 60, 140), (20, 110, 50)]   # RGB
    words = ["FRAGILE", "THIS SIDE UP", "HANDLE WITH CARE", "LOT", "MADE IN", "EXPORT", "KEEP DRY", "NET WT", "No.", "ITEM"]
    for _ in range(int(rng.integers(3, 7))):                      # text lines
        txt = str(rng.choice(words)) + " " + "".join(str(int(x)) for x in rng.integers(0, 10, size=int(rng.integers(0, 6))))
        fpx = int(rng.integers(H // 18 + 4, H // 7 + 6))
        x, y = int(rng.integers(0, max(1, W - fpx * 4))), int(rng.integers(0, max(1, H - fpx)))
        d.text((x, y), txt, fill=palette[int(rng.integers(0, len(palette)))], font=_font(fpx))
    for _ in range(int(rng.integers(1, 4))):                      # logos / boxes / arrows
        c = palette[int(rng.integers(0, len(palette)))]
        x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
        s = int(rng.integers(H // 10 + 4, H // 4 + 8))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            d.ellipse([x0, y0, x0 + s, y0 + s], outline=c, width=max(2, s // 10))
        elif kind == 1:
            d.rectangle([x0, y0, x0 + 2 * s, y0 + s], outline=c, width=max(2, s // 12))
        else:
            d.polygon([(x0, y0 + s), (x0 + s // 2, y0), (x0 + s, y0 + s)], fill=c)
    img.save(path)
    return path
