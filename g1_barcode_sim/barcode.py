"""Barcode label textures and USD attachment.

Generation needs only python-barcode / qrcode / Pillow. The USD part imports `pxr` lazily, so this
module can be used outside Isaac Sim (e.g. by scripts/generate_barcodes.py).

Symbols are drawn module-by-module at an integer number of pixels per module, so bars stay crisp
(no resampling) and the symbol keeps its true aspect ratio inside the configured pattern box.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FACE_AXES = {  # face label -> (outward normal, in-face "right" u axis, in-face "up" v axis), box-local frame
    "-x": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "+x": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "-y": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "+y": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "+z": ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
}
# u = right, v = up as seen by a viewer standing outside the face looking at it (so the printed
# pattern reads left-to-right): right = up x normal.


def barcode_content(fmt: str, scene_id: int, box_num: int, face: str, face_codes: dict) -> str:
    return fmt.format(scene=scene_id, box=box_num, face=face_codes[face])


def _code128_modules(content: str) -> str:
    import barcode
    return barcode.get("code128", content).build()[0]  # '1101...' bar/space modules, no quiet zone


def _qr_modules(content: str) -> np.ndarray:
    import qrcode
    qr = qrcode.QRCode(border=0, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(content)
    qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=bool)


def _font(px: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default(size=px)


@dataclass
class LabelImage:
    image: Image.Image
    content: str
    type: str
    label_size_m: tuple      # (w, h)
    pattern_size_m: tuple    # actual drawn symbol size (w, h)
    px_per_m: float


def make_label(content: str, cfg: dict, btype: str | None = None) -> LabelImage:
    """Render the white label (incl. quiet zone) with the symbol and optional human-readable text."""
    btype = btype or cfg["type"]
    lw, lh = cfg["label_size"][btype]
    pw, ph = cfg["pattern_size"][btype]
    long_px = cfg["texture_long_edge_px"]
    ppm = long_px / max(lw, lh)                       # pixels per metre
    W, H = int(round(lw * ppm)), int(round(lh * ppm))
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    text_h = int(round(cfg["text_height_frac"] * H)) if cfg["print_text"] else 0
    box_w, box_h = int(pw * ppm), int(ph * ppm)

    if btype == "code128":
        mods = _code128_modules(content)
        px = box_w // len(mods)                        # integer px per module
        if px < 1:
            raise ValueError(f"{content!r}: {len(mods)} modules do not fit {box_w}px; raise texture_long_edge_px")
        sym_w, sym_h = px * len(mods), box_h
        x0 = (W - sym_w) // 2
        y0 = (H - text_h - sym_h) // 2
        for i, m in enumerate(mods):
            if m == "1":
                draw.rectangle([x0 + i * px, y0, x0 + (i + 1) * px - 1, y0 + sym_h - 1], fill=0)
    elif btype == "qr":
        mat = _qr_modules(content)
        n = mat.shape[0]
        px = min(box_w, box_h - text_h) // n
        sym_w = sym_h = px * n
        x0 = (W - sym_w) // 2
        y0 = (H - text_h - sym_h) // 2
        for r in range(n):
            for c in range(n):
                if mat[r, c]:
                    draw.rectangle([x0 + c * px, y0 + r * px, x0 + (c + 1) * px - 1, y0 + (r + 1) * px - 1], fill=0)
    else:
        raise ValueError(f"unknown barcode type {btype!r}")

    if text_h:
        font = _font(int(text_h * 0.8))
        tw = draw.textlength(content, font=font)
        draw.text(((W - tw) / 2, y0 + sym_h + (H - (y0 + sym_h) - text_h) / 2), content, fill=0, font=font)
    return LabelImage(img.convert("RGB"), content, btype, (lw, lh), (sym_w / ppm, sym_h / ppm), ppm)


def decode_image(img) -> list[str]:
    from pyzbar.pyzbar import decode
    return [d.data.decode() for d in decode(img)]


# ---------------------------------------------------------------------------------------------
# USD attachment
# ---------------------------------------------------------------------------------------------
def placement_in_box(box_size, face: str, uv=(0.0, 0.0), rot_deg: float = 0.0, face_offset: float = 0.0015):
    """Label centre / axes in the box-local frame. Returns (center, normal, right, up) as np arrays.
    `rot_deg` rotates the label in-plane about the normal (counter-clockwise seen from outside)."""
    n, u, v = (np.array(a, dtype=float) for a in FACE_AXES[face])
    half = np.asarray(box_size, dtype=float) / 2.0
    center = n * (np.abs(n) @ half + face_offset) + uv[0] * u + uv[1] * v
    a = math.radians(rot_deg)
    right = math.cos(a) * u + math.sin(a) * v
    up = -math.sin(a) * u + math.cos(a) * v
    return center, n, right, up


def author_label(stage, prim_path: str, texture_path: str, size_wh, thickness: float,
                 center, normal, right, up, roughness: float = 0.8, metallic: float = 0.0, plate: bool = True):
    """Create a textured thin plate: front quad (texture) + white thin box behind it (plate=False: quad only, i.e.
    printed directly on the box). The quad's local frame: +X = right, +Y = up, +Z = normal (outward)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    xf = UsdGeom.Xform.Define(stage, prim_path)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*right, 0.0))
    m.SetRow(1, Gf.Vec4d(*up, 0.0))
    m.SetRow(2, Gf.Vec4d(*normal, 0.0))
    m.SetRow(3, Gf.Vec4d(*center, 1.0))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)

    w, h = size_wh
    quad = UsdGeom.Mesh.Define(stage, f"{prim_path}/face")
    quad.CreatePointsAttr([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)])
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    quad.CreateNormalsAttr([(0, 0, 1)] * 4)
    quad.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    quad.CreateDoubleSidedAttr(False)
    st = UsdGeom.PrimvarsAPI(quad).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])

    if plate:
        back = UsdGeom.Cube.Define(stage, f"{prim_path}/plate")
        back.CreateSizeAttr(1.0)
        bx = UsdGeom.Xformable(back)
        bx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -thickness / 2 - 1e-4))
        bx.AddScaleOp().Set(Gf.Vec3d(w, h, thickness))
        back.CreateDisplayColorAttr([(1.0, 1.0, 1.0)])

    mat = UsdShade.Material.Define(stage, f"{prim_path}/mat")
    sh = UsdShade.Shader.Define(stage, f"{prim_path}/mat/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    reader = UsdShade.Shader.Define(stage, f"{prim_path}/mat/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{prim_path}/mat/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(mat)
    return xf
