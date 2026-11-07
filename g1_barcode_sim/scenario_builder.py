"""Runtime (re)building of test-suite scenarios on the USD stage.

The base scene (lights, ground collider, the three backgrounds) is authored once before the simulation starts
(scene_builder.build_scene with `scenario_backgrounds`). Every reset(scenario=...) deletes `/World/BarcodeScene/Scenario`
and authors the table, boxes, decals and labels again from the scenario layout: prims (incl. static colliders)
added while the simulation runs are picked up by PhysX and the renderer (verified: the robot is stopped by a
collider added at runtime). Backgrounds are switched by visibility, lights rescaled per background.

  /World/BarcodeScene/Scenario/Table/{top,leg0..3}        white static colliders
  /World/BarcodeScene/Scenario/Boxes/<id>/collider        invisible exact-size static collider
                                         /visual           box-coloured cube (appearance), or the shipping-box
                                                           art asset scaled to the box ("cardbox")
                                         /decal_front|top  printed artwork / tape (appearance)
                                         /BC_mx|BC_px|BC_my|BC_py  barcode label on the -x/+x/-y/+y face
                                                           (texture; white plate if backed); further labels on the
                                                           same face: BC_mx_1, BC_mx_2, ...
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from isaaclab.sim.utils import add_labels

from .barcode import author_label
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .scene_builder import ROOT, _cuboid, _disable_physics_below, _local_bbox, _preview_material, _set_xform
from .testsuite.scenario import LABEL_PRIM
from .testsuite.textures import APPEARANCE_COLOR, TAPE_COLOR, label_texture, printed_decal

SCN = f"{ROOT}/Scenario"
BACKGROUNDS = {"warehouse": f"{ROOT}/Background", "none": f"{ROOT}/BG_none", "white": f"{ROOT}/BG_white"}


def author_extra_backgrounds(stage, cfg: dict):
    """`none`: neutral grey floor only. `white`: white floor + white walls around the work area (a box-shaped
    cyclorama). Visual only (the ground collider is shared); created invisible."""
    for p in ("none", "white"):   # typed Xform parents: an implicit (typeless) parent is not Imageable and ignores MakeInvisible
        UsdGeom.Xform.Define(stage, BACKGROUNDS[p])
    n = cfg["none"]
    _cuboid(stage, f"{BACKGROUNDS['none']}/floor", (0, 0, -0.005), (n["floor_size"], n["floor_size"], 0.01), color=n["floor_color"],
            roughness=0.9, collider=False)
    w = cfg["white"]
    L, Hh, c = w["size"], w["wall_height"], w["center"]
    _cuboid(stage, f"{BACKGROUNDS['white']}/floor", (c[0], c[1], -0.005), (L, L, 0.01), color=w["color"], roughness=1.0, collider=False)
    for i, (off, s) in enumerate([((L / 2, 0), (0.05, L)), ((-L / 2, 0), (0.05, L)), ((0, L / 2), (L, 0.05)), ((0, -L / 2), (L, 0.05))]):
        _cuboid(stage, f"{BACKGROUNDS['white']}/wall{i}", (c[0] + off[0], c[1] + off[1], Hh / 2), (s[0], s[1], Hh),
                color=w["color"], roughness=1.0, collider=False)
    for p in ("none", "white"):
        UsdGeom.Imageable(stage.GetPrimAtPath(BACKGROUNDS[p])).MakeInvisible()


def apply_background(stage, bg: dict, lighting: dict):
    """Show one background, hide the others, set the lights of that background times light_scale."""
    if bg["type"] not in BACKGROUNDS:
        raise ValueError(f"unknown background {bg['type']!r}")
    for k, path in BACKGROUNDS.items():
        im = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        if not im:
            raise RuntimeError(f"background prim {path} is not Imageable")
        im.MakeVisible() if k == bg["type"] else im.MakeInvisible()
        want = "inherited" if k == bg["type"] else "invisible"
        if im.ComputeVisibility() != want:
            raise RuntimeError(f"background {k}: visibility {im.ComputeVisibility()} != {want}")
    lt = lighting[bg["type"]]
    s = float(bg["light_scale"])
    stage.GetPrimAtPath(f"{ROOT}/Lights/Dome").GetAttribute("inputs:intensity").Set(lt["dome"] * s)
    stage.GetPrimAtPath(f"{ROOT}/Lights/Sun").GetAttribute("inputs:intensity").Set(lt["distant"] * s)


def _quad(stage, path, w, h, center, normal, right, up, material):
    xf = UsdGeom.Xform.Define(stage, path)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*[float(x) for x in right], 0.0)); m.SetRow(1, Gf.Vec4d(*[float(x) for x in up], 0.0))
    m.SetRow(2, Gf.Vec4d(*[float(x) for x in normal], 0.0)); m.SetRow(3, Gf.Vec4d(*[float(x) for x in center], 1.0))
    xf.AddTransformOp().Set(m)
    q = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    q.CreatePointsAttr([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)])
    q.CreateFaceVertexCountsAttr([4]); q.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    q.CreateNormalsAttr([(0, 0, 1)] * 4); q.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(q).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(q.GetPrim()).Bind(material)


def _texture_material(stage, path, tex_path, roughness=0.85):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    rd = UsdShade.Shader.Define(stage, f"{path}/st_reader")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, f"{path}/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def build_scenario(stage, layout: dict, fcfg: dict, bcfg_label: dict) -> dict:
    """Delete the previous scenario prims and author this one. Returns authored prim paths."""
    if stage.GetPrimAtPath(SCN).IsValid():
        stage.RemovePrim(SCN)
    UsdGeom.Xform.Define(stage, SCN)
    paths = {"furniture": {}, "boxes": {}, "barcodes": {}}
    for f in layout["furniture"]:
        fp = f"{SCN}/Table"
        UsdGeom.Xform.Define(stage, fp)
        add_labels(stage.GetPrimAtPath(fp), labels=["table"], instance_name="class")
        paths["furniture"][f["id"]] = [_cuboid(stage, f"{fp}/{p['name']}", p["center"], p["size"], color=p["color"],
                                               roughness=p["roughness"]).GetPath().pathString for p in f["parts"]]
    B = fcfg["barcode"]
    eps = 0.0008                                                   # decals sit 0.8 mm off the face (labels: 1.5 mm)
    for b in layout["boxes"]:
        bp = f"{SCN}/Boxes/{b['id']}"
        xf = UsdGeom.Xform.Define(stage, bp)
        _set_xform(xf.GetPrim(), b["pos"], b["yaw"])
        add_labels(xf.GetPrim(), labels=["box"], instance_name="class")
        d, w, h = b["size"]
        _cuboid(stage, f"{bp}/collider", (0, 0, 0), b["size"], collider=True, visible=False)
        app = b["appearance"]
        if app == "cardbox":
            # the shipping-box art asset scaled so its bounding box is exactly the box (as scene_builder box_visual)
            vp = stage.DefinePrim(f"{bp}/visual", "Xform")
            vp.GetReferences().AddReference(f"{ISAAC_NUCLEUS_DIR}/{fcfg['appearance']['cardbox_usd']}")
            _disable_physics_below(vp)
            mn, mx = _local_bbox(stage, vp)
            native = mx - mn
            if np.any(native <= 0):
                raise RuntimeError(f"box asset {fcfg['appearance']['cardbox_usd']} has an empty bounding box (not loaded?)")
            scale = np.asarray(b["size"]) / native
            _set_xform(vp, -(mn + mx) / 2 * scale, 0.0, scale)
        else:
            _cuboid(stage, f"{bp}/visual", (0, 0, 0), b["size"], color=APPEARANCE_COLOR[app], roughness=0.9, collider=False)
        front = dict(center=(-d / 2 - eps, 0, 0), normal=(-1, 0, 0), right=(0, -1, 0), up=(0, 0, 1))
        top = dict(center=(0, 0, h / 2 + eps), normal=(0, 0, 1), right=(0, 1, 0), up=(1, 0, 0))
        sid = layout["scenario"]["id"]
        if app == "printed":
            _quad(stage, f"{bp}/decal_front", w, h, **front, material=_texture_material(stage, f"{bp}/m_front", printed_decal(f"{sid}/{b['id']}/front", (w, h))))
            _quad(stage, f"{bp}/decal_top", w, d, **top, material=_texture_material(stage, f"{bp}/m_top", printed_decal(f"{sid}/{b['id']}/top", (w, d))))
        elif app == "kraft_tape":
            tw, drop = fcfg["appearance"]["tape_width"], fcfg["appearance"]["tape_drop"]
            tm = _preview_material(stage, f"{bp}/m_tape", TAPE_COLOR, fcfg["appearance"]["tape_roughness"])
            _quad(stage, f"{bp}/decal_top", tw, d, **top, material=tm)
            fr = dict(front); fr["center"] = (-d / 2 - eps, 0, h / 2 - min(drop, h) / 2)
            _quad(stage, f"{bp}/decal_front", tw, min(drop, h), **fr, material=tm)
        paths["boxes"][b["id"]] = bp
    for bc in layout["barcodes"]:
        bp = paths["boxes"][bc["box_id"]]
        lp, k = f"{bp}/{LABEL_PRIM[bc['face']]}", 0
        while stage.GetPrimAtPath(lp).IsValid():             # several labels on one face (to_layout checks they do not overlap)
            k += 1
            lp = f"{bp}/{LABEL_PRIM[bc['face']]}_{k}"
        app = next(x["appearance"] for x in layout["boxes"] if x["id"] == bc["box_id"])
        tex = label_texture(bc["code"], app, B["backing_border"], B["texture_px_per_module"])
        loc = bc["local"]
        author_label(stage, lp, tex, bc["size"], bcfg_label["label_thickness"], loc["center"], loc["normal"], loc["right"], loc["up"],
                     roughness=bcfg_label["material"]["roughness"], metallic=bcfg_label["material"]["metallic"],
                     plate=bc["label_backing"])
        add_labels(stage.GetPrimAtPath(lp), labels=["barcode"], instance_name="class")
        paths["barcodes"][bc["content"]] = lp
    return paths
