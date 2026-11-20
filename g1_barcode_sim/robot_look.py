"""Materials of a robot imported from URDF (robot variant g1_v2), applied after the articulation is spawned.

Isaac's URDF importer creates materials only for the URDF's named materials (white / dark) and a few Dex1 colours; parts
whose colour is given inline in their <visual> (the scanner bracket, scanner, D435, D405s, D405 mount) keep their mesh's
default material. And the named greys differ from the hand-tuned materials of unitree's G1 USD (robot variant g1_v1).

robot.yaml `looks`:
  materials: {<imported material name>: {<OmniPBR input>: value}}   shader inputs to override (e.g. the g1_v1 look)
  mesh_colors: {<mesh stem>: rgba}   bind a UsdPreviewSurface of that colour to these meshes
  urdf_inline_colors: true    bind a UsdPreviewSurface with the URDF inline colour to every visual that has one
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from pxr import Gf, Sdf, Usd, UsdShade


def urdf_inline_colors(urdf_path: str) -> dict:
    """{(link, mesh stem): rgba} of the URDF visuals whose <material> carries its own <color>."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        for v in link.findall("visual"):
            m, g = v.find("material"), v.find("geometry/mesh")
            c = m.find("color") if m is not None else None
            if c is not None and g is not None:
                stem = os.path.splitext(os.path.basename(g.get("filename")))[0]
                out[(link.get("name"), stem)] = [float(x) for x in c.get("rgba").split()]
    return out


def _preview_material(stage, path, rgba, roughness=0.6):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgba[:3]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def apply_looks(stage, robot_path: str, urdf_path: str, looks: dict) -> dict:
    """Returns what was changed (for logs)."""
    done = {"materials": [], "inline": [], "uninstanced": 0}
    # instanced <link>/visuals: the renderer kept the imported material look on instanced copies (measured: edited
    # materials showed only on un-instanced links), so every robot visual is made a plain prim first
    for p in list(Usd.PrimRange(stage.GetPrimAtPath(robot_path))):
        if p.GetName() == "visuals" and p.IsInstance():
            p.SetInstanceable(False)
            done["uninstanced"] += 1
    for name, inputs in (looks.get("materials") or {}).items():
        p = stage.GetPrimAtPath(f"{robot_path}/Looks/{name}/Shader")
        if not p.IsValid():
            raise RuntimeError(f"robot material {name} not found under {robot_path}/Looks (robot.yaml looks.materials)")
        sh = UsdShade.Shader(p)
        for k, v in inputs.items():
            typ = Sdf.ValueTypeNames.Color3f if isinstance(v, list) else Sdf.ValueTypeNames.Float
            sh.CreateInput(k, typ).Set(Gf.Vec3f(*v) if isinstance(v, list) else float(v))
        done["materials"].append(name)
    colors = urdf_inline_colors(urdf_path) if looks.get("urdf_inline_colors") else {}
    colors.update({("mesh", stem): rgba for stem, rgba in (looks.get("mesh_colors") or {}).items()})
    if colors:
        # merged fixed joints: a child link's visuals sit under its movable parent's <link>/visuals/<mesh stem>
        for (link, stem), rgba in colors.items():
            hits = [p for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies())
                    if p.GetName() == stem and p.GetParent().GetName() == "visuals"]
            if not hits:
                raise RuntimeError(f"no visual prim for mesh {stem} (link {link}) under {robot_path}")
            for p in hits:
                vis = p.GetParent()
                if vis.IsInstance():           # instanced visuals cannot carry per-mesh bindings
                    vis.SetInstanceable(False)
                p = stage.GetPrimAtPath(p.GetPath())
                # the importer's per-mesh DefaultMaterial is bound on the mesh itself: bind ours with stronger strength
                mat = _preview_material(stage, f"{robot_path}/Looks/urdf_{link}_{stem}", rgba)
                api = UsdShade.MaterialBindingAPI.Apply(p)
                api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                done["inline"].append(f"{link}:{stem}")
    return done
