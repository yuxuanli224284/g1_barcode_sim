"""Author a scene layout (layout.py) on the USD stage.

Exposed as an Isaac Lab spawner (`BarcodeSceneSpawnerCfg`) so the whole scene is created by the
InteractiveScene together with the robot, before the simulation starts. Everything lives under
`/World/BarcodeScene` (world frame, outside the per-env namespace):

  /World/BarcodeScene/Lights/{Dome,Sun}
  /World/BarcodeScene/Ground                     invisible static collider, top at z = 0
  /World/BarcodeScene/Background                 Simple_Warehouse reference (or geometry fallback)
  /World/BarcodeScene/Furniture/<id>/<part>      static cuboids (class=table|shelf)
  /World/BarcodeScene/Boxes/<id>                 Xform at the box pose (class=box)
        collider   exact-size cube (static, or rigid body when dynamic: true)
        visual     cardboard asset scaled to the box size (or a kraft-coloured cube)
        BC_<face>  label (class=barcode), child of the box so it follows it
"""
from __future__ import annotations

import math
import os
from dataclasses import MISSING

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import add_labels
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .barcode import author_label

ROOT = "/World/BarcodeScene"


def _set_xform(prim, pos=(0, 0, 0), yaw=0.0, scale=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    x.AddRotateZOp().Set(math.degrees(yaw))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in scale]))


def _preview_material(stage, path, color, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _cuboid(stage, path, center, size, yaw=0.0, color=None, roughness=0.8, collider=True, visible=True, material=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), center, yaw, size)
    if color is not None:
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        mat = material or _preview_material(stage, f"{path}_mat", color, roughness)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube).MakeInvisible()
    return cube.GetPrim()


def _disable_physics_below(prim):
    """Referenced art assets may carry their own rigid-body / collision APIs; the exact-size collider
    cube is the physics proxy, so switch those off."""
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)


def _local_bbox(stage, prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


def build_scene(stage, layout: dict, scfg: dict, bcfg: dict, texture_dir: str, asset_root: str = ISAAC_NUCLEUS_DIR):
    """Author everything. Returns a dict of authored prim paths (for GT read-back / checks)."""
    UsdGeom.Xform.Define(stage, ROOT)
    paths = {"boxes": {}, "barcodes": {}, "furniture": {}, "notes": []}

    # lights
    lt = scfg["lighting"]
    dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/Lights/Dome")
    dome.CreateIntensityAttr(lt["dome"]["intensity"])
    dome.CreateColorAttr(Gf.Vec3f(*lt["dome"]["color"]))
    sun = UsdLux.DistantLight.Define(stage, f"{ROOT}/Lights/Sun")
    sun.CreateIntensityAttr(lt["distant"]["intensity"])
    sun.CreateColorAttr(Gf.Vec3f(*lt["distant"]["color"]))
    rx, ry, rz = lt["distant"]["angle_deg"]
    sx = UsdGeom.Xformable(sun)
    sx.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))

    # ground collider (the warehouse art is used for looks only)
    _cuboid(stage, f"{ROOT}/Ground", (0, 0, -0.05), (60.0, 60.0, 0.1), collider=True, visible=False)

    # background
    bg = scfg["background"]
    if bg["kind"] == "simple_warehouse":
        p = stage.DefinePrim(f"{ROOT}/Background", "Xform")
        p.GetReferences().AddReference(f"{asset_root}/{bg['usd']}")
        _set_xform(p, bg["offset"], math.radians(bg["yaw_deg"]))
        _disable_physics_below(p)
    else:
        g = bg["geometry_fallback"]
        L, Hh = g["floor_size"], g["wall_height"]
        _cuboid(stage, f"{ROOT}/Background/floor", (0, 0, -0.005), (L, L, 0.01), color=(0.4, 0.4, 0.42), collider=False)
        for i, (c, s) in enumerate([((L / 2, 0, Hh / 2), (0.2, L, Hh)), ((-L / 2, 0, Hh / 2), (0.2, L, Hh)),
                                    ((0, L / 2, Hh / 2), (L, 0.2, Hh)), ((0, -L / 2, Hh / 2), (L, 0.2, Hh))]):
            _cuboid(stage, f"{ROOT}/Background/wall{i}", c, s, color=g["wall_color"], collider=True)

    if "scenario_backgrounds" in scfg:            # test-suite scenario mode: also author `none` / `white` (hidden)
        from .scenario_builder import author_extra_backgrounds
        author_extra_backgrounds(stage, scfg["scenario_backgrounds"])

    dynamic = bool(scfg.get("dynamic", False))

    # furniture
    for f in layout["furniture"]:
        fp = f"{ROOT}/Furniture/{f['id']}"
        UsdGeom.Xform.Define(stage, fp)
        add_labels(stage.GetPrimAtPath(fp), labels=[f["kind"]], instance_name="class")
        paths["furniture"][f["id"]] = []
        for part in f["parts"]:
            pp = _cuboid(stage, f"{fp}/{part['name']}", part["center"], part["size"], color=part["color"], roughness=part["roughness"])
            paths["furniture"][f["id"]].append(pp.GetPath().pathString)

    # boxes
    bv = scfg["box_visual"]
    kraft = None
    for b in layout["boxes"]:
        bp = f"{ROOT}/Boxes/{b['id']}"
        xf = UsdGeom.Xform.Define(stage, bp)
        _set_xform(xf.GetPrim(), b["pos"], b["yaw"])
        add_labels(xf.GetPrim(), labels=["box"], instance_name="class")
        col = _cuboid(stage, f"{bp}/collider", (0, 0, 0), b["size"], collider=True, visible=(bv["kind"] == "geometry"),
                      color=bv["geometry_color"] if bv["kind"] == "geometry" else None, roughness=0.9)
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(scfg.get("box_mass", 2.0)))
        if bv["kind"] == "asset":
            vp = stage.DefinePrim(f"{bp}/visual", "Xform")
            vp.GetReferences().AddReference(f"{asset_root}/{bv['usd']}")
            _disable_physics_below(vp)
            mn, mx = _local_bbox(stage, vp)
            native = mx - mn
            if np.any(native <= 0):
                raise RuntimeError(f"box asset {bv['usd']} has an empty bounding box")
            scale = np.asarray(b["size"]) / native
            center = (mn + mx) / 2
            # place the scaled asset so its bbox centre sits at the box origin
            _set_xform(vp, -center * scale, 0.0, scale)
            paths.setdefault("box_asset_native_size", native.tolist())
        paths["boxes"][b["id"]] = bp

    # barcode labels
    for bc in layout["barcodes"]:
        bp = paths["boxes"][bc["box_id"]]
        lp = f"{bp}/BC_{bc['face'].replace('+', 'p').replace('-', 'm')}"
        tex = os.path.join(texture_dir, f"{bc['content']}.png")
        if not os.path.exists(tex):
            raise FileNotFoundError(f"missing barcode texture {tex}; run scripts/generate_barcodes.py")
        loc = bc["local"]
        author_label(stage, lp, tex, bc["size"], bcfg["label_thickness"], loc["center"], loc["normal"], loc["right"], loc["up"],
                     roughness=bcfg["material"]["roughness"], metallic=bcfg["material"]["metallic"])
        add_labels(stage.GetPrimAtPath(lp), labels=["barcode"], instance_name="class")
        paths["barcodes"][bc["content"]] = lp
    return paths


def update_label_poses(stage, layout: dict, paths: dict):
    """Re-place labels after a (re-)randomisation; labels are visual-only, so this is safe at runtime."""
    from pxr import Gf
    for bc in layout["barcodes"]:
        prim = stage.GetPrimAtPath(paths["barcodes"][bc["content"]])
        loc = bc["local"]
        m = Gf.Matrix4d(1.0)
        m.SetRow(0, Gf.Vec4d(*loc["right"], 0.0)); m.SetRow(1, Gf.Vec4d(*loc["up"], 0.0))
        m.SetRow(2, Gf.Vec4d(*loc["normal"], 0.0)); m.SetRow(3, Gf.Vec4d(*loc["center"], 1.0))
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(m)


# ------------------------------------------------------------------------------------------------
# Isaac Lab spawner wrapper
# ------------------------------------------------------------------------------------------------
def _spawn_barcode_scene(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    # module-level record: the cfg object passed here may be a copy of the one the caller holds
    AUTHORED.clear()
    AUTHORED.update(build_scene(stage, cfg.layout, cfg.scene_cfg, cfg.barcode_cfg, cfg.texture_dir))
    root = stage.GetPrimAtPath(ROOT)
    # InteractiveScene wraps every AssetBaseCfg in an XformPrimView, which needs [translate, orient, scale]
    sim_utils.standardize_xform_ops(root)
    return root


AUTHORED: dict = {}


@configclass
class BarcodeSceneSpawnerCfg(sim_utils.SpawnerCfg):
    func = _spawn_barcode_scene
    layout: dict = MISSING
    scene_cfg: dict = MISSING
    barcode_cfg: dict = MISSING
    texture_dir: str = MISSING
