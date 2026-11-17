"""Convert the EngineAI T800 Pro URDF to USD with Isaac Lab's UrdfConverter, then describe the result.

  gpuq run -n 1 --note "HumanScan T800 urdf->usd" --eta 15m -- \
      /data/yicheng/HumanoidBarcodeScan/conda_envs/unitree_sim_env/bin/python t800/convert_urdf.py

Input : <SDK>/assets/resource/robot/t800pro/urdf/serial_t800pro.urdf (BSD-3, local only)
Output: /data/yicheng/HumanoidBarcodeScan/third_party/engineai/usd/t800pro/t800pro.usd (local only, never published)
        t800/out/usd_convert_report.json (link / joint / collider listing, total mass)

The URDF references its meshes as "meshes/<name>.dae" relative to its own directory, but the SDK keeps them one
level up (t800pro/meshes). The URDF is therefore copied into a staging directory next to a symlink to the mesh
folder; the file itself is not modified.

Converter settings (each one a stated choice):
  fix_base=False               floating humanoid
  merge_fixed_joints=True      the only fixed joints are J_FIXED_FOOT_L/R (massless LINK_FOOT_* frames, 1 g, no
                               geometry); they merge into LINK_ANKLE_ROLL_*. Every link the scanner / head camera
                               needs (LINK_WRIST_ROLL_R, LINK_HEAD_YAW) is a revolute child and is kept.
  joint_drive gains 0 / 0      the actuator models of the ArticulationCfg (t800_robot.py) set the gains; no
                               drive values are baked into the USD
  collider_type=convex_hull    Isaac Lab default; only the two ankle-roll links have mesh colliders (the feet)
  self_collision=False         as the G1 config of this project. NOT as the walk-policy MJCF: there every collision
                               geom has contype = conaffinity = 1 with no <exclude> / <pair> (nexclude = npair = 0), so
                               MuJoCo (and the real robot) has self-collision; with the arms at the walk-policy default
                               pose it never fires (15 s stand / walk / turn in MuJoCo: no non-ground contact). It
                               matters once the arm IK brings an arm near the torso: Isaac then lets the arm pass
                               through the torso (the URDF has primitive colliders on LINK_WAIST_YAW and on every arm
                               link), and only the clearance model (collision.py) guards against it. Can be enabled
                               later without re-converting (ArticulationRootPropertiesCfg.enabled_self_collisions).
  make_instanceable=True       Isaac Lab default
"""
from __future__ import annotations

import json
import os
import shutil
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

SDK = os.environ.get("T800_SDK_ROOT",
                     "/data/yicheng/HumanoidBarcodeScan/third_party/engineai/engineai_robotics_native_sdk")
ROBOT_DIR = os.path.join(SDK, "assets/resource/robot/t800pro")
URDF = os.path.join(ROBOT_DIR, "urdf/serial_t800pro.urdf")
USD_ROOT = "/data/yicheng/HumanoidBarcodeScan/third_party/engineai/usd"
STAGE_DIR = os.path.join(USD_ROOT, "urdf_staging")
OUT_DIR = os.path.join(USD_ROOT, "t800pro")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "usd_convert_report.json")


def stage_urdf() -> str:
    os.makedirs(STAGE_DIR, exist_ok=True)
    dst = os.path.join(STAGE_DIR, "serial_t800pro.urdf")
    shutil.copyfile(URDF, dst)
    link = os.path.join(STAGE_DIR, "meshes")
    if not os.path.islink(link):
        os.symlink(os.path.join(ROBOT_DIR, "meshes"), link)
    return dst


def describe(usd_path: str) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    links, joints, colliders, mass = [], [], [], 0.0
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(p).GetMassAttr().Get() if p.HasAPI(UsdPhysics.MassAPI) else None
            links.append({"path": str(p.GetPath()), "mass": m})
            mass += m or 0.0
        if p.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(p)
            joints.append({"path": str(p.GetPath()), "type": p.GetTypeName(),
                           "body0": [str(t) for t in j.GetBody0Rel().GetTargets()],
                           "body1": [str(t) for t in j.GetBody1Rel().GetTargets()]})
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim):
            colliders.append({"path": str(p.GetPath()), "type": p.GetTypeName()})
    return {"usd": usd_path, "default_prim": str(root.GetPath()), "n_links": len(links), "total_mass_kg": mass,
            "links": links, "n_joints": len(joints), "joints": joints, "colliders": colliders}


def main():
    cfg = UrdfConverterCfg(
        asset_path=stage_urdf(), usd_dir=OUT_DIR, usd_file_name="t800pro.usd", force_usd_conversion=True,
        fix_base=False, merge_fixed_joints=True, collider_type="convex_hull", self_collision=False,
        make_instanceable=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0), target_type="position"),
    )
    conv = UrdfConverter(cfg)
    rep = describe(conv.usd_path)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(f"CONVERTED {conv.usd_path}: {rep['n_links']} links, {rep['n_joints']} joints, "
          f"{len(rep['colliders'])} colliders, mass {rep['total_mass_kg']:.3f} kg; report {REPORT}", flush=True)


if __name__ == "__main__":
    main()
    # the USD is written by the converter; skip Kit's shutdown (it can hang, see g1_barcode_sim/env.py close())
    sys.stdout.flush()
    os._exit(0)
