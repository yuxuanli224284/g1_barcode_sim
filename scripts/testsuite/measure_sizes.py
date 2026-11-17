"""Measure (not assume) the sizes of a scenario's boxes / labels and of the robot standing at its start pose.

  CUDA_VISIBLE_DEVICES=<card> bash t800/run_isaac.sh -u scripts/testsuite/measure_sizes.py \
      --scenario wall6x5_L0_0000 --robot t800 --out t800/out/scenes/sizes_t800.json
  python scripts/testsuite/measure_sizes.py --robot t800 --cpu_only      # URDF + OBJ forward kinematics only

Three independent sources, printed side by side:
  spec / layout  the scenario JSON and to_layout (what the generator asked for)
  Isaac stage    UsdGeom.BBoxCache world bounds of the spawned box / label prims; the robot's link mesh bounds (USD,
                 link frame) placed at the links' physics poses after standing (the USD link xforms are stale under
                 CPU physics, as render_scenario.py), plus the stage metersPerUnit and every scale op on the robot
  URDF FK        (T800 only) the vendor URDF chain x the vendor OBJ meshes at the walk-policy default pose (no Isaac):
                 height, shoulder width, and the right-arm length from the shoulder-pitch axis to the farthest hand
                 vertex with the arm straight, plus the forward reach with the straight arm raised horizontally.
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="wall6x5_L0_0000")
ap.add_argument("--robot", choices=("g1", "t800"), required=True)
ap.add_argument("--out", default=None, help="JSON with every measured number")
ap.add_argument("--settle_s", type=float, default=1.0, help="stand before measuring (as render_scenario.py)")
ap.add_argument("--cpu_only", action="store_true")
args = ap.parse_args()

R = {"robot": args.robot, "scenario": args.scenario}


def obj_vertices(path):
    v = [list(map(float, ln.split()[1:4])) for ln in open(path) if ln.startswith("v ")]
    return np.array(v)


def t800_urdf_fk():
    """Robot mesh extent from the vendor URDF + OBJ meshes (DAE twins, <unit meter="1">), root frame at the URDF base."""
    from g1_barcode_sim.config import load_robot_cfg
    from g1_barcode_sim.t800_robot import _sdk, joint_tables, urdf_fk
    from g1_barcode_sim.camera import T_from_xyz_rpy
    rcfg = load_robot_cfg({"model": "t800"})
    urdf_path = _sdk(rcfg, rcfg["asset"]["urdf"])
    mesh_dir = os.path.join(os.path.dirname(os.path.dirname(urdf_path)), "meshes")
    root = ET.parse(urdf_path).getroot()
    q0 = dict(joint_tables(rcfg)["default_q"])
    for j in root.findall("joint"):
        q0.setdefault(j.get("name"), 0.0)

    def link_world(q):
        pts = {}
        for L in root.findall("link"):
            vis = L.find("visual")
            if vis is None:
                continue
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            if m.get("scale") not in (None, "1 1 1"):
                raise ValueError(f"{L.get('name')}: mesh scale {m.get('scale')}")
            o = vis.find("origin")
            Tv = T_from_xyz_rpy([float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
                                [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
            _, T = urdf_fk(urdf_path, L.get("name"), q)
            V = obj_vertices(os.path.join(mesh_dir, os.path.basename(m.get("filename")).replace(".dae", ".obj")))
            Tw = T @ Tv
            pts[L.get("name")] = V @ Tw[:3, :3].T + Tw[:3, 3]
        return pts

    out = {"urdf": urdf_path}
    P = link_world(q0)
    allp = np.vstack(list(P.values()))
    zmin = allp[:, 2].min()
    out["default_pose"] = {"height_m": float(allp[:, 2].max() - zmin), "root_height_m": float(-zmin),
                           "y_width_m": float(np.ptp(allp[:, 1])), "x_depth_m": float(np.ptp(allp[:, 0]))}
    sh = {s: urdf_fk(urdf_path, f"LINK_SHOULDER_PITCH_{s}", q0)[1][:3, 3] for s in "LR"}
    out["default_pose"]["shoulder_joint_spacing_m"] = float(np.linalg.norm(sh["L"] - sh["R"]))
    out["default_pose"]["shoulder_joint_height_m"] = float(sh["R"][2] - zmin)
    arm_links = [n for n in P if n.endswith("_L") and any(k in n for k in ("SHOULDER", "ELBOW"))]
    shoulder_pts = np.vstack([P[n] for n in arm_links if "SHOULDER" in n] +
                             [P[n.replace("_L", "_R")] for n in arm_links if "SHOULDER" in n])
    out["default_pose"]["shoulder_outer_width_m"] = float(np.ptp(shoulder_pts[:, 1]))
    torso = P["LINK_WAIST_YAW"] if "LINK_WAIST_YAW" in P else P["LINK_TORSO_YAW"]
    out["default_pose"]["torso_link_width_m"] = float(np.ptp(torso[:, 1]))
    # right arm: all 7 arm joints 0 (straight), then the shoulder pitch that points it forward (horizontal)
    arm = [f"J{n}_{k}_R" for n, k in zip(range(27, 34), ("SHOULDER_PITCH", "SHOULDER_ROLL", "SHOULDER_YAW", "ELBOW_PITCH",
                                                          "ELBOW_YAW", "WRIST_PITCH", "WRIST_ROLL"))]
    hand = [n for n in P if n.endswith("_R") and any(k in n for k in ("WRIST_ROLL", "INDEX", "MIDDLE", "THUMB"))]
    res = {}
    for label, pitch in (("straight_down", 0.0), ("forward_neg90", -math.pi / 2), ("forward_pos90", math.pi / 2)):
        q = dict(q0)
        for j in arm:
            q[j] = 0.0
        q[arm[0]] = pitch
        Pq = link_world(q)
        s = urdf_fk(urdf_path, "LINK_SHOULDER_PITCH_R", q)[1][:3, 3]
        H = np.vstack([Pq[n] for n in hand])
        d = np.linalg.norm(H - s, axis=1)
        tip = H[d.argmax()]
        wr = urdf_fk(urdf_path, "LINK_WRIST_ROLL_R", q)[1][:3, 3]
        res[label] = {"shoulder_to_farthest_hand_vertex_m": float(d.max()), "tip_minus_shoulder_xyz": (tip - s).round(4).tolist(),
                      "shoulder_to_wrist_roll_origin_m": float(np.linalg.norm(wr - s)),
                      "wrist_roll_minus_shoulder_xyz": (wr - s).round(4).tolist()}
    out["right_arm"] = res
    out["right_arm"]["note"] = "forward_* = shoulder pitch -/+ 90 deg with every other right-arm joint 0; the one whose " \
                               "tip has +x is the arm stretched forward"
    return out


if args.robot == "t800":
    R["urdf_fk"] = t800_urdf_fk()
    print(json.dumps(R["urdf_fk"], indent=1), flush=True)
if args.cpu_only:
    if args.out:
        json.dump(R, open(args.out, "w"), indent=1)
    sys.exit(0)

SY = os.path.expanduser("~/random/switchyard/clients")
sys.path.insert(0, SY)
from switchyard_progress import Run  # noqa: E402

gpus = [int(g) for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip().isdigit()]
run = Run(f"hbs-measure-sizes-{args.robot}", label=f"HumanoidBarcodeScan · measure sizes {args.scenario} ({args.robot})", gpus=gpus)
run.status("Starting Isaac Sim")
from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

env = G1BarcodeEnv("scenario", headless=True, seed=0, config_overrides={"robot": {"model": args.robot}})


def rng3(r):
    lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
    return {"min": lo.round(4).tolist(), "max": hi.round(4).tolist(), "size": (hi - lo).round(4).tolist()}


def main():
    from pxr import Usd, UsdGeom
    import omni.usd
    from g1_barcode_sim.camera import quat_wxyz_to_R
    from g1_barcode_sim.scenario_builder import SCN

    run.status("Loading scenario")
    env.reset(scenario=args.scenario)
    r = env.stand_still(args.settle_s)
    if r["status"] != "ok":
        raise RuntimeError(f"robot did not stand: {r}")
    stage = omni.usd.get_context().get_stage()
    R["stage_meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lay = env.layout

    # ---- boxes: layout vs stage
    boxes = {}
    for b in lay["boxes"]:
        bp = f"{SCN}/Boxes/{b['id']}"
        e = {"layout_size_xyz": [round(v, 4) for v in b["size"]], "layout_pos": [round(v, 4) for v in b["pos"]]}
        for part in ("visual", "collider"):
            p = stage.GetPrimAtPath(f"{bp}/{part}")
            if not p.IsValid():
                raise RuntimeError(f"{bp}/{part} missing")
            e[part] = rng3(cache.ComputeWorldBound(p).ComputeAlignedRange())
        boxes[b["id"]] = e
    R["boxes"] = boxes
    vis_err = max(abs(a - c) for e in boxes.values() for a, c in zip(e["visual"]["size"], e["layout_size_xyz"]))
    col_err = max(abs(a - c) for e in boxes.values() for a, c in zip(e["collider"]["size"], e["layout_size_xyz"]))
    R["box_size_max_abs_err_m"] = {"visual": vis_err, "collider": col_err}
    whole = cache.ComputeWorldBound(stage.GetPrimAtPath(f"{SCN}/Boxes")).ComputeAlignedRange()
    R["wall_stage_bound"] = rng3(whole)
    lb = lay["barcodes"][0]
    lab_prims = [p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{SCN}/Boxes/{lb['box_id']}")) if p.GetName().startswith("BC_")]
    R["label"] = {"box_id": lb["box_id"], "layout_size_wh": lb["size"], "layout_center": lb["center"],
                  "prims": {str(p.GetPath()): rng3(cache.ComputeWorldBound(p).ComputeAlignedRange()) for p in lab_prims}}

    # ---- robot
    rp = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    scales = {}
    for p in Usd.PrimRange(rp):
        if p.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(p).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    v = np.array(op.Get(), dtype=float)
                    if np.max(np.abs(v - 1)) > 1e-6:
                        scales[str(p.GetPath())] = v.tolist()
    R["robot_non_unit_scale_ops"] = scales
    refs = []
    for p in Usd.PrimRange(rp):
        for spec in p.GetPrimStack():
            if spec.layer and spec.layer.identifier not in refs:
                refs.append(spec.layer.identifier)
    R["robot_layers_meters_per_unit"] = {lid: UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(lid)) for lid in refs
                                         if lid.endswith((".usd", ".usda", ".usdc")) and os.path.exists(lid)}
    bp_, bq_ = env.robot.data.body_pos_w[0].cpu().numpy(), env.robot.data.body_quat_w[0].cpu().numpy()
    body, per = [], {}
    for i, name in enumerate(env.body_names):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"robot link prim {name} not found")
        rg = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if rg.IsEmpty():
            continue
        lo, hi = np.array(rg.GetMin()), np.array(rg.GetMax())
        loc = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = bp_[i] + loc @ quat_wxyz_to_R(bq_[i]).T
        per[name] = w
        body.append(w)
    body = np.vstack(body)
    st = env.frame_state()
    Twr = st["T_world_root"]
    Rwr = Twr[:3, :3]
    rel = (body - Twr[:3, 3]) @ Rwr                         # robot frame (x fwd, y left, z up)
    zfloor = 0.0                                             # ground collider top (scene_builder)
    names = env.body_names
    sh = {s: bp_[[k for k, n in enumerate(names) if ("SHOULDER_PITCH" in n or "shoulder_pitch" in n) and
                  (n.endswith(f"_{s}") or n.startswith("left" if s == "L" else "right"))][0]] for s in "LR"}
    hc = st["T_world_headcam"][:3, 3]
    R["robot_stage"] = {
        "body_aabb_world": {"min": body.min(0).round(4).tolist(), "max": body.max(0).round(4).tolist()},
        "height_top_minus_floor_m": float(body[:, 2].max() - zfloor), "lowest_point_z_m": float(body[:, 2].min()),
        "root_pos_world": Twr[:3, 3].round(4).tolist(), "root_yaw_deg": float(math.degrees(math.atan2(Rwr[1, 0], Rwr[0, 0]))),
        "root_height_m": float(Twr[2, 3] - zfloor), "lateral_width_robot_frame_m": float(np.ptp(rel[:, 1])),
        "depth_robot_frame_m": float(np.ptp(rel[:, 0])),
        "shoulder_pitch_link_origin_spacing_m": float(np.linalg.norm(sh["L"] - sh["R"])),
        "shoulder_pitch_link_origin_height_m": float(sh["R"][2] - zfloor),
        "head_camera_pos_world": hc.round(4).tolist(), "head_camera_height_m": float(hc[2] - zfloor),
        "head_camera_dist_to_wall_face_m": float(lay["boxes"][0]["pos"][0] - lay["boxes"][0]["size"][0] / 2 - hc[0]),
    }
    shoulder_links = [n for n in per if "shoulder" in n.lower()]
    sp = (np.vstack([per[n] for n in shoulder_links]) - Twr[:3, 3]) @ Rwr
    R["robot_stage"]["shoulder_links_outer_width_m"] = float(np.ptp(sp[:, 1]))
    if "T_world_tip" in st:
        R["robot_stage"]["scanner_tip_world"] = st["T_world_tip"][:3, 3].round(4).tolist()
    R["layout_robot_init"] = lay["robot_init"]
    print(json.dumps({k: v for k, v in R.items() if k != "boxes"}, indent=1), flush=True)
    b0 = next(iter(boxes.items()))
    print("box", b0[0], json.dumps(b0[1]), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(R, open(args.out, "w"), indent=1)
        print("->", args.out, flush=True)


try:
    main()
except BaseException as exc:        # report the failure to the dashboard, print it, exit non-zero (Kit hangs on close)
    import traceback
    traceback.print_exc()
    run.failed(repr(exc))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
run.done()
sys.stdout.flush(); os._exit(0)
