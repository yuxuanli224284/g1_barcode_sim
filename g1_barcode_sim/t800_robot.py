"""EngineAI T800 Pro articulation for the barcode scenes (robot_t800.yaml).

Every number comes from a vendor file (paths in robot_t800.yaml `asset` / `t800_walk`):
  joints / order        model.yaml limbs (43 revolute joints, serial ankles / wrists as in the vendor MJCF)
  default pose          walk.yaml default_joint_q, head joints replaced by t800_walk.head_hold
  kp / kd               walk.yaml joint_kp / joint_kd (what the walk runner sends to every joint)
  effort / vel. limits  URDF <limit effort velocity> (= MJCF actuator ctrlrange)
  armature              MJCF serial_links.xml joint armature (rotor inertia; the URDF has none)
  joint friction        0 (the MJCF `joint` default class with damping / frictionloss is never applied: no body uses
                        childclass="joint")
Physics: dt = MJCF timestep, policy every control_dt (decimation = control_dt / dt), CPU PhysX like the G1 config.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _sdk(rcfg, rel):
    return rel if os.path.isabs(rel) else os.path.join(rcfg["asset"]["sdk_root"], rel)


def joint_tables(rcfg) -> dict:
    """Per-joint values (dicts keyed by joint name) for the 43 model.yaml joints."""
    w = rcfg["t800_walk"]
    walk = yaml.safe_load(open(_sdk(rcfg, w["walk_yaml"])))
    model = yaml.safe_load(open(_sdk(rcfg, w["model_yaml"])))
    names = [j for limb in model["limbs"] for j in limb["joints"]]

    def flat(key):
        v = np.concatenate([np.asarray(g, dtype=float) for g in walk[key]])
        if v.shape != (len(names),):
            raise ValueError(f"walk.yaml {key} has {v.shape}, expected ({len(names)},)")
        return dict(zip(names, v.tolist()))

    default_q, kp, kd = flat("default_joint_q"), flat("joint_kp"), flat("joint_kd")
    for j, v in w["head_hold"].items():
        if j not in default_q:
            raise ValueError(f"head_hold joint {j} not in model.yaml")
        default_q[j] = float(v)
    urdf = ET.parse(_sdk(rcfg, rcfg["asset"]["urdf"])).getroot()
    effort, vel, lim = {}, {}, {}
    for j in urdf.findall("joint"):
        if j.get("type") == "fixed":
            continue
        L = j.find("limit")
        effort[j.get("name")], vel[j.get("name")] = float(L.get("effort")), float(L.get("velocity"))
        lim[j.get("name")] = (float(L.get("lower")), float(L.get("upper")))
    arm = {}
    for jn in ET.parse(_sdk(rcfg, rcfg["asset"]["mjcf_links"])).getroot().iter("joint"):
        arm[jn.get("name")] = float(jn.get("armature", 0.0))
    for tbl, src in ((effort, "URDF"), (arm, "MJCF")):
        missing = [n for n in names if n not in tbl]
        if missing:
            raise ValueError(f"{src} lacks joints {missing}")
    for n, v in default_q.items():
        lo, hi = lim[n]
        if not lo - 1e-9 <= v <= hi + 1e-9:
            raise ValueError(f"default / hold angle {n} = {v} outside the URDF limits [{lo}, {hi}]")
    return {"names": names, "default_q": default_q, "kp": kp, "kd": kd, "effort": {n: effort[n] for n in names},
            "velocity": {n: vel[n] for n in names}, "armature": {n: arm[n] for n in names},
            "control_dt": float(walk["control_dt"])}


def t800_articulation_cfg(rcfg):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    w = rcfg["t800_walk"]
    T = joint_tables(rcfg)
    names = T["names"]
    # exact names as regexes (re.fullmatch); none contains a regex metacharacter
    common = dict(joint_names_expr=list(names), stiffness=T["kp"], damping=T["kd"], effort_limit_sim=T["effort"],
                  velocity_limit_sim=T["velocity"], armature=T["armature"], friction=0.0)
    if w["actuator"] == "ideal_pd":
        act = IdealPDActuatorCfg(effort_limit=T["effort"], velocity_limit=T["velocity"], **common)
    elif w["actuator"] == "implicit":
        act = ImplicitActuatorCfg(**common)
    else:
        raise ValueError(f"t800_walk.actuator must be ideal_pd | implicit, got {w['actuator']!r}")
    usd = rcfg["asset"]["usd"]
    if not os.path.exists(usd):
        raise FileNotFoundError(f"{usd} missing: run t800/convert_urdf.py")
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=bool(w["self_collision"]), solver_position_iteration_count=int(w["solver_position_iterations"]),
                solver_velocity_iteration_count=int(w["solver_velocity_iterations"]))),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(rcfg["init_pos"]), joint_pos=dict(T["default_q"]),
                                                   joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=float(w["soft_joint_pos_limit_factor"]),
        actuators={"t800": act},
    )


def apply_t800_scene(cfg, rcfg):
    """Turn the upstream template cfg into a T800 scene skeleton: T800 articulation, physics dt / decimation of the
    walk policy, no contact sensor (nothing reads it; the template's is set up for the G1 bodies)."""
    T = joint_tables(rcfg)
    dt = float(rcfg["t800_walk"]["physics_dt"])
    dec = int(round(T["control_dt"] / dt))
    if abs(dec * dt - T["control_dt"]) > 1e-12:
        raise ValueError(f"control_dt {T['control_dt']} is not a multiple of physics_dt {dt}")
    cfg.scene.robot = t800_articulation_cfg(rcfg)
    if hasattr(cfg.scene, "contact_forces"):
        cfg.scene.contact_forces = None
    cfg.sim.dt = dt
    cfg.decimation = dec
    cfg.sim.render_interval = dec
    return cfg


def urdf_fk(urdf_path, link, q) -> tuple[str, np.ndarray]:
    """(root link, T_root_link) of `link` by the URDF joint chain at joint positions q (dict name -> rad / m; every
    movable joint of the chain must be in q)."""
    from .camera import T_from_xyz_rpy
    by_child = {j.find("child").get("link"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    T, n = np.eye(4), link
    while n in by_child:
        j = by_child[n]
        o = j.find("origin")
        Tj = T_from_xyz_rpy([float(v) for v in o.get("xyz", "0 0 0").split()], [float(v) for v in o.get("rpy", "0 0 0").split()])
        kind = j.get("type")
        if kind in ("revolute", "continuous", "prismatic"):
            a = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            a = a / np.linalg.norm(a)
            v = float(q[j.get("name")])
            M = np.eye(4)
            if kind == "prismatic":
                M[:3, 3] = a * v
            else:                                           # Rodrigues
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                M[:3, :3] = np.eye(3) + np.sin(v) * K + (1 - np.cos(v)) * K @ K
            Tj = Tj @ M
        elif kind != "fixed":
            raise ValueError(f"URDF joint {j.get('name')} type {kind} not handled")
        T, n = Tj @ T, j.find("parent").get("link")
    return n, T


# Pass criterion of the head-camera mount check: user specification 2026-11-10 (translation < 1 mm, rotation < 0.1 deg).
MOUNT_TOL_M, MOUNT_TOL_DEG = 1e-3, 0.1


def check_head_camera_mount(env) -> dict:
    """After spawn: the head camera must sit at URDF FK(root -> head_camera.parent_link, current joint angles) x the
    head_camera.chain offsets (robot_t800.yaml). Two independent paths are compared with that reference:
      obs     env.get_obs()["T_root_cam"] (physics link pose x configured offset; what the agent / tests read)
      render  physics pose of the camera prim's USD parent x the camera prim's USD local transform (OpenGL -> ROS);
              this is what the renderer draws from
    Raises RuntimeError above MOUNT_TOL_M / MOUNT_TOL_DEG; returns the errors."""
    import omni.usd
    from pxr import UsdGeom
    from .camera import T_from_xyz_rpy, inv_T

    rcfg, ccfg = env.cfg["robot"], env.cfg["camera"]
    hc = rcfg["head_camera"]
    names = list(env.robot.data.joint_names)
    qv = env.robot.data.joint_pos[0].cpu().numpy()
    root_link, T_root_link = urdf_fk(_sdk(rcfg, rcfg["asset"]["urdf"]), hc["parent_link"], dict(zip(names, qv.tolist())))
    root_body = env.body_names[0]
    if root_body != root_link:
        raise RuntimeError(f"articulation root body {root_body} != URDF root link {root_link}")
    T_off = np.eye(4)
    for j in hc["chain"]:
        T_off = T_off @ T_from_xyz_rpy(j["xyz"], j["rpy"])
    T_ref = T_root_link @ T_off

    T_rw = inv_T(env._T_world_root())
    T_obs = env.get_obs()["T_root_cam"]
    cam_path = f"/World/envs/env_0/Robot/{ccfg['prim_parent_link']}/{ccfg['prim_name']}"   # as barcode_task (env 0)
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(cam_path)
    if not prim.IsValid():
        raise RuntimeError(f"head camera prim {cam_path} not found")
    parent = prim.GetParent().GetName()
    if parent != hc["parent_link"]:
        raise RuntimeError(f"head camera prim parent {parent} != head_camera.parent_link {hc['parent_link']}")
    T_gl = np.array(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float).T    # Gf: row vectors
    T_ros = T_gl @ np.diag([1.0, -1.0, -1.0, 1.0])                                      # OpenGL (-z fwd, y up) -> ROS
    T_render = T_rw @ env._link_T(env.body_names.index(parent)) @ T_ros

    def err(T):
        dR = T_ref[:3, :3].T @ T[:3, :3]
        return {"trans_mm": float(np.linalg.norm(T[:3, 3] - T_ref[:3, 3]) * 1e3),
                "rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))))}
    res = {"obs": err(T_obs), "render": err(T_render), "T_root_cam_fk": np.round(T_ref, 6).tolist(),
           "T_link_cam_usd_ros": np.round(T_ros, 6).tolist(), "tol_mm": MOUNT_TOL_M * 1e3, "tol_deg": MOUNT_TOL_DEG,
           "joint_q_deg": {n: round(float(np.degrees(v)), 4) for n, v in zip(names, qv) if "HEAD" in n or "TORSO" in n}}
    bad = {k: v for k, v in res.items() if k in ("obs", "render")
           and (v["trans_mm"] > MOUNT_TOL_M * 1e3 or v["rot_deg"] > MOUNT_TOL_DEG)}
    if bad:
        raise RuntimeError(f"head camera mount does not match URDF FK + head_camera.chain: {bad}")
    return res
