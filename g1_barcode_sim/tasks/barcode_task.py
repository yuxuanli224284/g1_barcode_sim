"""Env cfg for the barcode scenes, derived from the upstream Dex1 Wholebody task cfg.

Kept from upstream: robot articulation (G1 29-DoF + Dex1, wholebody USD), contact sensor, sim
settings (dt 0.005, decimation 4, physics material), the head camera on d435_link.
Removed: the cylinder-task room, tables, cylinder, upstream wrist / third-person cameras, the camera-image
observation term (shared-memory writer) and the cylinder reward. Added: the barcode scene spawner,
one aux camera, optional fixed recording cameras (camera_head.yaml recording), the handheld scanner (scanner_mount + gun geometry) and the camera at the scanner head. Upstream files are not modified; we edit a copy of the cfg object.
Robot variant g1_v2 (robot.yaml asset.kind urdf): the articulation is imported from the lab's URDF instead of the upstream
USD (same actuator groups, gripper joints renamed), plus the hand cameras (robot.yaml hand_cameras).
"""
from __future__ import annotations

import copy
import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ..scene_builder import ROOT, BarcodeSceneSpawnerCfg


@configclass
class _PolicyObs(ObsGroup):
    joint_pos = ObsTerm(func=base_mdp.joint_pos)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class _Obs:
    policy: _PolicyObs = _PolicyObs()


@configclass
class _NoRewards:
    pass


def init_root_z(rcfg) -> float:
    return rcfg["gear_wbc"]["init_z"] if rcfg["locomotion"] == "gear_wbc" else rcfg["init_pos"][2]


_G15_SHORT = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]


def _apply_gear_wbc_robot(robot_cfg, gcfg):
    """Leg + waist actuators and default stance for the GR00T Decoupled WBC (the arm / gripper actuators stay upstream).
    Replaces the upstream "legs" (hips, knees, waist) and "feet" (ankles) groups by one group over the 15 policy joints."""
    import re
    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
    from ..gear_wbc import G15, load_gear_cfg
    ref = load_gear_cfg(gcfg)
    short = [f"{n}" for n in _G15_SHORT] * 2 + ["waist_yaw", "waist_roll", "waist_pitch"]
    kp = {j: float(gcfg["kp_override"].get(s, k)) for j, s, k in zip(G15, short, ref["kps"])}
    kd = {j: float(k) for j, k in zip(G15, ref["kds"])}
    eff = {j: float(gcfg["effort_limit"][s]) for j, s in zip(G15, short)}
    vel = {j: float(gcfg["velocity_limit"][s]) for j, s in zip(G15, short)}
    acts = dict(robot_cfg.actuators)
    for k in ("legs", "feet"):
        del acts[k]
    common = dict(joint_names_expr=list(G15), stiffness=kp, damping=kd, effort_limit_sim=eff, velocity_limit_sim=vel,
                  armature=float(gcfg["armature"]), friction=float(gcfg["friction"]))
    if gcfg["actuator"] == "ideal_pd":
        acts["gear_wbc"] = IdealPDActuatorCfg(effort_limit=eff, velocity_limit=vel, **common)
    elif gcfg["actuator"] == "implicit":
        acts["gear_wbc"] = ImplicitActuatorCfg(**common)
    else:
        raise ValueError(f"gear_wbc.actuator must be ideal_pd | implicit, got {gcfg['actuator']!r}")
    robot_cfg.actuators = acts
    # default stance: drop upstream init entries that touch the 15 policy joints, then set the GR00T default angles
    jp = {k: v for k, v in robot_cfg.init_state.joint_pos.items() if not any(re.fullmatch(k, j) for j in G15)}
    jp.update({j: float(a) for j, a in zip(G15, ref["default_angles"])})
    robot_cfg.init_state.joint_pos = jp


def robot_urdf_path(rcfg) -> str:
    """URDF of a robot variant imported from a description directory (robot.yaml asset)."""
    from ..config import PKG_ROOT
    a = rcfg["asset"]
    d = a["description_dir"] or os.environ.get("G1_V2_DESCRIPTION") or os.path.join(os.path.dirname(PKG_ROOT), "g1_description_v2")
    p = os.path.join(d, a["urdf"])
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found: prepare the description with scripts/prepare_g1_v2.py (or set $G1_V2_DESCRIPTION)")
    return p


def _apply_urdf_robot(robot_cfg, rcfg):
    """Swap the upstream USD for the variant's URDF (converted once into <description>/usd_cache) and rename the gripper
    joints; actuator groups, gains, limits and the default arm / leg pose stay upstream."""
    import isaaclab.sim as su
    a, up = rcfg["asset"], robot_cfg.spawn
    urdf = robot_urdf_path(rcfg)
    robot_cfg.spawn = su.UrdfFileCfg(
        asset_path=urdf, usd_dir=os.path.join(os.path.dirname(urdf), "usd_cache"),
        usd_file_name=os.path.splitext(os.path.basename(urdf))[0] + ".usd",
        fix_base=False, merge_fixed_joints=bool(a["merge_fixed_joints"]), self_collision=False,
        make_instanceable=False,      # the clearance model walks <link>/collisions gprims (instance proxies are skipped)
        joint_drive=su.UrdfConverterCfg.JointDriveCfg(
            gains=su.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)),   # gains: actuators below
        activate_contact_sensors=up.activate_contact_sensors, rigid_props=up.rigid_props,
        articulation_props=up.articulation_props)
    grip = [j for side in ("left", "right") for j in rcfg["gripper"]["joints"][side]]
    acts = dict(robot_cfg.actuators)
    if grip:
        h = copy.deepcopy(acts["hands"])
        h.joint_names_expr = list(grip)
        acts["hands"] = h
    else:
        del acts["hands"]
    robot_cfg.actuators = acts
    jp = {k: v for k, v in robot_cfg.init_state.joint_pos.items() if "_hand_Joint" not in k}   # upstream Dex1 names
    jp.update({j: float(rcfg["gripper"]["joint_closed"]) for j in grip})                        # upstream starts closed
    robot_cfg.init_state.joint_pos = jp


def check_head_mount(urdf: str, ccfg) -> None:
    """The head camera mount (camera_head.yaml `mount`) must be the URDF d435_joint the D435 visual hangs on, else the
    rendered cameras and the D435 body drift apart (e.g. after changing --d435_pitch_deg in prepare_g1_v2.py only)."""
    import re
    import numpy as np
    j = re.search(r'<joint name="d435_joint" type="fixed">\s*<origin xyz="([^"]*)" rpy="([^"]*)"', open(urdf).read())
    if j is None:
        raise ValueError(f"{urdf}: no fixed d435_joint")
    m = ccfg["mount"]
    if not (np.allclose([float(v) for v in j.group(1).split()], m["pos"], atol=1e-6)
            and np.allclose([float(v) for v in j.group(2).split()], m["rpy"], atol=1e-6)):
        raise ValueError(f"camera mount {m} != URDF d435_joint xyz {j.group(1)} rpy {j.group(2)} ({urdf})")


def make_env_cfg(cfg, layout, scfg, bcfg, rcfg, ccfg, texture_dir, scancfg):
    import math
    from ..camera import R_to_quat_wxyz
    from ..scanner import ScannerSpawnerCfg, transforms
    sc = cfg.scene
    for name in ("room_walls", "packing_table1", "packing_table2", "object", "light", "world_camera",
                 "left_wrist_camera", "right_wrist_camera", "robot_camera"):
        if hasattr(sc, name):
            setattr(sc, name, None)

    asset = rcfg.get("asset") or {"kind": "upstream_usd"}
    if asset["kind"] == "urdf":
        _apply_urdf_robot(sc.robot, rcfg)
        if ccfg.get("mount"):
            check_head_mount(robot_urdf_path(rcfg), ccfg)
    elif asset["kind"] != "upstream_usd":
        raise ValueError(f"robot asset kind must be upstream_usd | urdf, got {asset['kind']!r}")
    yaw = rcfg["init_yaw"]
    sc.robot.init_state.pos = tuple(rcfg["init_pos"][:2]) + (init_root_z(rcfg),)
    if rcfg["locomotion"] == "gear_wbc":
        _apply_gear_wbc_robot(sc.robot, rcfg["gear_wbc"])
    elif rcfg["locomotion"] != "unitree":
        raise ValueError(f"robot.yaml locomotion must be unitree | gear_wbc, got {rcfg['locomotion']!r}")
    sc.robot.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    # head camera from configs/camera_head.yaml
    hc = sc.front_camera
    hc.prim_path = f"{{ENV_REGEX_NS}}/Robot/{ccfg['prim_parent_link']}/{ccfg['prim_name']}"
    hc.width, hc.height = ccfg["width"], ccfg["height"]
    hc.update_period = 0.0 if ccfg["update_hz"] is None else 1.0 / ccfg["update_hz"]
    hc.data_types = ["rgb", "distance_to_image_plane"]
    hc.spawn = sim_utils.PinholeCameraCfg(focal_length=ccfg["focal_length"], focus_distance=ccfg["focus_distance"],
                                          horizontal_aperture=ccfg["horizontal_aperture"],
                                          clipping_range=tuple(ccfg["clipping_range"]))
    hc.offset = CameraCfg.OffsetCfg(pos=tuple(ccfg["offset_pos"]), rot=tuple(ccfg["offset_rot_wxyz"]),
                                    convention=ccfg["offset_convention"])
    # D435i infrared pair (camera_head.yaml model d435i): rgb + render depth of each imager feed the infrared image
    # synthesis (stereo_ir.py); same mount rotation as the colour camera, positions on d435_link
    st = ccfg.get("stereo")
    if st is not None:
        for side in ("left", "right"):
            setattr(sc, f"head_ir_{side}_camera", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{ccfg['prim_parent_link']}/ir_{side}_cam",
                width=st["width"], height=st["height"], update_period=hc.update_period,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=st["focal_length"], focus_distance=ccfg["focus_distance"],
                                                 horizontal_aperture=st["horizontal_aperture"],
                                                 clipping_range=tuple(ccfg["clipping_range"])),
                offset=CameraCfg.OffsetCfg(pos=tuple(st[f"{side}_pos"]), rot=tuple(st["rot_wxyz"]),
                                           convention=ccfg["offset_convention"])))

    # cameras on the hands (robot variant g1_v2: the D405 on the left wrist bracket)
    from ..camera import focal_for_hfov
    for name, h in (rcfg.get("hand_cameras") or {}).items():
        setattr(sc, f"hand_{name}_camera", CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{h['parent_link']}/{name}_cam", width=int(h["width"]), height=int(h["height"]),
            update_period=0.0, data_types=list(h["data_types"]),
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal_for_hfov(h["hfov_deg"], 20.0), horizontal_aperture=20.0,
                                             focus_distance=ccfg["focus_distance"], clipping_range=tuple(h["clipping_range"])),
            offset=CameraCfg.OffsetCfg(pos=tuple(h["pos"]), rot=tuple(h["rot_wxyz"]), convention="ros")))

    ac = ccfg["aux"]
    sc.aux_camera = CameraCfg(prim_path="/World/AuxCamera", width=ac["width"], height=ac["height"], update_period=0.0,
                              data_types=["rgb", "distance_to_image_plane"],
                              spawn=sim_utils.PinholeCameraCfg(focal_length=ac["focal_length"],
                                                               horizontal_aperture=ac["horizontal_aperture"],
                                                               clipping_range=tuple(ac["clipping_range"])),
                              offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"))

    # fixed recording cameras for the composite episode video (camera_head.yaml recording, off by default: like every
    # camera they render on every policy step). Same lens as the aux camera; pose and FOV are set per episode
    # (G1BarcodeEnv.pose_rec_camera) from the scenario layout by the video user.
    rc = ccfg["recording"]
    if rc["enabled"]:
        for name in rc["cameras"]:
            c = rc["cameras"][name]
            setattr(sc, f"rec_{name}_camera", CameraCfg(
                prim_path=f"/World/Rec_{name}_Camera", width=c["width"], height=c["height"], update_period=0.0,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=ac["focal_length"], horizontal_aperture=ac["horizontal_aperture"],
                                                 clipping_range=tuple(ac["clipping_range"])),
                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros")))

    # handheld scanner: scanner_mount Xform (+ gun parts) under the gripper link, camera at the scanner head
    mc = rcfg["scanner_mount"]
    sc.scanner = AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/{mc['parent_link']}/scanner_mount",
                              spawn=ScannerSpawnerCfg(parts=scancfg["parts"] if scancfg["enabled"] else []),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(mc["pos"]), rot=tuple(mc["quat_wxyz"])))
    if scancfg["enabled"]:
        wc = scancfg["camera"]
        T = transforms(mc, scancfg)["T_hand_cam"]
        sc.wrist_camera = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{mc['parent_link']}/{wc['prim_name']}", width=wc["width"], height=wc["height"],
            update_period=0.0, data_types=list(wc["data_types"]),
            spawn=sim_utils.PinholeCameraCfg(focal_length=wc["focal_length"], focus_distance=wc["focus_distance"],
                                             horizontal_aperture=wc["horizontal_aperture"],
                                             clipping_range=tuple(wc["clipping_range"])),
            offset=CameraCfg.OffsetCfg(pos=tuple(T[:3, 3].tolist()), rot=tuple(R_to_quat_wxyz(T[:3, :3])), convention="ros"))

    spawner = BarcodeSceneSpawnerCfg(layout=layout, scene_cfg=scfg, barcode_cfg=bcfg, texture_dir=texture_dir)
    sc.barcode_scene = AssetBaseCfg(prim_path=ROOT, spawn=spawner)

    cfg.sim.render.antialiasing_mode = ccfg["render"]["antialiasing_mode"]
    cfg.observations = _Obs()
    cfg.rewards = _NoRewards()
    cfg.episode_length_s = 1.0e6     # we never call env.step(); avoid any time-out bookkeeping
    return cfg, spawner
