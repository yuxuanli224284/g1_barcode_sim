"""Env cfg for the barcode scenes, derived from the upstream Dex1 Wholebody task cfg.

Kept from upstream: robot articulation (G1 29-DoF + Dex1, wholebody USD), contact sensor, sim
settings (dt 0.005, decimation 4, physics material), the head camera on d435_link.
Removed: the cylinder-task room, tables, cylinder, upstream wrist / third-person cameras, the camera-image
observation term (shared-memory writer) and the cylinder reward. Added: the barcode scene spawner,
one aux camera, optional fixed recording cameras (camera_head.yaml recording), the handheld scanner (scanner_mount + gun geometry) and the camera at the scanner head. Upstream files are not modified; we edit a copy of the cfg object.
"""
from __future__ import annotations

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


def make_env_cfg(cfg, layout, scfg, bcfg, rcfg, ccfg, texture_dir, scancfg):
    import math
    from ..camera import R_to_quat_wxyz
    from ..scanner import ScannerSpawnerCfg, transforms
    sc = cfg.scene
    for name in ("room_walls", "packing_table1", "packing_table2", "object", "light", "world_camera",
                 "left_wrist_camera", "right_wrist_camera", "robot_camera"):
        if hasattr(sc, name):
            setattr(sc, name, None)

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
