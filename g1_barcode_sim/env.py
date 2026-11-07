"""G1BarcodeEnv: the single entry point algorithms use to talk to the simulation.

Data flow: velocity command -> InProcessWholebodyWalker (upstream unitree Wholebody ONNX policy,
DDS replaced by in-process stand-ins, see locomotion.py) -> joint targets -> PhysX (CPU) ->
Isaac Lab sensors (head camera on d435_link, camera at the head of the handheld scanner, aux camera for
top view / debug renders).

Every public method returns plain Python / numpy structures.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_APP = None  # one Isaac Sim app per process


def _launch_app(headless: bool, device: str):
    global _APP
    if _APP is not None:
        return _APP
    import argparse
    from isaaclab.app import AppLauncher
    from .proc import die_with_parent_on_fork
    die_with_parent_on_fork()   # forked children must not outlive us holding GPU memory (proc.py)
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args([])
    args.headless = headless
    args.enable_cameras = True
    args.device = device
    import pinocchio  # noqa: F401  (upstream imports it before the app; pink tasks need it)
    _APP = AppLauncher(args).app
    return _APP


class G1BarcodeEnv:
    def __init__(self, scene: str = "scenario", headless: bool = False, seed: int = 0, config_overrides: dict | None = None):
        """scene: s1_table | s2_shelf | s3_stack (fixed scenes) or "scenario" (default: test-suite mode,
        load scenes with reset(scenario=...))."""
        from .config import PKG_ROOT, load_all
        self.cfg = load_all(scene, config_overrides)
        self.scene_name = scene
        self.seed = seed
        self.scenario_mode = scene == "scenario"
        self.step_hooks = []                     # callables run after every policy step (e.g. a video recorder)
        self.scenario = None                     # current test-suite scenario dict (scenario mode)
        rcfg, ccfg, scfg, bcfg = self.cfg["robot"], self.cfg["camera"], self.cfg["scene"], self.cfg["barcode"]
        if rcfg["end_effector"] != "dex1":
            raise NotImplementedError("only end_effector=dex1 is wired up in this release (see README: 修改与扩展)")
        self.app = _launch_app(headless, rcfg["physics_device"])

        import gymnasium as gym
        import torch
        from .locomotion import UNITREE_ROOT, InProcessWholebodyWalker
        os.environ["PROJECT_ROOT"] = UNITREE_ROOT
        import tasks  # noqa: F401  registers unitree tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from .layout import expand_scene
        from .tasks.barcode_task import make_env_cfg
        from .textures import ensure_textures

        self.torch = torch
        self.layout = expand_scene(scfg, bcfg, seed=seed)
        self.texture_dir = ensure_textures(self.layout, bcfg, os.path.join(PKG_ROOT, "assets", "barcodes"))
        base_cfg = parse_env_cfg(rcfg["templates"][rcfg["end_effector"]], device=rcfg["physics_device"], num_envs=1)
        env_cfg, self._spawner = make_env_cfg(base_cfg, self.layout, scfg, bcfg, rcfg, ccfg, self.texture_dir, self.cfg["scanner"])
        self.env = gym.make(rcfg["templates"][rcfg["end_effector"]], cfg=env_cfg).unwrapped
        self.env.sim.reset()
        self.env.reset()
        self.robot = self.env.scene["robot"]
        self.head_cam = self.env.scene["front_camera"]
        self.aux_cam = self.env.scene["aux_camera"]
        self.scanner_on = bool(self.cfg["scanner"]["enabled"])
        self.wrist_cam = self.env.scene["wrist_camera"] if self.scanner_on else None
        # fixed recording cameras (camera_head.yaml recording; off unless a video user enables them): name -> Camera
        self.rec_cams = ({n: self.env.scene[f"rec_{n}_camera"] for n in ccfg["recording"]["cameras"]}
                         if ccfg["recording"]["enabled"] else {})
        self.locomotion = rcfg["locomotion"]
        if self.locomotion == "gear_wbc":
            from .config import deep_merge
            rcfg["move_relative"] = deep_merge(rcfg["move_relative"], rcfg["gear_wbc"]["move_relative"])
            from .gear_wbc import GearWbcWalker
            if rcfg.get("mode_policies"):
                raise ValueError("mode_policies applies to the unitree walker only; GR00T switches Balance/Walk itself")
            self.walker = GearWbcWalker(self.env, rcfg["gear_wbc"])
            self.policy_dt = self.env.physics_dt * self.walker.decimation
        else:
            self.walker = InProcessWholebodyWalker(self.env, model_path=rcfg["policy_model"], height_cmd=rcfg["height_cmd"])
            self.policy_dt = self.env.physics_dt * 4   # upstream get_action() sub-steps physics 4x
        self._mode_policies, self._mode = None, None
        if rcfg.get("mode_policies"):
            self._mode_policies = {m: self.walker.load_extra_policy(p) for m, p in rcfg["mode_policies"].items()}
            self._set_mode("stand")
        self.body_names = list(self.robot.data.body_names)
        mc = rcfg["scanner_mount"]
        self._mount_body = self.body_names.index(mc["parent_link"])
        self.scanner_side = mc["parent_link"].split("_")[0]
        if self.scanner_side not in ("left", "right"):
            raise ValueError(f"scanner_mount.parent_link must be a left_/right_ link, got {mc['parent_link']}")
        self.scanner_mount_prim = f"/World/envs/env_0/Robot/{mc['parent_link']}/scanner_mount"
        from .camera import make_T
        from .scanner import transforms
        st = transforms(mc, self.cfg["scanner"])
        self._T_body_mount, self._T_body_tip, self._T_body_wcam = st["T_hand_G"], st["T_hand_tip"], st["T_hand_cam"]
        # Head camera pose = physics pose of its parent link x fixed mount offset. The Camera sensor's own
        # pos_w/quat_w are read from the USD prim, which is not updated while physics runs on CPU (it stays
        # at the spawn pose), whereas the renderer follows the robot -> never use head_cam.data.pos_w.
        if ccfg["offset_convention"] != "ros":
            raise ValueError("camera_head.yaml offset_convention must be 'ros' (T_link_cam is built from it)")
        self._cam_body = self.body_names.index(ccfg["prim_parent_link"])
        self._T_link_cam = make_T(ccfg["offset_pos"], ccfg["offset_rot_wxyz"])
        from .collision import RobotCollisionModel
        self.collision_model = RobotCollisionModel(self.robot, self.layout)
        self._scanner_grip_q = None
        if self.scanner_on:
            from .scanner import part_corners_G
            P = part_corners_G(self.cfg["scanner"])
            self.collision_model.add_body_points(mc["parent_link"], (self._T_body_mount[:3, :3] @ P.T).T + self._T_body_mount[:3, 3])
        from .arm import WristIK
        self.arm = WristIK(self, rcfg["arm_ik"])
        self._post_reset(seed)
        self.arm.measure_home(self._step_once, rcfg["arm_ik"]["settle_max_s"])
        # start the episode from a clean state: robot back at its init pose with default joints, odom / sim time reset
        self.teleport_root(rcfg["init_pos"][0], rcfg["init_pos"][1], rcfg["init_yaw"])
        self._post_reset(seed)

    # ------------------------------------------------------------------ internals
    def _post_reset(self, seed):
        rng = np.random.default_rng(seed)
        o = self.cfg["robot"]["odom"]
        self._odom_scale = 1.0 + o["pos_err_per_m"] * rng.choice([-1.0, 1.0])
        self._odom_yaw_rate = math.radians(o["yaw_err_deg_per_m"]) * rng.choice([-1.0, 1.0])
        self._depth_rng = np.random.default_rng(seed + 1)
        self._reset_walker_buffers()
        self.arm.reset()
        self.walker.set_velocity(0.0, 0.0, 0.0)
        self.walker.set_gripper_joint_targets(*self._default_gripper_targets())
        self._sim_time = 0.0
        self._last_min_gap = None   # clearance of the latest status check (frame_state); none yet in this episode
        self._step_once()  # one policy step so the cameras hold a rendered frame
        T = self._T_world_root()
        self._odom_T = T.copy()
        self._last_gt_T = T.copy()

    def _default_gripper_targets(self):
        """(left, right) joint targets after a reset: open, except the scanner hand, which closes on the handle."""
        t = {"left": self._gripper_joint(1.0), "right": self._gripper_joint(1.0)}
        if self.scanner_on:
            if self._scanner_grip_q is None:
                from .scanner import grip_joint_target
                g = self.cfg["robot"]["gripper"]
                self._scanner_grip_q = grip_joint_target(self.robot, self.collision_model, self.scanner_side, g["pad_links"],
                                                         g["joints"][self.scanner_side], self.cfg["scanner"]["handle_width"])
            t[self.scanner_side] = self._scanner_grip_q
        return t["left"], t["right"]

    def _link_T(self, body_idx):
        from .camera import make_T
        p = self.robot.data.body_link_pose_w[0, body_idx].cpu().numpy()
        return make_T(p[:3], p[3:])

    def _set_mode(self, mode: str):
        """walk / stand policy switch (robot.yaml mode_policies); no-op when a single policy is configured."""
        if self._mode_policies is None or mode == self._mode:
            return
        self.walker.policy = self._mode_policies[mode]
        self._mode = mode

    def _reset_walker_buffers(self):
        w = self.walker
        if self.locomotion == "gear_wbc":
            w.reset()
            return
        w.actor_obs_buffer.reset()
        w.action_buffer.reset()
        w.action_buffer.compute(self.torch.zeros(1, w.num_actions_all, device=self.env.device))

    def _gripper_joint(self, opening):
        g = self.cfg["robot"]["gripper"]
        opening = float(np.clip(opening, 0.0, 1.0))
        return g["joint_closed"] + (g["joint_open"] - g["joint_closed"]) * opening

    def _step_once(self):
        ts = self.robot.data._sim_timestamp  # upstream get_action() swallows exceptions; detect a stalled step
        if hasattr(self, "arm"):
            self.arm.update()
        self.walker.step_policy()
        if self.robot.data._sim_timestamp <= ts:
            raise RuntimeError("walker policy step did not advance physics (see '[DDSActionProvider] Get DDS action failed' above)")
        self._sim_time += self.policy_dt
        self._update_odom()
        for h in self.step_hooks:
            h()

    def _T_world_root(self):
        from .camera import make_T
        p = self.robot.data.root_pos_w[0].cpu().numpy()
        q = self.robot.data.root_quat_w[0].cpu().numpy()
        return make_T(p, q)

    def _update_odom(self):
        from .camera import T_from_xy_yaw, inv_T, yaw_of
        if not hasattr(self, "_last_gt_T"):
            return
        T = self._T_world_root()
        d = inv_T(T_from_xy_yaw(*self._last_gt_T[:2, 3], yaw_of(self._last_gt_T))) @ T_from_xy_yaw(*T[:2, 3], yaw_of(T))
        dx, dy, dyaw = d[0, 3], d[1, 3], yaw_of(d)
        dist = math.hypot(dx, dy)
        step = T_from_xy_yaw(self._odom_scale * dx, self._odom_scale * dy, dyaw + self._odom_yaw_rate * dist)
        O = T_from_xy_yaw(*self._odom_T[:2, 3], yaw_of(self._odom_T)) @ step
        O[2, 3] = T[2, 3]
        # roll / pitch are observable (IMU gravity direction), so the odometry pose carries the true tilt: with the yaw
        # removed, R = Rz(yaw) Ry(pitch) Rx(roll) leaves Ry Rx. Only x, y, yaw drift. Without it a pelvis pitched by a
        # squat or a torso command would put every odometry-anchored target off by the tilt.
        O[:3, :3] = O[:3, :3] @ (T_from_xy_yaw(0.0, 0.0, -yaw_of(T))[:3, :3] @ T[:3, :3])
        self._odom_T = O
        self._last_gt_T = T

    def _tilt_deg(self):
        g = self.robot.data.projected_gravity_b[0].cpu().numpy()
        return float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))

    def _check_status(self):
        f = self.cfg["robot"]["fall"]
        z = float(self.robot.data.root_pos_w[0, 2])
        tilt = self._tilt_deg()
        if tilt > f["tilt_deg"]:
            return "fell", f"tilt {tilt:.1f} deg > {f['tilt_deg']}"
        if f["pelvis_min_z"] is not None and z < f["pelvis_min_z"]:
            return "fell", f"pelvis z {z:.3f} < {f['pelvis_min_z']}"
        gap = self.collision_model.min_gap()
        self._last_min_gap = gap            # kept for frame_state() (recorders must not recompute it)
        if gap["gap"] < self.cfg["robot"]["collision_gap"]:
            return "collision", f"{gap['body']} within {gap['gap']:.3f} m of {gap['obstacle']}"
        return "ok", ""

    def _clip_vel(self, vx, vy, wz):
        L = self.cfg["robot"]["velocity_limits"]
        return float(np.clip(vx, -L["vx"], L["vx"])), float(np.clip(vy, -L["vy"], L["vy"])), float(np.clip(wz, -L["wz"], L["wz"]))

    # ------------------------------------------------------------------ public API
    def reset(self, seed: int | None = None, scenario: dict | str | None = None) -> dict:
        """Fixed scenes: robot back to its start, labels re-placed for `seed`. Scenario mode: `scenario` = scenario
        dict, scenario JSON path or scenario id; the table, boxes, labels and background are
        rebuilt and the robot starts at the scenario's start pose. reset() without a scenario in scenario mode
        restarts the current scenario."""
        if self.scenario_mode:
            if scenario is None and self.scenario is None:
                raise ValueError("scenario mode: pass reset(scenario=...) (dict, JSON path or scenario id)")
            return self._reset_scenario(self.scenario if scenario is None else scenario)
        if scenario is not None:
            raise ValueError(f"reset(scenario=...) needs G1BarcodeEnv(scene='scenario'), this env is {self.scene_name!r}")
        from .layout import expand_scene
        from .scene_builder import AUTHORED, update_label_poses
        import omni.usd
        if seed is not None:
            self.seed = seed
        self.env.reset()
        # the upstream task has no reset events, so env.reset() leaves the robot where it is
        ri = self.cfg["robot"]
        self.teleport_root(ri["init_pos"][0], ri["init_pos"][1], ri["init_yaw"])
        self.layout = expand_scene(self.cfg["scene"], self.cfg["barcode"], seed=self.seed)
        update_label_poses(omni.usd.get_context().get_stage(), self.layout, AUTHORED)
        self.collision_model.set_layout(self.layout)
        for attr in ("_last_gt_T",):
            if hasattr(self, attr):
                delattr(self, attr)
        self._post_reset(self.seed)
        return self.get_obs()

    def _reset_scenario(self, sc) -> dict:
        import json
        import omni.usd
        from .scenario_builder import apply_background, build_scenario
        from .testsuite import load_factors, load_scenario
        from .testsuite.scenario import to_layout
        if isinstance(sc, str):
            sc = json.load(open(sc)) if os.path.exists(sc) else load_scenario(sc)
        if not hasattr(self, "_ts_factors"):
            self._ts_factors = load_factors()
        self.env.reset()
        lay = to_layout(sc, self._ts_factors)
        stage = omni.usd.get_context().get_stage()
        self._scenario_paths = build_scenario(stage, lay, self._ts_factors, self.cfg["barcode"])
        apply_background(stage, sc["background"], self.cfg["scene"]["scenario_lighting"])
        self.layout, self.scenario, self.seed = lay, sc, int(sc["seed"])
        self.collision_model.set_layout(lay)
        ri = lay["robot_init"]
        self.teleport_root(ri["pos"][0], ri["pos"][1], ri["yaw"])
        if hasattr(self, "_last_gt_T"):
            delattr(self, "_last_gt_T")
        self._post_reset(self.seed)
        return self.get_obs()

    def get_obs(self) -> dict:
        from .camera import inv_T, make_T, postprocess_depth
        cc = self.cfg["camera"]
        rgb = self.head_cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        raw = self.head_cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze(-1)
        depth = postprocess_depth(raw, cc["depth"]["valid_range"], cc["depth"]["noise"], self._depth_rng)
        T_wr = self._T_world_root()
        T_wc = self._link_T(self._cam_body) @ self._T_link_cam
        T_wb = self._link_T(self._mount_body)
        T_rw = inv_T(T_wr)
        jn = self.robot.data.joint_names
        jp = self.robot.data.joint_pos[0].cpu().numpy()
        g = self.cfg["robot"]["gripper"]
        def opening(side):
            q = np.mean([jp[jn.index(j)] for j in g["joints"][side]])
            return float((q - g["joint_closed"]) / (g["joint_open"] - g["joint_closed"]))
        obs = {
            "rgb": rgb, "depth": depth.astype(np.float32),
            "K": self.head_cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64),
            "T_root_cam": T_rw @ T_wc,
            "T_world_root_gt": T_wr,            # evaluation only
            "T_world_root_odom": self._odom_T.copy(),
            "T_root_scanner_mount": T_rw @ T_wb @ self._T_body_mount,
            "gripper_state": {"left": opening("left"), "right": opening("right")},
            "wrist_pos_root": {sd: self.arm.wrist_pos_root(sd) for sd in ("left", "right")},
            "wrist_target_root": {sd: (None if self.arm.target[sd] is None else self.arm.target[sd].copy()) for sd in ("left", "right")},
            "timestamp": self._sim_time,
        }
        if self.scanner_on:
            # camera at the scanner head; pose from the link pose (Camera.data.pos_w is stale with CPU physics)
            wc = self.cfg["scanner"]["camera"]
            out = self.wrist_cam.data.output
            obs["wrist_rgb"] = out["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            if "distance_to_image_plane" in out:
                wd = out["distance_to_image_plane"][0].cpu().numpy().squeeze(-1)
                obs["wrist_depth"] = (wd if wc["depth_valid_range"] is None else
                                      postprocess_depth(wd, wc["depth_valid_range"])).astype(np.float32)
            obs["wrist_K"] = self.wrist_cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)
            obs["T_root_wrist_cam"] = T_rw @ T_wb @ self._T_body_wcam
            obs["T_root_scanner_tip"] = T_rw @ T_wb @ self._T_body_tip
            obs["scanner_aim_target"] = self.arm.aim_target_root(self.scanner_side)   # current root frame
            obs["scanner_aim_error"] = self.arm.aim_error(self.scanner_side)
        return obs

    def frame_state(self) -> dict:
        """Cheap per-frame state for recorders / evaluation (evaluation only: ground-truth poses). Poses and intrinsics
        only: no image copies, no depth post-processing, so it never draws from _depth_rng (get_obs does when depth
        noise is on) and calling it does not change what the harness sees. Same pose formulas as get_obs."""
        T_wr = self._T_world_root()
        cc = self.cfg["camera"]
        out = {"t_sim": self._sim_time, "T_world_root": T_wr, "T_world_root_odom": self._odom_T.copy(),
               "T_world_headcam": self._link_T(self._cam_body) @ self._T_link_cam,
               "head_K": self.head_cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64),
               "head_size": [cc["height"], cc["width"]],
               "min_gap": None if self._last_min_gap is None else
               {"gap": float(self._last_min_gap["gap"]), "body": self._last_min_gap["body"],
                "obstacle": self._last_min_gap["obstacle"]},
               "height_cmd": float(self.walker.height_cmd) if hasattr(self.walker, "height_cmd") else None}
        if self.scanner_on:
            T_wb = self._link_T(self._mount_body)
            wc = self.cfg["scanner"]["camera"]
            out.update({"T_world_tip": T_wb @ self._T_body_tip, "T_world_wristcam": T_wb @ self._T_body_wcam,
                        "wrist_K": self.wrist_cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64),
                        "wrist_size": [wc["height"], wc["width"]]})
        return out

    def camera_rgb(self, name: str) -> np.ndarray:
        """Latest rendered rgb of a camera without any other processing: "head", "wrist" or a recording camera name.
        Every policy step renders all cameras, so in a step hook this is the frame of that step."""
        if name == "head":
            cam = self.head_cam
        elif name == "wrist":
            if self.wrist_cam is None:
                raise ValueError("no wrist camera (configs/scanner.yaml enabled: false)")
            cam = self.wrist_cam
        else:
            cam = self.rec_cams[name]
        return cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)

    def pose_rec_camera(self, name: str, T_world_cam_ros, hfov_deg: float) -> np.ndarray:
        """Place a fixed recording camera (ROS optical pose) with a horizontal FOV; returns its intrinsic matrix as set.
        No render here: the caller renders (settle_renders) once after posing all of them."""
        from .camera import R_to_quat_wxyz
        torch = self.torch
        cam = self.rec_cams[name]
        W, H = cam.cfg.width, cam.cfg.height
        f = W / 2 / math.tan(math.radians(hfov_deg) / 2)
        K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]], dtype=torch.float32, device=self.env.device)
        cam.set_intrinsic_matrices(K, focal_length=cam.cfg.spawn.focal_length)
        pos = torch.tensor([T_world_cam_ros[:3, 3]], dtype=torch.float32, device=self.env.device)
        quat = torch.tensor([R_to_quat_wxyz(T_world_cam_ros[:3, :3])], dtype=torch.float32, device=self.env.device)
        cam.set_world_poses(pos, quat, convention="ros")
        return cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)

    def settle_renders(self) -> None:
        """aux_render_passes renders without a physics step (RTX temporal AA needs them after a camera jump), then
        refresh the recording-camera buffers."""
        for _ in range(self.cfg["camera"].get("aux_render_passes", 2)):
            self.env.sim.render()
        for cam in self.rec_cams.values():
            cam.update(0.0, force_recompute=True)

    def step(self, vx: float, vy: float, wz: float, duration: float) -> dict:
        cmd = self._clip_vel(vx, vy, wz)
        self._set_mode("walk" if any(abs(c) > 0 for c in cmd) else "stand")
        self.walker.set_velocity(*cmd)
        n = max(1, int(round(duration / self.policy_dt)))
        status, reason = "ok", ""
        for _ in range(n):
            self._step_once()
            status, reason = self._check_status()
            if status != "ok":
                break
        self.walker.set_velocity(0.0, 0.0, 0.0)
        return {"status": status, "reason": reason or f"walked {n * self.policy_dt:.2f} s at {cmd}", "obs": self.get_obs(),
                "command": cmd}

    def set_body_pose(self, height: float | None = None, torso_rpy=None, duration: float = 1.0, settle: float = 1.0) -> dict:
        """Squat / stand up and lean the torso (GR00T walker only), standing still: the height and torso commands
        are ramped linearly from their current values over `duration` s, then held for `settle` s; the status check
        runs every policy step as in step(). Returns the commands and the achieved pelvis height / tilt."""
        w = self.walker
        if not hasattr(w, "set_body_command"):
            raise RuntimeError(f"walker {type(w).__name__} has no body (height / torso) command")
        h0, r0 = float(w.height_cmd), np.asarray(w.rpy_cmd, dtype=float).copy()
        h1 = h0 if height is None else float(height)
        r1 = r0 if torso_rpy is None else np.asarray(torso_rpy, dtype=float).reshape(3)
        w.set_body_command(h1, r1)                      # validates the targets before anything moves
        w.set_body_command(h0, r0)
        self._set_mode("stand")
        w.set_velocity(0.0, 0.0, 0.0)
        n_ramp = max(1, int(round(duration / self.policy_dt)))
        n_hold = int(round(settle / self.policy_dt))
        status, reason = "ok", ""
        for k in range(n_ramp + n_hold):
            a = min(1.0, (k + 1) / n_ramp)
            w.set_body_command(h0 + a * (h1 - h0), r0 + a * (r1 - r0))
            self._step_once()
            status, reason = self._check_status()
            if status != "ok":
                break
        T = self._T_world_root()
        return {"status": status, "reason": reason or f"body command height {h1:.3f} rpy {np.round(r1, 3).tolist()}",
                "height_cmd": float(w.height_cmd), "rpy_cmd": np.asarray(w.rpy_cmd, float).tolist(),
                "pelvis_z": float(T[2, 3]), "tilt_deg": self._tilt_deg(), "obs": self.get_obs()}

    def stand_still(self, duration: float) -> dict:
        r = self.step(0.0, 0.0, 0.0, duration)
        r["reason"] = r["reason"] if r["status"] != "ok" else f"stood {duration:.2f} s"
        return r

    def move_relative(self, dx: float, dy: float, dyaw: float, timeout: float = 10.0, pos_tol: float | None = None,
                      yaw_tol: float | None = None, speed_scale: float = 1.0) -> dict:
        """P-controller on the odom estimate towards (dx, dy, dyaw) given in the start root frame. pos_tol / yaw_tol
        override robot.yaml move_relative tolerances (m / rad) for this call; speed_scale (0, 1] scales the velocity
        limits (short precise moves overshoot less when slower; the dead-zone minimum commands still apply)."""
        if not 0 < speed_scale <= 1:
            raise ValueError("speed_scale must be in (0, 1]")
        from .camera import T_from_xy_yaw, inv_T, wrap, yaw_of
        mc = dict(self.cfg["robot"]["move_relative"])
        if pos_tol is not None:
            mc["pos_tol"] = float(pos_tol)
        if yaw_tol is not None:
            mc["yaw_tol"] = float(yaw_tol)
        O0 = T_from_xy_yaw(*self._odom_T[:2, 3], yaw_of(self._odom_T))
        G0 = T_from_xy_yaw(*self._T_world_root()[:2, 3], yaw_of(self._T_world_root()))
        goal = O0 @ T_from_xy_yaw(dx, dy, dyaw)
        t, status, reason = 0.0, "timeout", f"not within tolerance after {timeout:.1f} s"
        L = {k: v * speed_scale for k, v in self.cfg["robot"]["velocity_limits"].items()}

        def err():
            O = T_from_xy_yaw(*self._odom_T[:2, 3], yaw_of(self._odom_T))
            e = inv_T(O) @ goal
            return e[0, 3], e[1, 3], wrap(yaw_of(e))

        def within(ex, ey, eyaw):
            return math.hypot(ex, ey) < mc["pos_tol"] and abs(eyaw) < mc["yaw_tol"]

        # The walking policy needs time to stop: once the command drops to zero the base keeps moving (measured
        # 10-15 cm at 0.4 m/s) and it cannot walk backwards to undo an overshoot. So (1) a round ends when the stop
        # point predicted from the current base velocity (v * stop_time_s) is within tolerance, (2) the robot then
        # settles, and (3) the remaining error is corrected in a further round (max_rounds).
        for rnd in range(mc["max_rounds"]):
            vcmd = (0.0, 0.0, 0.0)
            while t < timeout:
                ex, ey, eyaw = err()
                # predicted stop point from the *commanded* linear velocity (the measured base velocity swings with the
                # gait and stopped rounds prematurely); no prediction for yaw (no turn-inertia measurement)
                px, py, pyaw = ex - vcmd[0] * mc["stop_time_s"], ey - vcmd[1] * mc["stop_time_s"], eyaw
                if within(px, py, pyaw) or within(ex, ey, eyaw):
                    break
                # saturated P control: full speed until within the braking distance, then linear ramp-down.
                # (A unit-gain P law sinks into the policy's low-speed dead zone long before the goal.)
                cmd = [L["vx"] * px / mc["brake_dist_lin"], L["vy"] * py / mc["brake_dist_lin"], L["wz"] * pyaw / mc["brake_dist_ang"]]
                # dead-zone compensation: an axis still outside its tolerance gets at least the smallest command the
                # policy actually executes (measured, robot.yaml move_relative.min_cmd)
                tol = [mc["pos_tol"] / math.sqrt(2)] * 2 + [mc["yaw_tol"]]
                for i, (e_i, key) in enumerate(zip((px, py, pyaw), ("vx", "vy", "wz"))):
                    if abs(e_i) > tol[i]:
                        c = float(np.clip(cmd[i], -L[key], L[key]))
                        cmd[i] = math.copysign(max(abs(c), mc["min_cmd"][key]), e_i)
                    else:
                        cmd[i] = 0.0
                vcmd = self._clip_vel(*cmd)
                self._set_mode("walk" if any(abs(c) > 0 for c in vcmd) else "stand")
                self.walker.set_velocity(*vcmd)
                self._step_once()
                t += self.policy_dt
                s_, r_ = self._check_status()
                if s_ != "ok":
                    status, reason = s_, r_
                    break
            if status in ("fell", "collision"):
                break
            self.walker.set_velocity(0.0, 0.0, 0.0)
            self._set_mode("stand")
            hold = 0.0
            for _ in range(int(round(mc["settle_max_s"] / self.policy_dt))):       # settle
                self._step_once()
                t += self.policy_dt
                v = self.robot.data.root_lin_vel_b[0].cpu().numpy()
                hold = hold + self.policy_dt if math.hypot(v[0], v[1]) < mc["settle_speed"] else 0.0
                if hold >= mc["settle_hold_s"]:
                    break
            if within(*err()):
                status, reason = "ok", f"within tolerance after {rnd + 1} round(s)"
                break
            if t >= timeout:
                break
            ex, ey, eyaw = err()
            reason = f"not within tolerance after {rnd + 1} round(s) / {t:.1f} s (remaining {math.hypot(ex, ey):.3f} m, {math.degrees(eyaw):.1f} deg)"
        self.walker.set_velocity(0.0, 0.0, 0.0)
        self._set_mode("stand")
        O = T_from_xy_yaw(*self._odom_T[:2, 3], yaw_of(self._odom_T))
        G = T_from_xy_yaw(*self._T_world_root()[:2, 3], yaw_of(self._T_world_root()))
        d_odom, d_gt = inv_T(O0) @ O, inv_T(G0) @ G
        req = np.array([dx, dy, dyaw])
        od = np.array([d_odom[0, 3], d_odom[1, 3], wrap(yaw_of(d_odom))])
        gd = np.array([d_gt[0, 3], d_gt[1, 3], wrap(yaw_of(d_gt))])
        return {"status": status, "reason": reason, "elapsed_s": t,
                "requested": req.tolist(), "odom_displacement": od.tolist(), "gt_displacement": gd.tolist(),
                "gt_error": {"pos_m": float(np.linalg.norm(gd[:2] - req[:2])), "yaw_rad": float(abs(wrap(gd[2] - req[2])))},
                "odom_error_vs_gt": {"pos_m": float(np.linalg.norm(od[:2] - gd[:2])), "yaw_rad": float(abs(wrap(od[2] - gd[2])))}}

    def set_gripper(self, side: str, opening: float) -> dict:
        """The hand holding the scanner keeps its grip: side=<scanner side> is an error, side=both moves only the
        other gripper (the reply says so)."""
        if side not in ("left", "right", "both"):
            return {"status": "error", "reason": f"side must be left|right|both, got {side!r}"}
        sides = ["left", "right"] if side == "both" else [side]
        note = ""
        if self.scanner_on and self.scanner_side in sides:
            if side != "both":
                return {"status": "error", "reason": f"the {side} gripper holds the scanner (configs/scanner.yaml enabled: true)"}
            sides.remove(self.scanner_side)
            note = f"{self.scanner_side} gripper holds the scanner, left unchanged"
        q = self._gripper_joint(opening)
        tgt = {"left": self.walker.gripper_dds.left, "right": self.walker.gripper_dds.right}
        for sd in sides:
            tgt[sd] = q
        self.walker.set_gripper_joint_targets(tgt["left"], tgt["right"])
        return {"status": "ok", "side": side, "applied_to": sides, "opening": float(np.clip(opening, 0, 1)),
                "joint_target": q, "reason": note}

    def set_wrist_target(self, side: str, pos_root) -> dict:
        """Wrist (`<side>_hand_base_link`) position target in the root frame; tracked by IK every step.
        side: left | right | both. Clamped to the default wrist position +- robot.yaml arm_ik.workspace_half_extent."""
        sides = ("left", "right") if side == "both" else (side,)
        if any(sd not in ("left", "right") for sd in sides):
            return {"status": "error", "reason": f"side must be left|right|both, got {side!r}"}
        return {"status": "ok", "target_root": {sd: self.arm.set_target(sd, pos_root).tolist() for sd in sides}}

    def move_wrist(self, side: str, dx: float, dy: float, dz: float) -> dict:
        """Shift the wrist target by (dx, dy, dz) in the root frame (from the current target, or from the
        default wrist position if the arm has no target yet)."""
        sides = ("left", "right") if side == "both" else (side,)
        if any(sd not in ("left", "right") for sd in sides):
            return {"status": "error", "reason": f"side must be left|right|both, got {side!r}"}
        out = {}
        for sd in sides:
            base = self.arm.target[sd] if self.arm.target[sd] is not None else self.arm.home_pos(sd)
            out[sd] = self.arm.set_target(sd, base + np.array([dx, dy, dz])).tolist()
        return {"status": "ok", "target_root": out}

    def reset_arms(self) -> dict:
        """Arms back to the default (upstream) pose, IK targets cleared."""
        self.arm.reset()
        return {"status": "ok"}

    def get_ground_truth(self, d: float = 0.20) -> dict:
        """Evaluation only. Scenario mode adds target_box_id, task1_gt_obs_pose (d_obs rule
        on the true face with the measured head camera) and task2_gt_wrist_target (true barcode centre + offset *
        normal). Target "all" also returns "faces": every exposed side of the stack in the
        tour order -x, -y, +x, +y with its union face, gt_obs_pose and label contents (testsuite.geometry.stack_faces)."""
        from .ground_truth import build_gt
        gt = build_gt(self.layout, d)
        if self.scenario is not None:
            import yaml
            from .config import PKG_ROOT
            from .testsuite import geometry as TG
            F = self._ts_factors
            cam = TG.head_camera(yaml.safe_load(open(os.path.join(PKG_ROOT, F["head_camera_nominal"]))))
            S = F["task2_start"]
            # a face counts as hidden where another box reaches into the slab a label would occupy (as to_layout)
            depth = F["barcode"]["face_offset"] + self.cfg["barcode"]["label_thickness"]
            if self.layout["target_box_id"] == "all":
                # task 3 over every box: the union of the exposed front faces; no single target barcode / wrist target
                fc, n, corners = TG.front_face_union(self.layout["boxes"], depth)
                tid, tbc, wt = "all", None, None
                # every exposed side of the stack in the tour order -x, -y, +x, +y (obs pose per side)
                gt["faces"] = TG.stack_faces(self.layout["boxes"], self.layout["barcodes"], cam, S["d_obs_range"],
                                             S["d_obs_step"], S["fov_margin_frac"], depth)
            else:
                tb = next(b for b in self.layout["boxes"] if b["id"] == self.layout["target_box_id"])
                bc = next(b for b in self.layout["barcodes"] if b["box_id"] == tb["id"])
                fc, n, corners = TG.front_face(tb)
                tid, tbc, wt = tb["id"], bc["content"], TG.wrist_target(bc, S["wrist_offset"])
            gt.update({"scenario_id": self.scenario["id"], "task": self.scenario["task"], "target": self.scenario["target"],
                       "target_box_id": tid, "target_barcode": tbc,
                       "target_face": {"center": fc.tolist(), "normal": n.tolist(), "corners": corners.tolist()},
                       "task1_gt_obs_pose": TG.obs_pose(fc, n, corners, cam, S["d_obs_range"], S["d_obs_step"], S["fov_margin_frac"]),
                       "task2_gt_wrist_target": wt})
        return gt

    def teleport_root(self, x: float, y: float, yaw: float) -> None:
        torch = self.torch
        from .tasks.barcode_task import init_root_z
        z = init_root_z(self.cfg["robot"])
        pose = torch.tensor([[x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=self.env.device)
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.env.device))
        q0 = self.robot.data.default_joint_pos.clone()
        if self._scanner_grip_q is not None:   # scanner hand starts closed on the handle (no ~6 s re-grip after a reset)
            jn = list(self.robot.data.joint_names)
            for j in self.cfg["robot"]["gripper"]["joints"][self.scanner_side]:
                q0[0, jn.index(j)] = self._scanner_grip_q
        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(self.robot.data.default_joint_vel))
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        # advance the data timestamps (no physics step) so robot.data re-reads the written pose;
        # update(dt=0) would leave the timestamped buffers (body poses etc.) stale
        self.env.scene.update(dt=self.env.physics_dt)
        self._reset_walker_buffers()
        self.arm.reset()
        T = self._T_world_root()
        self._odom_T, self._last_gt_T = T.copy(), T.copy()

    def render_aux(self, T_world_cam_ros, hfov_deg: float | None = None) -> dict:
        """Render the aux camera at a given pose (ROS optical convention), optionally with a given
        horizontal FOV (default: camera_head.yaml aux intrinsics). Physics does not advance."""
        from .camera import R_to_quat_wxyz
        torch = self.torch
        ac = self.cfg["camera"]["aux"]
        if hfov_deg is None:
            hfov_deg = math.degrees(2 * math.atan(ac["horizontal_aperture"] / 2 / ac["focal_length"]))
        if getattr(self, "_aux_hfov", None) != hfov_deg:
            W, H = ac["width"], ac["height"]
            f = W / 2 / math.tan(math.radians(hfov_deg) / 2)
            K = torch.tensor([[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]], dtype=torch.float32, device=self.env.device)
            self.aux_cam.set_intrinsic_matrices(K, focal_length=ac["focal_length"])
            self._aux_hfov = hfov_deg
        pos = torch.tensor([T_world_cam_ros[:3, 3]], dtype=torch.float32, device=self.env.device)
        quat = torch.tensor([R_to_quat_wxyz(T_world_cam_ros[:3, :3])], dtype=torch.float32, device=self.env.device)
        self.aux_cam.set_world_poses(pos, quat, convention="ros")
        for _ in range(self.cfg["camera"].get("aux_render_passes", 2)):
            self.env.sim.render()
        self.aux_cam.update(0.0, force_recompute=True)
        return {"rgb": self.aux_cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8),
                "depth": self.aux_cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze(-1),
                "K": self.aux_cam.data.intrinsic_matrices[0].cpu().numpy()}

    def topdown(self) -> dict:
        from .camera import look_at_ros
        t = self.cfg["scene"]["topdown"]
        T = look_at_ros(np.array([t["center"][0], t["center"][1], t["height"]]), np.array([t["center"][0], t["center"][1], 0.0]),
                        world_up=(0.0, 1.0, 0.0))   # image right = +x (robot faces right), image up = +y
        out = self.render_aux(T)
        out["T_world_cam"] = T
        return out

    def oblique_views(self) -> dict:
        """Oblique overview renders defined in the scene YAML (`views`). Returns {name: rgb}."""
        from .camera import look_at_ros
        v = self.cfg["scene"]["views"]
        cx, cy = self.cfg["scene"]["topdown"]["center"]
        target = np.array([cx, cy, v["target_z"]])
        out = {}
        for item in v["list"]:
            eye = np.array([cx, cy, 0.0]) + np.asarray(item["eye_offset"], dtype=float)
            out[item["name"]] = self.render_aux(look_at_ros(eye, target), hfov_deg=item["hfov"])["rgb"]
        return out

    def save_snapshot(self, path: str) -> None:
        import cv2
        os.makedirs(path, exist_ok=True)
        obs = self.get_obs()
        cv2.imwrite(os.path.join(path, "head_rgb.png"), cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(path, "head_depth.png"), depth_colormap(obs["depth"], self.cfg["camera"]["depth"]["valid_range"]))
        np.save(os.path.join(path, "head_depth.npy"), obs["depth"])
        if self.scanner_on:
            cv2.imwrite(os.path.join(path, "wrist_rgb.png"), cv2.cvtColor(obs["wrist_rgb"], cv2.COLOR_RGB2BGR))
            if "wrist_depth" in obs:
                wd = obs["wrist_depth"]
                rng = self.cfg["scanner"]["camera"]["depth_valid_range"] or [0.0, float(np.percentile(wd[np.isfinite(wd)], 95))]
                cv2.imwrite(os.path.join(path, "wrist_depth.png"), depth_colormap(np.nan_to_num(wd, posinf=0.0), rng))
        top = self.topdown()
        cv2.imwrite(os.path.join(path, "topdown.png"), cv2.cvtColor(top["rgb"], cv2.COLOR_RGB2BGR))
        for name, rgb in self.oblique_views().items():
            cv2.imwrite(os.path.join(path, f"view_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def aim_scanner(self, tip_pos_root, dir_root, anchor: str = "odom") -> dict:
        """Scanner aim (IK, tracked every step): put the scanner tip (exit-window centre) at `tip_pos_root` with the
        scan direction along `dir_root`, both given in the current root frame. Roll about the scan direction is
        left free. anchor="odom" (default) keeps the target fixed in space (odometry frame) when the base drifts;
        "root" moves it with the body. Replaces any wrist target of the scanner hand; reset_arms() ends it.
        Progress: obs["scanner_aim_error"]."""
        if not self.scanner_on:
            return {"status": "error", "reason": "scanner disabled (configs/scanner.yaml enabled: false)"}
        t = self.arm.set_aim(self.scanner_side, tip_pos_root, dir_root, self._T_body_tip, anchor)
        return {"status": "ok", "side": self.scanner_side, "anchor": anchor, "tip_pos_root": t["pos"].tolist(),
                "dir_root": t["dir"].tolist()}

    def aim_scanner_at(self, point_root, normal_root, distance: float = 0.20, anchor: str = "odom") -> dict:
        """Aim the scanner at a label: tip at `distance` in front of `point_root` along the label's outward normal
        `normal_root`, scanning along -normal (perpendicular incidence). Root frame at the time of the call."""
        n = np.asarray(normal_root, dtype=float)
        n = n / np.linalg.norm(n)
        return self.aim_scanner(np.asarray(point_root, dtype=float) + distance * n, -n, anchor)

    def scan(self) -> dict:
        """Trigger the scanner: decode the current scanner-camera image (pyzbar at several image rotations, i.e.
        omnidirectional like the area imager of a handheld scanner). `content` is the code nearest the image centre (where a real scanner's aiming pattern is);
        `decoded` lists every code in view. `gt` (evaluation only) intersects the scan ray with the GT labels:
        which label it hits, at what distance and incidence angle (occlusion not considered)."""
        if not self.scanner_on:
            return {"status": "error", "reason": "scanner disabled (configs/scanner.yaml enabled: false)"}
        import cv2
        from pyzbar import pyzbar
        obs = self.get_obs()
        img = obs["wrist_rgb"]
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        codes = {}
        step = self.cfg["scanner"]["scan"]["rotation_step_deg"]
        # omnidirectional decoding (as the area imager does): zbar alone reads only codes close to an image axis,
        # so decode the image at several rotations (canvas enlarged, nothing cropped) and map hits back
        for ang in np.arange(0.0, 90.0, step):
            M = cv2.getRotationMatrix2D((W / 2, H / 2), float(ang), 1.0)
            cs, sn = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(np.ceil(H * sn + W * cs)), int(np.ceil(H * cs + W * sn))
            M[0, 2] += nw / 2 - W / 2
            M[1, 2] += nh / 2 - H / 2
            Minv = cv2.invertAffineTransform(M)
            for r in pyzbar.decode(cv2.warpAffine(gray, M, (nw, nh), borderValue=255)):
                content = r.data.decode()
                if content in codes:
                    continue
                pts = np.array([(p.x, p.y) for p in r.polygon], dtype=float)
                poly = (Minv[:, :2] @ pts.T).T + Minv[:, 2]                      # pyzbar polygon in the original image
                c = poly.mean(axis=0)                                            # centre in the original image
                codes[content] = {"content": content, "type": r.type, "center_px": c.tolist(), "polygon": poly.tolist(),
                                  "rotation_deg": float(ang),
                                  "offset_from_image_center_px": float(np.hypot(c[0] - W / 2, c[1] - H / 2))}
        codes = sorted(codes.values(), key=lambda d: d["offset_from_image_center_px"])
        T_wt = obs["T_world_root_gt"] @ obs["T_root_scanner_tip"]
        return {"status": "ok" if codes else "no_read", "content": codes[0]["content"] if codes else None,
                "decoded": codes, "T_root_scanner_tip": obs["T_root_scanner_tip"], "timestamp": obs["timestamp"],
                "gt": self._scan_ray_gt(T_wt[:3, 3], T_wt[:3, 0])}

    def _scan_ray_gt(self, p, d) -> dict:
        """First GT label hit by the ray p + t d (t > 0), evaluation only."""
        best = None
        for bc in self.layout["barcodes"]:
            c, n = np.asarray(bc["center"]), np.asarray(bc["normal"])
            den = float(d @ n)
            if den >= 0:          # ray must come from the label's front side
                continue
            t = float((c - p) @ n) / den
            if t <= 0:
                continue
            q = p + t * d - c
            u, v = float(q @ np.asarray(bc["right"])), float(q @ np.asarray(bc["up"]))
            if abs(u) <= bc["size"][0] / 2 and abs(v) <= bc["size"][1] / 2 and (best is None or t < best["distance_m"]):
                best = {"content": bc["content"], "distance_m": t, "incidence_deg": float(np.degrees(np.arccos(-den))),
                        "hit_offset_m": [u, v]}
        return {"hit": best, "note": "evaluation only (GT); occlusion not considered"}

    def close(self) -> None:
        global _APP
        self.env.close()
        if _APP is not None:
            # A plain SimulationApp.close() never returns once camera annotators are active (replicator
            # orchestrator shutdown blocks; reproduced with a single camera).
            # skip_cleanup=True is the documented immediate-exit path: it ends the process.
            sys.stdout.flush(); sys.stderr.flush()
            _APP.close(skip_cleanup=True)
            _APP = None


def depth_colormap(depth, valid_range):
    import cv2
    lo, hi = valid_range
    d = np.clip((depth - lo) / (hi - lo), 0, 1)
    vis = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth <= 0] = 0
    return vis
