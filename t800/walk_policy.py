"""Python port of EngineAI's T800 Pro RL walking runner (route A: vendor walk policy).

Faithful re-implementation of
  engineai_robotics_native_sdk/src/runner/rl_walking_example/src/rl_walking_example_runner.cc   ("runner.cc")
  engineai_robotics_native_sdk/src/data/_param/rl_walking_example_param/rl_walking_example_param.h ("param.h")
with every number read from
  assets/config/t800pro/rl_walking_example/default.yaml ("walk.yaml")
  assets/config/t800pro/model/default.yaml              ("model.yaml", joint order of the full body)
No constant is hard-coded here: all values come from the two yaml files.

Joint space
-----------
The runner reads/writes `data_store_->joint_info` (runner.cc:229-230, 529) which is the SERIAL joint space
of `model.yaml` limbs (43 joints, order left_leg, right_leg, waist, left_arm, left_hand, right_arm,
right_hand, head; model.yaml:8-76). The parallel-ankle / parallel-wrist mechanisms are converted to motor
space by a separate runner (joint_motor_transform_runner/default.yaml:6 `enable_parallel_converter: true`)
downstream of this runner. The official MJCF (serial_t800pro.xml) models the ankles/wrists as SERIAL joints,
so in simulation no parallel conversion is applied: the policy targets are used directly as joint targets.

Single-step observation (num_observations = 72, walk.yaml:3; assembled at runner.cc:383-389)
------------------------------------------------------------------------------------------
  idx  0..21 : (q - default_q)[active]       x observation_scale_dof_pos = 1.0  (walk.yaml:33, param.h:14-16)
  idx 22..43 : dq[active]                    x observation_scale_dof_vel = 0.05 (walk.yaml:34, param.h:17-18)
  idx 44..65 : previous action (clipped)     x 1                               (param.h:19; runner.cc:387, 465)
  idx 66..68 : body-frame angular velocity   x observation_scale_angular_vel=1.0 (walk.yaml:32, param.h:20)
  idx 69..71 : body-frame projected gravity  x observation_scale_quat = 1.0     (walk.yaml:35, param.h:21)
               (-R^T e_z, = (0,0,-1) upright; runner.cc:374-375)
  then clipped to +-observation_clip = 100 (walk.yaml:36; runner.cc:397-399).
  [active] = the 22 `active_joint_names` (walk.yaml:4-28) in that order:
    0-5 left leg (hip pitch/roll/yaw, knee, ankle pitch/roll), 6-11 right leg,
    12-16 left arm (shoulder pitch/roll/yaw, elbow pitch/yaw), 17-21 right arm (same).
    Waist, wrists, fingers and head are NOT active: they get q_des = default (runner.cc:470-473).

History (num_include_obs_steps = 15, walk.yaml:29; runner.cc:402-416)
  72x15 matrix, column = time step, oldest at column 0, newest at column 14. On the first call every
  column is filled with the current observation (runner.cc:402-409).

Network input (1083 = 72*15 + 3; runner.cc:449-458)
  [obs_{t-14} (72), obs_{t-13} (72), ..., obs_t (72), command_obs (3)]  (column-major flatten of the
  72x15 matrix = time-major; runner.cc:454-456)
  command_obs = command * [observation_scale_linear_vel, observation_scale_linear_vel,
                           observation_scale_angular_vel] = command * [2.0, 2.0, 1.0]
                (param.h:22-24, walk.yaml:31-32; runner.cc:445)
  There is NO phase / clock input: the gait timing is internal to the policy (history-based).

Command (runner.cc:300-318)
  command = stick * command_scale_pos (stick>=0) or stick * command_scale_neg (stick<0)
    command_scale_pos = [1.2, 0.4, 1.0], command_scale_neg = [0.6, 0.4, 1.0] (walk.yaml:84-85)
  then (enable_remote_command_lpf = true, walk.yaml:80) a first-order low-pass filter with
  sampling 40 Hz, cut-off 0.1 Hz (walk.yaml:81-82). NOTE: math/first_order_low_pass_filter.h is NOT
  shipped in the SDK, so its exact discretisation is unknown; we use the standard
  y += a (x - y), a = Ts / (Ts + 1/(2 pi fc)), Ts = 1/fs, state reset to 0 -- see FirstOrderLPF.

Action (runner.cc:462-473)
  a = clip(net(obs), +-action_clip = 100) (walk.yaml:41)
  q_des = default_q (all 43);  q_des[active] += a * action_scale (walk.yaml:72-76)
  kp/kd = joint_kp / joint_kd (walk.yaml:52-70), dq_des = 0, tau_ff = 0 (runner.cc:525-530)
  tau = kp (q_des - q) + kd (0 - dq) is computed by the low-level driver (runner.cc:515-516).

Startup transition (runner.cc:195-198, 479-497): for time < transition_time = 0.5 s (walk.yaml:83)
  q_des, kp, kd are linearly blended from the command that was active when the runner was entered.

control_dt = 0.01 s (walk.yaml:77): one act() call per 10 ms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml

SDK_ROOT = os.environ.get(
    "T800_SDK_ROOT",
    "/data/yicheng/HumanoidBarcodeScan/third_party/engineai/engineai_robotics_native_sdk",
)
CONFIG_ROOT = os.path.join(SDK_ROOT, "assets/config/t800pro")
DEFAULT_WALK_YAML = os.path.join(CONFIG_ROOT, "rl_walking_example/default.yaml")
DEFAULT_MODEL_YAML = os.path.join(CONFIG_ROOT, "model/default.yaml")


def _flatten(groups) -> np.ndarray:
    """common::ConcatenateVectors (runner.cc:167-174): flatten the per-limb groups."""
    return np.concatenate([np.asarray(g, dtype=np.float64) for g in groups])


def _rpy_to_matrix(rpy) -> np.ndarray:
    """math::RotationMatrixd(RollPitchYawd(rpy)) = Rz(yaw) Ry(pitch) Rx(roll) (Drake-style RPY)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class FirstOrderLPF:
    """Stand-in for math::FirstOrderLowPassFilter<Vector3d>(sampling_freq, cutoff_freq).

    The C++ header is not part of the shipped SDK; this is the textbook RC discretisation.
    It is updated once per act() call exactly as the runner does (runner.cc:315-317), i.e. at
    1/control_dt = 100 Hz even though the configured sampling frequency is 40 Hz (walk.yaml:81) --
    we reproduce that mismatch rather than "fixing" it.

    The yaml lists this filter under "# sim to real fine tune parameters" (walk.yaml:79-82) and the
    runner applies it to the REMOTE (gamepad) command only (runner.cc:315-317). It changes the vx/vy
    start threshold, the minimum step length and the coasting distance after a stop (see
    mujoco_check.py --lpf both). A navigation layer that sends velocity commands directly can pass
    enable_command_lpf=False and shape its own commands; the default (None) follows the yaml.
    """

    def __init__(self, sampling_freq: float, cutoff_freq: float):
        ts = 1.0 / sampling_freq
        rc = 1.0 / (2.0 * np.pi * cutoff_freq)
        self.alpha = ts / (ts + rc)
        self.y = np.zeros(3)

    def reset(self):
        self.y = np.zeros(3)

    def update(self, x: np.ndarray) -> np.ndarray:
        self.y = self.y + self.alpha * (np.asarray(x, dtype=np.float64) - self.y)
        return self.y.copy()


class MNNPolicy:
    """Runs the .mnn MLP with the MNN session API (input 'input' (1,1083) -> output 'output' (1,22))."""

    def __init__(self, path: str):
        import MNN  # local import: only this class needs MNN

        self._MNN = MNN
        self.interp = MNN.Interpreter(path)
        self.session = self.interp.createSession()
        ins = self.interp.getSessionInputAll(self.session)
        outs = self.interp.getSessionOutputAll(self.session)
        if len(ins) != 1 or len(outs) != 1:
            raise RuntimeError(f"expected 1 input / 1 output, got {list(ins)} / {list(outs)}")
        (self.in_name, in_t), = ins.items()
        (self.out_name, out_t), = outs.items()
        self.in_shape = tuple(in_t.getShape())
        self.out_shape = tuple(out_t.getShape())
        self.in_t, self.out_t = in_t, out_t

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        MNN = self._MNN
        x = np.ascontiguousarray(obs, dtype=np.float32).reshape(self.in_shape)
        tmp_in = MNN.Tensor(self.in_shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
        self.in_t.copyFrom(tmp_in)
        self.interp.runSession(self.session)
        tmp_out = MNN.Tensor(self.out_shape, MNN.Halide_Type_Float,
                             np.zeros(self.out_shape, dtype=np.float32), MNN.Tensor_DimensionType_Caffe)
        self.out_t.copyToHostTensor(tmp_out)
        return np.array(tmp_out.getNumpyData(), dtype=np.float32).reshape(-1)


class NumpyMLPPolicy:
    """The same MLP evaluated with numpy from the weights exported by export_policy_npz.py (no MNN needed; the Isaac
    Sim env has no MNN). float32 throughout, as MNN's CPU backend. Layers: y = act(x W^T + b); act = ELU(alpha) or
    none, exactly as listed in the .npz (read from the MNN graph, not assumed)."""

    def __init__(self, path: str):
        d = np.load(path)
        n = int(d["n_layers"])
        self.layers = []
        for i in range(n):
            act = str(d[f"act{i}"])
            if act not in ("elu", "none"):
                raise ValueError(f"layer {i}: activation {act!r} not supported")
            self.layers.append((d[f"W{i}"].astype(np.float32), d[f"b{i}"].astype(np.float32), act,
                                np.float32(d[f"alpha{i}"])))
        self.in_shape = (1, self.layers[0][0].shape[1])
        self.out_shape = (1, self.layers[-1][0].shape[0])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(obs, dtype=np.float32).reshape(-1)
        for W, b, act, alpha in self.layers:
            x = W @ x + b
            if act == "elu":
                x = np.where(x > 0, x, alpha * np.expm1(np.minimum(x, 0)))
        return x.astype(np.float32)


@dataclass
class PolicyOutput:
    q_des: np.ndarray   # (43,) full-body joint position targets, model.yaml order
    kp: np.ndarray      # (43,)
    kd: np.ndarray      # (43,)
    action: np.ndarray  # (22,) clipped raw policy action
    obs: np.ndarray     # (1083,) network input (for debugging)


class T800WalkPolicy:
    def __init__(self, policy_path: str | None = None, cfg_yaml: str = DEFAULT_WALK_YAML,
                 model_yaml: str = DEFAULT_MODEL_YAML, enable_command_lpf: bool | None = None):
        """policy_path: None = the yaml's .mnn (MNN runtime); a path ending in .npz = the same network exported by
        export_policy_npz.py, evaluated with numpy (NumpyMLPPolicy)."""
        with open(cfg_yaml) as f:
            self.cfg = yaml.safe_load(f)
        with open(model_yaml) as f:
            mcfg = yaml.safe_load(f)
        c = self.cfg

        # full-body joint order = model.yaml limbs (runner.cc:148 joint_id_in_total_limb)
        self.limb_names = [l["name"] for l in mcfg["limbs"]]
        self.joint_names = [j for l in mcfg["limbs"] for j in l["joints"]]
        self.num_total_joints = len(self.joint_names)
        jid = {n: i for i, n in enumerate(self.joint_names)}

        self.active_joint_names = list(c["active_joint_names"])
        self.num_actions = len(self.active_joint_names)                       # param.h:12
        self.active_idx = np.array([jid[n] for n in self.active_joint_names])  # runner.cc:145-149
        self.num_obs = int(c["num_observations"])
        self.num_steps = int(c["num_include_obs_steps"])
        if self.num_obs != 3 * self.num_actions + 6:
            raise ValueError(f"num_observations {self.num_obs} != 3*{self.num_actions}+6")

        self.default_q = _flatten(c["default_joint_q"])
        self.kp = _flatten(c["joint_kp"])
        self.kd = _flatten(c["joint_kd"])
        self.action_scale = _flatten(c["action_scale"])
        for name, v, n in [("default_joint_q", self.default_q, self.num_total_joints),
                           ("joint_kp", self.kp, self.num_total_joints),
                           ("joint_kd", self.kd, self.num_total_joints),
                           ("action_scale", self.action_scale, self.num_actions)]:
            if v.shape != (n,):
                raise ValueError(f"{name} has {v.shape}, expected ({n},)")

        na = self.num_actions
        self.obs_scale = np.concatenate([                                      # param.h:13-21
            np.full(na, c["observation_scale_dof_pos"]),
            np.full(na, c["observation_scale_dof_vel"]),
            np.ones(na),
            np.full(3, c["observation_scale_angular_vel"]),
            np.full(3, c["observation_scale_quat"]),
        ]).astype(np.float64)
        self.command_obs_scale = np.array([c["observation_scale_linear_vel"],   # param.h:22-24
                                           c["observation_scale_linear_vel"],
                                           c["observation_scale_angular_vel"]], dtype=np.float64)
        self.obs_clip = float(c["observation_clip"])
        self.action_clip = float(c["action_clip"])
        self.control_dt = float(c["control_dt"])
        self.command_scale_pos = np.asarray(c["command_scale_pos"], dtype=np.float64)
        self.command_scale_neg = np.asarray(c["command_scale_neg"], dtype=np.float64)
        self.imu_install_bias = np.asarray(c.get("imu_install_bias", [0, 0, 0]), dtype=np.float64)
        self.transition_time = c.get("transition_time")                         # optional (param.h:62)
        self.disable_transition_idx = np.array(
            [jid[n] for n in (c.get("disable_transition_joints") or [])], dtype=int)
        self.enable_command_lpf = (bool(c["enable_remote_command_lpf"])
                                   if enable_command_lpf is None else bool(enable_command_lpf))
        self.lpf = FirstOrderLPF(float(c["remote_command_sampling_frequency"]),
                                 float(c["remote_command_cut_off_frequency"]))

        if policy_path is None:
            policy_path = os.path.join(CONFIG_ROOT, c["policy_file"])
        self.net = NumpyMLPPolicy(policy_path) if policy_path.endswith(".npz") else MNNPolicy(policy_path)
        expect_in = self.num_obs * self.num_steps + 3
        if self.net.in_shape[-1] != expect_in or self.net.out_shape[-1] != na:
            raise ValueError(f"model io {self.net.in_shape}->{self.net.out_shape}, expected "
                             f"(1,{expect_in})->(1,{na})")
        self.reset()

    # ------------------------------------------------------------------ lifecycle (runner.cc:105-207)
    def reset(self, initial_q_cmd=None, initial_kp=None, initial_kd=None):
        """Enter(). initial_* = joint command active before entering (used by the startup blend).
        Defaults: default pose with the policy gains (then the blend only attenuates the action)."""
        self.time = np.float32(0.0)  # header declares `float time_` (runner.h:38)
        self.is_first_time = True
        self.obs_hist = np.zeros((self.num_obs, self.num_steps))
        self.action = np.zeros(self.num_actions)
        self.command_raw = np.zeros(3)
        self.command = np.zeros(3)
        self.lpf.reset()
        self.initial_q = self.default_q.copy() if initial_q_cmd is None else np.asarray(initial_q_cmd, float).copy()
        self.initial_kp = self.kp.copy() if initial_kp is None else np.asarray(initial_kp, float).copy()
        self.initial_kd = self.kd.copy() if initial_kd is None else np.asarray(initial_kd, float).copy()
        self.enable_transition = (self.transition_time is not None
                                  and self.transition_time > self.control_dt)  # runner.cc:196-198

    # ------------------------------------------------------------------ commands (runner.cc:300-318)
    def set_stick(self, lx: float, ly: float, ry: float):
        s = np.array([lx, ly, ry], dtype=np.float64)
        self.command_raw = np.where(s >= 0, s * self.command_scale_pos, s * self.command_scale_neg)

    def set_command(self, vx: float, vy: float, wz: float):
        """Physical command (m/s, m/s, rad/s). Equivalent to a stick value v/scale; refuses values that
        need |stick| > 1, i.e. outside the range the operator could command on the robot."""
        v = np.array([vx, vy, wz], dtype=np.float64)
        stick = np.where(v >= 0, v / self.command_scale_pos, v / self.command_scale_neg)
        if np.any(np.abs(stick) > 1.0 + 1e-9):
            raise ValueError(f"command {v} outside stick range: max +{self.command_scale_pos} "
                             f"/ -{self.command_scale_neg}")
        self.command_raw = v

    # ------------------------------------------------------------------ helpers
    def imu_to_obs(self, imu_quat_wxyz, imu_gyro):
        """runner.cc:356-375: IMU quaternion (world->imu orientation) + gyro (imu frame) ->
        body-frame angular velocity and projected gravity."""
        r_install = _rpy_to_matrix(self.imu_install_bias)
        r_local = quat_wxyz_to_matrix(imu_quat_wxyz)
        r_real = r_local @ r_install.T
        w_real = r_real.T @ r_local @ np.asarray(imu_gyro, dtype=np.float64)
        g_real = -r_real.T @ np.array([0.0, 0.0, 1.0])
        return w_real, g_real

    # ------------------------------------------------------------------ main step (runner.cc:227-239)
    def act(self, q, dq, base_ang_vel_body, projected_gravity_body) -> PolicyOutput:
        """q, dq: (43,) full-body serial joint state in model.yaml order."""
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        if q.shape != (self.num_total_joints,) or dq.shape != (self.num_total_joints,):
            raise ValueError(f"q/dq must be ({self.num_total_joints},), got {q.shape}/{dq.shape}")

        # UpdateRemoteCommand (runner.cc:300-318)
        self.command = self.lpf.update(self.command_raw) if self.enable_command_lpf else self.command_raw.copy()

        # CalculateObservation (runner.cc:350-417)
        single = np.concatenate([
            (q - self.default_q)[self.active_idx],
            dq[self.active_idx],
            self.action,
            np.asarray(base_ang_vel_body, dtype=np.float64),
            np.asarray(projected_gravity_body, dtype=np.float64),
        ])
        single = np.clip(single * self.obs_scale, -self.obs_clip, self.obs_clip)
        if self.is_first_time:
            self.is_first_time = False
            self.action = np.zeros(self.num_actions)
            self.obs_hist[:] = single[:, None]
        else:
            self.obs_hist[:, :-1] = self.obs_hist[:, 1:]
            self.obs_hist[:, -1] = single

        # CalculateMotorCommand (runner.cc:441-498)
        obs = np.concatenate([self.obs_hist.T.reshape(-1), self.command * self.command_obs_scale])
        self.action = np.clip(self.net(obs).astype(np.float64), -self.action_clip, self.action_clip)
        q_des = self.default_q.copy()
        q_des[self.active_idx] += self.action * self.action_scale
        kp, kd = self.kp.copy(), self.kd.copy()
        if self.enable_transition and self.time < self.transition_time:
            q_pol, kp_pol, kd_pol = q_des.copy(), kp.copy(), kd.copy()
            ratio = np.float32(self.time / np.float32(self.transition_time))
            q_des = ratio * q_des + (1.0 - ratio) * self.initial_q
            kp = ratio * kp + (1.0 - ratio) * self.initial_kp
            kd = ratio * kd + (1.0 - ratio) * self.initial_kd
            if self.disable_transition_idx.size:
                d = self.disable_transition_idx
                q_des[d], kp[d], kd[d] = q_pol[d], kp_pol[d], kd_pol[d]

        self.time = np.float32(self.time + np.float32(self.control_dt))  # runner.cc:238
        return PolicyOutput(q_des=q_des, kp=kp, kd=kd, action=self.action.copy(), obs=obs.astype(np.float32))


if __name__ == "__main__":
    pol = T800WalkPolicy()
    print("model io:", pol.net.in_name, pol.net.in_shape, "->", pol.net.out_name, pol.net.out_shape)
    print("active joints:", pol.active_joint_names)
    print("lpf alpha per call:", pol.lpf.alpha)
    out = pol.act(pol.default_q, np.zeros(43), np.zeros(3), np.array([0, 0, -1.0]))
    print("first action:", np.round(out.action, 4))
