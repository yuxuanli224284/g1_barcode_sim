"""Offline checks of the T800 integration (no Isaac Sim): robot selection by config (G1 default untouched), head-camera
mount, and the T800 walker (joint mapping, arm spoof, external arm / head targets, feed-forward indexing).
Fake articulation with a shuffled joint order. Needs the local vendor files (skipped when absent).
run: python -m pytest tests/test_t800.py"""
import os
import sys
import types

import numpy as np
import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from g1_barcode_sim.config import CFG_DIR, deep_merge, load_all, load_camera_cfg, load_robot_cfg  # noqa: E402

T800 = load_robot_cfg({"model": "t800"})
HAVE_VENDOR = os.path.exists(T800["t800_walk"]["policy"]) and os.path.isdir(T800["asset"]["sdk_root"])


# ------------------------------------------------------------------ config selection
def test_g1_default_is_robot_yaml():
    raw = yaml.safe_load(open(os.path.join(CFG_DIR, "robot.yaml")))
    assert load_robot_cfg() == raw and raw["model"] == "g1"
    assert load_robot_cfg({"height_cmd": 0.7}) == deep_merge(raw, {"height_cmd": 0.7})


def test_robot_yaml_cannot_switch_the_default(monkeypatch):
    """The robot is chosen by the per-run override only: a robot.yaml edited to another model is rejected, so the G1
    default (all G1 code, tests and scripts call load_robot_cfg() without a model) cannot change globally."""
    import g1_barcode_sim.config as C
    real = C._load

    def edited(path):
        d = real(path)
        return dict(d, model="t800") if os.path.basename(path) == "robot.yaml" else d
    monkeypatch.setattr(C, "_load", edited)
    with pytest.raises(ValueError, match="must declare model: g1"):
        C.load_robot_cfg()
    assert C.load_robot_cfg({"model": "t800"})["model"] == "t800"     # the T800 file is unaffected


def test_g1_camera_unchanged():
    assert load_all("s1_table")["camera"] == load_camera_cfg()


def test_t800_selected_by_override():
    assert T800["model"] == "t800" and T800["locomotion"] == "t800_walk" and T800["gripper"] is None
    with pytest.raises(ValueError):
        load_robot_cfg({"model": "h1"})


def test_t800_camera_mount_is_o2s_left_rgb():
    """T800 head camera = EngineAI O2S LEFT RGB optical frame (third_party/engineai/SOURCE.md): in LINK_HEAD_YAW at
    (0.0919, 0.0255, 0.0637) m (SOURCE.md rounds to 0.1 mm -> tolerance 0.05 mm), optical axis 0.13963 rad = 8.0 deg
    below the head +x axis, image x = head right (-y), no roll."""
    from g1_barcode_sim.camera import quat_wxyz_to_R
    cam = load_all("s1_table", {"robot": {"model": "t800"}})["camera"]
    base = load_camera_cfg()
    for k in ("width", "height", "focal_length", "horizontal_aperture", "clipping_range", "depth"):
        assert cam[k] == base[k]                             # intrinsics / depth settings kept
    assert cam["prim_parent_link"] == "LINK_HEAD_YAW" and cam["offset_convention"] == "ros"
    assert np.allclose(cam["offset_pos"], [0.0919, 0.0255, 0.0637], atol=5e-5)
    R = quat_wxyz_to_R(cam["offset_rot_wxyz"])
    z = R[:, 2]
    assert np.degrees(np.arctan2(-z[2], np.hypot(z[0], z[1]))) == pytest.approx(np.degrees(0.13963), abs=1e-9)
    assert abs(z[1]) < 1e-12 and np.allclose(R[:, 0], [0, -1, 0], atol=1e-12)
    assert "head_camera" not in load_robot_cfg()             # G1 keeps camera_head.yaml (no robot-file mount)


@pytest.mark.skipif(not HAVE_VENDOR, reason="vendor files absent")
def test_urdf_fk_head_chain():
    """URDF FK used by the post-spawn mount check: LINK_BASE -> J12_TORSO_YAW (z 0.1069647) -> J41_HEAD_PITCH
    (-0.0113, 0, 0.3868533) -> J42_HEAD_YAW (z 0.0400713); head pitch about +y (positive = nose down)."""
    from g1_barcode_sim.t800_robot import _sdk, urdf_fk
    urdf = _sdk(T800, T800["asset"]["urdf"])
    root, T = urdf_fk(urdf, "LINK_HEAD_YAW", {"J12_TORSO_YAW": 0.0, "J41_HEAD_PITCH": 0.0, "J42_HEAD_YAW": 0.0})
    assert root == "LINK_BASE" and np.allclose(T[:3, 3], [-0.0113, 0.0, 0.1069647 + 0.3868533 + 0.0400713])
    assert np.allclose(T[:3, :3], np.eye(3))
    _, T = urdf_fk(urdf, "LINK_HEAD_YAW", {"J12_TORSO_YAW": 0.0, "J41_HEAD_PITCH": 0.3, "J42_HEAD_YAW": 0.0})
    assert T[2, 0] == pytest.approx(-np.sin(0.3))            # head x axis tilted down


# ------------------------------------------------------------------ walker
class FakeView:
    def __init__(self, n):
        self.g = torch.arange(n + 6, dtype=torch.float32)[None] * 0.1        # distinct value per generalized dof
        self.c = torch.ones(1, n + 6) * 0.01

    def get_gravity_compensation_forces(self):
        return self.g

    def get_coriolis_and_centrifugal_compensation_forces(self):
        return self.c


class FakeRobot:
    def __init__(self, names, default):
        n = len(names)
        self.data = types.SimpleNamespace(
            joint_names=list(names), default_joint_pos=torch.as_tensor(default, dtype=torch.float32)[None].clone(),
            joint_pos=torch.as_tensor(default, dtype=torch.float32)[None].clone(), joint_vel=torch.zeros(1, n),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]), root_link_ang_vel_b=torch.zeros(1, 3))
        self.root_physx_view = FakeView(n)
        self.pos_targets, self.eff_targets = [], []

    def set_joint_position_target(self, t):
        self.pos_targets.append(t.clone())

    def set_joint_effort_target(self, t):
        self.eff_targets.append(t.clone())


class _Scene(dict):
    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


def make_walker(**walk_over):
    from g1_barcode_sim.t800_robot import joint_tables
    from g1_barcode_sim.t800_walker import T800Walker
    rcfg = deep_merge(T800, {"t800_walk": walk_over})
    T = joint_tables(rcfg)
    names = list(T["names"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(names))
    isaac = [names[i] for i in perm]
    robot = FakeRobot(isaac, [T["default_q"][n] for n in isaac])
    env = types.SimpleNamespace(physics_dt=float(rcfg["t800_walk"]["physics_dt"]), scene=_Scene(robot=robot),
                                sim=types.SimpleNamespace(step=lambda render: None, render=lambda: None),
                                observation_manager=types.SimpleNamespace(compute=lambda: None))
    return T800Walker(env, rcfg), robot, isaac


@pytest.mark.skipif(not HAVE_VENDOR, reason="vendor files not present")
def test_walker_mapping_and_targets():
    w, robot, isaac = make_walker()
    assert w.decimation == 5
    pol = w.pol
    assert [isaac[i] for i in w.idx43] == pol.joint_names
    # external arm command and head hold reach the targets; legs take the policy output
    w.robot_dds.positions[15:29] = list(np.linspace(-0.3, 0.3, 14))
    w.step_policy()
    assert len(robot.pos_targets) == 5 and len(robot.eff_targets) == 5
    tgt = robot.pos_targets[-1][0].numpy()
    by = {n: tgt[i] for i, n in enumerate(isaac)}
    for k, n in enumerate(w.arm14):
        assert by[n] == pytest.approx(np.linspace(-0.3, 0.3, 14)[k], abs=1e-6)
    for n, v in T800["t800_walk"]["head_hold"].items():
        assert by[n] == pytest.approx(v)
    legs = [n for n in pol.active_joint_names if n not in w.arm14]
    for n in legs:
        assert by[n] == pytest.approx(w.last_out.q_des[pol.joint_names.index(n)], abs=1e-6)
    # feed-forward only on the 14 arm joints and the held head joints, = gravity + coriolis compensation of that dof
    # (column j + 6)
    eff = robot.eff_targets[-1][0].numpy()
    view = robot.root_physx_view
    ff_joints = set(w.arm14) | set(T800["t800_walk"]["head_hold"])
    assert T800["t800_walk"]["head_ff"] == T800["t800_walk"]["arm_ff"] == "gravity_coriolis"
    for i, n in enumerate(isaac):
        exp = float(view.g[0, i + 6] + view.c[0, i + 6]) if n in ff_joints else 0.0
        assert eff[i] == pytest.approx(exp, abs=1e-6)


@pytest.mark.skipif(not HAVE_VENDOR, reason="vendor files not present")
def test_feedforward_modes_per_group():
    w, robot, isaac = make_walker(head_ff="gravity", arm_ff="none")
    w.step_policy()
    eff = robot.eff_targets[-1][0].numpy()
    view = robot.root_physx_view
    for i, n in enumerate(isaac):
        exp = float(view.g[0, i + 6]) if n in T800["t800_walk"]["head_hold"] else 0.0
        assert eff[i] == pytest.approx(exp, abs=1e-6)
    w2, robot2, _ = make_walker(head_ff="none", arm_ff="none")
    w2.step_policy()
    assert robot2.eff_targets == []
    with pytest.raises(ValueError):
        make_walker(head_ff="spring")


@pytest.mark.skipif(not HAVE_VENDOR, reason="vendor files not present")
def test_arm_spoof_observation():
    w, robot, isaac = make_walker()
    # move the real arm joints: with spoof the policy must not see it
    for n in w.arm14:
        robot.data.joint_pos[0, isaac.index(n)] += 0.5
    q, dq, _, _ = w.observe()
    assert np.allclose(q[w.pol_arm_m], w.pol.default_q[w.pol_arm_m]) and np.allclose(dq[w.pol_arm_m], 0)
    w.infer()
    out = w.last_out
    assert np.allclose(w.virt_q, out.q_des[w.pol_arm_m])
    assert np.allclose(w.virt_dq, (out.q_des[w.pol_arm_m] - w.pol.default_q[w.pol_arm_m]) / w.pol.control_dt)
    wr, robot_r, isaac_r = make_walker(arm_obs="real")
    for n in wr.arm14:
        robot_r.data.joint_pos[0, isaac_r.index(n)] += 0.5
    q, _, _, _ = wr.observe()
    assert np.allclose(q[wr.pol_arm_m], wr.pol.default_q[wr.pol_arm_m] + 0.5, atol=1e-6)


@pytest.mark.skipif(not HAVE_VENDOR, reason="vendor files not present")
def test_policy_owned_arms_have_no_feedforward():
    w, robot, isaac = make_walker(arm_owner="policy", head_ff="none")
    w.robot_dds.positions[15:29] = [1.0] * 14
    w.step_policy()
    assert robot.eff_targets == []
    tgt = robot.pos_targets[-1][0].numpy()
    for n in w.arm14:
        assert tgt[isaac.index(n)] == pytest.approx(w.last_out.q_des[w.pol.joint_names.index(n)], abs=1e-6)


@pytest.mark.skipif(not HAVE_VENDOR, reason="vendor files not present")
def test_velocity_outside_stick_range_raises():
    w, _, _ = make_walker()
    w.set_velocity(0.5, 0.4, 1.0)
    with pytest.raises(ValueError):
        w.set_velocity(0.0, 0.5, 0.0)
