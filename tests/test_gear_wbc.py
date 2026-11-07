"""Offline checks of the GR00T walker (no Isaac Sim): joint mapping, 86-dim frame layout, zero-padded history,
Balance/Walk switch, target writing. Fake robot with the joint order of our G1-29 + Dex1 articulation.
run: python -m pytest tests/test_gear_wbc.py"""
import os
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1_barcode_sim.config import load_robot_cfg  # noqa: E402
from g1_barcode_sim.gear_wbc import G29, GearWbcWalker  # noqa: E402

# Isaac Lab joint order of our articulation (log isaaclab_2026-11-05_23-09-40.log:63)
ISAAC_NAMES = [None] * 33
for g, i in zip(G29, [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]):
    ISAAC_NAMES[i] = g
ISAAC_NAMES[29:] = ["left_hand_Joint1_1", "left_hand_Joint2_1", "right_hand_Joint1_1", "right_hand_Joint2_1"]


class FakeRobot:
    def __init__(self):
        n = len(ISAAC_NAMES)
        self.data = types.SimpleNamespace(
            joint_names=list(ISAAC_NAMES), default_joint_pos=torch.zeros(1, n), joint_pos=torch.zeros(1, n),
            joint_vel=torch.zeros(1, n), root_link_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]))
        self.targets = []

    def set_joint_position_target(self, t):
        self.targets.append(t.clone())


class FakeEnv:
    physics_dt = 0.005

    def __init__(self):
        self.robot = FakeRobot()
        self.scene = _Scene(self.robot)
        self.sim = types.SimpleNamespace(step=lambda render: None, render=lambda: None)
        self.observation_manager = types.SimpleNamespace(compute=lambda: None)


class _Scene(dict):
    def __init__(self, robot):
        super().__init__(robot=robot)

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


@pytest.fixture
def walker():
    return GearWbcWalker(FakeEnv(), load_robot_cfg()["gear_wbc"])


def test_mapping(walker):
    assert [ISAAC_NAMES[i] for i in walker.idx29] == G29
    assert list(walker.idx15) == [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8]


def test_frame_layout(walker):
    r = walker.robot.data
    r.joint_pos[0, walker.idx29[3]] = 0.5          # left knee
    r.joint_vel[0, walker.idx29[20]] = 2.0         # left wrist pitch
    r.root_link_ang_vel_b[0] = torch.tensor([0.2, 0.0, 0.0])
    walker.set_velocity(0.1, -0.2, 0.4)
    f = walker.build_frame()
    assert f.shape == (86,)
    np.testing.assert_allclose(f[0:3], [0.2, -0.4, 0.2], atol=1e-6)        # x cmd_scale [2, 2, 0.5]
    assert f[3] == pytest.approx(0.74) and np.all(f[4:7] == 0)
    np.testing.assert_allclose(f[7:10], [0.1, 0, 0], atol=1e-6)            # x 0.5
    np.testing.assert_allclose(f[10:13], [0, 0, -1])
    assert f[13 + 3] == pytest.approx(0.5 - 0.3)                            # q - default (knee default 0.3)
    assert f[13 + 0] == pytest.approx(0.1)                                  # hip pitch 0 - (-0.1)
    assert f[42 + 20] == pytest.approx(0.1)                                 # dq x 0.05
    assert np.all(f[71:86] == 0)


def test_history_and_switch(walker):
    walker.set_gripper_joint_targets(0.0, 0.0)
    walker.set_velocity(0.0, 0.0, 0.04)
    walker.step_policy()
    assert walker.active == "balance"
    assert all(np.all(h == 0) for h in list(walker.history)[:-1])         # zero frames left-padded
    a1 = walker.last_raw.copy()
    walker.set_velocity(0.05, 0.0, 0.0)
    walker.step_policy()
    assert walker.active == "walk"                                         # ||cmd|| == 0.05 is not < 0.05
    np.testing.assert_array_equal(walker.history[-1][71:86], a1)          # frame carries the previous raw output
    walker.reset()
    assert all(np.all(h == 0) for h in walker.history) and np.all(walker.last_action == 0)


def test_targets(walker):
    walker.set_gripper_joint_targets(0.01, -0.02)
    walker.robot_dds.positions[15 + 3] = 0.9                               # left elbow
    walker.step_policy()
    t = walker.robot.targets
    assert len(t) == 4                                                     # control_decimation substeps
    t = t[-1][0].numpy()
    np.testing.assert_allclose(t[walker.idx15], walker.last_raw * 0.25 + walker.default15, atol=1e-6)
    assert t[walker.idx29[18]] == pytest.approx(0.9)
    assert t[ISAAC_NAMES.index("left_hand_Joint1_1")] == pytest.approx(0.01)
    assert t[ISAAC_NAMES.index("right_hand_Joint2_1")] == pytest.approx(-0.02)


def test_body_command_in_frame_bounds_and_reset(walker):
    """height / torso commands land at frame indices 3 and 4-6; out-of-range values raise; reset() stands up."""
    walker.set_body_command(0.45, [0.0, 0.4, 0.0])
    f = walker.build_frame()
    assert f[3] == pytest.approx(0.45) and f[4:7] == pytest.approx([0.0, 0.4, 0.0])
    lo, hi = walker.height_range
    with pytest.raises(ValueError):
        walker.set_body_command(lo - 0.01)
    with pytest.raises(ValueError):
        walker.set_body_command(None, [0.0, float(walker.rpy_limit[1]) + 0.01, 0.0])
    assert walker.height_cmd == pytest.approx(0.45)             # a rejected command changes nothing
    walker.reset()
    assert walker.height_cmd == pytest.approx(walker.height_default) and walker.rpy_cmd == pytest.approx(walker.rpy_default)
