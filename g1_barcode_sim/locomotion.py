"""In-process walking controller that reuses unitree_sim_isaaclab's Wholebody policy unchanged.

Upstream `DDSRLActionProvider` reads three things from DDS: the run command (vx, vy, wz, height),
the arm joint command, and the gripper command. We subclass it and replace only `_setup_dds` with
local stand-ins exposing the same getter methods, so the observation building, ONNX policy
inference, action delay buffer and the 4x physics sub-stepping are the upstream code, verbatim.

Must be imported after the Isaac Sim app is running (it imports isaaclab / the unitree repo).
"""
from __future__ import annotations

import os
import sys

# the upstream unitree_sim_isaaclab checkout: $UNITREE_SIM_ROOT, else a sibling directory of this repository
UNITREE_ROOT = os.environ.get("UNITREE_SIM_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "unitree_sim_isaaclab")
if not os.path.isdir(os.path.join(UNITREE_ROOT, "action_provider")):
    raise FileNotFoundError(f"unitree_sim_isaaclab not found at {UNITREE_ROOT}: clone it (see README) and set UNITREE_SIM_ROOT, "
                            "or place it next to this repository")
os.environ.setdefault("PROJECT_ROOT", UNITREE_ROOT)
if UNITREE_ROOT not in sys.path:
    sys.path.insert(0, UNITREE_ROOT)

from action_provider.action_provider_wh_dds import DDSRLActionProvider  # noqa: E402


class _RunCommand:
    """Stands in for the `run_command` DDS object. Upstream resets the command to zero after every
    read (a DDS publisher must keep re-sending); here the last command is simply held."""

    def __init__(self, height: float):
        self.cmd = [0.0, 0.0, 0.0, height]

    def get_run_command(self):
        return {"run_command": list(self.cmd)}

    def write_run_command(self, _cmd):
        pass


class _ArmCommand:
    """Stands in for the `g129` lowcmd DDS object: 29 motor positions, arms at motor index 15..28."""

    def __init__(self):
        self.positions = None  # filled by the walker once the joint mapping is known

    def get_robot_command(self):
        if self.positions is None:
            return None
        return {"motor_cmd": {"positions": list(self.positions)}}


class _GripperCommand:
    """Stands in for the `dex1` DDS object. Values are Isaac Lab joint targets (upstream converts
    the DDS 0..5.4 range to joint angles before this point)."""

    def __init__(self):
        self.left = None
        self.right = None

    def get_gripper_command(self):
        if self.left is None:
            return None
        return {"left_gripper_cmd": {"positions": [self.left]}, "right_gripper_cmd": {"positions": [self.right]}}


class _Args:
    def __init__(self, model_path: str):
        self.robot_type = "g129"
        self.enable_dex1_dds = True
        self.enable_dex3_dds = False
        self.enable_inspire_dds = False
        self.enable_wholebody_dds = True
        self.model_path = model_path


class InProcessWholebodyWalker(DDSRLActionProvider):
    """`step_policy()` advances physics by one policy step (decimation x physics_dt)."""

    def __init__(self, env, model_path: str = "assets/model/policy.onnx", height_cmd: float = 0.8):
        self._height_cmd = height_cmd
        super().__init__(env, _Args(model_path))
        # Hold the arms at the robot's default joint positions (the upstream arm command is absolute).
        default = self.env.scene["robot"].data.default_joint_pos[0]
        positions = [0.0] * 29
        for name, idx in self.arm_joint_mapping.items():
            positions[idx + 15] = float(default[self.joint_to_index[name]])
        self.robot_dds.positions = positions

    def _setup_dds(self):  # override: no DDS, local stand-ins with the same interface
        self.robot_dds = _ArmCommand()
        self.gripper_dds = _GripperCommand()
        self.run_command_dds = _RunCommand(self._height_cmd)

    # --- public control surface -------------------------------------------------------------
    def set_velocity(self, vx: float, vy: float, wz: float):
        self.run_command_dds.cmd[:3] = [float(vx), float(vy), float(wz)]

    def set_gripper_joint_targets(self, left: float, right: float):
        self.gripper_dds.left, self.gripper_dds.right = float(left), float(right)

    def load_extra_policy(self, model_path: str):
        """Load another ONNX policy with the same observation / action layout (upstream loader)."""
        return self.load_policy(f"{UNITREE_ROOT}/{model_path}")

    def step_policy(self):
        self.get_action(self.env)
