(robot)=
# Robot and end effectors

- **Model**: Unitree G1 with 29 body degrees of freedom (6 per leg, 3 in the waist, 7 per arm); each hand has a Unitree Dex1 two-finger parallel gripper with 2 prismatic joints; 33 joints in total.
- **Model source**: `assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd` from unitree_sim_isaaclab (configuration `G129_CFG_WITH_DEX1_WHOLEBODY`), read from the upstream repository at runtime. Self-collision is disabled.
- **End effectors**: `configs/robot.yaml: end_effector` currently supports only `dex1`; selecting `dex3` or `inspire` raises `NotImplementedError`.
- **Grippers**: joint targets are −0.02 for open and 0.024 for closed (Isaac Lab joint values). The API's `opening` ∈ [0, 1] (0 closed, 1 open) maps linearly to these two values. The actuators use the upstream implicit PD.
- **Scanner in the left hand**: the scanner is rigidly attached to a `scanner_mount` node under the left hand's `left_hand_base_link`, with the grip point between the two finger pads. At startup, the joint target at which both finger pads just touch the 3.0 cm wide handle is computed from the live positions of the finger-pad colliders; it is written directly on reset, so no re-grasping is needed. The left hand holding the scanner ignores gripper commands: `set_gripper("left", …)` returns an error, and `set_gripper("both", …)` acts on the right hand only.
- **Without the scanner**: set `configs/scanner.yaml: enabled: false`, or pass `config_overrides={"scanner": {"enabled": False}}` to the constructor. The left gripper is then controllable again, no scanner camera is created, and `scan()` and `aim_scanner()` return an error.

```{figure} ../../images/s1_table_scanner_closeup_reach.jpg
:alt: Close-up of the scanner: the Dex1 gripper holds the gray handle, the scanner head points up, the yellow nose and dark exit window face forward
:width: 640px

Scanner close-up: the handle is held between the two Dex1 fingers with the head on top; the exit window sits in the middle of the yellow nose (the scan direction is along the fingers).
```
