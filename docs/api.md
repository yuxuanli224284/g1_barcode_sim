(api)=
# Python API：`G1BarcodeEnv`

所有接口都在 `g1_barcode_sim/env.py` 的 `G1BarcodeEnv` 类中（也可以 `from g1_barcode_sim import G1BarcodeEnv`，按需导入）。一个进程只能创建一个环境，`close()` 会结束整个进程。

(api-quick)=
## 快速开始

```python
import math
import numpy as np
from g1_barcode_sim import G1BarcodeEnv
from g1_barcode_sim.camera import inv_T

# 内置场景必须显式传 scene；不传时默认 "scenario"（测试场景模式）
env = G1BarcodeEnv(scene="s1_table", headless=True, seed=0)
obs = env.reset()
print(obs["rgb"].shape, obs["depth"].shape)       # (480, 640, 3) (480, 640)

r = env.step(0.3, 0.0, 0.0, duration=1.0)         # root 系下 0.3 m/s 前进 1 s，结束后速度清零
print(r["status"], r["reason"])                   # ok / fell / collision

r = env.move_relative(0.3, 0.0, 0.0, timeout=20.0)  # 基于里程计的闭环，再前移 0.3 m
print(r["status"], r["gt_error"])

r = env.set_body_pose(height=0.50)                # 下蹲到 0.50 m 高度命令（GR00T 控制器）
print(r["pelvis_z"], r["tilt_deg"])
env.set_body_pose(height=0.74)                    # 站起来

# 扫码（示例直接用 GT 站位；实际算法应从头部相机估计标签位姿）
bc = env.get_ground_truth()["barcodes"][0]
c, n = np.array(bc["center"]), np.array(bc["normal"])
env.teleport_root(c[0] + 0.55 * n[0], c[1] + 0.55 * n[1] - 0.20, math.atan2(-n[1], -n[0]))
env.stand_still(0.5)
T = inv_T(env.get_obs()["T_world_root_gt"])       # world -> root
env.aim_scanner_at(T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ n, distance=0.20)
env.stand_still(3.0)                              # 等瞄准收敛
res = env.scan()
print(res["status"], res["content"])
env.close()                                       # 结束整个进程
```

测试场景模式：

```python
from g1_barcode_sim import G1BarcodeEnv

env = G1BarcodeEnv(headless=True)                 # scene 默认为 "scenario"
obs = env.reset(scenario="t1_L0_0000")            # 场景 id、JSON 路径或 dict 均可
gt = env.get_ground_truth()                       # 额外含 target_box_id、task1_gt_obs_pose 等（仅评测）
obs = env.reset()                                 # 不带参数：重开当前场景
obs = env.reset(scenario="stack4f_L0_0001")       # 直接切换到另一个测试场景，无需新进程
```

(api-methods)=
## 方法一览

```{list-table}
:header-rows: 1
:widths: 45 55

* - 方法（签名与 `env.py` 一致）
  - 说明与返回
* - `G1BarcodeEnv(scene: str = "scenario", headless: bool = False, seed: int = 0, config_overrides: dict | None = None)`
  - `scene`：`s1_table` / `s2_shelf` / `s3_stack` 或 `"scenario"`。`config_overrides` 的顶层键为 `scene` / `barcode` / `robot` / `camera` / `scanner`，按 `_common.yaml` ← `<scene>.yaml` ← overrides 的顺序深度合并
* - `reset(seed: int | None = None, scenario: dict | str | None = None) -> dict`
  - 返回首帧观测。内置场景：机器人回到起点、关节回默认，按 seed 重新放置标签（开启随机化时），里程计和仿真时间清零；传 `scenario` 会报错。测试场景模式：重建桌子、箱子、标签和背景，机器人放到场景起始位姿；首次必须传 `scenario`
* - `get_obs() -> dict`
  - 观测字典，见{ref}`api-obs`
* - `step(vx: float, vy: float, wz: float, duration: float) -> dict`
  - root 系速度命令，持续 `duration` 秒后清零。返回 `{status: ok|fell|collision, reason, obs, command}`
* - `stand_still(duration: float) -> dict`
  - 零速度站立，返回格式同 `step`
* - `move_relative(dx: float, dy: float, dyaw: float, timeout: float = 10.0, pos_tol: float | None = None, yaw_tol: float | None = None, speed_scale: float = 1.0) -> dict`
  - 里程计闭环相对移动（起始 root 系）。返回 `{status: ok|timeout|fell|collision, reason, elapsed_s, requested, odom_displacement, gt_displacement, gt_error{pos_m, yaw_rad}, odom_error_vs_gt}`
* - `set_body_pose(height: float | None = None, torso_rpy=None, duration: float = 1.0, settle: float = 1.0) -> dict`
  - 原地下蹲 / 倾斜躯干（仅 GR00T 控制器）。返回 `{status, reason, height_cmd, rpy_cmd, pelvis_z, tilt_deg, obs}`
* - `set_gripper(side: str, opening: float) -> dict`
  - `side`：`left` / `right` / `both`；`opening` 0 闭合 … 1 张开。只设目标，夹爪在后续步进中运动。返回 `{status, side, applied_to, opening, joint_target, reason}`
* - `set_wrist_target(side: str, pos_root) -> dict`
  - 手腕位置目标（root 系），每步由 IK 跟踪。返回 `{status, target_root}`
* - `move_wrist(side: str, dx: float, dy: float, dz: float) -> dict`
  - 在当前目标（或默认手腕位置）基础上平移
* - `reset_arms() -> dict`
  - 双臂回默认姿态，清除手腕和瞄准目标
* - `aim_scanner(tip_pos_root, dir_root, anchor: str = "odom") -> dict`
  - 枪口放到 `tip_pos_root`，扫描方向沿 `dir_root`（调用时的 root 系）。返回 `{status, side, anchor, tip_pos_root, dir_root}`
* - `aim_scanner_at(point_root, normal_root, distance: float = 0.20, anchor: str = "odom") -> dict`
  - 枪口放在标签中心沿外法向 `distance` 处，垂直对准
* - `scan() -> dict`
  - 扣扳机，见{ref}`scan`
* - `get_ground_truth(d: float = 0.20) -> dict`
  - GT（仅评测），见{ref}`gt`
* - `teleport_root(x: float, y: float, yaw: float) -> None`
  - 调试用瞬移：关节回默认（持枪手保持夹紧），手臂目标清除，里程计重置为真值
* - `render_aux(T_world_cam_ros, hfov_deg: float | None = None) -> dict`
  - 辅助相机在任意位姿渲染，返回 `{rgb, depth, K}`，不推进物理。位姿可用 `g1_barcode_sim.camera.look_at_ros(eye, target)` 生成
* - `topdown() -> dict` / `oblique_views() -> dict`
  - 俯视图（图像右方为 +x、上方为 +y）/ 场景 YAML `views` 定义的斜视图
* - `save_snapshot(path: str) -> None`
  - 写出 `head_rgb.png`、`head_depth.png`（伪彩色）、`head_depth.npy`、`wrist_rgb.png`、`wrist_depth.png`、`topdown.png`、`view_*.png`
* - `camera_rgb(name: str)`
  - 取单个相机的 RGB：`"head"`、`"wrist"` 或录像相机名
* - `close() -> None`
  - 关闭环境并结束进程
```

常用属性：`env.policy_dt`（0.02 s）、`env.layout`（当前场景布局）、`env.cfg`（合并后的配置）、`env.scanner_side`、`env.arm`（手臂 IK 控制器）、`env.step_hooks`（每个策略步后调用的回调列表，可用于录像）。

(api-obs)=
## 观测字典

| 键 | 说明 |
| --- | --- |
| `rgb` | 头部相机，480×640×3 uint8 |
| `depth` | 头部深度，480×640 float32，单位 m，0.2–3.0 m 之外为 0 |
| `K` | 头部相机内参 3×3 |
| `T_root_cam` | 头部相机光学系在 root 下的位姿（由当前关节角算出） |
| `T_world_root_odom` | 里程计位姿（带漂移） |
| `T_world_root_gt` | 真值位姿，**仅评测** |
| `T_root_scanner_mount` | 扫码枪 G 系在 root 下的位姿 |
| `gripper_state` | `{left, right}`，开度 0–1 |
| `wrist_pos_root` / `wrist_target_root` | 手腕位置 / IK 目标（无目标时为 `None`） |
| `timestamp` | 仿真时间（s） |
| `wrist_rgb`、`wrist_depth`、`wrist_K` | 枪头相机图像、深度（沿光轴，m）和内参（装枪时才有，下同） |
| `T_root_wrist_cam`、`T_root_scanner_tip` | 枪头相机光学系、枪口（x 轴 = 扫描方向）在 root 下的位姿 |
| `scanner_aim_target`、`scanner_aim_error` | 当前瞄准目标 `{pos, dir}`（已换算到当前 root 系）与误差 `{pos_m, angle_deg}`；未瞄准时为 `None` |

(api-teleop)=
## 键盘控制

`scripts/teleop.py` 用于手动检查底盘、手腕、夹爪和扫码枪，可选录像。有显示器时在 OpenCV 窗口里直接按键；没有显示器（或加 `--headless`）时，在终端输入字母后按回车。

```bash
python scripts/teleop.py --scene s1_table                                  # 交互控制
python scripts/teleop.py --scene s1_table --record outputs/my_run.mp4      # 交互控制 + 录像
python scripts/teleop.py --scene s1_table --demo --record outputs/demo.mp4 # 内置控制测试序列
python scripts/teleop.py --scene s1_table --demo_scanner --record outputs/scanner_demo.mp4  # 扫码枪测试序列
python scripts/teleop.py --scene s1_table --policy policy1.onnx            # 改用 unitree 上游行走策略
```

| 按键 | 作用 |
| --- | --- |
| {kbd}`w` / {kbd}`s`、{kbd}`a` / {kbd}`d`、{kbd}`q` / {kbd}`e`、空格 | 前进 / 后退、左移 / 右移、左转 / 右转、停止。每次按键持续 `--hold` 秒（默认 0.5 s），速度取限幅值 |
| {kbd}`1` / {kbd}`2` / {kbd}`3` | 后续手腕按键作用于左臂 / 右臂 / 双臂（默认双臂） |
| {kbd}`i` / {kbd}`k`、{kbd}`j` / {kbd}`l`、{kbd}`u` / {kbd}`o` | 手腕在 root 系下 ±x（前后）、±y（左右）、±z（上下）移动，每次 `--wrist_step`（默认 3 cm） |
| {kbd}`r` | 手臂回默认姿态 |
| {kbd}`g` / {kbd}`h` | 夹爪张开 / 闭合（只作用于不持枪的右手） |
| {kbd}`f` | 瞄准模式开 / 关。打开时枪保持当前位姿；选了 {kbd}`1` 或 {kbd}`3` 时，{kbd}`i` {kbd}`k` {kbd}`j` {kbd}`l` {kbd}`u` {kbd}`o` 移动的是枪口 |
| {kbd}`t` / {kbd}`v`、{kbd}`c` / {kbd}`b` | 瞄准方向上 / 下、左 / 右，每次 `--aim_step_deg`（默认 5°） |
| {kbd}`n` | 扫码（结果打印在终端并显示在画面上） |
| {kbd}`p` / {kbd}`x` | 保存快照到 `outputs/teleop/<scene>/` / 退出 |

`scripts/run_scene.py` 是更简单的场景查看器：{kbd}`W`/{kbd}`S` 前后，{kbd}`A`/{kbd}`D` 平移，{kbd}`Q`/{kbd}`E` 转向，空格停止，{kbd}`O`/{kbd}`C` 张开 / 闭合夹爪（持枪手不动），{kbd}`P` 保存快照，{kbd}`X` 退出。headless 时每 `--save_every` 秒（仿真时间）把头部图像写到 `outputs/run_scene/<scene>/head_latest.png`；摔倒或碰撞时自动 reset。

(api-scripts)=
## 脚本一览

```{list-table}
:header-rows: 1
:widths: 40 45 15

* - 脚本
  - 用途
  - 需要 Isaac Sim
* - `scripts/run_scene.py --scene S [--seed N] [--randomize] [--max_sim_s T]`
  - 打开内置场景，键盘控制
  - 是
* - `scripts/teleop.py`
  - 底盘 + 手腕 + 夹爪 + 扫码枪手动控制、测试序列、录像
  - 是
* - `scripts/check_env.py [--scenes ...]`
  - 端到端自检，报告写到 `outputs/check_report.md`
  - 是
* - `scripts/render_views.py [--scenes ...]`
  - 渲染俯视图、斜视图和头部相机图到 `outputs/<scene>/`
  - 是
* - `scripts/record_walk.py --scene S [--out F] [--fps 25]`
  - 按固定动作序列行走并录像（第三人称 + 头部相机画中画）
  - 是
* - `scripts/generate_barcodes.py [--type code128|qr] [--scenes ...]`
  - 批量生成内置场景的标签贴图并解码校验
  - 否
* - `scripts/testsuite/list.py` / `show.py` / `generate.py`
  - 列出、查看、生成测试场景集（见{ref}`suite-tools`）
  - 否
* - `scripts/testsuite/preview.py`
  - 渲染测试场景预览图
  - 是
* - `scripts/testsuite/measure_head_camera.py`
  - 重新测量头部相机标称参数（只在改了相机或机器人后需要）
  - 是
* - `examples/goto_barcode.py`、`examples/scanner_camera.py`
  - 最小示例：用 `move_relative` 走到条码参考站位；瞄准并扫码
  - 是
```
