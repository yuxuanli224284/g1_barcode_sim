# 行走与手臂控制

(wbc)=
## 全身控制与行走

### GR00T Decoupled WBC（默认）

下肢和腰部由 NVIDIA GR00T Decoupled WBC 的两个 ONNX 策略控制：Balance（站立）和 Walk（行走），文件在 `assets/gear_wbc/`，来自 [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)，未做修改。策略在 CPU 上用 onnxruntime 推理。

- **切换**：速度命令的模长 ‖(vx, vy, wz)‖ < 0.05 时用 Balance，否则用 Walk，自动切换；两个网络共用同一份观测历史和上一步动作。
- **频率**：物理步长 0.005 s（200 Hz）；decimation 4，策略、手臂 IK 和相机渲染都是 50 Hz（`env.policy_dt = 0.02`）。启动时检查物理步长与 GR00T 参考配置一致。
- **输出**：`[1, 15]`，12 个腿关节 + 3 个腰关节的位置目标：目标 = 输出 × 0.25 + 默认角。默认角为髋 pitch −0.1、膝 0.3、踝 pitch −0.2，其余 0 rad。
- **执行器**：这 15 个关节用一组显式 PD（IdealPD），kp/kd 取自 GR00T 参考配置：髋 150/2、膝 200/4、踝 40/2、腰 250/5。力矩上限（N·m）：髋 88、膝 139、踝 50、腰 yaw 88 / roll 50 / pitch 50。

**输入**为 `[1, 516]`：最近 6 帧观测，每帧 86 维，从旧到新排列，reset 后用零帧补齐。每帧的组成：

| 分量 | 维数 | 缩放 |
| --- | --- | --- |
| 速度命令 (vx, vy, wz) | 3 | × [2, 2, 0.5] |
| 身体高度命令 | 1 | 原值 |
| 躯干 roll / pitch / yaw 命令 | 3 | 原值 |
| 骨盆角速度 | 3 | × 0.5 |
| 重力方向投影 | 3 | 原值 |
| 29 个关节角 − 默认角 | 29 | 原值 |
| 29 个关节速度 | 29 | × 0.05 |
| 上一步策略原始输出 | 15 | 原值 |

(body-pose)=
### 下蹲与躯干姿态

`env.set_body_pose(height=None, torso_rpy=None, duration=1.0, settle=1.0)` 让机器人原地下蹲或倾斜躯干：在 `duration` 秒内把高度和躯干命令线性过渡到目标，再保持 `settle` 秒，每个策略步检查摔倒和碰撞。高度命令范围 0.20–0.74 m（默认 0.74 m 为站立）；躯干命令上限 |roll| ≤ 0.4、|pitch| ≤ 0.6、|yaw| ≤ 0.8 rad。超出范围直接报错，不会悄悄截断。reset 或 teleport 后恢复站立。

| 高度命令 (m) | 0.74 | 0.60 | 0.50 | 0.40 | 0.30 | 0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| 实测骨盆高度 (m) | 0.747 | 0.563 | 0.471 | 0.379 | 0.281 | 0.249 |

实测条件：原地，每级 1 s 过渡 + 2 s 保持，每级一次。各级都没有摔倒、双脚没有滑动；蹲得越深骨盆前倾越大（最大约 26°）。0.20 m 这一级没有测过。躯干 pitch 命令 0.2 / 0.4 / 0.6 rad 时实测躯干前倾约 12° / 27° / 43°（0.6 时腰 pitch 到达关节限位）。

(base-motion)=
### 底盘移动

- `step(vx, vy, wz, duration)`：以 root（骨盆）坐标系下的速度走 `duration` 秒，结束后速度命令清零。命令先截断到 |vx| ≤ 0.4 m/s、|vy| ≤ 0.5 m/s、|wz| ≤ 1.0 rad/s。每个策略步检查一次状态，摔倒或碰撞时提前结束。命令在进程内一直保持，不像上游 DDS 接口那样读一次就清零。
- `stand_still(duration)`：等于 `step(0, 0, 0, duration)`。
- `move_relative(dx, dy, dyaw, timeout=10.0, pos_tol=None, yaw_tol=None, speed_scale=1.0)`：基于里程计的闭环移动，目标在起始 root 坐标系下给出。控制律是饱和 P 控制：误差大于制动距离（0.15 m / 0.30 rad）时用满限幅，之内线性减速；某一轴误差还在容差外时，命令不低于最小值（vx 0.15、vy 0.3、wz 0.6），避免低速时走不动。每轮结束先停稳（速度 < 0.03 m/s 持续 0.3 s，最多等 3 s）再复查误差，最多 3 轮。默认容差 3 cm / 0.03 rad。
- **里程计**：`T_world_root_odom` 不是状态估计器的输出，而是在真值上叠加确定性漂移：每米 ±2% 位移误差、每米 ±1° 航向误差，正负号由 reset 的 seed 决定；roll / pitch 用真值。
- **状态判定**：躯干倾斜超过 45° 判为 `fell`；机器人碰撞体（含扫码枪外形）的任一顶点进入家具或纸箱时判为 `collision`。`collision` 只是状态报告，不会阻止物理接触；间隙只按顶点计算，是上界。

(unitree-policy)=
### 备选：unitree 上游行走策略

设 `configs/robot.yaml: locomotion: unitree`（或 `teleop.py --policy policy1.onnx`）改用 unitree_sim_isaaclab 自带的 Wholebody 行走策略（`assets/model/policy1.onnx` 或 `policy.onnx`，位于上游仓库）。它只控制 12 个腿关节、腰部固定，没有高度和躯干命令，所以 `set_body_pose` 不可用。实现方式是继承上游的 `DDSRLActionProvider`，只把 DDS 通信换成进程内的命令对象，观测构造、推理和子步进都是上游原代码。在本项目的测试中，GR00T 站立漂移更小、`move_relative` 到位率更高、能后退、低速命令也能执行，所以作为默认。

| 参数 | 默认值 | 配置位置 |
| --- | --- | --- |
| 行走控制器 | `gear_wbc`（可选 `unitree`） | `robot.yaml: locomotion` |
| Balance / Walk 切换阈值 | ‖(vx, vy, wz)‖ = 0.05 | `robot.yaml: gear_wbc.switch_norm` |
| 速度限幅 | vx 0.4 m/s、vy 0.5 m/s、wz 1.0 rad/s | `robot.yaml: velocity_limits` |
| 高度命令范围 | [0.20, 0.74] m | `robot.yaml: gear_wbc.height_cmd_range` |
| 躯干命令上限 | roll 0.4、pitch 0.6、yaw 0.8 rad | `robot.yaml: gear_wbc.rpy_cmd_limit_rad` |
| `move_relative` 容差 / 最多轮数 | 3 cm / 0.03 rad / 3 轮 | `robot.yaml: move_relative` |
| 摔倒判定 | 倾斜 > 45° | `robot.yaml: fall.tilt_deg` |
| 里程计漂移 | 2% / m、1° / m | `robot.yaml: odom` |
| 物理设备 | CPU | `robot.yaml: physics_device` |

(arm)=
## 手臂控制

手臂不受行走策略控制。每只手臂 7 个关节，由 IK 或关节指令直接给出位置目标，再由上游的 implicit PD 执行器跟踪。每个策略步（50 Hz）先根据当前实测关节状态更新一次 IK，再运行一次行走策略，所以 IK 是逐步闭环的。

1. **手腕位置**：`set_wrist_target(side, pos_root)`、`move_wrist(side, dx, dy, dz)`。阻尼最小二乘微分 IK，只控制 `{side}_hand_base_link`（夹爪基座）在 root 坐标系下的位置，不控制姿态。每步 dq = gain·Jᵀ(JJᵀ + λ²I)⁻¹e，gain 0.1、λ 0.05，雅可比来自 PhysX。目标被限制在默认手腕位置 ±0.25 m 的框内；够不到时停在最近的可达点。
2. **扫码枪瞄准（左臂）**：`aim_scanner(tip_pos_root, dir_root, anchor="odom")`、`aim_scanner_at(point_root, normal_root, distance=0.20, anchor="odom")`。5 维任务：枪口位置 3 维 + 扫描方向 2 维，不约束绕扫描轴的滚转；方向误差乘以 0.20 m/rad（标称扫描距离）与位置误差加权。`anchor="odom"` 时目标固定在里程计坐标系中，底盘漂移会被逐步补偿；`"root"` 时目标随身体移动。瞄准模式不受手腕框限制，残差在 `obs["scanner_aim_error"]` 中报告。
3. **关节空间**：`env.arm.set_joint_targets(side, q, n_steps)`，在 `n_steps` 个策略步内从当前指令线性插值到 7 维关节目标 `q`。`env` 上没有对应的封装方法。

通用限制：每步关节变化不超过 0.05 rad（即 2.5 rad/s），并夹到关节软限位内。`reset_arms()` 清除目标，双臂回到默认姿态（肩 pitch 0.35、肩 roll ±0.18、肘 0.87 rad，双手在髋部前方，手腕约在 root 系 (0.0, ±0.22, −0.09) m）。
