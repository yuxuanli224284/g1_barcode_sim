(robot)=
# 机器人与末端

- **型号**：Unitree G1，本体 29 个自由度（每条腿 6、腰 3、每条手臂 7）；两只手各一个 Unitree Dex1 两指平行夹爪，每个夹爪 2 个棱柱关节；整机 33 个关节。
- **模型来源**：unitree_sim_isaaclab 的 `assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd`（配置 `G129_CFG_WITH_DEX1_WHOLEBODY`），运行时从上游仓库读取。不开启自碰撞。
- **末端执行器**：`configs/robot.yaml: end_effector` 目前只接通了 `dex1`，选 `dex3` 或 `inspire` 会抛出 `NotImplementedError`。
- **夹爪**：关节目标张开 −0.02、闭合 0.024（Isaac Lab 关节值）。API 的开度 `opening` ∈ [0, 1]（0 闭合、1 张开）线性映射到这两个值。执行器沿用上游 implicit PD。
- **左手持枪**：扫码枪刚性挂在左手 `left_hand_base_link` 下的 `scanner_mount` 节点上，握持点在两指垫中间。启动时根据指垫碰撞体的实时位置算出"两侧指垫正好贴住 3.0 cm 宽枪柄"的关节目标，reset 时直接写入，不需要重新夹取。持枪的左手不响应夹爪指令：`set_gripper("left", …)` 返回 error，`set_gripper("both", …)` 只作用于右手。
- **不装枪**：`configs/scanner.yaml: enabled: false`，或构造时传 `config_overrides={"scanner": {"enabled": False}}`。这时左夹爪恢复可控，不创建枪头相机，`scan()` 和 `aim_scanner()` 返回 error。

```{figure} ../images/s1_table_scanner_closeup_reach.jpg
:alt: 扫码枪特写：Dex1 夹爪握住灰色枪柄，枪头朝上，黄色枪鼻和深色出光窗朝前
:width: 640px

扫码枪特写：枪柄夹在 Dex1 两指之间，枪头在上，黄色枪鼻中间是出光窗（扫描方向沿手指方向）。
```
