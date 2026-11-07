(limits)=
# 已知限制与许可证

## 已知限制

- **行走**：低速时后退、横移和转向的跟踪不足（只完成命令位移的 30–70%），左转与右转响应不对称；从 0.4 m/s 停下后底盘还会滑行约 10–15 cm；左臂抬起行走时航向会向右偏 2–4.5°。`move_relative` 的闭环和多轮复查可以弥补大部分误差。
- **下蹲**：高度命令下限 0.20 m 没有实测过，实测最低到 0.25 m（骨盆约 0.25 m）。
- **里程计**不是估计器，而是真值加确定性漂移（每米 2% / 1°）。
- **相机**：头部相机参数取自上游仿真配置（水平视场 105.5°，比真机 D435i 宽），枪头相机按 DS2208 规格估算，两者都没有与真机标定。
- **扫码**：`scan()` 是对渲染画面的图像解码，没有模拟真机的解码距离和角度规格；默认只解码原图，条码长轴需与画面横轴大致对齐（约 ±17° 以内）。
- **扫码枪**只有外观，没有质量和碰撞体，手臂负载比真机（约 160 g）轻；与场景的接触只在间隙检测中体现。
- **碰撞**：机器人不开启自碰撞；摔倒只按倾斜角判定；间隙只按顶点计算（上界）。仓库背景只作外观，碰撞已关闭。
- **末端**只接通了 Dex1；手腕 IK 只控制位置。
- **标签**：GT 只支持四个竖直面（顶面 `+z` 没有水平站位方向）；内置场景 GT 的 `pattern_size` 为 `null`。
- **运行**：物理在 CPU 上运行，每个策略步渲染头部、枪头和辅助相机，仿真速度明显低于实时（取决于硬件）；一个进程只能创建一个环境，`close()` 结束进程；场景依赖在线 Isaac 资产（可改用几何体）。

(license)=
## 许可证与第三方组件

```{list-table}
:header-rows: 1
:widths: 25 30 30 15

* - 组件
  - 来源
  - 许可证
  - 是否随本仓库分发
* - GR00T WBC 策略权重 `GR00T-WholeBodyControl-{Balance,Walk}.onnx`
  - [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)（`decoupled_wbc/sim2mujoco/resources/robots/g1`），未修改
  - NVIDIA Open Model License。Licensed by NVIDIA Corporation under the NVIDIA Open Model License；见 `assets/gear_wbc/NOTICE` 与 `NVIDIA_Open_Model_License.txt`
  - 是（`assets/gear_wbc/`）
* - 参考配置 `g1_gear_wbc.sim2mujoco.yaml`
  - 同上
  - Apache-2.0
  - 是
* - unitree_sim_isaaclab（G1 + Dex1 USD 模型、任务模板、上游行走策略）
  - [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
  - Apache-2.0
  - 否，运行时通过 `UNITREE_SIM_ROOT` 引用
* - 仓库背景 `Simple_Warehouse/warehouse.usd`、纸箱 `SM_CardBoxD_04.usd`
  - NVIDIA Isaac Sim 资产库
  - 按 Isaac Sim 资产条款
  - 否，运行时在线引用
* - 本仓库代码
  - 本仓库
  - MIT License（见仓库根目录 LICENSE）
  - —
```

扫码枪外形和枪头相机参数参照 Zebra DS2208 的公开规格，Zebra 与 DS2208 是其所有者的商标。NVIDIA、Isaac Sim、GR00T 是 NVIDIA Corporation 的商标，此处仅用于说明来源。
