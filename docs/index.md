# g1_barcode_sim

**Isaac Sim 5.1 · Isaac Lab · Unitree G1 + Dex1**

G1 人形机器人仓库扫码仿真环境：左手握一把手持扫码枪，在桌面、货架和地面箱堆上寻找纸箱的条码标签，走过去、对准并扫码。本文档介绍安装、实现细节、Python API、内置场景和测试场景集。

中文 | [English](https://yuxuanli224284.github.io/g1_docs/barcode_sim/en/)

```{figure} images/scene_s1_table_left_rear.jpg
:alt: s1_table 场景：仓库地面上一张白桌，桌上两个纸箱，G1 机器人站在桌前，左手握着黄黑色扫码枪

内置场景 `s1_table`，机器人左后方视角。G1 站在原点、面朝 +x，左手握扫码枪；白桌上两个纸箱朝向机器人的一面各贴一张 Code128 标签。
```

## 简介

g1_barcode_sim 是一个基于 Isaac Sim 5.1 / Isaac Lab 的仿真环境，用来开发和评测人形机器人的"找码—接近—扫码"能力。场景里有一台 Unitree G1，左手的 Dex1 夹爪握着一把手持扫码枪；纸箱放在桌上、货架上或直接堆在地上，箱面贴着 Code128 或 QR 条码标签。使用者通过 Python 接口控制机器人行走、下蹲、移动手臂和瞄准，再"扣扳机"扫码，环境返回对枪头相机画面的真实解码结果。

环境本身**不包含**找码、导航或扫码策略；它提供传感器、控制接口、可复现的场景和仅供评测用的 ground truth（GT）。

文档中的数值来自仓库内的代码与配置（`configs/`、`g1_barcode_sim/`、`suites/*/manifest.json`）。

```{toctree}
:maxdepth: 2
:caption: 入门

overview
installation
```

```{toctree}
:maxdepth: 2
:caption: 仿真实现

modules/robot
modules/locomotion
modules/cameras
modules/scanner
modules/barcodes
modules/ground_truth
```

```{toctree}
:maxdepth: 2
:caption: 接口

api
```

```{toctree}
:maxdepth: 2
:caption: 场景与测试

scenes
task_suites
```

```{toctree}
:maxdepth: 2
:caption: 其他

limitations
```
