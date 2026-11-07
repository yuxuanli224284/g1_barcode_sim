(gt)=
# 坐标系与 ground truth

## 坐标约定

| 项 | 约定 |
| --- | --- |
| 世界坐标系 | z 向上，地面 z = 0，单位米、弧度。场景都在 `/World/BarcodeScene` 下，只有 1 个 env，原点即世界原点 |
| root 坐标系 | 骨盆（pelvis）：x 向前、y 向左、z 向上。`step`、`move_relative`、手腕和瞄准目标都以它为参考 |
| 机器人起点 | 内置场景在原点、yaw = 0，面朝 +x（场景物体所在方向）。测试场景见{ref}`suite-schema` |
| 相机光学系 | ROS 约定：x 右、y 下、z 前 |
| 变换命名 | `T_a_b` 表示 b 在 a 中的位姿，`p_a = T_a_b · p_b`，例如 `T_world_cam = T_world_root · T_root_cam` |
| 四元数 | `wxyz`；箱子只绕 z 旋转 |
| 标签坐标系 | `normal` 为外法向，`up` 为图案的"上"，`right = up × normal` |

(gt-fields)=
## Ground truth

`env.get_ground_truth(d=0.20)` 返回当前场景的 GT。

```{warning}
**GT 只用于评测**，被测算法不应读取；同理，观测里的 `T_world_root_gt` 和 `scan()` 返回的 `gt` 也只用于评测。
```

`check_env.py` 会把 GT 写到 `outputs/<scene>/gt.json`。

| 字段 | 说明 |
| --- | --- |
| `scene`、`seed`、`randomized` | 场景名（或场景 id）、reset 种子、标签是否随机化 |
| `robot_init` | `{pos: [x, y, 0], yaw}`，初始 root 位姿 |
| `boxes[]` | `{id, size: [x 深, y 宽, z 高], pos（箱子中心）, quat_wxyz, distractor}` |
| `barcodes[].content`、`type` | 条码内容（场景内唯一）与码制 |
| `barcodes[].box_id`、`face` | 所在箱子和面 |
| `barcodes[].center`、`normal`、`up` | 标签正面中心、外法向、图案"上"方向（世界坐标，`up` 已含 `rot_deg`） |
| `barcodes[].size`、`pattern_size` | 标签外形 [w, h]（含底板或静区）；实际条码图案 [w, h]（内置场景目前为 `null`） |
| `barcodes[].height_above_ground`、`distractor` | 中心离地高度；是否为干扰箱上的标签 |
| `barcodes[].target_root_pose_d020` | 参考站位 `{pos: [x, y, 0], yaw}`：法向投影到水平面后归一化为 n_h，pos = center 水平投影 + d·n_h，yaw 面朝标签。后缀是 d 的厘米数，`d=0.40` 时为 `target_root_pose_d040` |

测试场景模式下另有：`scenario_id`、`task`、`target`、`target_box_id`、`target_barcode`、`target_face`（目标箱正面 `{center, normal, corners[4]}`）、`task1_gt_obs_pose`（GT 观察位：在 0.60–1.00 m 内以 0.05 m 步长取最小距离，使头部相机画面装下目标面的 4 个角并留 5% 边距）、`task2_gt_wrist_target`（条码中心 + 0.20 m × 法向，扫描方向为 −法向）。`target` 为 `"all"`（箱堆）时，`target_box_id` 为 `"all"`、`target_barcode` 为 `null`，并额外给出 `faces`：按 −x、−y、+x、+y 的巡视顺序列出箱堆每个外露侧面的中心、法向、四角、观察位和该面上的条码内容。

`s1_table`，seed 0 的 GT（节选）：

```text
{
 "scene": "s1_table", "seed": 0, "randomized": false,
 "robot_init": {"pos": [0.0, 0.0, 0.0], "yaw": 0.0},
 "boxes": [{"id": "B01", "size": [0.3, 0.4, 0.3], "pos": [1.5, -0.25, 0.9],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "distractor": false}, ...],
 "barcodes": [{"content": "S1-B01-F0", "type": "code128", "box_id": "B01", "face": "-x",
   "center": [1.3485, -0.25, 0.9], "normal": [-1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0],
   "size": [0.1, 0.05], "pattern_size": null, "height_above_ground": 0.9, "distractor": false,
   "target_root_pose_d020": {"pos": [1.1485, -0.25, 0.0], "yaw": 0.0}}, ...]
}
```

`center.x` = 箱子中心 1.5 − 半深 0.15 − 面偏移 0.0015 = 1.3485。
