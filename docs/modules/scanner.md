(scanner)=
# 扫码枪与扫码判定

## 扫码枪模型

枪由 5 个长方体拼成，外包络 16 × 6.6 × 10.5 cm，尺寸参照 Zebra DS2208（只参照外形和规格，不是复制来的资产）。枪只有外观，没有碰撞体和质量；但各部件的角点加进了机器人的间隙模型，撞到桌子或箱子时 `step()` 会返回 `collision`。

| 部件 | 中心（G 系, m） | 尺寸 (m) |
| --- | --- | --- |
| handle 枪柄 | (0, 0, −0.0175) | 0.040 × 0.030 × 0.105 |
| head 枪头 | (0.020, 0, 0.060) | 0.100 × 0.066 × 0.050 |
| nose 枪鼻 | (0.0625, 0, 0.060) | 0.015 × 0.070 × 0.056 |
| window 出光窗 | (0.0705, 0, 0.060) | 0.001 × 0.050 × 0.034 |
| trigger 扳机 | (0.026, 0, −0.030) | 0.012 × 0.012 × 0.020 |

| 坐标系 | 定义 |
| --- | --- |
| hand | `left_hand_base_link`（左 Dex1 夹爪基座），手指沿 +y_hand 伸出 |
| G（`scanner_mount`） | 原点在握持点（hand 系 (0, 0.110, 0)，两指垫中间），相对 hand 绕 z 转 +90°：+x_G = 扫描方向（沿手指），+z_G = 枪柄指向枪头，y_G = z_G × x_G |
| tip | 出光窗中心，G 系 (0.071, 0, 0.060)，轴向与 G 相同 |
| 枪头相机 | ROS 光学系：+z（前）= +x_G，+x（右）= −y_G，+y（下）= −z_G |

(scan)=
## 扫码判定：`env.scan()`

1. **触发**：调用 `scan()` 相当于扣扳机。取枪头相机当前的 RGB 图，转灰度，用 pyzbar（zbar）解码，不限定码制。
2. **结果完全来自图像解码**：没有任何距离、角度或视场阈值。对不准、太近（标签出画）、太远（模块太小）、入射太斜、遮挡，都会自然导致解码失败。
3. **方向**：默认配置 `scanner.yaml: scan.rotation_step_deg: 90` 只解码原图，不做旋转补偿，模拟真实扫码枪不读与画面明显倾斜的码。实测 zbar 能读出与图像轴夹角约 ±17° 以内的一维码，所以瞄准时应让画面水平方向对齐条码长轴。若改成 15，会在 0°、15°、…、75° 下分别解码（全向成像器模型）。
4. **选码**：同一内容只保留一次；画面里有多个码时，`content` 取离画面中心最近的那个，模拟真机的瞄准点。
5. **GT 诊断**：额外把扫描射线（tip 的 x 轴）与所有 GT 标签矩形求交，只计从正面入射的命中，给出命中的标签、距离和入射角。**只用于评测和排查失败原因**，不考虑遮挡，也不影响 `status`。

| 返回字段 | 说明 |
| --- | --- |
| `status` | `"ok"`（至少解出一个码）或 `"no_read"`；扫码枪关闭时为 `"error"` |
| `content` | 选中的条码内容（如 `S1-B01-F0`），没解出时为 `None` |
| `decoded` | 画面中所有解出的码，按离画面中心的距离排序。每项含 `content`、`type`（如 `CODE128`、`QRCODE`）、`center_px`、`polygon`、`rotation_deg`、`offset_from_image_center_px` |
| `T_root_scanner_tip` | 扣扳机时的枪口位姿（4×4） |
| `timestamp` | 仿真时间（s） |
| `gt` | `{"hit": {content, distance_m, incidence_deg, hit_offset_m} 或 None, "note"}`，仅供评测 |

改编自 `examples/scanner_camera.py`：站到标签前、瞄准、扫码、保存快照。

```python
import math
import numpy as np
from g1_barcode_sim.env import G1BarcodeEnv
from g1_barcode_sim.camera import inv_T

env = G1BarcodeEnv("s1_table", headless=True)
bc = env.get_ground_truth()["barcodes"][0]        # 示例直接用 GT；实际算法应从头部相机估计标签位姿
c, n = np.array(bc["center"]), np.array(bc["normal"])
env.teleport_root(c[0] + 0.55 * n[0], c[1] + 0.55 * n[1] - 0.20,
                  math.atan2(-n[1], -n[0]))         # 标签位于左肩正前方
env.stand_still(0.5)
T = inv_T(env.get_obs()["T_world_root_gt"])        # world -> root
env.aim_scanner_at(T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ n, distance=0.20)
env.stand_still(3.0)                               # 瞄准约 1–2 s 收敛
obs = env.get_obs()
print(obs["scanner_aim_error"])                    # {"pos_m": ..., "angle_deg": ...}
r = env.scan()
print(r["status"], r["content"], r["gt"]["hit"])
env.save_snapshot("outputs/my_snapshot")           # 头部 / 枪头 RGB + 深度、俯视图、斜视图
env.close()
```

| 想改的内容 | 位置 |
| --- | --- |
| 枪头相机内参 | `configs/scanner.yaml: camera`（`width` / `height` / `focal_length` / `horizontal_aperture`） |
| 枪的外形 | `scanner.yaml: parts`；同时改 `tip.pos`，让枪口落在出光窗上 |
| 解码旋转步长 | `scanner.yaml: scan.rotation_step_deg` |
| 握持位置 | `configs/robot.yaml: scanner_mount` |
| 瞄准时方向与位置误差的权重 | `robot.yaml: arm_ik.aim_weight` |
