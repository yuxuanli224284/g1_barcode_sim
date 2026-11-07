(barcodes)=
# 条码与标签

## 码制与贴图生成

| 码制 | `type` | 生成库 | 细节 |
| --- | --- | --- | --- |
| Code 128 | `code128`（默认） | python-barcode | 得到含起始符、校验符和终止符的模块序列，不含静区 |
| QR Code | `qr` | qrcode | 纠错等级 M，自动选版本，无边框 |

- **逐模块绘制**：每个模块占整数个像素，不做重采样，所以条边锐利、比例准确。内置场景按 `configs/barcode.yaml` 绘制（贴图长边 1024 px，图案下方印可读文字）；测试场景每模块固定 4 px，不印文字。
- **解码自检**：每张贴图生成后都用 pyzbar 解码，结果必须恰好等于原内容，否则抛出 `RuntimeError`（批量脚本 `generate_barcodes.py` 则以退出码 1 结束）。测试场景的无底板标签没有白色静区，自检时先在四周补 12 个模块宽的静区再解码。`check_env.py` 还会在每个标签正前方 0.3 m 处放一台辅助相机渲染，再解码一次（端到端检查）。
- **缓存**：内置场景贴图在 `assets/barcodes/<type>/`；测试场景贴图按参数哈希缓存在 `assets/barcodes/cache/`。两者都可删除，用到时自动重新生成。

(label-geometry)=
## 标签几何

标签是箱子 prim 的子节点，随箱子移动。它由一个带贴图的四边形（单面）和可选的白色底板组成：

- **底板（backing）**：白色薄片，厚 1 mm（`barcode.yaml: label_thickness`）。内置场景总是带底板，标签外形为 Code128 10 × 5 cm、QR 6 × 6 cm（含静区），条码图案最大 8 × 3 cm / 5 × 5 cm。测试场景由 `label_backing` 决定：有底板时标签 = 图案 + 每边 1 cm 白边；无底板时图案的浅色模块直接用箱面颜色，相当于印在箱子上。
- **face_offset**：标签正面离箱面 1.5 mm，避免 z-fighting。
- **放置参数**：`u`、`v` 从面中心量起（米），`u` 沿该面的"右"方向，`v` 向上；`rot_deg` 为面内旋转，从箱外看逆时针为正。
- **材质**：roughness 0.8、metallic 0；带语义标签 `class=barcode`（箱子 `class=box`，家具 `class=table` / `shelf`），可用于生成分割真值。

| 面 | 外法向 n | 面内"右"轴 u | 面内"上"轴 v | 内置场景面号 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `-x` | (−1, 0, 0) | (0, −1, 0) | (0, 0, 1) | 0 | 正面，朝向机器人起点 |
| `+x` | (1, 0, 0) | (0, 1, 0) | (0, 0, 1) | 1 | 背面 |
| `-y` | (0, −1, 0) | (1, 0, 0) | (0, 0, 1) | 2 | 机器人右侧的侧面 |
| `+y` | (0, 1, 0) | (−1, 0, 0) | (0, 0, 1) | 3 | 机器人左侧的侧面 |

方向都在箱子局部坐标系下（箱子 yaw = 0 时与世界系重合）。u、v 按"站在箱外看这个面"定义，满足 right = up × normal，所以图案从左往右读。

- **内置场景**：每个箱面最多 1 张标签，内容由模板 `S{scene}-B{box:02d}-F{face}` 自动生成（如 `S1-B01-F0`）。标签旋转后的四个角都必须落在面内。可选随机化（`barcode.yaml: randomize.enabled` 或 `run_scene.py --randomize`）：每次 reset 给 u、v 叠加 ±3 cm、rot 叠加 ±10° 的均匀扰动，同一 seed 结果相同。
- **测试场景**：同一个面可以贴多张标签，内容任意。加载时检查：标签旋转后的外接框加 1 cm 边距必须在面内；同一面上的多张标签外接框不能重叠；不能贴在被相邻箱子挡住的内侧面上。prim 名为 `BC_mx`、`BC_px`、`BC_my`、`BC_py`，同面第 2 张起依次为 `BC_mx_1`、`BC_mx_2`……

## 添加自己的条码

### 方式 A：内置场景 YAML

内容由模板生成，不能逐个指定；`id` 必须写成 `B` 加数字。

```yaml
# configs/scenes/s1_table.yaml
boxes:
  - {id: B03, size: [0.30, 0.30, 0.30],
     place_on: {furniture: T01, y: 0.0, front_setback: 0.05},
     barcodes: [{face: "-x", uv: [0.03, -0.02], rot_deg: 15},   # 内容 S1-B03-F0
                {face: "+y"}]}                                  # 内容 S1-B03-F3
```

改完后可运行 `python scripts/generate_barcodes.py --scenes s1_table`（加 `--type qr` 生成 QR）。要换编号规则，改 `barcode.yaml` 的 `content_format`（占位符 `{scene}`、`{box}`、`{face}`）。

### 方式 B：测试场景 JSON（内容任意）

在场景的 `barcodes` 数组里加一项（字段见{ref}`suite-schema`）：

```json
{"box_id": "B0", "face": "-x", "type": "code128",
 "width": 0.08, "aspect": 0.375, "label_backing": true,
 "u": 0.0, "v": 0.0, "rot_deg": 0.0,
 "content": "MYCODE123", "content_id": "MY-0000-B0"}
```

QR 不需要 `aspect`，`width` 就是边长。Code128 只能编码 ASCII；`content` 在同一场景内必须唯一。

### 方式 C：只生成标签图（不需要 Isaac Sim）

```python
from g1_barcode_sim.config import load_barcode_cfg
from g1_barcode_sim.barcode import make_label, decode_image

lab = make_label("HELLO-123", load_barcode_cfg(), "code128")
lab.image.save("hello.png")
assert decode_image(lab.image) == ["HELLO-123"]
print(lab.pattern_size_m)          # 实际图案尺寸（米）
```
