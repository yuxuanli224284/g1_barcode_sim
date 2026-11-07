(suites)=
# Task suites（测试场景集）

Task suite 是一组可复现的测试场景，每个场景由一个 JSON 文件完整描述，用 `env.reset(scenario=...)` 加载，同一进程内可以连续切换。目前有两类：**桌面场景**（`t1_*`、`t2_*`，共 19 套，由生成器按因素表产出）和**箱堆场景**（`stack*`，共 6 套，手写）。Suite 会持续扩充，新的场景类型和难度等级会陆续加入。

```{note}
场景 JSON 里只描述场景本身，不含 GT 目标；GT 在 reset 之后由 `get_ground_truth()` 计算，只用于评测。
```

(suite-tasks)=
## 任务定义与分级

| task | 场景 | 起点 | 要完成什么 | 评测用 GT |
| --- | --- | --- | --- | --- |
| 1 | 桌上两个箱子，`target` 指定左 / 右哪个是目标 | 离桌沿 2–5 m，可横向偏移 | 找到目标箱，走到能看清它正面的观察位 | `task1_gt_obs_pose` |
| 2 | 同上 | 目标箱前的观察位附近（GT 观察位加噪声和横向偏移），保证目标箱面和条码都在头部画面内 | 扫出目标箱上的条码 | `target_barcode`、`task2_gt_wrist_target` |
| 3 | 地面箱堆，`target: "all"` | 箱堆前 2.5 m | 扫描箱堆上的全部条码；有 `scan_faces: "all"` 时需绕堆扫描四个侧面 | `faces`、每张标签的位姿 |

任务 2 的起点由 GT 观察位生成，所以可以脱离任务 1 单独测试。"左 / 右"以机器人视角为准：左为 +y。

### 分级 L0 / L1 / L2

| Level | 含义 | 用途 |
| --- | --- | --- |
| L0 | 标称场景，只在一个小网格上变化（如起始距离 × 目标左右） | 调通流程 |
| L1 | 单因素扫描：一次只改一个因素，其余取标称值。连续因素在范围内等间距取点，离散因素逐个取值 | 定位是哪个因素导致失败，画"成功率—因素值"曲线 |
| L2 | 该任务的所有因素在各自范围内均匀随机采样，目标左右交替（各占一半） | 测总体成功率 |

任务 1 的 8 个因素：`robot_dist`、`robot_lateral`、`box_shape`、`box_appearance`、`placement`、`barcode_size`、`background`、`lighting`。任务 2 的 7 个因素：`barcode_pos`、`barcode_size`、`barcode_type`、`box_shape`、`box_lateral`、`start_noise`、`background`。标称值与范围都在 `configs/testsuite/factors.yaml`：

| 参数 | 标称值 | 范围 / 取值 |
| --- | --- | --- |
| 机器人到桌沿距离 `dist_to_table` | 3.0 m | 2.0–5.0 m |
| 横向偏移 `lateral_offset` | 0 | −1.5–1.5 m |
| 初始朝向 `yaw_deg` | 0 | 生成器固定为 0（手写场景可设非零） |
| 箱宽 w（沿 y）/ 箱高 h / 箱深 d（沿 x） | 0.40 / 0.30 / 0.30 m | 0.20–0.60 / 0.15–0.50 / 0.20–0.50 m |
| 箱子外观 | `kraft` | `kraft` / `kraft_tape` / `white_box` / `printed` |
| 桌深 / 桌宽 / 桌高 | 0.60 / 1.20 / 0.75 m | 0.50–0.80 / 1.00–1.80 / 0.70–0.80 m（白色哑光，固定） |
| 箱前表面到桌沿 `setback` | 0.05 m | 0–0.20 m |
| 两箱间距 `gap` | 0.10 m | 0.05–0.60 m |
| 两箱整体横向中心 `group_y` | 0 | 两箱都不出桌面的范围内均匀采样 |
| 条码面内旋转 | 0° | 基准 {0°, 90°} 加 ±10° 扰动 |
| 码制 | `code128` | `code128` / `qr` |
| Code128 图案宽 / 高宽比 | 0.08 m / 0.375 | 0.04–0.12 m / 0.25–0.60 |
| QR 边长 | 0.05 m | 0.03–0.08 m |
| 内容长度 | 9 字符 | 6–20 字符，字母表 `ABCDEFGHJKLMNPQRSTUVWXYZ23456789`（去掉易混淆字符） |
| 白色底板 | 有 | 有 / 无 |
| 背景 | `warehouse` | `warehouse` / `none`（灰色地面）/ `white`（白色地面和墙） |
| 光照倍率 | 1.0 | 0.7–1.3 |
| 任务 2 起点噪声 | ±0.03 m / ±3° | `start_noise` 最大放大到 ±0.08 m / ±8° |
| 任务 2 观察距离 `d_obs` | — | 0.60–1.00 m，步长 0.05 m，画面四周留 5% 边距 |

**合法性检查**：生成时自动检查，不通过就重新采样（每个场景最多 50 次），拒绝原因记在 manifest 的 `rejections` 中。检查项包括：两箱和间距不超出桌宽、箱子不伸出桌面、标签（含 1 cm 边距）放得进箱面、条码中心离地 ≤ 1.3 m、机器人占地圆（半径 0.35 m）不与桌子或箱子相交、任务 2 起点处箱面四角和条码四角全部投影在头部画面内。L1 中无法实现的扫描点直接跳过，记入 `skipped_infeasible`；扫描出的箱子在标称桌面上放不下时，桌子会在自身范围内放大到刚好放下，并记到 `factor.table_adjusted`。

**可复现性**：每个场景的随机数流由 `SHA-256(套件名|seed|序号|尝试次数)` 派生，与 Python 哈希种子和生成顺序无关；每个套件在内存中生成两次，逐字节一致才写盘；manifest 不含时间戳，`code_hash` 记录生成器源码和配置的哈希。

(suite-schema)=
## 场景 JSON 格式

文件位于 `suites/<suite>/scenarios/<id>.json`。手写场景与生成器产出的场景使用同一格式。

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - 字段
  - 类型
  - 说明
* - `id`
  - str
  - `<suite>_<4 位序号>`，如 `t1_L2_0007`。加载时去掉末尾 `_NNNN` 定位套件目录
* - `suite`、`index`
  - str、int
  - 套件名、场景在套件内的序号
* - `task`
  - int
  - 1、2 或 3，见{ref}`suite-tasks`
* - `seed`
  - int
  - 套件的基础种子（目前都为 0），reset 时设为环境的 seed
* - `level`
  - str
  - `L0` / `L1` / `L2`；不是 L0 时 GT 的 `randomized` 为 true
* - `factors_varied`
  - list
  - L1 为被扫描的那个因素；L2 为该任务的全部因素；L0 和手写场景为空
* - `factor`
  - dict / null
  - L0 记录网格取值，如 `{"dist_to_table": 2.0, "target": "left"}`；L1 为 `{"name", "value"}`（可能附 `table_adjusted`）；L2 为 `null`；手写场景为 `{"note": "设计说明"}`
* - `background`
  - dict
  - `{"type": "warehouse"|"none"|"white", "light_scale": float}`。灯光 = 该背景的基准值 × `light_scale`
* - `table`
  - dict / null
  - `{"depth", "width", "height", "appearance": "white_matte"}`。`null` 表示没有桌子，箱子直接放在地面
* - `robot_init`
  - dict
  - `{"dist_to_table", "lateral_offset", "yaw_deg"}`。桌沿（或箱堆前表面）固定在 x = 1.30 并平行于 y 轴；机器人起点为 (1.30 − `dist_to_table`, `lateral_offset`)，`yaw_deg` 逆时针为正（+30 表示左转 30°），0 表示面朝 +x
* - `boxes[]`
  - list
  - 见下表
* - `barcodes[]`
  - list
  - 见下表
* - `target`
  - str
  - `"left"` / `"right"`（机器人视角）或 `"all"`（task 3，全部箱子）
* - `task2_start`
  - dict / null
  - 只有任务 2 有值：`pose_world{x, y, yaw_deg}`、`noise{dx, dy, dyaw_deg, bound_pos, bound_yaw_deg}`、`lateral_offset`、`lateral_feasible[lo, hi]`、`obs_d`、`obs_fits`。有值时起点取 `pose_world`，忽略 `robot_init`
* - `scan_faces`
  - str，可选
  - 值为 `"all"` 时表示要扫描箱堆的四个侧面（目前出现在 `stack4f_L0`、`stack4c_L0`）。仿真本身不读取这个字段，它是交给被测系统的任务说明
```

| `boxes[i]` 字段 | 说明 |
| --- | --- |
| `id` | 生成器写 `B0` / `B1`（按 y 从小到大，B0 在机器人右侧）；手写场景可自定义，如 `LF2` |
| `size_wdh` | `[w（沿 y）, d（沿 x）, h]`，米。注意与 GT 中 `size` 的 `[x, y, z]` 顺序不同 |
| `appearance` | `kraft`（牛皮纸色）、`kraft_tape`（顶缝封箱胶带，沿前表面向下 6 cm）、`white_box`（白色箱）、`printed`（印有 FRAGILE / THIS SIDE UP 等文字、批号、logo 和箭头的印刷箱）、`cardbox`（Isaac 纸箱美术资产 `SM_CardBoxD_04.usd`，按箱子尺寸缩放，与内置场景同款） |
| `setback` | 箱子前表面到 x = 1.30 的距离，米 |
| `y_center` | 箱子中心的 y 坐标，米 |
| `on` | 可选。前面已出现的箱子 id，表示叠放在它的顶面上（不能伸出下面箱子的边缘）；`null` 或省略时放在桌面或地面 |

| `barcodes[i]` 字段 | 说明 |
| --- | --- |
| `box_id` | 所在箱子 |
| `face` | 可选，`-x` / `+x` / `-y` / `+y`，默认 `-x`。同一面可有多张，不能重叠 |
| `type` | `code128` 或 `qr` |
| `width`、`aspect` | Code128 图案宽度与高宽比；QR 只用 `width`（边长） |
| `label_backing` | 是否带白色底板（每边比图案宽 1 cm） |
| `u`、`v`、`rot_deg` | 面内位置（从面中心量起，米）和旋转，定义见{ref}`label-geometry`。对 −x 面，`u` 的正方向是机器人的右手方向 −y |
| `content`、`content_id` | 条码实际编码的字符串；可读编号（如 `T1L2-0007-B0`） |

`suites/t2_L0/scenarios/t2_L0_0000.json`（压缩显示）：

```text
{"id": "t2_L0_0000", "suite": "t2_L0", "task": 2, "seed": 0, "index": 0, "level": "L0",
 "factors_varied": [], "factor": {"group_y": -0.15, "target": "left"},
 "background": {"type": "warehouse", "light_scale": 1.0},
 "table": {"depth": 0.6, "width": 1.2, "height": 0.75, "appearance": "white_matte"},
 "robot_init": {"dist_to_table": 3.0, "lateral_offset": 0.0, "yaw_deg": 0.0},
 "boxes": [{"id": "B0", "size_wdh": [0.4, 0.3, 0.3], "appearance": "kraft", "setback": 0.05, "y_center": -0.4},
           {"id": "B1", "size_wdh": [0.4, 0.3, 0.3], "appearance": "kraft", "setback": 0.05, "y_center": 0.1}],
 "barcodes": [{"box_id": "B0", "type": "code128", "width": 0.08, "label_backing": true, "aspect": 0.375,
               "u": 0.0, "v": 0.0, "rot_deg": 0.0, "content": "ABVK68LYK", "content_id": "T2L0-0000-B0"},
              {"box_id": "B1", ..., "content": "H2BXK949C", "content_id": "T2L0-0000-B1"}],
 "target": "left",
 "task2_start": {"pose_world": {"x": 0.75, "y": 0.1, "yaw_deg": 0.0},
                 "noise": {"dx": 0.0, "dy": 0.0, "dyaw_deg": 0.0, "bound_pos": 0.0, "bound_yaw_deg": 0.0},
                 "lateral_offset": 0.0, "lateral_feasible": [-0.4, 0.4], "obs_d": 0.6, "obs_fits": true}}
```

(suite-table)=
## 桌面场景（`t1_*`、`t2_*`）

共 19 套，全部由生成器产出：一张白色哑光桌，桌上两个箱子，每箱一张条码，都贴在朝向机器人的 −x 面。标称配置为箱子 0.40 × 0.30 × 0.30 m（w × d × h）、`kraft` 外观、`warehouse` 背景、Code128 带底板、条码在箱面中心。"左 / 右"列是目标箱在左侧和右侧的场景数。

| Suite | task | 场景数 | 变化内容 | 外观 / 条码 | 左 / 右 |
| --- | --- | --- | --- | --- | --- |
| `t1_L0` | 1 | 8 | 起始距离 {2, 3, 4, 5} m × 目标 {左, 右} | 标称 | 4 / 4 |
| `t1_L1_robot_dist` | 1 | 12 | 起始距离 2.0 → 5.0 m | 标称 | 6 / 6 |
| `t1_L1_robot_lateral` | 1 | 13 | 横向偏移 −1.5 → 1.5 m，步长 0.25 m | 标称 | 7 / 6 |
| `t1_L1_box_shape` | 1 | 12 | 目标箱从最小（0.2 × 0.2 × 0.15）放大到最大（0.6 × 0.5 × 0.5），w / d / h 同步插值 | 标称 | 6 / 6 |
| `t1_L1_box_appearance` | 1 | 12 | 外观 kraft / kraft_tape / white_box / printed 各 3 次 | 两箱外观相同 | 8 / 4 |
| `t1_L1_placement` | 1 | 12 | 依次扫 setback 0 → 0.2、gap 0.05 → 0.6、group_y 最左 → 最右，各 4 点；第 6、7 个场景桌子自动加宽到 1.2167 m 和 1.4 m | 标称 | 6 / 6 |
| `t1_L1_barcode_size` | 1 | 10 | 目标条码宽 0.04 → 0.12 m | Code128 | 5 / 5 |
| `t1_L1_background` | 1 | 12 | 背景 warehouse / none / white 各 4 次 | 标称 | 6 / 6 |
| `t1_L1_lighting` | 1 | 10 | 光照倍率 0.7 → 1.3 | warehouse 背景 | 5 / 5 |
| `t1_L2` | 1 | 100 | 全部 8 个因素随机组合 | 四种外观；Code128 102 / QR 98 张；有底板 96 / 无底板 104 张 | 50 / 50 |
| `t2_L0` | 2 | 8 | `group_y` {−0.15, −0.05, 0.05, 0.15} m × 目标 {左, 右}；从 GT 观察位出发，无噪声 | 标称 | 4 / 4 |
| `t2_L1_barcode_pos` | 2 | 12 | 条码在面内的位置：4 档 u × 3 档 v（按允许范围的比例） | 标称 | 6 / 6 |
| `t2_L1_barcode_size` | 2 | 10 | 条码宽 0.04 → 0.12 m | Code128 | 5 / 5 |
| `t2_L1_barcode_type` | 2 | 12 | {Code128, QR} × {有底板, 无底板}，各 3 次 | QR 边长取标称 0.05 m | 8 / 4 |
| `t2_L1_box_shape` | 2 | 10 | 目标箱从最小放大（定义 12 点；最大两档在 1.0 m 观察距离内看不全整个箱面，被跳过） | 标称 | 5 / 5 |
| `t2_L1_box_lateral` | 2 | 12 | 观察位相对目标箱的横向偏移，在可见区间内从一端扫到另一端 | 标称 | 6 / 6 |
| `t2_L1_start_noise` | 2 | 12 | 起点噪声上界从 ±0.03 m / ±3° 放大到 ±0.08 m / ±8° | 标称 | 6 / 6 |
| `t2_L1_background` | 2 | 12 | 背景 warehouse / none / white 各 4 次 | 标称 | 6 / 6 |
| `t2_L2` | 2 | 100 | 全部 7 个因素随机组合（`t2_L2_0056`、`0057` 在 0.6–1.0 m 内找不到能留 5% 边距的距离，观察距离取上限 1.0 m） | 四种外观；Code128 93 / QR 107 张；有底板 102 / 无底板 98 张 | 50 / 50 |

L1 的左右并非都各占一半：目标按序号每 4 个交替一次，12 个场景时得到"左、右、左"三组，所以是 8 / 4。

每张卡片是该 suite 中一个代表场景的机器人左后方视角：

::::{container} cards
```{figure} images/suite_t1_L0_left_rear.jpg
:alt: t1_L0 代表场景

**`t1_L0`** · task 1 · L0 · 8 个场景：起始距离 × 目标左右
```
```{figure} images/suite_t1_L1_robot_dist_left_rear.jpg
:alt: t1_L1_robot_dist 代表场景

**`t1_L1_robot_dist`** · task 1 · L1 · 12 个场景：起始距离 2–5 m
```
```{figure} images/suite_t1_L1_robot_lateral_left_rear.jpg
:alt: t1_L1_robot_lateral 代表场景

**`t1_L1_robot_lateral`** · task 1 · L1 · 13 个场景：横向偏移 ±1.5 m
```
```{figure} images/suite_t1_L1_box_shape_left_rear.jpg
:alt: t1_L1_box_shape 代表场景

**`t1_L1_box_shape`** · task 1 · L1 · 12 个场景：目标箱尺寸
```
```{figure} images/suite_t1_L1_box_appearance_left_rear.jpg
:alt: t1_L1_box_appearance 代表场景

**`t1_L1_box_appearance`** · task 1 · L1 · 12 个场景：四种箱子外观
```
```{figure} images/suite_t1_L1_placement_left_rear.jpg
:alt: t1_L1_placement 代表场景

**`t1_L1_placement`** · task 1 · L1 · 12 个场景：退距、间距、整体横移
```
```{figure} images/suite_t1_L1_barcode_size_left_rear.jpg
:alt: t1_L1_barcode_size 代表场景

**`t1_L1_barcode_size`** · task 1 · L1 · 10 个场景：条码宽 4–12 cm
```
```{figure} images/suite_t1_L1_background_left_rear.jpg
:alt: t1_L1_background 代表场景

**`t1_L1_background`** · task 1 · L1 · 12 个场景：仓库 / 无背景 / 纯白
```
```{figure} images/suite_t1_L1_lighting_left_rear.jpg
:alt: t1_L1_lighting 代表场景

**`t1_L1_lighting`** · task 1 · L1 · 10 个场景：光照 ×0.7–1.3
```
```{figure} images/suite_t1_L2_left_rear.jpg
:alt: t1_L2 代表场景

**`t1_L2`** · task 1 · L2 · 100 个场景：全部因素随机
```
```{figure} images/suite_t2_L0_left_rear.jpg
:alt: t2_L0 代表场景

**`t2_L0`** · task 2 · L0 · 8 个场景：箱组横移 × 目标左右
```
```{figure} images/suite_t2_L1_barcode_pos_left_rear.jpg
:alt: t2_L1_barcode_pos 代表场景

**`t2_L1_barcode_pos`** · task 2 · L1 · 12 个场景：条码面内位置
```
```{figure} images/suite_t2_L1_barcode_size_left_rear.jpg
:alt: t2_L1_barcode_size 代表场景

**`t2_L1_barcode_size`** · task 2 · L1 · 10 个场景：条码宽 4–12 cm
```
```{figure} images/suite_t2_L1_barcode_type_left_rear.jpg
:alt: t2_L1_barcode_type 代表场景

**`t2_L1_barcode_type`** · task 2 · L1 · 12 个场景：码制 × 有无底板
```
```{figure} images/suite_t2_L1_box_shape_left_rear.jpg
:alt: t2_L1_box_shape 代表场景

**`t2_L1_box_shape`** · task 2 · L1 · 10 个场景：目标箱尺寸
```
```{figure} images/suite_t2_L1_box_lateral_left_rear.jpg
:alt: t2_L1_box_lateral 代表场景

**`t2_L1_box_lateral`** · task 2 · L1 · 12 个场景：目标箱相对横向位置
```
```{figure} images/suite_t2_L1_start_noise_left_rear.jpg
:alt: t2_L1_start_noise 代表场景

**`t2_L1_start_noise`** · task 2 · L1 · 12 个场景：起点噪声幅度
```
```{figure} images/suite_t2_L1_background_left_rear.jpg
:alt: t2_L1_background 代表场景

**`t2_L1_background`** · task 2 · L1 · 12 个场景：仓库 / 无背景 / 纯白
```
```{figure} images/suite_t2_L2_left_rear.jpg
:alt: t2_L2 代表场景

**`t2_L2`** · task 2 · L2 · 100 个场景：全部因素随机
```
::::

(suite-stack)=
## 箱堆场景（`stack*`）

共 6 套，全部手写（manifest 中 `definition.note` 为 `hand-written`，不受生成器管理，`generate.py --suite all` 不会改写它们）。共同设定：`task: 3`、`level: L0`、`table: null`（箱子直接放在地面）、`target: "all"`、`warehouse` 背景、光照倍率 1.0；所有箱子 0.40 × 0.40 × 0.38 m（w × d × h）；标签都是 Code128，图案宽 8 cm、高宽比 0.375、带底板（标签外形 10 × 5 cm）；机器人起始距离 2.5 m（站在 x = −1.2）。

| Suite | 场景数 | 箱堆结构 | 外观 | 每场景标签数（面分布） | `scan_faces` | 起始位姿 |
| --- | --- | --- | --- | --- | --- | --- |
| `stack3_L0` | 2 | 1 列 3 层，3 箱，位于 y = 0 | `kraft` | 3（全在 −x） | — | 正对 |
| `stack9_L0` | 1 | 3 列 × 3 层，9 箱，宽 1.2 m、高 1.14 m | `kraft` | 5（全在 −x，每层至少 1 个） | — | 正对 |
| `stack9p_L0` | 1 | 同 `stack9_L0` | `printed` 5 箱 + `kraft_tape` 4 箱，交替 | 5（同上） | — | 正对 |
| `stack9c_L0` | 1 | 同 `stack9_L0` | `cardbox` | 5（同上） | — | 正对 |
| `stack4f_L0` | 3 | 2 列 × 前后 2 排 × 3 层，12 箱，独立摆放；占地 x 1.30–2.10、y −0.4–0.4，高 1.14 m | `kraft` | 12（−x、−y、+x、+y 各 3） | `all` | 0000 正对；0001、0002 不平行于箱堆 |
| `stack4c_L0` | 3 | 同 `stack4f_L0` | `cardbox` | 0000：12（四面各 3）；0001：6（只有 −x、+x 各 3）；0002：60（−x、+x 各 30） | `all` | 正对 |

### stack3_L0：单列三层

::::{container} figpair
```{figure} images/suite_stack3_L0_left_rear.jpg
:alt: stack3_L0：地面上三个纸箱叠成一摞

左后方视角
```
```{figure} images/suite_stack3_L0_head.jpg
:alt: stack3_L0 头部相机画面

头部相机（起点）
```
::::

- `stack3_L0_0000`：三个箱子叠成一摞，每箱正面一张码，离地分别为顶箱 1.10 m（需要抬臂）、中箱 0.57 m（需要下蹲）、底箱 0.30 m（需要深蹲）。
- `stack3_L0_0001`：同一摞箱子，但底箱条码在 0.19 m，低于机器人深蹲后的可达范围（约 0.27 m）。用于检验被测系统能否识别并报告够不着的条码，然后继续处理其余条码。

### stack9_L0 / stack9p_L0 / stack9c_L0：3 × 3 箱墙

::::{container} cards
```{figure} images/suite_stack9_L0_left_rear.jpg
:alt: stack9_L0：3×3 牛皮纸色箱墙

**`stack9_L0`** · kraft · 5 labels：牛皮纸色箱墙，前表面 5 个码，每层至少一个，不是每箱都有。条码高度：顶层 1.07–1.10 m、中层 0.57 m、底层 0.30–0.32 m
```
```{figure} images/suite_stack9p_L0_left_rear.jpg
:alt: stack9p_L0：印刷箱与封箱胶带箱交替的箱墙

**`stack9p_L0`** · printed + kraft_tape · 5 labels：同一箱墙和 5 个码，箱子换成印刷箱（警示文字、批号、logo、箭头）与封箱胶带箱交替，增加视觉干扰
```
```{figure} images/suite_stack9c_L0_left_rear.jpg
:alt: stack9c_L0：Isaac 纸箱资产组成的箱墙

**`stack9c_L0`** · cardbox · 5 labels：同一箱墙和 5 个码，全部换成 Isaac 纸箱资产（牛皮纸色、封箱胶带、印刷运单贴纸）
```
::::

### stack4f_L0：独立 2 × 2 × 3 箱堆，四面贴码

::::{container} figpair
```{figure} images/suite_stack4f_L0_left_rear.jpg
:alt: stack4f_L0：地面上独立的 2×2×3 牛皮纸色箱堆

左后方视角
```
```{figure} images/suite_stack4f_L0_head.jpg
:alt: stack4f_L0 头部相机画面

头部相机（起点）
```
::::

- `stack4f_L0_0000`：12 个码，每个竖直侧面 3 个，每层一个，在该面的两列之间交替；标签只贴在外露面上。条码高度：顶层 1.07–1.10 m、中层 0.57 m、底层 0.33 m。巡视顺序（GT 中 `faces` 的顺序）为前（−x）、右（−y）、后（+x）、左（+y）。机器人正对箱堆。
- `stack4f_L0_0001`、`stack4f_L0_0002`：同一箱堆和 12 个码，但机器人起始朝向不平行于箱堆。0001 偏右 0.5 m（`lateral_offset: -0.5`）并左转 30°（`yaw_deg: 30`）；0002 偏左 0.7 m（`lateral_offset: 0.7`，已在箱堆 +y 侧面之外）并右转 35°（`yaw_deg: -35`）。两者都保证箱堆前表面仍在头部画面内、且在 3.0 m 深度量程内。

### stack4c_L0：纸箱资产版，标签数量不同

::::{container} figpair
```{figure} images/suite_stack4c_L0_left_rear.jpg
:alt: stack4c_L0：由 Isaac 纸箱资产组成的 2×2×3 箱堆，箱面上有印刷运单贴纸

左后方视角
```
```{figure} images/suite_stack4c_L0_head.jpg
:alt: stack4c_L0 头部相机画面

头部相机（起点）
```
::::

箱堆与 `stack4f_L0` 相同，所有箱子用 Isaac 纸箱资产绘制（外观 `cardbox`）。资产贴图上印着带类似条码图案的运单贴纸，但这些图案**不是可解码的码**，是贴近真实仓库的干扰物。机器人都正对箱堆。

- `stack4c_L0_0000`：四个侧面都有标签，与 `stack4f_L0_0000` 的 12 个码相同（每面 3 个）。
- `stack4c_L0_0001`：只有前（−x）和后（+x）两面有标签，各 3 个；左右两侧没有。
- `stack4c_L0_0002`：前、后两面各 30 个标签（共 60），侧面没有。每个箱面上是 3 列（u = −0.13 / 0 / +0.13 m）× 若干行：底箱 1 行（离地 0.33 m）、中箱 2 行（0.51 / 0.67 m）、顶箱 2 行（0.83 / 1.07 m），每张标签按种子随机抖动不超过 1 cm；每面合计 6 列 × 5 行，标签（含底板 10 × 5 cm）相邻间隔约 3 cm。用于测试密集多码时的选码与去重。

(suite-tools)=
## 加载、查看、生成与预览

### 在代码中加载

```python
from g1_barcode_sim import G1BarcodeEnv
from g1_barcode_sim.testsuite import list_suites, load_suite, load_scenario

env = G1BarcodeEnv(headless=True)                 # 默认 scene="scenario"
for sc in load_suite("t1_L0"):                    # 按 id 顺序返回场景 dict
    obs = env.reset(scenario=sc)                  # dict、JSON 路径或场景 id 均可
    gt = env.get_ground_truth()                   # 仅评测
    ...
obs = env.reset(scenario="t1_L0_0000")            # 直接用 id
```

每次 reset 会删除旧场景并按 JSON 重建桌子、箱子和标签，切换背景（灯光 = 背景基准值 × `light_scale`，三种背景的 dome / distant 基准值为 warehouse 3000 / 1000、none 280 / 90、white 1000 / 300），再把机器人放到起点。只读写或检查场景、不需要仿真时，只导入 `g1_barcode_sim.testsuite` 即可，这部分是纯 Python。只有测试场景模式支持 `reset(scenario=...)`，在内置场景的环境上传会报错。

### 命令行工具

```bash
# 纯 Python（不需要 Isaac Sim）
python scripts/testsuite/list.py                          # 列出所有 suite：task、level、场景数、拒绝次数
python scripts/testsuite/show.py --id t1_L2_0007          # 打印一个场景的 JSON
python scripts/testsuite/generate.py --suite t1_L0        # 重新生成一个 suite
python scripts/testsuite/generate.py --suite t1_L1 --factor background   # L1 只生成某个因素
python scripts/testsuite/generate.py --suite t1_L2 --num 100 --seed 0    # L2 指定数量和种子
python scripts/testsuite/generate.py --suite all          # factors.yaml 中定义的全部桌面 suite（不动 stack*）

# 需要 Isaac Sim
python scripts/testsuite/preview.py --suite t2_L1_barcode_size
python scripts/testsuite/preview.py --suite stack4f_L0 stack4c_L0 --oblique
python scripts/testsuite/preview.py --suite all --max 3 --out_dir outputs/previews
```

| `preview.py` 参数 | 说明 |
| --- | --- |
| `--suite NAME [NAME ...]` | 一个或多个 suite 名，或 `all`（含手写的箱堆 suite） |
| `--max N` | 每个 suite 只渲染前 N 个场景 |
| `--oblique` | 每个场景另外渲染 4 张斜上方视角：`left_rear`、`right_rear`、`left_front`、`right_front`，以站在起点的机器人为参照命名（`left_rear` = 机器人左后方），画面框住机器人起点和所有箱子 |
| `--out_dir DIR` | 输出到 `DIR/<suite>/`，默认 `suites/<suite>/previews/` |
| `--settle_s S` | reset 后先推进多少秒仿真时间再渲染，让贴图加载完，默认 0.5 |

每个场景输出 `<id>_head.png`（起点头部相机）、`<id>_top.png`（俯视图，图像右方为 +x），加 `--oblique` 时还有 4 张斜视图；每个 suite 另有拼图 `overview.png`（每格标注序号、目标首字母 L / R / A、背景和因素值）。

**生成输出**：`suites/<suite>/scenarios/<id>.json`（重新生成前会删除旧的 `scenarios/`）和 `manifest.json`（suite 信息、`rejections`、`warnings`、`skipped_infeasible`、`code_hash`、`head_camera_nominal`、`scenario_ids`）。生成任务 2 套件需要 `configs/testsuite/head_camera_nominal.yaml`（站立时头部相机的内参、`T_root_cam` 和骨盆高度），仓库已附带；只有改了相机或机器人才需要用 `measure_head_camera.py` 重新测量。

### 手写新的场景

1. 复制一份现有场景（如 `stack4f_L0_0000.json`），修改 `boxes`（用 `on` 叠放）和 `barcodes`（用 `face` 选面，同面可多张）。
2. `id` 写成 `<suite>_NNNN`，放到 `suites/<suite>/scenarios/` 下，`load_scenario` 才能按 id 找到它。
3. 同时写一份 `manifest.json`，`list.py` 和 `preview.py --suite all` 依据它列出 suite。
4. 运行 `python -m pytest tests/test_stack_faces.py`，确认所有场景都能通过几何检查。

新增一个生成器因素：在 `factors.yaml` 加入因素名、标称值和范围；在 `testsuite/sampler.py` 中实现采样；需要仿真支持时在 `scenario_builder.py` 中实现搭建；在合法性检查中补充约束。
