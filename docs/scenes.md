(scenes)=
# 内置场景

三个内置场景共用一个仓库背景（Isaac 自带的 Simple_Warehouse，只作外观，碰撞已关闭）和同一套灯光（Dome 3000 + Distant 1000）。机器人从世界原点出发，面朝 +x。纸箱有一个与箱子尺寸一致的静态碰撞体，外观用 Isaac 的纸箱资产 `SM_CardBoxD_04.usd` 缩放到箱子尺寸。场景配置在 `configs/scenes/<scene>.yaml`，公共部分在 `_common.yaml`。每个场景配两张图：机器人左后方视角，以及初始位姿下的头部相机画面。

| 场景 | 布局 | 条码 | 用途 |
| --- | --- | --- | --- |
| `s1_table` | 白桌 1 张，桌上纸箱 2 个 | 2 张，都在朝向机器人的 −x 面，离地 0.90 m | 最简单的接近、瞄准、扫码 |
| `s2_shelf` | 四层货架，纸箱 9 个 | 7 张目标码（−x 面，离地 0.225–1.15 m）+ 2 张干扰码（顶层，贴在背面） | 不同高度的搜索，干扰条码 |
| `s3_stack` | 地面 3×3×3 纸箱堆 | 36 张，四个外露侧面每箱一张 | 绕堆巡视，多面扫码 |

(scene-s1)=
## s1_table

::::{container} figpair
```{figure} images/scene_s1_table_left_rear.jpg
:alt: s1_table 左后方视角：白桌上两个纸箱，机器人站在桌前约 1.3 m

左后方视角
```
```{figure} images/scene_s1_table_head.jpg
:alt: s1_table 头部相机画面：正前方白桌与两个纸箱

头部相机（初始位姿）
```
::::

家具
: 白色哑光桌 T01：深 0.60 × 宽 1.20 × 高 0.75 m，前沿在 x = 1.30，横向居中于 y = 0

纸箱
: B01、B02，尺寸 0.30 × 0.40 × 0.30 m（x 深 × y 宽 × z 高），离桌沿 5 cm，中心在 (1.50, −0.25, 0.90) 和 (1.50, +0.25, 0.90)

条码
: `S1-B01-F0`（右）、`S1-B02-F0`（左），都在 −x 面中心，标签中心 x = 1.3485、离地 0.90 m

起点
: 原点，面朝 +x；标签在正前方约 1.35 m

用途
: 调通"接近—瞄准—扫码"流程；`check_env.py` 的行走和扫码检查都在这个场景里跑

```bash
python scripts/run_scene.py --scene s1_table                # 键盘控制（加 --randomize 开启标签随机化）
python scripts/teleop.py --scene s1_table --demo_scanner    # 扫码枪测试序列
python scripts/render_views.py --scenes s1_table            # 俯视 / 斜视 / 头部图 -> outputs/s1_table/
python examples/goto_barcode.py                             # 用 move_relative 走到第一张标签的参考站位
```

(scene-s2)=
## s2_shelf

::::{container} figpair
```{figure} images/scene_s2_shelf_left_rear.jpg
:alt: s2_shelf 左后方视角：蓝色四层货架上放着九个纸箱，机器人在货架前

左后方视角
```
```{figure} images/scene_s2_shelf_head.jpg
:alt: s2_shelf 头部相机画面：正前方的货架与纸箱

头部相机（初始位姿）
```
::::

家具
: 货架 SH01：0.60 × 2.00 × 2.00 m，前沿在 x = 2.00，四层板面高 0.10 / 0.55 / 1.00 / 1.45 m

纸箱
: 9 个，边长 0.25–0.40 m，箱面比板沿退后 3 cm，横向间隔 ≥ 10 cm

条码
: B01–B07 在 −x 面各贴 1 张，离地 0.25、0.275、0.225、0.75、0.675、1.15、1.125 m；顶层 B08、B09 是干扰箱，标签贴在背面 +x（离地 1.60 m），从机器人一侧看不到，GT 中 `distractor: true`

起点
: 原点，面朝 +x；货架在正前方 2 m

用途
: 不同高度（需要下蹲或抬臂）的搜索与扫码，以及干扰条码

```bash
python scripts/run_scene.py --scene s2_shelf
python scripts/record_walk.py --scene s2_shelf              # 行走录像 -> outputs/walk_s2_shelf.mp4
```

(scene-s3)=
## s3_stack

::::{container} figpair
```{figure} images/scene_s3_stack_left_rear.jpg
:alt: s3_stack 左后方视角：地面上 3×3×3 的立方纸箱堆，机器人在箱堆前

左后方视角
```
```{figure} images/scene_s3_stack_head.jpg
:alt: s3_stack 头部相机画面：正前方的纸箱堆

头部相机（初始位姿）
```
::::

箱堆
: 3 × 3 × 3 共 27 个立方纸箱，边长 0.44 m，间隙 5 mm；前沿在 x = 1.50，占地约 1.34 × 1.34 m（x 1.50–2.835，y ±0.67），高约 1.33 m

条码
: 前（−x）、后（+x）、左（+y）、右（−y）四个外露面上每个箱子各 1 张，共 36 张；离地 0.22 / 0.665 / 1.11 m。箱号按 z → y → x 的顺序递增

起点
: 原点，面朝 +x；箱堆前表面在正前方 1.5 m

用途
: 绕堆巡视、多面扫码

```bash
python scripts/run_scene.py --scene s3_stack
python -c "from g1_barcode_sim import G1BarcodeEnv; env = G1BarcodeEnv('s3_stack', headless=True); print(len(env.get_ground_truth()['barcodes'])); env.close()"
```

```{tip}
纸箱资产的贴图上自带几张印刷的运单贴纸，看起来像条码但不能解码，是天然的干扰。若不需要，在 `_common.yaml` 中设 `box_visual.kind: geometry` 换成纯色纸箱。新增内置场景：新建 `configs/scenes/s4_xxx.yaml`（需要 `scene_id`、`name`，以及 `furniture` / `boxes` / `stacks` 中的一项或几项），然后 `G1BarcodeEnv("s4_xxx")`；放不下（深度不够、超出层间净空、横向越界、标签伸出箱面）时会直接报错。
```
