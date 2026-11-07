(scenes)=
# Built-in scenes

The three built-in scenes share one warehouse background (Isaac's Simple_Warehouse, visual only, with collisions disabled) and the same lighting (Dome 3000 + Distant 1000). The robot starts at the world origin facing +x. Each box has a static collider matching its size; its appearance is Isaac's cardboard box asset `SM_CardBoxD_04.usd` scaled to the box size. Scene configurations are in `configs/scenes/<scene>.yaml`, with shared settings in `_common.yaml`. Each scene is shown in two images: a view from behind the robot's left shoulder, and the head camera image at the start pose.

| Scene | Layout | Barcodes | Purpose |
| --- | --- | --- | --- |
| `s1_table` | 1 white table with 2 boxes on it | 2, both on the −x face facing the robot, 0.90 m above the floor | The simplest approach, aim and scan |
| `s2_shelf` | Four-level shelf with 9 boxes | 7 target codes (−x face, 0.225–1.15 m above the floor) + 2 distractor codes (top level, on the back face) | Search at different heights, distractor barcodes |
| `s3_stack` | 3×3×3 stack of boxes on the floor | 36, one per box on each of the four exposed side faces | Walking around the stack, scanning multiple faces |

(scene-s1)=
## s1_table

::::{container} figpair
```{figure} ../images/scene_s1_table_left_rear.jpg
:alt: s1_table from behind the robot's left shoulder: two boxes on a white table, the robot standing about 1.3 m in front of the table

Left-rear view
```
```{figure} ../images/scene_s1_table_head.jpg
:alt: s1_table head camera image: the white table and two boxes straight ahead

Head camera (start pose)
```
::::

Furniture
: White matte table T01: depth 0.60 × width 1.20 × height 0.75 m, front edge at x = 1.30, centered laterally at y = 0

Boxes
: B01 and B02, size 0.30 × 0.40 × 0.30 m (x depth × y width × z height), 5 cm back from the table edge, centers at (1.50, −0.25, 0.90) and (1.50, +0.25, 0.90)

Barcodes
: `S1-B01-F0` (right) and `S1-B02-F0` (left), both at the center of the −x face; label centers at x = 1.3485, 0.90 m above the floor

Start
: Origin, facing +x; the labels are about 1.35 m straight ahead

Purpose
: Getting the approach, aim and scan pipeline working; the walking and scanning checks of `check_env.py` run in this scene

```bash
python scripts/run_scene.py --scene s1_table                # keyboard control (add --randomize for label randomization)
python scripts/teleop.py --scene s1_table --demo_scanner    # scanner test sequence
python scripts/render_views.py --scenes s1_table            # top-down / oblique / head images -> outputs/s1_table/
python examples/goto_barcode.py                             # walk to the reference standing pose of the first label with move_relative
```

(scene-s2)=
## s2_shelf

::::{container} figpair
```{figure} ../images/scene_s2_shelf_left_rear.jpg
:alt: s2_shelf from behind the robot's left shoulder: nine boxes on a blue four-level shelf, the robot in front of the shelf

Left-rear view
```
```{figure} ../images/scene_s2_shelf_head.jpg
:alt: s2_shelf head camera image: the shelf and boxes straight ahead

Head camera (start pose)
```
::::

Furniture
: Shelf SH01: 0.60 × 2.00 × 2.00 m, front edge at x = 2.00, four shelf boards at heights 0.10 / 0.55 / 1.00 / 1.45 m

Boxes
: 9 boxes with side lengths of 0.25–0.40 m; box faces are set back 3 cm from the shelf edge, with lateral gaps ≥ 10 cm

Barcodes
: B01–B07 each carry 1 label on the −x face, at 0.25, 0.275, 0.225, 0.75, 0.675, 1.15 and 1.125 m above the floor. B08 and B09 on the top level are distractor boxes whose labels are on the back face +x (1.60 m above the floor), not visible from the robot's side; in the GT they have `distractor: true`

Start
: Origin, facing +x; the shelf is 2 m straight ahead

Purpose
: Searching and scanning at different heights (requiring squatting or raising the arm), and distractor barcodes

```bash
python scripts/run_scene.py --scene s2_shelf
python scripts/record_walk.py --scene s2_shelf              # walking video -> outputs/walk_s2_shelf.mp4
```

(scene-s3)=
## s3_stack

::::{container} figpair
```{figure} ../images/scene_s3_stack_left_rear.jpg
:alt: s3_stack from behind the robot's left shoulder: a 3×3×3 stack of cubic boxes on the floor, the robot in front of the stack

Left-rear view
```
```{figure} ../images/scene_s3_stack_head.jpg
:alt: s3_stack head camera image: the box stack straight ahead

Head camera (start pose)
```
::::

Stack
: 3 × 3 × 3 = 27 cubic boxes with 0.44 m sides and 5 mm gaps; front edge at x = 1.50, footprint about 1.34 × 1.34 m (x 1.50–2.835, y ±0.67), height about 1.33 m

Barcodes
: One label per box on each of the four exposed faces (front −x, back +x, left +y, right −y), 36 in total, at 0.22 / 0.665 / 1.11 m above the floor. Box numbers increase in z → y → x order

Start
: Origin, facing +x; the front face of the stack is 1.5 m straight ahead

Purpose
: Walking around the stack and scanning multiple faces

```bash
python scripts/run_scene.py --scene s3_stack
python -c "from g1_barcode_sim import G1BarcodeEnv; env = G1BarcodeEnv('s3_stack', headless=True); print(len(env.get_ground_truth()['barcodes'])); env.close()"
```

```{tip}
The texture of the cardboard box asset includes a few printed shipping labels that look like barcodes but cannot be decoded, which makes them natural distractors. If you do not want them, set `box_visual.kind: geometry` in `_common.yaml` to use plain-colored boxes. To add a built-in scene, create `configs/scenes/s4_xxx.yaml` (it needs `scene_id`, `name`, and one or more of `furniture` / `boxes` / `stacks`), then call `G1BarcodeEnv("s4_xxx")`. If something does not fit (insufficient depth, exceeding the clearance between shelf levels, out of bounds laterally, a label extending past the box face), an error is raised immediately.
```
