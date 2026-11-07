(suites)=
# Task suites

A task suite is a set of reproducible test scenarios. Each scenario is fully described by one JSON file, loaded with `env.reset(scenario=...)`, and scenarios can be switched one after another within the same process. There are currently two kinds: **table scenarios** (`t1_*`, `t2_*`, 19 suites, produced by a generator from a factor table) and **stack scenarios** (`stack*`, 6 suites, hand-written). The suites will keep growing, with new scenario types and difficulty levels added over time.

```{note}
A scenario JSON describes only the scene itself and contains no GT targets. GT is computed by `get_ground_truth()` after reset and is for evaluation only.
```

(suite-tasks)=
## Task definitions and levels

| task | Scene | Start | Goal | GT for evaluation |
| --- | --- | --- | --- | --- |
| 1 | Two boxes on a table; `target` says whether the left or right one is the target | 2–5 m from the table edge, possibly with a lateral offset | Find the target box and walk to an observation pose from which its front face is clearly visible | `task1_gt_obs_pose` |
| 2 | Same as above | Near the observation pose in front of the target box (GT observation pose plus noise and a lateral offset), guaranteeing that the target box face and the barcode are within the head camera image | Scan the barcode on the target box | `target_barcode`, `task2_gt_wrist_target` |
| 3 | Box stack on the floor, `target: "all"` | 2.5 m in front of the stack | Scan all barcodes on the stack; with `scan_faces: "all"` the robot must walk around the stack and scan all four side faces | `faces`, the pose of every label |

The start of task 2 is generated from the GT observation pose, so it can be tested independently of task 1. "Left / right" is from the robot's point of view: left is +y.

### Levels L0 / L1 / L2

| Level | Meaning | Use |
| --- | --- | --- |
| L0 | Nominal scenarios, varied only over a small grid (e.g. start distance × target left/right) | Getting the pipeline working |
| L1 | Single-factor sweeps: one factor changes at a time while the others stay at their nominal values. Continuous factors take evenly spaced points within their range; discrete factors take each value in turn | Finding which factor causes failures; plotting success rate against factor value |
| L2 | All factors of the task sampled uniformly within their ranges, with the target alternating left/right (half each) | Measuring the overall success rate |

The 8 factors of task 1: `robot_dist`, `robot_lateral`, `box_shape`, `box_appearance`, `placement`, `barcode_size`, `background`, `lighting`. The 7 factors of task 2: `barcode_pos`, `barcode_size`, `barcode_type`, `box_shape`, `box_lateral`, `start_noise`, `background`. Nominal values and ranges are all in `configs/testsuite/factors.yaml`:

| Parameter | Nominal | Range / values |
| --- | --- | --- |
| Robot distance to table edge `dist_to_table` | 3.0 m | 2.0–5.0 m |
| Lateral offset `lateral_offset` | 0 | −1.5–1.5 m |
| Initial heading `yaw_deg` | 0 | Fixed to 0 by the generator (hand-written scenarios may use non-zero values) |
| Box width w (along y) / height h / depth d (along x) | 0.40 / 0.30 / 0.30 m | 0.20–0.60 / 0.15–0.50 / 0.20–0.50 m |
| Box appearance | `kraft` | `kraft` / `kraft_tape` / `white_box` / `printed` |
| Table depth / width / height | 0.60 / 1.20 / 0.75 m | 0.50–0.80 / 1.00–1.80 / 0.70–0.80 m (white matte, fixed) |
| Box front face to table edge `setback` | 0.05 m | 0–0.20 m |
| Gap between the two boxes `gap` | 0.10 m | 0.05–0.60 m |
| Lateral center of the box pair `group_y` | 0 | Sampled uniformly within the range that keeps both boxes on the table |
| In-plane barcode rotation | 0° | Base {0°, 90°} plus a ±10° perturbation |
| Symbology | `code128` | `code128` / `qr` |
| Code128 pattern width / aspect ratio (height/width) | 0.08 m / 0.375 | 0.04–0.12 m / 0.25–0.60 |
| QR side length | 0.05 m | 0.03–0.08 m |
| Content length | 9 characters | 6–20 characters, alphabet `ABCDEFGHJKLMNPQRSTUVWXYZ23456789` (easily confused characters removed) |
| White backing | Yes | Yes / no |
| Background | `warehouse` | `warehouse` / `none` (gray floor) / `white` (white floor and walls) |
| Lighting scale | 1.0 | 0.7–1.3 |
| Task 2 start noise | ±0.03 m / ±3° | `start_noise` enlarges it up to ±0.08 m / ±8° |
| Task 2 observation distance `d_obs` | — | 0.60–1.00 m in 0.05 m steps, with a 5% margin around the image |

**Validity checks**: performed automatically during generation; a scenario that fails is resampled (at most 50 times per scenario), and the rejection reasons are recorded in the manifest's `rejections`. The checks: the two boxes plus their gap fit within the table width; no box overhangs the table; the label (with a 1 cm margin) fits on the box face; the barcode center is ≤ 1.3 m above the floor; the robot's footprint circle (radius 0.35 m) does not intersect the table or boxes; at the task 2 start, all four corners of the box face and of the barcode project into the head camera image. Infeasible L1 sweep points are skipped and recorded in `skipped_infeasible`. When a swept box does not fit on the nominal table, the table is enlarged within its own range just enough to fit, which is recorded in `factor.table_adjusted`.

**Reproducibility**: the random stream of each scenario is derived from `SHA-256(suite name|seed|index|attempt)`, independent of Python's hash seed and of generation order. Each suite is generated twice in memory and written to disk only if both are byte-identical. The manifest contains no timestamps; `code_hash` records a hash of the generator source and configuration.

(suite-schema)=
## Scenario JSON format

Files live in `suites/<suite>/scenarios/<id>.json`. Hand-written and generated scenarios use the same format.

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Type
  - Description
* - `id`
  - str
  - `<suite>_<4-digit index>`, e.g. `t1_L2_0007`. At load time the trailing `_NNNN` is stripped to locate the suite directory
* - `suite`, `index`
  - str, int
  - Suite name; index of the scenario within the suite
* - `task`
  - int
  - 1, 2 or 3, see {ref}`suite-tasks`
* - `seed`
  - int
  - Base seed of the suite (currently always 0), used as the environment seed at reset
* - `level`
  - str
  - `L0` / `L1` / `L2`; for anything other than L0, `randomized` in the GT is true
* - `factors_varied`
  - list
  - For L1, the swept factor; for L2, all factors of the task; empty for L0 and hand-written scenarios
* - `factor`
  - dict / null
  - L0 records the grid values, e.g. `{"dist_to_table": 2.0, "target": "left"}`; L1 is `{"name", "value"}` (possibly with `table_adjusted`); L2 is `null`; hand-written scenarios use `{"note": "design notes"}`
* - `background`
  - dict
  - `{"type": "warehouse"|"none"|"white", "light_scale": float}`. Lighting = base value of that background × `light_scale`
* - `table`
  - dict / null
  - `{"depth", "width", "height", "appearance": "white_matte"}`. `null` means there is no table and the boxes stand on the floor
* - `robot_init`
  - dict
  - `{"dist_to_table", "lateral_offset", "yaw_deg"}`. The table edge (or the front face of the stack) is fixed at x = 1.30, parallel to the y axis; the robot starts at (1.30 − `dist_to_table`, `lateral_offset`). `yaw_deg` is counterclockwise positive (+30 means turned 30° to the left); 0 means facing +x
* - `boxes[]`
  - list
  - See the table below
* - `barcodes[]`
  - list
  - See the table below
* - `target`
  - str
  - `"left"` / `"right"` (from the robot's point of view) or `"all"` (task 3, all boxes)
* - `task2_start`
  - dict / null
  - Only set for task 2: `pose_world{x, y, yaw_deg}`, `noise{dx, dy, dyaw_deg, bound_pos, bound_yaw_deg}`, `lateral_offset`, `lateral_feasible[lo, hi]`, `obs_d`, `obs_fits`. When set, the start pose is `pose_world` and `robot_init` is ignored
* - `scan_faces`
  - str, optional
  - `"all"` means all four side faces of the stack must be scanned (currently in `stack4f_L0` and `stack4c_L0`). The simulation itself does not read this field; it is part of the task description given to the system under test
```

| `boxes[i]` field | Description |
| --- | --- |
| `id` | The generator writes `B0` / `B1` (in increasing y, so B0 is on the robot's right); hand-written scenarios may use their own ids, e.g. `LF2` |
| `size_wdh` | `[w (along y), d (along x), h]` in meters. Note that this order differs from the `[x, y, z]` order of `size` in the GT |
| `appearance` | `kraft` (kraft paper color), `kraft_tape` (packing tape over the top seam, running 6 cm down the front face), `white_box` (white box), `printed` (printed box with FRAGILE / THIS SIDE UP text, a lot number, a logo and arrows), `cardbox` (Isaac's cardboard box art asset `SM_CardBoxD_04.usd`, scaled to the box size, same as in the built-in scenes) |
| `setback` | Distance from the box front face to x = 1.30, meters |
| `y_center` | y coordinate of the box center, meters |
| `on` | Optional. The id of a box defined earlier; this box is stacked on its top face (it must not overhang the edges of the box below). `null` or omitted means the box stands on the table or floor |

| `barcodes[i]` field | Description |
| --- | --- |
| `box_id` | Box the label is on |
| `face` | Optional, `-x` / `+x` / `-y` / `+y`, default `-x`. A face can carry several labels, which must not overlap |
| `type` | `code128` or `qr` |
| `width`, `aspect` | Code128 pattern width and aspect ratio (height/width); QR uses only `width` (side length) |
| `label_backing` | Whether the label has a white backing (1 cm wider than the pattern on each side) |
| `u`, `v`, `rot_deg` | In-plane position (from the face center, meters) and rotation, defined in {ref}`label-geometry`. On the −x face, positive `u` points to the robot's right, −y |
| `content`, `content_id` | The string actually encoded in the barcode; a human-readable id (e.g. `T1L2-0007-B0`) |

`suites/t2_L0/scenarios/t2_L0_0000.json` (condensed):

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
## Table scenarios (`t1_*`, `t2_*`)

19 suites, all produced by the generator: one white matte table with two boxes on it, one barcode per box, all on the −x face facing the robot. The nominal configuration is a 0.40 × 0.30 × 0.30 m box (w × d × h), `kraft` appearance, `warehouse` background, Code128 with backing, and the barcode at the center of the box face. The "Left / right" column gives the number of scenarios with the target box on the left and on the right.

| Suite | task | Scenarios | What varies | Appearance / barcodes | Left / right |
| --- | --- | --- | --- | --- | --- |
| `t1_L0` | 1 | 8 | Start distance {2, 3, 4, 5} m × target {left, right} | Nominal | 4 / 4 |
| `t1_L1_robot_dist` | 1 | 12 | Start distance 2.0 → 5.0 m | Nominal | 6 / 6 |
| `t1_L1_robot_lateral` | 1 | 13 | Lateral offset −1.5 → 1.5 m in 0.25 m steps | Nominal | 7 / 6 |
| `t1_L1_box_shape` | 1 | 12 | Target box scaled from smallest (0.2 × 0.2 × 0.15) to largest (0.6 × 0.5 × 0.5), with w / d / h interpolated together | Nominal | 6 / 6 |
| `t1_L1_box_appearance` | 1 | 12 | Appearance kraft / kraft_tape / white_box / printed, 3 each | Both boxes share the appearance | 8 / 4 |
| `t1_L1_placement` | 1 | 12 | Sweeps setback 0 → 0.2, gap 0.05 → 0.6 and group_y leftmost → rightmost in turn, 4 points each; in scenarios 6 and 7 the table is widened automatically to 1.2167 m and 1.4 m | Nominal | 6 / 6 |
| `t1_L1_barcode_size` | 1 | 10 | Target barcode width 0.04 → 0.12 m | Code128 | 5 / 5 |
| `t1_L1_background` | 1 | 12 | Background warehouse / none / white, 4 each | Nominal | 6 / 6 |
| `t1_L1_lighting` | 1 | 10 | Lighting scale 0.7 → 1.3 | warehouse background | 5 / 5 |
| `t1_L2` | 1 | 100 | All 8 factors randomized together | All four appearances; Code128 102 / QR 98 labels; with backing 96 / without 104 labels | 50 / 50 |
| `t2_L0` | 2 | 8 | `group_y` {−0.15, −0.05, 0.05, 0.15} m × target {left, right}; starting at the GT observation pose, no noise | Nominal | 4 / 4 |
| `t2_L1_barcode_pos` | 2 | 12 | In-plane barcode position: 4 u levels × 3 v levels (as fractions of the allowed range) | Nominal | 6 / 6 |
| `t2_L1_barcode_size` | 2 | 10 | Barcode width 0.04 → 0.12 m | Code128 | 5 / 5 |
| `t2_L1_barcode_type` | 2 | 12 | {Code128, QR} × {with backing, without backing}, 3 each | QR side length at the nominal 0.05 m | 8 / 4 |
| `t2_L1_box_shape` | 2 | 10 | Target box scaled up from the smallest size (12 points defined; the two largest are skipped because the whole box face does not fit in the image within the 1.0 m observation distance) | Nominal | 5 / 5 |
| `t2_L1_box_lateral` | 2 | 12 | Lateral offset of the observation pose relative to the target box, swept from one end of the visible interval to the other | Nominal | 6 / 6 |
| `t2_L1_start_noise` | 2 | 12 | Start noise bound enlarged from ±0.03 m / ±3° to ±0.08 m / ±8° | Nominal | 6 / 6 |
| `t2_L1_background` | 2 | 12 | Background warehouse / none / white, 4 each | Nominal | 6 / 6 |
| `t2_L2` | 2 | 100 | All 7 factors randomized together (for `t2_L2_0056` and `0057` no distance in 0.6–1.0 m leaves a 5% margin, so the observation distance is set to the upper bound 1.0 m) | All four appearances; Code128 93 / QR 107 labels; with backing 102 / without 98 labels | 50 / 50 |

In L1 the targets are not always split evenly: the target alternates every 4 scenarios by index, so 12 scenarios give three groups "left, right, left", i.e. 8 / 4.

Each card shows one representative scenario of the suite, viewed from behind the robot's left shoulder:

::::{container} cards
```{figure} ../images/suite_t1_L0_left_rear.jpg
:alt: Representative scenario of t1_L0

**`t1_L0`** · task 1 · L0 · 8 scenarios: start distance × target left/right
```
```{figure} ../images/suite_t1_L1_robot_dist_left_rear.jpg
:alt: Representative scenario of t1_L1_robot_dist

**`t1_L1_robot_dist`** · task 1 · L1 · 12 scenarios: start distance 2–5 m
```
```{figure} ../images/suite_t1_L1_robot_lateral_left_rear.jpg
:alt: Representative scenario of t1_L1_robot_lateral

**`t1_L1_robot_lateral`** · task 1 · L1 · 13 scenarios: lateral offset ±1.5 m
```
```{figure} ../images/suite_t1_L1_box_shape_left_rear.jpg
:alt: Representative scenario of t1_L1_box_shape

**`t1_L1_box_shape`** · task 1 · L1 · 12 scenarios: target box size
```
```{figure} ../images/suite_t1_L1_box_appearance_left_rear.jpg
:alt: Representative scenario of t1_L1_box_appearance

**`t1_L1_box_appearance`** · task 1 · L1 · 12 scenarios: four box appearances
```
```{figure} ../images/suite_t1_L1_placement_left_rear.jpg
:alt: Representative scenario of t1_L1_placement

**`t1_L1_placement`** · task 1 · L1 · 12 scenarios: setback, gap, lateral shift of the pair
```
```{figure} ../images/suite_t1_L1_barcode_size_left_rear.jpg
:alt: Representative scenario of t1_L1_barcode_size

**`t1_L1_barcode_size`** · task 1 · L1 · 10 scenarios: barcode width 4–12 cm
```
```{figure} ../images/suite_t1_L1_background_left_rear.jpg
:alt: Representative scenario of t1_L1_background

**`t1_L1_background`** · task 1 · L1 · 12 scenarios: warehouse / no background / plain white
```
```{figure} ../images/suite_t1_L1_lighting_left_rear.jpg
:alt: Representative scenario of t1_L1_lighting

**`t1_L1_lighting`** · task 1 · L1 · 10 scenarios: lighting ×0.7–1.3
```
```{figure} ../images/suite_t1_L2_left_rear.jpg
:alt: Representative scenario of t1_L2

**`t1_L2`** · task 1 · L2 · 100 scenarios: all factors randomized
```
```{figure} ../images/suite_t2_L0_left_rear.jpg
:alt: Representative scenario of t2_L0

**`t2_L0`** · task 2 · L0 · 8 scenarios: lateral shift of the pair × target left/right
```
```{figure} ../images/suite_t2_L1_barcode_pos_left_rear.jpg
:alt: Representative scenario of t2_L1_barcode_pos

**`t2_L1_barcode_pos`** · task 2 · L1 · 12 scenarios: in-plane barcode position
```
```{figure} ../images/suite_t2_L1_barcode_size_left_rear.jpg
:alt: Representative scenario of t2_L1_barcode_size

**`t2_L1_barcode_size`** · task 2 · L1 · 10 scenarios: barcode width 4–12 cm
```
```{figure} ../images/suite_t2_L1_barcode_type_left_rear.jpg
:alt: Representative scenario of t2_L1_barcode_type

**`t2_L1_barcode_type`** · task 2 · L1 · 12 scenarios: symbology × backing
```
```{figure} ../images/suite_t2_L1_box_shape_left_rear.jpg
:alt: Representative scenario of t2_L1_box_shape

**`t2_L1_box_shape`** · task 2 · L1 · 10 scenarios: target box size
```
```{figure} ../images/suite_t2_L1_box_lateral_left_rear.jpg
:alt: Representative scenario of t2_L1_box_lateral

**`t2_L1_box_lateral`** · task 2 · L1 · 12 scenarios: lateral position relative to the target box
```
```{figure} ../images/suite_t2_L1_start_noise_left_rear.jpg
:alt: Representative scenario of t2_L1_start_noise

**`t2_L1_start_noise`** · task 2 · L1 · 12 scenarios: start noise magnitude
```
```{figure} ../images/suite_t2_L1_background_left_rear.jpg
:alt: Representative scenario of t2_L1_background

**`t2_L1_background`** · task 2 · L1 · 12 scenarios: warehouse / no background / plain white
```
```{figure} ../images/suite_t2_L2_left_rear.jpg
:alt: Representative scenario of t2_L2

**`t2_L2`** · task 2 · L2 · 100 scenarios: all factors randomized
```
::::

(suite-stack)=
## Stack scenarios (`stack*`)

6 suites, all hand-written (`definition.note` in the manifest is `hand-written`; they are not managed by the generator, and `generate.py --suite all` does not rewrite them). Common settings: `task: 3`, `level: L0`, `table: null` (boxes stand directly on the floor), `target: "all"`, `warehouse` background, lighting scale 1.0; every box is 0.40 × 0.40 × 0.38 m (w × d × h); all labels are Code128 with an 8 cm wide pattern, aspect ratio 0.375 and a backing (label outline 10 × 5 cm); the robot starts 2.5 m away (standing at x = −1.2).

| Suite | Scenarios | Stack structure | Appearance | Labels per scenario (by face) | `scan_faces` | Start pose |
| --- | --- | --- | --- | --- | --- | --- |
| `stack3_L0` | 2 | 1 column, 3 levels, 3 boxes, at y = 0 | `kraft` | 3 (all on −x) | — | Facing the stack |
| `stack9_L0` | 1 | 3 columns × 3 levels, 9 boxes, 1.2 m wide, 1.14 m high | `kraft` | 5 (all on −x, at least 1 per level) | — | Facing the stack |
| `stack9p_L0` | 1 | Same as `stack9_L0` | 5 `printed` + 4 `kraft_tape` boxes, alternating | 5 (as above) | — | Facing the stack |
| `stack9c_L0` | 1 | Same as `stack9_L0` | `cardbox` | 5 (as above) | — | Facing the stack |
| `stack4f_L0` | 3 | 2 columns × 2 rows (front/back) × 3 levels, 12 boxes, free-standing; footprint x 1.30–2.10, y −0.4–0.4, 1.14 m high | `kraft` | 12 (3 each on −x, −y, +x, +y) | `all` | 0000 facing the stack; 0001 and 0002 not parallel to the stack |
| `stack4c_L0` | 3 | Same as `stack4f_L0` | `cardbox` | 0000: 12 (3 per face); 0001: 6 (only −x and +x, 3 each); 0002: 60 (30 each on −x and +x) | `all` | Facing the stack |

### stack3_L0: single column of three

::::{container} figpair
```{figure} ../images/suite_stack3_L0_left_rear.jpg
:alt: stack3_L0: three cardboard boxes stacked on the floor

Left-rear view
```
```{figure} ../images/suite_stack3_L0_head.jpg
:alt: stack3_L0 head camera image

Head camera (start)
```
::::

- `stack3_L0_0000`: three boxes stacked in one column, one code on the front of each box, at 1.10 m above the floor for the top box (requires raising the arm), 0.57 m for the middle box (requires squatting) and 0.30 m for the bottom box (requires a deep squat).
- `stack3_L0_0001`: the same column, but the bottom barcode is at 0.19 m, below the robot's reach even in a deep squat (about 0.27 m). It tests whether the system under test can recognize and report an unreachable barcode and then continue with the remaining ones.

### stack9_L0 / stack9p_L0 / stack9c_L0: 3 × 3 box wall

::::{container} cards
```{figure} ../images/suite_stack9_L0_left_rear.jpg
:alt: stack9_L0: a 3×3 wall of kraft-colored boxes

**`stack9_L0`** · kraft · 5 labels: a wall of kraft-colored boxes with 5 codes on the front, at least one per level, not one on every box. Barcode heights: top level 1.07–1.10 m, middle level 0.57 m, bottom level 0.30–0.32 m
```
```{figure} ../images/suite_stack9p_L0_left_rear.jpg
:alt: stack9p_L0: a box wall alternating printed boxes and taped boxes

**`stack9p_L0`** · printed + kraft_tape · 5 labels: the same wall and 5 codes, with the boxes replaced by printed boxes (warning text, lot numbers, logos, arrows) alternating with taped boxes, adding visual clutter
```
```{figure} ../images/suite_stack9c_L0_left_rear.jpg
:alt: stack9c_L0: a box wall made of Isaac cardboard box assets

**`stack9c_L0`** · cardbox · 5 labels: the same wall and 5 codes, with all boxes replaced by the Isaac cardboard box asset (kraft color, packing tape, printed shipping labels)
```
::::

### stack4f_L0: free-standing 2 × 2 × 3 stack, labels on four faces

::::{container} figpair
```{figure} ../images/suite_stack4f_L0_left_rear.jpg
:alt: stack4f_L0: a free-standing 2×2×3 stack of kraft-colored boxes on the floor

Left-rear view
```
```{figure} ../images/suite_stack4f_L0_head.jpg
:alt: stack4f_L0 head camera image

Head camera (start)
```
::::

- `stack4f_L0_0000`: 12 codes, 3 on each vertical side face, one per level, alternating between the two columns of that face; labels are only on exposed faces. Barcode heights: top level 1.07–1.10 m, middle level 0.57 m, bottom level 0.33 m. The patrol order (the order of `faces` in the GT) is front (−x), right (−y), back (+x), left (+y). The robot faces the stack.
- `stack4f_L0_0001`, `stack4f_L0_0002`: the same stack and 12 codes, but the robot's initial heading is not parallel to the stack. 0001 is offset 0.5 m to the right (`lateral_offset: -0.5`) and turned 30° to the left (`yaw_deg: 30`); 0002 is offset 0.7 m to the left (`lateral_offset: 0.7`, already beyond the stack's +y side face) and turned 35° to the right (`yaw_deg: -35`). Both keep the front face of the stack within the head camera image and within the 3.0 m depth range.

### stack4c_L0: cardboard asset version with different label counts

::::{container} figpair
```{figure} ../images/suite_stack4c_L0_left_rear.jpg
:alt: stack4c_L0: a 2×2×3 stack built from Isaac cardboard box assets, with printed shipping labels on the box faces

Left-rear view
```
```{figure} ../images/suite_stack4c_L0_head.jpg
:alt: stack4c_L0 head camera image

Head camera (start)
```
::::

The stack is the same as in `stack4f_L0`, with all boxes rendered from the Isaac cardboard box asset (appearance `cardbox`). The asset texture includes shipping labels with barcode-like patterns, but these patterns are **not decodable codes**; they are realistic warehouse distractors. The robot faces the stack in all scenarios.

- `stack4c_L0_0000`: labels on all four side faces, the same 12 codes as `stack4f_L0_0000` (3 per face).
- `stack4c_L0_0001`: labels only on the front (−x) and back (+x) faces, 3 each; none on the left and right sides.
- `stack4c_L0_0002`: 30 labels each on the front and back faces (60 in total), none on the sides. Each box face carries 3 columns (u = −0.13 / 0 / +0.13 m) × several rows: 1 row on the bottom box (0.33 m above the floor), 2 rows on the middle box (0.51 / 0.67 m) and 2 rows on the top box (0.83 / 1.07 m), each label jittered by at most 1 cm according to the seed. Each face thus has 6 columns × 5 rows, with about 3 cm between adjacent labels (10 × 5 cm including backing). It tests code selection and deduplication with many codes close together.

(suite-tools)=
## Loading, inspecting, generating and previewing

### Loading in code

```python
from g1_barcode_sim import G1BarcodeEnv
from g1_barcode_sim.testsuite import list_suites, load_suite, load_scenario

env = G1BarcodeEnv(headless=True)                 # default scene="scenario"
for sc in load_suite("t1_L0"):                    # returns scenario dicts in id order
    obs = env.reset(scenario=sc)                  # a dict, JSON path or scenario id
    gt = env.get_ground_truth()                   # evaluation only
    ...
obs = env.reset(scenario="t1_L0_0000")            # by id
```

Each reset deletes the previous scenario, rebuilds the table, boxes and labels from the JSON, switches the background (lighting = background base value × `light_scale`; the dome / distant base values are warehouse 3000 / 1000, none 280 / 90, white 1000 / 300), and then places the robot at the start pose. To read, write or check scenarios without simulation, importing `g1_barcode_sim.testsuite` is enough; this part is pure Python. Only task-suite mode supports `reset(scenario=...)`; passing it to an environment created with a built-in scene raises an error.

### Command-line tools

```bash
# Pure Python (no Isaac Sim needed)
python scripts/testsuite/list.py                          # list all suites: task, level, scenario count, rejection count
python scripts/testsuite/show.py --id t1_L2_0007          # print the JSON of one scenario
python scripts/testsuite/generate.py --suite t1_L0        # regenerate one suite
python scripts/testsuite/generate.py --suite t1_L1 --factor background   # L1: generate only one factor
python scripts/testsuite/generate.py --suite t1_L2 --num 100 --seed 0    # L2: set count and seed
python scripts/testsuite/generate.py --suite all          # all table suites defined in factors.yaml (stack* untouched)

# Isaac Sim required
python scripts/testsuite/preview.py --suite t2_L1_barcode_size
python scripts/testsuite/preview.py --suite stack4f_L0 stack4c_L0 --oblique
python scripts/testsuite/preview.py --suite all --max 3 --out_dir outputs/previews
```

| `preview.py` argument | Description |
| --- | --- |
| `--suite NAME [NAME ...]` | One or more suite names, or `all` (including the hand-written stack suites) |
| `--max N` | Render only the first N scenarios of each suite |
| `--oblique` | Additionally render 4 oblique views from above per scenario: `left_rear`, `right_rear`, `left_front`, `right_front`, named relative to the robot standing at the start (`left_rear` = behind the robot's left shoulder); each view frames the robot's start position and all boxes |
| `--out_dir DIR` | Write to `DIR/<suite>/`; default `suites/<suite>/previews/` |
| `--settle_s S` | Simulation time to advance after reset before rendering, so textures finish loading; default 0.5 |

Each scenario produces `<id>_head.png` (head camera at the start) and `<id>_top.png` (top-down view, image right is +x), plus 4 oblique views with `--oblique`. Each suite also gets a mosaic `overview.png` (each tile labeled with the index, the target initial L / R / A, the background and the factor value).

**Generation output**: `suites/<suite>/scenarios/<id>.json` (the old `scenarios/` is deleted before regeneration) and `manifest.json` (suite information, `rejections`, `warnings`, `skipped_infeasible`, `code_hash`, `head_camera_nominal`, `scenario_ids`). Generating task 2 suites requires `configs/testsuite/head_camera_nominal.yaml` (head camera intrinsics, `T_root_cam` and pelvis height while standing), which ships with the repository; it only needs to be re-measured with `measure_head_camera.py` after changing the camera or robot.

### Writing new scenarios by hand

1. Copy an existing scenario (e.g. `stack4f_L0_0000.json`) and edit `boxes` (use `on` to stack) and `barcodes` (use `face` to choose a face; a face can carry several labels).
2. Set `id` to `<suite>_NNNN` and put the file in `suites/<suite>/scenarios/` so that `load_scenario` can find it by id.
3. Also write a `manifest.json`; `list.py` and `preview.py --suite all` use it to list suites.
4. Run `python -m pytest tests/test_stack_faces.py` to make sure every scenario passes the geometry checks.

To add a generator factor: add the factor name, nominal value and range to `factors.yaml`; implement its sampling in `testsuite/sampler.py`; implement scene construction in `scenario_builder.py` if the simulation needs to support it; and add constraints to the validity checks.
