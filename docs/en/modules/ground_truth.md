(gt)=
# Coordinate frames and ground truth

## Conventions

| Item | Convention |
| --- | --- |
| World frame | z up, floor at z = 0, meters and radians. All scenes live under `/World/BarcodeScene`; there is only 1 env, whose origin is the world origin |
| Root frame | Pelvis: x forward, y left, z up. `step`, `move_relative`, and wrist and aim targets all refer to it |
| Robot start pose | Built-in scenes: at the origin with yaw = 0, facing +x (towards the scene objects). Task suites: see {ref}`suite-schema` |
| Camera optical frame | ROS convention: x right, y down, z forward |
| Transform naming | `T_a_b` is the pose of b in a, `p_a = T_a_b · p_b`, e.g. `T_world_cam = T_world_root · T_root_cam` |
| Quaternions | `wxyz`; boxes rotate about z only |
| Label frame | `normal` is the outward normal, `up` is the "up" direction of the pattern, `right = up × normal` |

(gt-fields)=
## Ground truth

`env.get_ground_truth(d=0.20)` returns the GT of the current scene.

```{warning}
**GT is for evaluation only** and must not be read by the algorithm under test. The same applies to `T_world_root_gt` in the observation and to the `gt` returned by `scan()`.
```

`check_env.py` writes the GT to `outputs/<scene>/gt.json`.

| Field | Description |
| --- | --- |
| `scene`, `seed`, `randomized` | Scene name (or scenario id), reset seed, whether labels are randomized |
| `robot_init` | `{pos: [x, y, 0], yaw}`, initial root pose |
| `boxes[]` | `{id, size: [x depth, y width, z height], pos (box center), quat_wxyz, distractor}` |
| `barcodes[].content`, `type` | Barcode content (unique within the scene) and symbology |
| `barcodes[].box_id`, `face` | Box and face the label is on |
| `barcodes[].center`, `normal`, `up` | Center of the label's front surface, outward normal, "up" direction of the pattern (world frame; `up` includes `rot_deg`) |
| `barcodes[].size`, `pattern_size` | Label outline [w, h] (including backing or quiet zone); actual barcode pattern [w, h] (currently `null` for built-in scenes) |
| `barcodes[].height_above_ground`, `distractor` | Height of the center above the floor; whether the label is on a distractor box |
| `barcodes[].target_root_pose_d020` | Reference standing pose `{pos: [x, y, 0], yaw}`: the normal is projected onto the horizontal plane and normalized to n_h; pos = horizontal projection of center + d·n_h, and yaw faces the label. The suffix is d in centimeters, e.g. `target_root_pose_d040` for `d=0.40` |

Task-suite mode adds: `scenario_id`, `task`, `target`, `target_box_id`, `target_barcode`, `target_face` (front face of the target box, `{center, normal, corners[4]}`), `task1_gt_obs_pose` (GT observation pose: the smallest distance in 0.60–1.00 m, in 0.05 m steps, at which the head camera image contains all 4 corners of the target face with a 5% margin) and `task2_gt_wrist_target` (barcode center + 0.20 m × normal, with the scan direction along −normal). When `target` is `"all"` (stacks), `target_box_id` is `"all"`, `target_barcode` is `null`, and `faces` is added: it lists every exposed side face of the stack in the patrol order −x, −y, +x, +y, with its center, normal, four corners, observation pose and the barcode contents on that face.

GT of `s1_table`, seed 0 (excerpt):

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

`center.x` = box center 1.5 − half depth 0.15 − face offset 0.0015 = 1.3485.
