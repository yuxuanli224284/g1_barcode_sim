(scanner)=
# Scanner and scan evaluation

## Scanner model

The scanner is built from 5 boxes with an overall envelope of 16 × 6.6 × 10.5 cm, sized after the Zebra DS2208 (only its shape and specifications are followed; it is not a copied asset). The scanner is visual only, with no collider and no mass, but the corners of its parts are added to the robot's clearance model, so `step()` returns `collision` when it hits a table or box.

| Part | Center (G frame, m) | Size (m) |
| --- | --- | --- |
| handle | (0, 0, −0.0175) | 0.040 × 0.030 × 0.105 |
| head | (0.020, 0, 0.060) | 0.100 × 0.066 × 0.050 |
| nose | (0.0625, 0, 0.060) | 0.015 × 0.070 × 0.056 |
| window (exit window) | (0.0705, 0, 0.060) | 0.001 × 0.050 × 0.034 |
| trigger | (0.026, 0, −0.030) | 0.012 × 0.012 × 0.020 |

| Frame | Definition |
| --- | --- |
| hand | `left_hand_base_link` (base of the left Dex1 gripper); the fingers extend along +y_hand |
| G (`scanner_mount`) | Origin at the grip point (hand frame (0, 0.110, 0), between the two finger pads), rotated +90° about z relative to hand: +x_G = scan direction (along the fingers), +z_G = from handle to head, y_G = z_G × x_G |
| tip | Center of the exit window, G frame (0.071, 0, 0.060), axes aligned with G |
| Scanner camera | ROS optical frame: +z (forward) = +x_G, +x (right) = −y_G, +y (down) = −z_G |

(scan)=
## Scan evaluation: `env.scan()`

1. **Trigger**: calling `scan()` is pulling the trigger. The current RGB image of the scanner camera is converted to grayscale and decoded with pyzbar (zbar), with no restriction on symbology.
2. **The result comes entirely from image decoding**: there are no distance, angle or field-of-view thresholds. Poor aim, being too close (label outside the image), too far (modules too small), too oblique an angle of incidence, or occlusion all naturally cause decoding to fail.
3. **Orientation**: the default `scanner.yaml: scan.rotation_step_deg: 90` decodes only the original image without rotation compensation, mimicking real scanners that do not read codes clearly tilted relative to the image. zbar was measured to read 1D codes within about ±17° of the image axes, so when aiming, align the image's horizontal direction with the long axis of the barcode. Setting it to 15 decodes at 0°, 15°, …, 75° (an omnidirectional imager model).
4. **Code selection**: each content is kept only once; when several codes are in the image, `content` is the one closest to the image center, mimicking the aiming point of a real scanner.
5. **GT diagnostic**: the scan ray (the tip's x axis) is additionally intersected with all GT label rectangles, counting only hits from the front side, and the hit label, distance and angle of incidence are reported. **This is for evaluation and failure analysis only**; it ignores occlusion and does not affect `status`.

| Return field | Description |
| --- | --- |
| `status` | `"ok"` (at least one code decoded) or `"no_read"`; `"error"` when the scanner is disabled |
| `content` | Content of the selected barcode (e.g. `S1-B01-F0`); `None` if nothing was decoded |
| `decoded` | All codes decoded in the image, sorted by distance from the image center. Each item contains `content`, `type` (e.g. `CODE128`, `QRCODE`), `center_px`, `polygon`, `rotation_deg`, `offset_from_image_center_px` |
| `T_root_scanner_tip` | Scanner tip pose when the trigger was pulled (4×4) |
| `timestamp` | Simulation time (s) |
| `gt` | `{"hit": {content, distance_m, incidence_deg, hit_offset_m} or None, "note"}`, evaluation only |

Adapted from `examples/scanner_camera.py`: stand in front of a label, aim, scan and save a snapshot.

```python
import math
import numpy as np
from g1_barcode_sim.env import G1BarcodeEnv
from g1_barcode_sim.camera import inv_T

env = G1BarcodeEnv("s1_table", headless=True)
bc = env.get_ground_truth()["barcodes"][0]        # the example uses GT directly; a real algorithm should estimate the label pose from the head camera
c, n = np.array(bc["center"]), np.array(bc["normal"])
env.teleport_root(c[0] + 0.55 * n[0], c[1] + 0.55 * n[1] - 0.20,
                  math.atan2(-n[1], -n[0]))         # label straight ahead of the left shoulder
env.stand_still(0.5)
T = inv_T(env.get_obs()["T_world_root_gt"])        # world -> root
env.aim_scanner_at(T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ n, distance=0.20)
env.stand_still(3.0)                               # aiming converges in about 1–2 s
obs = env.get_obs()
print(obs["scanner_aim_error"])                    # {"pos_m": ..., "angle_deg": ...}
r = env.scan()
print(r["status"], r["content"], r["gt"]["hit"])
env.save_snapshot("outputs/my_snapshot")           # head / scanner RGB + depth, top-down and oblique views
env.close()
```

| What to change | Where |
| --- | --- |
| Scanner camera intrinsics | `configs/scanner.yaml: camera` (`width` / `height` / `focal_length` / `horizontal_aperture`) |
| Scanner shape | `scanner.yaml: parts`; also update `tip.pos` so the tip stays on the exit window |
| Decoding rotation step | `scanner.yaml: scan.rotation_step_deg` |
| Grip position | `configs/robot.yaml: scanner_mount` |
| Weight of direction vs. position error when aiming | `robot.yaml: arm_ik.aim_weight` |
