(barcodes)=
# Barcodes and labels

## Symbologies and texture generation

| Symbology | `type` | Generator library | Details |
| --- | --- | --- | --- |
| Code 128 | `code128` (default) | python-barcode | Module sequence including start, check and stop characters, without quiet zones |
| QR Code | `qr` | qrcode | Error correction level M, automatic version, no border |

- **Module-by-module drawing**: every module spans an integer number of pixels with no resampling, so bar edges are sharp and proportions exact. Built-in scenes are drawn according to `configs/barcode.yaml` (1024 px along the long side of the texture, with human-readable text below the pattern); task-suite scenarios use a fixed 4 px per module and no text.
- **Decoding self-check**: every texture is decoded with pyzbar after generation, and the result must equal the original content exactly; otherwise a `RuntimeError` is raised (the batch script `generate_barcodes.py` exits with code 1 instead). Task-suite labels without backing have no white quiet zone, so a quiet zone 12 modules wide is padded around them before the self-check decode. `check_env.py` additionally places an auxiliary camera 0.3 m in front of each label, renders it and decodes it again (an end-to-end check).
- **Cache**: built-in scene textures are in `assets/barcodes/<type>/`; task-suite textures are cached by parameter hash in `assets/barcodes/cache/`. Both can be deleted and are regenerated automatically when needed.

(label-geometry)=
## Label geometry

A label is a child of the box prim and moves with the box. It consists of a textured quad (single-sided) and an optional white backing:

- **Backing**: a thin white sheet, 1 mm thick (`barcode.yaml: label_thickness`). Built-in scenes always use a backing; the label outline is 10 × 5 cm for Code128 and 6 × 6 cm for QR (including the quiet zone), with the barcode pattern at most 8 × 3 cm / 5 × 5 cm. In task-suite scenarios this is set by `label_backing`: with a backing, label = pattern + a 1 cm white margin on each side; without one, the light modules of the pattern take the box face color, as if printed directly on the box.
- **face_offset**: the label's front surface is 1.5 mm off the box face to avoid z-fighting.
- **Placement parameters**: `u` and `v` are measured from the face center (meters), with `u` along the face's "right" direction and `v` pointing up; `rot_deg` is the in-plane rotation, counterclockwise positive as seen from outside the box.
- **Material**: roughness 0.8, metallic 0; semantic label `class=barcode` (boxes `class=box`, furniture `class=table` / `shelf`), usable for generating segmentation ground truth.

| Face | Outward normal n | In-plane "right" axis u | In-plane "up" axis v | Face number in built-in scenes | Notes |
| --- | --- | --- | --- | --- | --- |
| `-x` | (−1, 0, 0) | (0, −1, 0) | (0, 0, 1) | 0 | Front, facing the robot's start pose |
| `+x` | (1, 0, 0) | (0, 1, 0) | (0, 0, 1) | 1 | Back |
| `-y` | (0, −1, 0) | (1, 0, 0) | (0, 0, 1) | 2 | Side face on the robot's right |
| `+y` | (0, 1, 0) | (−1, 0, 0) | (0, 0, 1) | 3 | Side face on the robot's left |

All directions are in the box's local frame (which coincides with the world frame when the box yaw = 0). u and v are defined as seen when standing outside the box looking at the face, satisfying right = up × normal, so the pattern reads from left to right.

- **Built-in scenes**: at most 1 label per box face, with content generated automatically from the template `S{scene}-B{box:02d}-F{face}` (e.g. `S1-B01-F0`). All four corners of the rotated label must lie within the face. Optional randomization (`barcode.yaml: randomize.enabled` or `run_scene.py --randomize`): at each reset, uniform perturbations of ±3 cm are added to u and v and ±10° to rot; the same seed gives the same result.
- **Task-suite scenarios**: a face can carry several labels with arbitrary content. Checks at load time: the bounding box of the rotated label plus a 1 cm margin must lie within the face; the bounding boxes of labels on the same face must not overlap; labels must not be placed on inner faces hidden by an adjacent box. Prims are named `BC_mx`, `BC_px`, `BC_my`, `BC_py`, and from the second label on the same face onward `BC_mx_1`, `BC_mx_2`, …

## Adding your own barcodes

### Option A: built-in scene YAML

Content is generated from the template and cannot be set per label; `id` must be `B` followed by a number.

```yaml
# configs/scenes/s1_table.yaml
boxes:
  - {id: B03, size: [0.30, 0.30, 0.30],
     place_on: {furniture: T01, y: 0.0, front_setback: 0.05},
     barcodes: [{face: "-x", uv: [0.03, -0.02], rot_deg: 15},   # content S1-B03-F0
                {face: "+y"}]}                                  # content S1-B03-F3
```

Afterwards you can run `python scripts/generate_barcodes.py --scenes s1_table` (add `--type qr` for QR). To change the numbering scheme, edit `content_format` in `barcode.yaml` (placeholders `{scene}`, `{box}`, `{face}`).

### Option B: task-suite scenario JSON (arbitrary content)

Add an entry to the scenario's `barcodes` array (fields in {ref}`suite-schema`):

```json
{"box_id": "B0", "face": "-x", "type": "code128",
 "width": 0.08, "aspect": 0.375, "label_backing": true,
 "u": 0.0, "v": 0.0, "rot_deg": 0.0,
 "content": "MYCODE123", "content_id": "MY-0000-B0"}
```

QR does not need `aspect`; `width` is the side length. Code128 can only encode ASCII; `content` must be unique within a scenario.

### Option C: generate only the label image (no Isaac Sim needed)

```python
from g1_barcode_sim.config import load_barcode_cfg
from g1_barcode_sim.barcode import make_label, decode_image

lab = make_label("HELLO-123", load_barcode_cfg(), "code128")
lab.image.save("hello.png")
assert decode_image(lab.image) == ["HELLO-123"]
print(lab.pattern_size_m)          # actual pattern size (meters)
```
