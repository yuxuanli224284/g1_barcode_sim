(limits)=
# Known limitations and licenses

## Known limitations

- **Walking**: at low speeds, backward, sideways and turning motions are tracked poorly (only 30–70% of the commanded displacement is achieved), and left and right turns respond asymmetrically. After stopping from 0.4 m/s the base still slides about 10–15 cm. Walking with the left arm raised makes the heading drift 2–4.5° to the right. The closed loop and repeated re-checks of `move_relative` compensate for most of these errors.
- **Squatting**: the lower limit of the height command, 0.20 m, has not been tested; the lowest measured level is 0.25 m (pelvis at about 0.25 m).
- **Odometry** is not an estimator but the true pose plus deterministic drift (2% / 1° per meter).
- **Cameras**: the head camera parameters come from the upstream simulation configuration (105.5° horizontal field of view, wider than a real D435i); the scanner camera is estimated from DS2208 specifications. Neither has been calibrated against real hardware.
- **Scanning**: `scan()` decodes the rendered image and does not model the decoding range and angle specifications of a real scanner. By default only the original image is decoded, so the long axis of the barcode must be roughly aligned with the image's horizontal axis (within about ±17°).
- **The scanner** is visual only, with no mass or collider, so the arm load is lighter than on the real robot (about 160 g). Contact with the scene only shows up in the clearance check.
- **Collisions**: robot self-collision is disabled; falls are detected only from the tilt angle; clearance is computed only from vertices (an upper bound). The warehouse background is visual only, with collisions disabled.
- **End effectors**: only Dex1 is wired up; wrist IK controls position only.
- **Labels**: GT supports only the four vertical faces (the top face `+z` has no horizontal standing direction); `pattern_size` is `null` in the GT of built-in scenes.
- **Runtime**: physics runs on the CPU, and the head, scanner and auxiliary cameras are rendered at every policy step, so simulation is noticeably slower than real time (hardware dependent). A process can create only one environment, and `close()` terminates the process. Scenes depend on online Isaac assets (geometric primitives can be used instead).

(license)=
## Licenses and third-party components

```{list-table}
:header-rows: 1
:widths: 25 30 30 15

* - Component
  - Source
  - License
  - Distributed with this repository
* - GR00T WBC policy weights `GR00T-WholeBodyControl-{Balance,Walk}.onnx`
  - [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (`decoupled_wbc/sim2mujoco/resources/robots/g1`), unmodified
  - NVIDIA Open Model License. Licensed by NVIDIA Corporation under the NVIDIA Open Model License; see `assets/gear_wbc/NOTICE` and `NVIDIA_Open_Model_License.txt`
  - Yes (`assets/gear_wbc/`)
* - Reference configuration `g1_gear_wbc.sim2mujoco.yaml`
  - Same as above
  - Apache-2.0
  - Yes
* - unitree_sim_isaaclab (G1 + Dex1 USD model, task template, upstream locomotion policy)
  - [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
  - Apache-2.0
  - No, referenced at runtime via `UNITREE_SIM_ROOT`
* - Warehouse background `Simple_Warehouse/warehouse.usd`, cardboard box `SM_CardBoxD_04.usd`
  - NVIDIA Isaac Sim asset library
  - Isaac Sim asset terms
  - No, referenced online at runtime
* - Code in this repository
  - This repository
  - MIT License (see LICENSE in the repository root)
  - —
```

The scanner's shape and the scanner camera parameters follow the public specifications of the Zebra DS2208; Zebra and DS2208 are trademarks of their owner. NVIDIA, Isaac Sim and GR00T are trademarks of NVIDIA Corporation and are used here only to identify sources.
