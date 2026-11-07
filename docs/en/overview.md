(overview)=
# Overview

## What the environment contains

| Component | Contents | Details |
| --- | --- | --- |
| Robot | Unitree G1 with 29 body degrees of freedom plus two Dex1 two-finger grippers (33 joints in total). The left hand permanently holds the scanner; the right gripper can open and close | {ref}`robot` |
| Whole-body control | NVIDIA GR00T Decoupled WBC by default (two ONNX policies, Balance and Walk). It controls the legs and waist and supports walking, squatting and torso tilt; the upstream unitree locomotion policy is available as an alternative | {ref}`wbc` |
| Arms | Damped least-squares IK: wrist position control, scanner aiming (position + direction), joint-space trajectories | {ref}`arm` |
| Cameras | Head RGB-D camera; scanner camera (the scanner's imager); off-robot auxiliary cameras for top-down and oblique views | {ref}`cameras` |
| Scanning | `scan()` decodes the current scanner camera image with pyzbar and returns the code closest to the image center, plus a GT diagnostic based on ray geometry | {ref}`scanner` |
| Barcodes | Code128 / QR, generated procedurally and verified by decoding each image. Labels can go on any of the four vertical faces of a box; in task-suite scenarios one face can carry several labels | {ref}`barcodes` |
| Scenes | 3 built-in scenes (table, shelf, floor stack) | {ref}`scenes` |
| Task suites | Reproducible test scenarios: table scenarios `t1_*`, `t2_*` (19 suites) and stack scenarios `stack*` (6 suites) | {ref}`suites` |
| Ground truth | Poses of boxes and labels, reference standing poses and observation poses, for evaluation only | {ref}`gt` |

## Repository layout

```text
g1_barcode_sim/
├── g1_barcode_sim/        Python package: env.py (G1BarcodeEnv), gear_wbc.py (GR00T locomotion), arm.py (IK),
│                          scanner.py, camera.py, barcode.py, scene_builder.py, scenario_builder.py,
│                          tasks/ (task configs registered with Isaac Lab), testsuite/ (scenario generation and loading, pure Python)
├── configs/               robot / camera_head / scanner / barcode.yaml, scenes/*.yaml, testsuite/*.yaml
├── scripts/               run_scene.py, teleop.py, check_env.py, render_views.py, record_walk.py,
│                          generate_barcodes.py; testsuite/ holds list / show / generate / preview and other tools
├── suites/                Task suites: one manifest.json + scenarios/*.json per suite
├── examples/              goto_barcode.py (walk up to a barcode), scanner_camera.py (aim and scan)
├── tests/                 CPU unit tests (no Isaac Sim needed)
├── assets/gear_wbc/       GR00T WBC policies (ONNX), reference config, license and NOTICE
└── docs/                  This documentation and its images
```

The repository has no `setup.py` / `pyproject.toml`. The scripts add the repository root to `sys.path` themselves; to use the package from your own code, run from the repository root or add the root to `PYTHONPATH`.

(impl)=
## How the simulation is built

The environment uses the upstream task `Isaac-Move-Cylinder-G129-Dex1-Wholebody` as a template: at startup it copies that task's configuration, removes the upstream room, table, cylinder and cameras, and adds this project's scene, cameras, scanner and controllers. The upstream files themselves are not modified. Default parameters live in the YAML files under `configs/` and can be overridden with `config_overrides` when the environment is constructed. Each part is described in detail on the pages of the "Simulation" section.
