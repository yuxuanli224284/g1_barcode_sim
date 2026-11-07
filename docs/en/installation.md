(install)=
# Installation

This repository runs on top of Isaac Sim 5.1 + Isaac Lab and reuses the G1 model and task template of [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) at runtime, so all three must be fully installed. The steps below are ordered by dependency; every third-party repository is pinned to a tested commit.

(install-req)=
## Hardware and system requirements

- A Linux machine with an NVIDIA GPU that meets the official Isaac Sim 5.1 hardware requirements. A single scene uses about 5 GB of GPU memory.
- conda, git, git-lfs, cmake, a C/C++ toolchain (e.g. `build-essential`) and unzip.
- Several tens of GB of free disk space: the Isaac Sim pip packages and extension cache are large, and the unitree_sim_isaaclab USD assets are about 1.2 GB to download and about 1.7 GB unpacked.
- Access to NVIDIA's online Isaac asset library: the default scenes load the warehouse background and cardboard box models from it (see {ref}`install-faq` for an offline alternative).
- Physics runs on the CPU (as upstream does); rendering uses the GPU.

(install-steps)=
## Installation steps

We recommend placing this repository and unitree_sim_isaaclab under the same parent directory. Then `UNITREE_SIM_ROOT` does not need to be set, because the code looks for the upstream repository next to this one by default:

```text
workspace/
├── IsaacLab/                 step 3
├── cyclonedds/               step 4
├── unitree_sdk2_python/      step 4
├── unitree_sim_isaaclab/     step 5 (default location: sibling of this repository)
└── g1_barcode_sim/           this repository
```

### 1. Create a conda environment (Python 3.11)

The Isaac Sim 5.1 pip packages only support Python 3.11.

```bash
conda create -y -n g1sim python=3.11
conda activate g1sim
python -m pip install --upgrade pip
```

### 2. Install Isaac Sim 5.1.0

See the official Isaac Sim pip installation guide for details. The EULA must be accepted on first run; afterwards the environment variable `OMNI_KIT_ACCEPT_EULA=YES` skips the prompt.

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
isaacsim            # optional: launch once to check that it opens (downloads and caches extensions)
```

### 3. Install Isaac Lab (pinned commit)

Use the commit specified in the unitree_sim_isaaclab documentation (7 commits after v2.3.2). The install script also installs torch (2.7.0+cu128).

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 80094be3245aa5c8376a7464d29cb4412ea518f5
./isaaclab.sh --install
cd ..
```

### 4. Build CycloneDDS 0.10.x and install unitree_sdk2_python

This simulation does not use DDS communication, but upstream modules import `unitree_sdk2py` and `cyclonedds`, so both must be installed.

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install
cd ../..
export CYCLONEDDS_HOME=$PWD/cyclonedds/install

git clone https://github.com/unitreerobotics/unitree_sdk2_python
cd unitree_sdk2_python
git checkout 814556d15970dd2ecf1c9984e845ca02ab07e206
pip install -e .
cd ..
```

### 5. Install unitree_sim_isaaclab (pinned commit) and download the USD assets

`fetch_assets.sh` downloads the robot models and other assets from the HuggingFace dataset `unitreerobotics/unitree_sim_isaaclab_usds` and requires git-lfs. This simulation does not need the upstream teleimager submodule (only upstream's own `sim_main.py` and XR teleoperation use it).

```bash
git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
cd unitree_sim_isaaclab
git checkout e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc
pip install -r requirements.txt
bash fetch_assets.sh             # creates assets/robots, assets/model, etc.
cd ..
```

### 6. Install the additional dependencies of this repository

pyzbar needs the zbar shared library (the conda-forge package is used here; the system package `libzbar0` also works). The pyarrow pin is explained in {ref}`install-faq`. The human-readable text on barcode labels uses the DejaVu font if available (e.g. the system package `fonts-dejavu`) and falls back to Pillow's default font otherwise; the barcode itself is not affected.

```bash
conda install -y -c conda-forge zbar libstdcxx-ng
pip install python-barcode==0.16.1 qrcode==7.4.2 pyzbar==0.1.9 \
            pillow opencv-python pyyaml scipy imageio-ffmpeg pytest
pip install --no-deps "pyarrow==23.0.1"
```

### 7. Clone this repository and set environment variables

```bash
git clone <repository URL> g1_barcode_sim
cd g1_barcode_sim
export OMNI_KIT_ACCEPT_EULA=YES
# Only needed if unitree_sim_isaaclab is not at ../unitree_sim_isaaclab:
export UNITREE_SIM_ROOT=/path/to/unitree_sim_isaaclab
```

### 8. Generate the barcode textures (optional)

On first startup the environment generates any missing textures automatically and verifies each one by decoding it. You can also generate them in advance:

```bash
python scripts/generate_barcodes.py              # Code128; exit code 0 means every texture decoded correctly
python scripts/generate_barcodes.py --type qr
```

(install-env)=
## Environment variables

| Variable | Purpose | Required |
| --- | --- | --- |
| `UNITREE_SIM_ROOT` | Directory of the unitree_sim_isaaclab clone. If unset, the sibling directory `../unitree_sim_isaaclab` is used. If the directory has no `action_provider/`, creating the environment raises `FileNotFoundError`. The code also sets it as `PROJECT_ROOT`, which upstream uses | Only if the upstream repository is not at the default location |
| `OMNI_KIT_ACCEPT_EULA` | Read by Isaac Sim; `YES` skips the EULA prompt | Recommended |
| `PROGRESS_CLIENT_DIR` | Optional directory of a progress-reporting plugin for `check_env.py` and `testsuite/preview.py`. If unset, nothing is reported; runs are unaffected | Usually not |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA variable that selects which GPU to use on multi-GPU machines | Optional |
| `CYCLONEDDS_HOME` | Only used when installing unitree_sdk2_python | During installation |
| `DISPLAY` | `run_scene.py` and `teleop.py` switch to headless mode automatically when there is no display | No |

(install-versions)=
## Version table

The following combination has been tested. The commits in the table are the ones pinned in the installation steps; the other Python package versions are given for reference when troubleshooting.

| Component | Version / commit | Notes |
| --- | --- | --- |
| Python | 3.11 | Required by the Isaac Sim 5.1 pip packages |
| Isaac Sim | 5.1.0 | `isaacsim[all,extscache]==5.1.0` |
| Isaac Lab | `80094be3245aa5c8376a7464d29cb4412ea518f5` | 2.3.2 series (7 commits after v2.3.2); isaaclab 0.54.3 |
| unitree_sim_isaaclab | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Upstream files are not modified |
| unitree_sdk2_python | `814556d15970dd2ecf1c9984e845ca02ab07e206` | unitree_sdk2py 1.0.1 |
| CycloneDDS | `releases/0.10.x` | Python package cyclonedds 0.10.2 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 | Installed by the Isaac Lab install script |
| numpy | 1.26.0 | |
| onnxruntime | 1.22.1 | GR00T policy inference (CPU) |
| pin (pinocchio) / pin-pink | 2.7.0 / 3.1.0 | Installed by Isaac Lab |
| gymnasium | 1.2.1 | |
| scipy | 1.15.3 | Collision clearance computation |
| python-barcode / qrcode / pyzbar | 0.16.1 / 7.4.2 / 0.1.9 | Barcode generation and decoding |
| zbar (conda-forge) | 0.23.95 | Shared library for pyzbar |
| opencv-python / Pillow / PyYAML | 4.11.0.86 / 11.3.0 / 6.0.2 | |
| pyarrow | 23.0.1 | Pinned manually, see {ref}`install-faq` |
| rerun-sdk / pyzmq | 0.20.1 / 27.0.0 | From the upstream requirements.txt; not used directly by this simulation |
| imageio-ffmpeg | 0.8.1 | Optional: converts recordings to H.264 when no system ffmpeg is available |
| GR00T WBC ONNX | Shipped with the repository | `assets/gear_wbc/`, from NVlabs/GR00T-WholeBodyControl, unmodified |

(install-verify)=
## Verifying the installation

### 1. CPU unit tests (no Isaac Sim or GPU needed)

22 tests that finish in a few seconds: `test_gear_wbc.py` checks the input layout of the GR00T policies and the Balance/Walk switching; `test_stack_faces.py` checks that every existing task-suite scenario converts to a valid layout, plus four-face labeling of stacks and overlap detection for multiple labels on one face.

```bash
python -m pytest tests/
```

### 2. Key packages and upstream assets

```bash
python -c "from pyzbar import pyzbar; print('zbar ok')"
python -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ['isaacsim','isaaclab','torch','unitree_sdk2py','onnxruntime','python-barcode','qrcode','pyzbar']]"
test -f "${UNITREE_SIM_ROOT:-../unitree_sim_isaaclab}/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd" && echo robot-usd ok
python scripts/testsuite/list.py                 # pure Python: lists all task suites
```

### 3. End-to-end self-check (GPU required)

`check_env.py` starts one subprocess per built-in scene and checks standing, walking, the gripper, consistency of the scanner camera pose, scanning, GT and head-camera visibility, and whether the rendered barcodes decode (the walking and scanning checks run only in `s1_table`). All three scenes take about 15 minutes in total (hardware dependent). The report is written to `outputs/check_report.md`; exit code 0 means every subprocess succeeded.

```bash
python scripts/check_env.py                      # or test a single scene: --scenes s1_table
python scripts/run_scene.py --scene s1_table --max_sim_s 10   # exit after 10 s of simulated time
```

(install-faq)=
## Troubleshooting

```{list-table}
:header-rows: 1
:widths: 35 65

* - Symptom
  - Cause and fix
* - `` FileNotFoundError: unitree_sim_isaaclab not found at … ``
  - The upstream repository is not a sibling of this repository. Clone it to `../unitree_sim_isaaclab` or set `UNITREE_SIM_ROOT`
* - `` libstdc++.so.6: version `GLIBCXX_3.4.30' not found ``
  - The system libstdc++ is too old: `conda install -c conda-forge libstdcxx-ng`
* - `ImportError: pyarrow requires NumPy 2.0`
  - rerun-sdk 0.20.1 from the upstream requirements needs numpy<2, while recent pyarrow needs numpy 2. Pin it with `pip install --no-deps "pyarrow==23.0.1"`
* - `Could not locate cyclonedds` when installing unitree_sdk2_python
  - Build CycloneDDS first as in step 4 and set `CYCLONEDDS_HOME`
* - Building egl_probe fails with CMake ≥ 4 (`Compatibility with CMake < 3.5`)
  - Run `export CMAKE_POLICY_VERSION_MINIMUM=3.5` and retry
* - cmake cannot find a compiler or uses odd compiler flags
  - Some conda base environments export placeholder compiler variables. Run `unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS` before building
* - A later pip install upgraded dependencies that Isaac Sim pins exactly (e.g. click, psutil, typing_extensions, cryptography, packaging, numpy)
  - Reinstall the `==` versions listed in the `isaacsim-kernel` / `isaacsim-core` metadata with `pip install --no-deps`
* - pyzbar cannot find the zbar shared library
  - `conda install -c conda-forge zbar`, or install the system package `libzbar0`
* - No network access; the warehouse background or box models do not load
  - The scenes reference `Environments/Simple_Warehouse/warehouse.usd` and `Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd` from the online Isaac asset library. Offline, set `background.kind: geometry` (geometric floor plus walls) and `box_visual.kind: geometry` (kraft-colored cuboids) in `configs/scenes/_common.yaml`. Alternatively, download the local asset pack as described in the official Isaac Sim documentation and point the asset root to it (not verified in this project). Note: task-suite scenarios with the `cardbox` appearance (`stack9c_L0`, `stack4c_L0`) must load this box asset and raise `RuntimeError` if loading fails
* - Changed the size or text parameters in `configs/barcode.yaml`, but the labels did not change
  - At startup the environment only fills in missing textures and reuses existing PNGs. After changing parameters, run `python scripts/generate_barcodes.py` once (it always overwrites)
* - Switching built-in scenes within one process
  - A process can create only one `G1BarcodeEnv`, and `close()` terminates the whole process (Isaac Sim's normal shutdown hangs while camera annotators are running). Start a new process to change the built-in scene; task-suite scenarios can be switched directly with `reset(scenario=...)`
* - Recordings are mp4v-encoded and do not play in a browser
  - `teleop.py --record` first writes mp4v and then converts it to H.264 with ffmpeg. Install a system ffmpeg or `imageio-ffmpeg`
```
