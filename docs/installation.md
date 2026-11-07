(install)=
# 环境配置

本仓库运行在 Isaac Sim 5.1 + Isaac Lab 之上，并在运行时复用 [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) 的 G1 模型和任务模板，所以需要完整安装这三者。下面的步骤按依赖顺序排列；所有第三方仓库都固定到经过测试的 commit。

(install-req)=
## 硬件与系统要求

- 一台满足 Isaac Sim 5.1 官方硬件要求、带 NVIDIA GPU 的 Linux 机器。单个场景约占 5 GB 显存。
- 已安装 conda、git、git-lfs、cmake、C/C++ 编译工具链（如 `build-essential`）和 unzip。
- 预留数十 GB 磁盘空间：Isaac Sim 的 pip 包与扩展缓存较大，unitree_sim_isaaclab 的 USD 资产下载约 1.2 GB、解压后约 1.7 GB。
- 能访问 NVIDIA 的在线 Isaac 资产库：默认场景从那里加载仓库背景和纸箱模型（离线方案见{ref}`install-faq`）。
- 物理仿真跑在 CPU 上（与上游一致），渲染使用 GPU。

(install-steps)=
## 安装步骤

建议把本仓库和 unitree_sim_isaaclab 放在同一个父目录下。这样不需要设置 `UNITREE_SIM_ROOT`，代码默认就在本仓库旁边找上游仓库：

```text
workspace/
├── IsaacLab/                 第 3 步
├── cyclonedds/               第 4 步
├── unitree_sdk2_python/      第 4 步
├── unitree_sim_isaaclab/     第 5 步（默认位置：本仓库的同级目录）
└── g1_barcode_sim/           本仓库
```

### 1. 创建 conda 环境（Python 3.11）

Isaac Sim 5.1 的 pip 包只支持 Python 3.11。

```bash
conda create -y -n g1sim python=3.11
conda activate g1sim
python -m pip install --upgrade pip
```

### 2. 安装 Isaac Sim 5.1.0

详见 Isaac Sim 官方的 pip 安装文档。首次运行要接受 EULA；之后可以用环境变量 `OMNI_KIT_ACCEPT_EULA=YES` 跳过交互。

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
isaacsim            # 可选：首次启动一次，确认能打开（会下载并缓存扩展）
```

### 3. 安装 Isaac Lab（固定 commit）

使用 unitree_sim_isaaclab 文档指定的 commit（v2.3.2 之后 7 个提交）。安装脚本会同时装好 torch（2.7.0+cu128）。

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 80094be3245aa5c8376a7464d29cb4412ea518f5
./isaaclab.sh --install
cd ..
```

### 4. 编译 CycloneDDS 0.10.x，安装 unitree_sdk2_python

本仿真不走 DDS 通信，但上游模块在 import 时会加载 `unitree_sdk2py` 和 `cyclonedds`，所以必须安装。

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

### 5. 安装 unitree_sim_isaaclab（固定 commit）并下载 USD 资产

`fetch_assets.sh` 从 HuggingFace 数据集 `unitreerobotics/unitree_sim_isaaclab_usds` 下载机器人模型等资产，需要 git-lfs。本仿真不需要上游的 teleimager 子模块（只有上游自己的 `sim_main.py` 和 XR 遥操作会用到）。

```bash
git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
cd unitree_sim_isaaclab
git checkout e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc
pip install -r requirements.txt
bash fetch_assets.sh             # 生成 assets/robots、assets/model 等
cd ..
```

### 6. 安装本仓库的额外依赖

pyzbar 需要 zbar 共享库（这里用 conda-forge 的包，也可以用系统包 `libzbar0`）。pyarrow 的版本锁定见{ref}`install-faq`。条码标签上的可读文字优先使用 DejaVu 字体（如系统包 `fonts-dejavu`），没有时回退到 Pillow 默认字体，不影响条码本身。

```bash
conda install -y -c conda-forge zbar libstdcxx-ng
pip install python-barcode==0.16.1 qrcode==7.4.2 pyzbar==0.1.9 \
            pillow opencv-python pyyaml scipy imageio-ffmpeg pytest
pip install --no-deps "pyarrow==23.0.1"
```

### 7. 克隆本仓库，设置环境变量

```bash
git clone <本仓库地址> g1_barcode_sim
cd g1_barcode_sim
export OMNI_KIT_ACCEPT_EULA=YES
# 只有 unitree_sim_isaaclab 不在 ../unitree_sim_isaaclab 时才需要：
export UNITREE_SIM_ROOT=/path/to/unitree_sim_isaaclab
```

### 8. 生成条码贴图（可选）

环境首次启动时会自动生成缺失的贴图并逐张解码校验；也可以提前批量生成：

```bash
python scripts/generate_barcodes.py              # Code128，退出码 0 表示全部解码正确
python scripts/generate_barcodes.py --type qr
```

(install-env)=
## 环境变量

| 变量 | 作用 | 是否需要 |
| --- | --- | --- |
| `UNITREE_SIM_ROOT` | unitree_sim_isaaclab 克隆目录。未设置时使用本仓库的同级目录 `../unitree_sim_isaaclab`。目录里找不到 `action_provider/` 时，创建环境会直接抛出 `FileNotFoundError`。代码会把它同时设为上游使用的 `PROJECT_ROOT` | 上游仓库不在默认位置时需要 |
| `OMNI_KIT_ACCEPT_EULA` | 由 Isaac Sim 读取，设为 `YES` 时跳过 EULA 交互 | 建议设置 |
| `PROGRESS_CLIENT_DIR` | `check_env.py` 和 `testsuite/preview.py` 的可选进度上报插件目录。不设置时不上报，对运行没有影响 | 一般不需要 |
| `CUDA_VISIBLE_DEVICES` | 标准 CUDA 变量，多卡机器上选择使用哪张卡 | 可选 |
| `CYCLONEDDS_HOME` | 只在安装 unitree_sdk2_python 时使用 | 安装时需要 |
| `DISPLAY` | `run_scene.py`、`teleop.py` 在没有显示器时自动进入 headless 模式 | 不需要 |

(install-versions)=
## 版本表

以下是经过测试的组合。表中 commit 是安装步骤里固定的版本；其余 Python 包版本仅供排查问题时参考。

| 组件 | 版本 / commit | 说明 |
| --- | --- | --- |
| Python | 3.11 | Isaac Sim 5.1 pip 包要求 |
| Isaac Sim | 5.1.0 | `isaacsim[all,extscache]==5.1.0` |
| Isaac Lab | `80094be3245aa5c8376a7464d29cb4412ea518f5` | 2.3.2 系列（v2.3.2 之后 7 个提交）；isaaclab 0.54.3 |
| unitree_sim_isaaclab | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | 上游文件不做任何修改 |
| unitree_sdk2_python | `814556d15970dd2ecf1c9984e845ca02ab07e206` | unitree_sdk2py 1.0.1 |
| CycloneDDS | `releases/0.10.x` | Python 包 cyclonedds 0.10.2 |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 | 由 Isaac Lab 安装脚本装入 |
| numpy | 1.26.0 | |
| onnxruntime | 1.22.1 | GR00T 策略推理（CPU） |
| pin（pinocchio） / pin-pink | 2.7.0 / 3.1.0 | 由 Isaac Lab 安装 |
| gymnasium | 1.2.1 | |
| scipy | 1.15.3 | 碰撞间隙计算 |
| python-barcode / qrcode / pyzbar | 0.16.1 / 7.4.2 / 0.1.9 | 条码生成与解码 |
| zbar（conda-forge） | 0.23.95 | pyzbar 的共享库 |
| opencv-python / Pillow / PyYAML | 4.11.0.86 / 11.3.0 / 6.0.2 | |
| pyarrow | 23.0.1 | 手动锁定，见{ref}`install-faq` |
| rerun-sdk / pyzmq | 0.20.1 / 27.0.0 | 来自上游 requirements.txt，本仿真不直接使用 |
| imageio-ffmpeg | 0.8.1 | 可选：系统没有 ffmpeg 时用于把录像转成 H.264 |
| GR00T WBC ONNX | 随仓库附带 | `assets/gear_wbc/`，来自 NVlabs/GR00T-WholeBodyControl，未修改 |

(install-verify)=
## 验证安装

### 1. CPU 单元测试（不需要 Isaac Sim 和 GPU）

共 22 个测试，几秒内完成：`test_gear_wbc.py` 检查 GR00T 策略的输入布局和 Balance/Walk 切换；`test_stack_faces.py` 检查所有现有测试场景都能转换成合法布局、箱堆四面贴码和同面多标签的重叠检测。

```bash
python -m pytest tests/
```

### 2. 关键包与上游资产

```bash
python -c "from pyzbar import pyzbar; print('zbar ok')"
python -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ['isaacsim','isaaclab','torch','unitree_sdk2py','onnxruntime','python-barcode','qrcode','pyzbar']]"
test -f "${UNITREE_SIM_ROOT:-../unitree_sim_isaaclab}/assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd" && echo robot-usd ok
python scripts/testsuite/list.py                 # 纯 Python：列出所有测试场景集
```

### 3. 端到端自检（需要 GPU）

`check_env.py` 为每个内置场景开一个子进程，检查站立、行走、夹爪、扫码枪相机位姿一致性、扫码、GT 和头部相机可见性，以及渲染后的条码能否解码（行走和扫码检查只在 `s1_table` 中运行）。三个场景合计约 15 分钟（与硬件有关），报告写到 `outputs/check_report.md`，退出码 0 表示全部子进程成功。

```bash
python scripts/check_env.py                      # 或只测一个：--scenes s1_table
python scripts/run_scene.py --scene s1_table --max_sim_s 10   # 跑 10 s 仿真时间后退出
```

(install-faq)=
## 常见问题

```{list-table}
:header-rows: 1
:widths: 35 65

* - 现象
  - 原因与处理
* - `` FileNotFoundError: unitree_sim_isaaclab not found at … ``
  - 上游仓库不在本仓库的同级目录。克隆到 `../unitree_sim_isaaclab`，或设置 `UNITREE_SIM_ROOT`
* - `` libstdc++.so.6: version `GLIBCXX_3.4.30' not found ``
  - 系统 libstdc++ 太旧：`conda install -c conda-forge libstdcxx-ng`
* - `ImportError: pyarrow requires NumPy 2.0`
  - 上游 requirements 里的 rerun-sdk 0.20.1 要求 numpy<2，新版 pyarrow 却要求 numpy 2。用 `pip install --no-deps "pyarrow==23.0.1"` 锁定
* - 安装 unitree_sdk2_python 时 `Could not locate cyclonedds`
  - 先按第 4 步编译 CycloneDDS，并设置 `CYCLONEDDS_HOME`
* - CMake ≥ 4 时编译 egl_probe 失败（`Compatibility with CMake < 3.5`）
  - `export CMAKE_POLICY_VERSION_MINIMUM=3.5` 后重试
* - cmake 找不到编译器或用了奇怪的编译选项
  - 有些 conda base 环境会导出占位的编译器变量。编译前执行 `unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS`
* - 之后的 pip 安装把 Isaac Sim 精确锁定的依赖（如 click、psutil、typing_extensions、cryptography、packaging、numpy）升级了
  - 按 `isaacsim-kernel` / `isaacsim-core` 元数据中的 `==` 版本，用 `pip install --no-deps` 装回
* - pyzbar 报找不到 zbar 共享库
  - `conda install -c conda-forge zbar`，或安装系统包 `libzbar0`
* - 无法联网，仓库背景或纸箱模型加载不出来
  - 场景从 Isaac 在线资产库引用 `Environments/Simple_Warehouse/warehouse.usd` 和 `Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd`。离线时在 `configs/scenes/_common.yaml` 中设 `background.kind: geometry`（几何地面加墙）和 `box_visual.kind: geometry`（牛皮纸色长方体）。也可以按 Isaac Sim 官方文档下载本地资产包并把资产根目录指向本地（未在本项目中验证）。注意：外观为 `cardbox` 的测试场景（`stack9c_L0`、`stack4c_L0`）必须加载这个纸箱资产，加载失败时会抛出 `RuntimeError`
* - 改了 `configs/barcode.yaml` 的尺寸或文字参数，标签没有变化
  - 环境启动时只补齐缺失的贴图，已有的 PNG 会直接复用。改参数后运行一次 `python scripts/generate_barcodes.py`（总是覆盖写盘）
* - 想在同一个进程里切换内置场景
  - 一个进程只能创建一个 `G1BarcodeEnv`，`close()` 会结束整个进程（相机 annotator 运行时 Isaac Sim 的正常关闭流程会卡住）。换内置场景请新开进程；测试场景之间可以直接 `reset(scenario=...)` 切换
* - 录像是 mp4v 编码，浏览器播放不了
  - `teleop.py --record` 先写 mp4v，再用 ffmpeg 转成 H.264。安装系统 ffmpeg 或 `imageio-ffmpeg` 即可
```
