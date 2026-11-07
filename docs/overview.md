(overview)=
# 概览

## 环境里有什么

| 模块 | 内容 | 详见 |
| --- | --- | --- |
| 机器人 | Unitree G1，29 个本体自由度 + 两只 Dex1 两指夹爪（共 33 个关节）。左手固定握扫码枪，右手夹爪可开合 | {ref}`robot` |
| 全身控制 | 默认 NVIDIA GR00T Decoupled WBC（Balance / Walk 两个 ONNX 策略），控制腿和腰，支持行走、下蹲和躯干倾斜；可切换为 unitree 上游行走策略 | {ref}`wbc` |
| 手臂 | 阻尼最小二乘 IK：手腕位置控制、扫码枪瞄准（位置 + 方向）、关节空间轨迹 | {ref}`arm` |
| 相机 | 头部 RGB-D 相机；枪头相机（扫码枪的成像器）；不在机器人上的辅助相机，用于俯视图和斜视图 | {ref}`cameras` |
| 扫码 | `scan()` 用 pyzbar 解码枪头相机当前画面，返回离画面中心最近的码；另附射线几何的 GT 诊断 | {ref}`scanner` |
| 条码 | Code128 / QR，程序生成并逐张解码自检；标签可贴在箱子的四个竖直面上，测试场景中一个面可贴多张 | {ref}`barcodes` |
| 场景 | 3 个内置场景（桌面、货架、地面箱堆） | {ref}`scenes` |
| 测试场景集 | 可复现的 task suites：桌面场景 `t1_*`、`t2_*`（共 19 套），箱堆场景 `stack*`（6 套） | {ref}`suites` |
| Ground truth | 箱子、标签位姿、参考站位和观察位，只用于评测 | {ref}`gt` |

## 目录结构

```text
g1_barcode_sim/
├── g1_barcode_sim/        Python 包：env.py（G1BarcodeEnv）、gear_wbc.py（GR00T 行走）、arm.py（IK）、
│                          scanner.py、camera.py、barcode.py、scene_builder.py、scenario_builder.py、
│                          tasks/（注册到 Isaac Lab 的任务配置）、testsuite/（场景生成与加载，纯 Python）
├── configs/               robot / camera_head / scanner / barcode.yaml，scenes/*.yaml，testsuite/*.yaml
├── scripts/               run_scene.py、teleop.py、check_env.py、render_views.py、record_walk.py、
│                          generate_barcodes.py；testsuite/ 下是 list / show / generate / preview 等工具
├── suites/                测试场景集：每套一个 manifest.json + scenarios/*.json
├── examples/              goto_barcode.py（走到条码前）、scanner_camera.py（瞄准并扫码）
├── tests/                 CPU 单元测试（不需要 Isaac Sim）
├── assets/gear_wbc/       GR00T WBC 策略（ONNX）、参考配置、许可证与 NOTICE
└── docs/                  本文档与图片
```

仓库没有 `setup.py` / `pyproject.toml`。脚本会自行把仓库根目录加入 `sys.path`；在自己的代码里使用时，请在仓库根目录运行，或把根目录加入 `PYTHONPATH`。

(impl)=
## 仿真的搭建方式

环境以上游任务 `Isaac-Move-Cylinder-G129-Dex1-Wholebody` 为模板：启动时复制它的配置，删掉上游的房间、桌子、圆柱和相机，再加入本项目的场景、相机、扫码枪和控制器。上游文件本身不做修改。默认参数集中在 `configs/` 下的 YAML 里，构造环境时也可以用 `config_overrides` 覆盖。各部分的细节见"仿真实现"一节的各页。
