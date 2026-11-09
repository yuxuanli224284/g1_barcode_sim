# g1_barcode_sim

**Isaac Sim 5.1 · Isaac Lab · Unitree G1 + Dex1**

A warehouse barcode-scanning simulation environment for the G1 humanoid. The robot holds a handheld barcode scanner in its left hand, searches for barcode labels on cardboard boxes placed on tables, on shelves or in stacks on the floor, walks up to them, aims and scans. This documentation covers installation, implementation details, the Python API, the built-in scenes and the task suites.

[中文](https://yuxuanli224284.github.io/g1_docs/barcode_sim/) | English

```{figure} ../images/scene_s1_table_left_rear.jpg
:alt: The s1_table scene: a white table on a warehouse floor with two cardboard boxes on it; the G1 robot stands in front of the table, holding a yellow-and-black barcode scanner in its left hand

Built-in scene `s1_table`, viewed from behind the robot's left shoulder. G1 stands at the origin facing +x with the scanner in its left hand; each of the two boxes on the white table has a Code128 label on the side facing the robot.
```

## Introduction

g1_barcode_sim is a simulation environment built on Isaac Sim 5.1 / Isaac Lab for developing and evaluating the "find, approach, scan" capability of humanoid robots. The scene contains a Unitree G1 whose left Dex1 gripper holds a handheld barcode scanner. Cardboard boxes sit on a table, on a shelf or directly on the floor, with Code128 or QR labels on their faces. Through the Python interface you make the robot walk, squat, move its arms and aim, then "pull the trigger" to scan; the environment returns the actual decoding result of the scanner camera image.

The environment itself does **not** include any barcode search, navigation or scanning policy. It provides sensors, control interfaces, reproducible scenes and ground truth (GT) intended for evaluation only.

The numbers in this documentation are taken from the code and configuration in the repository (`configs/`, `g1_barcode_sim/`, `suites/*/manifest.json`).

```{toctree}
:maxdepth: 2
:caption: Getting started

overview
installation
```

```{toctree}
:maxdepth: 2
:caption: Simulation

modules/robot
modules/locomotion
modules/cameras
modules/scanner
modules/barcodes
modules/ground_truth
```

```{toctree}
:maxdepth: 2
:caption: Interface

api
```

```{toctree}
:maxdepth: 2
:caption: Scenes and evaluation

scenes
task_suites
```

```{toctree}
:maxdepth: 2
:caption: Other

limitations
```
