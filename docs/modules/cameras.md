(cameras)=
# 相机

| 相机 | 安装位置 | 分辨率 | 焦距 / 水平孔径 | 视场 H × V | fx | 裁剪范围 | 输出 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 头部 `front_cam` | G1 头部 `d435_link` | 640×480 | 7.6 / 20 mm | 105.5° × 约 89° | 243.2 px | 0.1 m – 远 | RGB + 深度 |
| 枪头 `wrist_cam` | 出光窗中心前 0.5 mm，光轴 = 扫描方向 | 640×480 | 33.98 / 20 mm | 32.8° × 约 24.9° | 1087 px | 0.01 – 100 m | RGB + 深度 |
| 辅助 aux | 不在机器人上，按需摆放 | 1280×720 | 18.15 / 20.955 mm | 60°（可指定）× 约 36° | 约 1109 px | 0.05 – 100 m | RGB + 深度 |

- **头部相机**参数沿用 unitree_sim_isaaclab 的 G1 头部相机预设（安装偏移按 ROS 约定），视场比真机 D435i（约 69°）宽，没有与真机标定对齐。深度为 `distance_to_image_plane`，NaN / inf 置 0，有效范围 [0.2, 3.0] m 之外置 0。可选深度噪声默认关闭（`camera_head.yaml: depth.noise`）：σ = 0.002·z² 的高斯噪声，加上深度边缘像素以 1% 概率变成空洞。抗锯齿使用 DLAA（原生分辨率）。
- **枪头相机**就是扫码枪的成像器。没有真机标定，参数按 Zebra DS2208 成像器的规格取值。视场窄：距离 d 处画面宽约 0.59·d，所以 10 cm 宽的标签要在枪口离标签约 0.17 m 以上才能完整入画。深度默认不截断。
- **辅助相机**只用于可视化：`render_aux(T_world_cam_ros, hfov_deg=None)` 在任意位姿渲染并返回 `{rgb, depth, K}`，不推进物理；`topdown()` 俯视图；`oblique_views()` 按场景 YAML 的 `views` 渲染斜视图。另有默认关闭的录像相机（`camera_head.yaml: recording`）。
- 所有相机每个策略步都渲染。相机位姿由物理 link 位姿乘固定外参得到（物理在 CPU 上运行时 Camera prim 自身的位姿不会更新），通过 `get_obs()` 的 `T_root_cam`、`T_root_wrist_cam` 提供。

```{figure} ../images/s1_table_head_rgb_annotated.jpg
:alt: 头部相机画面：前方白桌上两个纸箱，绿色圆点标出两张可见条码 S1-B02-F0 和 S1-B01-F0
:width: 640px

`s1_table` 初始位姿下的头部相机画面。绿点是按 GT 投影、且未被遮挡的标签中心（由 `check_env.py` 标注）；画面下方是机器人自身和左手的扫码枪。
```
