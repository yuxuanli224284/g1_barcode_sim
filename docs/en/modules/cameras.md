(cameras)=
# Cameras

| Camera | Mounting | Resolution | Focal length / horizontal aperture | Field of view H × V | fx | Clipping range | Output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Head `front_cam` | G1 head `d435_link` | 640×480 | 7.6 / 20 mm | 105.5° × about 89° | 243.2 px | 0.1 m – far | RGB + depth |
| Scanner `wrist_cam` | 0.5 mm in front of the exit window center, optical axis = scan direction | 640×480 | 33.98 / 20 mm | 32.8° × about 24.9° | 1087 px | 0.01 – 100 m | RGB + depth |
| Auxiliary (aux) | Off the robot, placed as needed | 1280×720 | 18.15 / 20.955 mm | 60° (configurable) × about 36° | about 1109 px | 0.05 – 100 m | RGB + depth |

- **The head camera** uses the G1 head camera preset of unitree_sim_isaaclab (mounting offset in the ROS convention). Its field of view is wider than that of a real D435i (about 69°), and it has not been calibrated against the real camera. Depth is `distance_to_image_plane`; NaN / inf are set to 0, and values outside the valid range [0.2, 3.0] m are set to 0. Optional depth noise is off by default (`camera_head.yaml: depth.noise`): Gaussian noise with σ = 0.002·z², plus pixels on depth edges turning into holes with 1% probability. Anti-aliasing uses DLAA (native resolution).
- **The scanner camera** is the scanner's imager. It has no real-hardware calibration; its parameters follow the specifications of the Zebra DS2208 imager. The field of view is narrow: at distance d the image is about 0.59·d wide, so a 10 cm wide label fits entirely in the image only when the scanner tip is at least about 0.17 m from the label. Depth is not clipped by default.
- **Auxiliary cameras** are for visualization only: `render_aux(T_world_cam_ros, hfov_deg=None)` renders at an arbitrary pose and returns `{rgb, depth, K}` without stepping physics; `topdown()` renders a top-down view; `oblique_views()` renders the oblique views defined under `views` in the scene YAML. There is also a recording camera, off by default (`camera_head.yaml: recording`).
- All cameras are rendered at every policy step. Camera poses are obtained by multiplying the physical link pose by a fixed extrinsic (with physics on the CPU, the Camera prim's own pose is not updated) and are provided as `T_root_cam` and `T_root_wrist_cam` in `get_obs()`.

```{figure} ../../images/s1_table_head_rgb_annotated.jpg
:alt: Head camera image: two boxes on the white table ahead, with green dots marking the two visible barcodes S1-B02-F0 and S1-B01-F0
:width: 640px

Head camera image of `s1_table` at the start pose. Green dots are the label centers projected from GT that are not occluded (annotated by `check_env.py`); the robot's own body and the scanner in its left hand are at the bottom of the image.
```
