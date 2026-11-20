"""Robot preview in a scenario: the robot at the scenario start, the scanner arm raised to a label, and images to check
the robot model against the real one.

    python scripts/robot_preview.py --variant g1_v2 --scenario scanroom6x5_L0_0000 --label C3R4 --out outputs/g1_v2_preview

Steps: reset to the scenario (robot at its start pose), stand 1 s, aim the scanner at the centre of the label on box
--label from --dist in front of it (env.aim_scanner_at: perpendicular incidence), hold until the aim error has settled,
trigger one scan. Writes into --out:
  overview_left_rear.png, overview_right_rear.png    whole scene and robot from behind-left / behind-right
  head_front_left.png, head_left.png                 head close-ups (the D435 on the head)
  wrist_left/front_left/rear_left/top.png         left-wrist close-ups (bracket, scanner, D405)
  cam_head_color.png, cam_head_ir_left.png           head camera views (D435i colour, left infrared)
  cam_<hand camera>.png                              hand cameras (g1_v2: cam_left_d405.png)
  cam_scanner.png                                    scanner camera (the imager at the exit window)
  sheet.png                                          all of the above, labelled
  preview.json                                       aim target / error, scan result, poses
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="g1_v2")
ap.add_argument("--scenario", default="scanroom6x5_L0_0000")
ap.add_argument("--label", default="C3R4", help="box id whose label the scanner is aimed at")
ap.add_argument("--dist", type=float, default=0.20, help="scanner window distance from the label (m)")
ap.add_argument("--hold", type=float, default=4.0, help="seconds to hold the aim before the images")
ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "g1_v2_preview"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

import numpy as np  # noqa: E402
from g1_barcode_sim.env import G1BarcodeEnv  # noqa: E402

t0 = time.time()
env = G1BarcodeEnv("scenario", headless=True, seed=0, config_overrides={"robot": {"variant": args.variant}})
import cv2  # noqa: E402
from g1_barcode_sim.camera import inv_T, look_at_ros, make_T  # noqa: E402

rep = {"variant": args.variant, "scenario": args.scenario, "label_box": args.label, "dist": args.dist,
       "startup_s": round(time.time() - t0, 1)}
env.reset(scenario=args.scenario)
rep["stand"] = env.stand_still(1.0)["status"]
# head camera tilt while standing (world): optical axis below horizontal, camera face to the floor = 90 - that
_Tc = env._link_T(env._cam_body) @ env._T_link_cam
_z = _Tc[:3, 2]
rep["head_cam_axis_below_horizontal_deg"] = round(float(np.degrees(np.arcsin(-_z[2]))), 2)
rep["head_cam_face_to_floor_deg"] = round(90.0 - rep["head_cam_axis_below_horizontal_deg"], 2)
rep["head_cam_pos_world"] = _Tc[:3, 3].round(4).tolist()

bc = next(b for b in env.layout["barcodes"] if b["box_id"] == args.label)
T_rw = inv_T(env._T_world_root())
p_root = T_rw[:3, :3] @ np.asarray(bc["center"]) + T_rw[:3, 3]
n_root = T_rw[:3, :3] @ np.asarray(bc["normal"])
rep["aim_cmd"] = env.aim_scanner_at(p_root, n_root, distance=args.dist)
rep["hold"] = env.stand_still(args.hold)["status"]
side = env.scanner_side
rep["aim_error"] = env.arm.aim_error(side)
rep["target_label"] = {"content": bc["content"], "center_world": bc["center"], "size": bc["size"]}
rep["scan"] = {k: v for k, v in env.scan().items() if k in ("status", "content", "decoded", "reason")}
obs = env.get_obs()
T_wr = env._T_world_root()
rep["root_world"] = T_wr[:3, 3].round(4).tolist()
rep["scanner_tip_world"] = (T_wr @ obs["T_root_scanner_tip"])[:3, 3].round(4).tolist()
rep["clearance_min_gap"] = env._last_min_gap


def save(name, img):
    cv2.imwrite(os.path.join(args.out, f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img)
    return name


def link_world(name):
    i = env.body_names.index(name)
    p = env.robot.data.body_link_pose_w[0, i].cpu().numpy()
    return make_T(p[:3], p[3:])


R = T_wr[:3, :3]                      # root axes in the world (x forward, y left)
shots = []
# whole scene: from behind the robot, to its left / right, high enough to see the top row
front_x = min(b["pos"][0] - b["size"][0] / 2 for b in env.layout["boxes"])
top = max(b["pos"][2] + b["size"][2] / 2 for b in env.layout["boxes"])
scene_c = np.array([front_x, 0.0, top / 2])
for nm, off in (("overview_left_rear", (-1.7, 1.5, 1.2)), ("overview_right_rear", (-1.7, -1.5, 1.2))):
    eye = T_wr[:3, 3] + R @ np.array(off)
    eye[2] = off[2] + 0.6
    shots.append(save(nm, env.render_aux(look_at_ros(eye, (T_wr[:3, 3] + scene_c) / 2 + np.array([0.2, 0, 0.1])), hfov_deg=62.0)["rgb"]))
# head: the camera module (colour camera position) seen from front-left and from the left
T_head = link_world(env.cfg["camera"]["prim_parent_link"]) @ env._T_link_cam
c = T_head[:3, 3]
for nm, off in (("head_front_left", (0.28, 0.22, 0.10)), ("head_left", (0.0, 0.42, 0.02))):
    shots.append(save(nm, env.render_aux(look_at_ros(c + R @ np.array(off), c - R @ np.array([0.03, 0, 0.05])), hfov_deg=38.0)["rgb"]))
# left wrist: centre of the parts on the wrist (clearance boxes of that link: bracket, scanner, D405), whole assembly in view
wl = env.cfg["robot"]["scanner_mount"]["parent_link"]
T_w = link_world(wl)
bx = [b for b in env.cfg["robot"].get("clearance_boxes") or [] if b["link"] == wl]
lc = (np.mean([np.asarray(b["center"]) for b in bx], 0) if bx else np.asarray(env.cfg["robot"]["scanner_mount"]["pos"]) * 0.6)
wc = (T_w @ np.r_[lc, 1.0])[:3]
for nm, off in (("wrist_left", (0.0, 0.50, 0.04)), ("wrist_front_left", (0.16, 0.45, 0.12)), ("wrist_rear_left", (-0.35, 0.30, 0.22)),
                ("wrist_top", (-0.08, 0.10, 0.50))):
    shots.append(save(nm, env.render_aux(look_at_ros(wc + R @ np.array(off), wc), hfov_deg=42.0)["rgb"]))
# sensor views
obs = env.get_obs()
shots.append(save("cam_head_color", obs["rgb"]))
if env.ir_cams is not None:
    shots.append(save("cam_head_ir_left", env.head_stereo_obs()["ir_left"]))
for n in env.hand_cams:
    shots.append(save(f"cam_{n}", env.hand_camera_obs(n)["rgb"]))
if env.scanner_on:
    shots.append(save("cam_scanner", obs["wrist_rgb"]))

# contact sheet: tiles 640 wide, labelled
tiles = []
for s in shots:
    im = cv2.imread(os.path.join(args.out, f"{s}.png"))
    im = cv2.resize(im, (640, int(round(im.shape[0] * 640 / im.shape[1]))))
    im = cv2.copyMakeBorder(im, 0, 360 - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) if im.shape[0] < 360 else im[:360]
    cv2.rectangle(im, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(im, s, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(im)
while len(tiles) % 3:
    tiles.append(np.full_like(tiles[0], 255))
sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
cv2.imwrite(os.path.join(args.out, "sheet.png"), sheet)
rep["images"] = shots
rep["wall_s"] = round(time.time() - t0, 1)
json.dump(rep, open(os.path.join(args.out, "preview.json"), "w"), indent=1, default=str)
print(json.dumps(rep, default=str))
print("PREVIEW_DONE", flush=True)
env.close()
