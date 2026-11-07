"""Example: walk to a barcode's GT standing pose with move_relative and print the error."""
import math, os, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1_barcode_sim.env import G1BarcodeEnv
from g1_barcode_sim.camera import inv_T, T_from_xy_yaw, yaw_of, wrap

env = G1BarcodeEnv("s1_table", headless=True, seed=0)
obs = env.reset()
gt = env.get_ground_truth(d=0.40)   # d = 0.20 时站定后最小间隙只有 1.7–4.4 cm，比 move_relative 的到位误差（约 4 cm）还小，示例里留出余量
bc = gt["barcodes"][0]
tgt = bc["target_root_pose_d040"]
# 目标在当前 root 坐标系下的相对位姿（示例直接用 GT；真实算法应从图像里估计）
T_w_r = obs["T_world_root_gt"]
rel = inv_T(T_from_xy_yaw(T_w_r[0, 3], T_w_r[1, 3], yaw_of(T_w_r))) @ T_from_xy_yaw(*tgt["pos"][:2], tgt["yaw"])
r = env.move_relative(rel[0, 3], rel[1, 3], yaw_of(rel), timeout=30.0)
T = env.get_obs()["T_world_root_gt"]
print(r["status"], "到位误差 %.3f m, %.1f°" % (np.linalg.norm(T[:2, 3] - np.array(tgt["pos"][:2])),
      math.degrees(abs(wrap(yaw_of(T) - tgt["yaw"])))))
env.close()
