"""Example: stand in front of a barcode, aim the scanner at it, scan."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from g1_barcode_sim.env import G1BarcodeEnv
from g1_barcode_sim.camera import inv_T

# policy.onnx keeps the base nearly still while the arm moves
env = G1BarcodeEnv("s1_table", headless=True)
bc = env.get_ground_truth()["barcodes"][0]          # the example uses GT; a real algorithm estimates the label pose
c, n = np.array(bc["center"]), np.array(bc["normal"])  # from the head camera
# stand 0.55 m in front of the label plane with the label 0.20 m to the left (in front of the scanner hand)
env.teleport_root(c[0] + 0.55 * n[0], c[1] + 0.55 * n[1] - 0.20, math.atan2(-n[1], -n[0]))
env.stand_still(0.5)
T_root_world = inv_T(env.get_obs()["T_world_root_gt"])
c_root = T_root_world[:3, :3] @ c + T_root_world[:3, 3]
n_root = T_root_world[:3, :3] @ n
print(env.aim_scanner_at(c_root, n_root, distance=0.20))   # tip 0.20 m in front of the label, scanning along -normal
env.stand_still(3.0)                                         # aim IK converges in ~1-2 s
obs = env.get_obs()
print("aim error", obs["scanner_aim_error"])                # {"pos_m", "angle_deg"}
r = env.scan()
print("scan:", r["status"], r["content"], "| GT ray:", r["gt"]["hit"])
env.close()
