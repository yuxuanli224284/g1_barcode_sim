"""Compare two isaac_smoke trajectories (root pose + all joint positions every policy step).
python t800/compare_traj.py out/isaac_g1_base_traj.npz out/isaac_g1_mod_traj.npz"""
import json
import sys

import numpy as np

a, b = (np.load(p) for p in sys.argv[1:3])
res = {"files": sys.argv[1:3], "steps": [len(a["t"]), len(b["t"])],
       "same_joint_names": bool(list(a["joint_names"]) == list(b["joint_names"]))}
n = min(len(a["t"]), len(b["t"]))
for k in ("t", "pos", "yaw", "q"):
    d = np.abs(a[k][:n] - b[k][:n])
    res[f"max_abs_diff_{k}"] = float(d.max())
res["identical"] = bool(res["steps"][0] == res["steps"][1] and res["same_joint_names"]
                        and all(res[f"max_abs_diff_{k}"] == 0.0 for k in ("t", "pos", "yaw", "q")))
print("COMPARE_JSON " + json.dumps(res))
