"""Measure the head camera of the standing robot (intrinsics, T_root_cam, pelvis height) and write
configs/testsuite/head_camera_nominal.yaml, used by the generator for task-2 start poses / visibility checks and by
the evaluation channel for task1_gt_obs_pose. Averaged over 2 s of standing (the head sways slightly).

python scripts/testsuite/measure_head_camera.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
from g1_barcode_sim.env import G1BarcodeEnv

env = G1BarcodeEnv("s1_table", headless=True, seed=0)
try:
    import yaml
    from g1_barcode_sim.camera import R_to_quat_wxyz, quat_wxyz_to_R
    env.stand_still(2.0)
    Ts, zs = [], []
    for _ in range(int(round(2.0 / env.policy_dt))):
        env._step_once()
        o = env.get_obs()
        Ts.append(o["T_root_cam"]); zs.append(float(o["T_world_root_gt"][2, 3]))
    Ts = np.array(Ts)
    t = Ts[:, :3, 3].mean(axis=0)
    q = np.mean([R_to_quat_wxyz(T[:3, :3]) for T in Ts], axis=0); q /= np.linalg.norm(q)
    T = np.eye(4); T[:3, :3] = quat_wxyz_to_R(q); T[:3, 3] = t
    o = env.get_obs()
    out = {"source": "scripts/testsuite/measure_head_camera.py: standing robot (s1_table), mean over 2 s; T_root_cam = "
                     "head camera optical frame (ROS) in the pelvis frame",
           "width": int(o["rgb"].shape[1]), "height": int(o["rgb"].shape[0]), "K": np.round(o["K"], 4).tolist(),
           "T_root_cam": np.round(T, 5).tolist(), "root_height": round(float(np.mean(zs)), 4),
           "sway_std_m": round(float(Ts[:, :3, 3].std(axis=0).max()), 5)}
    p = os.path.join(ROOT, "configs", "testsuite", "head_camera_nominal.yaml")
    yaml.safe_dump(out, open(p, "w"), sort_keys=False)
    print("WROTE", p, out, flush=True)
except BaseException:
    import traceback; traceback.print_exc(); sys.stdout.flush(); os._exit(1)
sys.stdout.flush(); os._exit(0)
