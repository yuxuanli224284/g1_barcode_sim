"""Self-check of the simulator. Writes outputs/check_report.md + images.

python scripts/check_env.py                      # everything (each scene in its own process)
python scripts/check_env.py --scenes s1_table    # subset
Every scene runs in a subprocess because one process can launch Isaac Sim only once.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs")
SWITCHYARD = os.environ.get("PROGRESS_CLIENT_DIR", "")   # optional progress reporter (switchyard_progress.Run); unset = none


def progress(**kw):
    if SWITCHYARD and os.path.isdir(SWITCHYARD):
        sys.path.insert(0, SWITCHYARD)
        from switchyard_progress import Run
        run = Run("hbs-check-env", label="HumanoidBarcodeScan · check_env", gpus=[int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.isdigit()])
        if "failed" in kw:
            run.failed(kw["failed"])
        elif kw.get("done_all"):
            run.done()
        else:
            run.progress(kw["done"], kw["total"], unit="scenes", status=kw.get("status", "Running"))


# ----------------------------------------------------------------------------- worker (one scene)
def worker(scene: str, control: bool):
    import numpy as np
    from g1_barcode_sim.env import G1BarcodeEnv

    od = os.path.join(OUT, scene)
    os.makedirs(od, exist_ok=True)
    res = {"scene": scene}
    t0 = time.time()
    env = G1BarcodeEnv(scene, headless=True, seed=0)
    # modules below need the running Isaac Sim app
    import cv2
    import omni.usd
    from pxr import Usd, UsdGeom
    from g1_barcode_sim.camera import look_at_ros, visibility
    from g1_barcode_sim.barcode import decode_image
    from g1_barcode_sim.ground_truth import save_gt
    from g1_barcode_sim.scene_builder import AUTHORED
    res["startup_s"] = round(time.time() - t0, 1)
    cc = env.cfg["camera"]
    stage = omni.usd.get_context().get_stage()

    def usd_world(path):
        return np.array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T

    def box_positions():
        return {k: usd_world(p)[:3, 3].copy() for k, p in AUTHORED["boxes"].items()}

    # --- 1. physics 5 s
    b0 = box_positions()
    r = env.stand_still(5.0)
    b1 = box_positions()
    disp = max(float(np.linalg.norm(b1[k] - b0[k])) for k in b0)
    res["physics_5s"] = {"robot_status": r["status"], "reason": r["reason"], "max_box_displacement_m": disp,
                         "dynamic": bool(env.cfg["scene"].get("dynamic", False)),
                         "pass": r["status"] == "ok" and disp < 1e-3}

    # --- 2. GT
    gt = env.get_ground_truth()
    save_gt(gt, os.path.join(od, "gt.json"))
    contents = [b["content"] for b in gt["barcodes"]]
    required = ["content", "type", "box_id", "face", "center", "normal", "up", "size", "height_above_ground", "target_root_pose_d020"]
    complete = all(all(k in b and b[k] not in (None, []) for k in required) for b in gt["barcodes"])
    targets = [b for b in gt["barcodes"] if not b["distractor"]]
    distractors = [b for b in gt["barcodes"] if b["distractor"]]
    usd_err = max(float(np.linalg.norm(usd_world(AUTHORED["barcodes"][b["content"]])[:3, 3] - np.array(b["center"]))) for b in gt["barcodes"])
    res["gt"] = {"n_barcodes": len(contents), "n_targets": len(targets), "n_distractors": len(distractors),
                 "unique": len(set(contents)) == len(contents), "complete": complete,
                 "max_target_height_m": max(b["height_above_ground"] for b in targets),
                 "distractor_heights_m": [b["height_above_ground"] for b in distractors],
                 "gt_vs_usd_label_pos_err_m": usd_err}
    res["gt"]["pass"] = complete and res["gt"]["unique"] and res["gt"]["max_target_height_m"] <= 1.3 and usd_err < 1e-4

    # --- 3. snapshots
    env.save_snapshot(od)
    obs = env.get_obs()

    # --- 4. visibility from the head camera
    K = obs["K"]
    T_wc = obs["T_world_root_gt"] @ obs["T_root_cam"]
    raw = env.head_cam.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze(-1)
    H, W = raw.shape
    vis = [visibility(K, T_wc, W, H, raw, b, cc["visibility_depth_tol"]) for b in env.layout["barcodes"]]
    img = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
    for v in vis:
        if v["in_image"]:
            col = (0, 255, 0) if v["visible"] else (0, 0, 255)
            cv2.circle(img, (int(v["u"]), int(v["v"])), 5, col, 2)
            cv2.putText(img, v["content"], (int(v["u"]) + 6, int(v["v"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(od, "head_rgb_annotated.png"), img)
    n_vis = sum(v["visible"] for v in vis)
    res["visibility"] = {"n_visible": n_vis, "n_total": len(vis), "items": vis, "pass": n_vis >= 1}

    # --- 5. texture correctness: 1280x720 aux camera 0.3 m in front of each label
    dec_items = []
    lw = env.cfg["barcode"]["label_size"][env.cfg["barcode"]["type"]][0]
    dec_hfov = math.degrees(2 * math.atan(lw / cc["decode_label_width_frac"] / 2 / cc["decode_distance"]))
    res["decode_hfov_deg"] = dec_hfov
    os.makedirs(os.path.join(od, "decode"), exist_ok=True)
    for b in env.layout["barcodes"]:
        c, n, u = (np.asarray(b[k]) for k in ("center", "normal", "up"))
        T = look_at_ros(c + cc["decode_distance"] * n, c, world_up=u)
        out = env.render_aux(T, hfov_deg=dec_hfov)
        got = decode_image(out["rgb"])
        ok = got == [b["content"]]
        if not ok:
            cv2.imwrite(os.path.join(od, "decode", f"{b['content']}.png"), cv2.cvtColor(out["rgb"], cv2.COLOR_RGB2BGR))
        dec_items.append({"content": b["content"], "decoded": got, "pass": ok})
    if env.layout["barcodes"]:
        cv2.imwrite(os.path.join(od, "decode", "example.png"), cv2.cvtColor(out["rgb"], cv2.COLOR_RGB2BGR))
    n_ok = sum(d["pass"] for d in dec_items)
    res["decode"] = {"n_pass": n_ok, "n_total": len(dec_items), "rate": n_ok / max(1, len(dec_items)),
                     "failures": [d for d in dec_items if not d["pass"]], "pass": n_ok == len(dec_items)}

    # --- 6. standing-pose clearance at target_root_pose_d020
    gaps = []
    for b in gt["barcodes"]:
        tp = b["target_root_pose_d020"]
        env.teleport_root(tp["pos"][0], tp["pos"][1], tp["yaw"])
        g0 = env.collision_model.min_gap()
        r = env.stand_still(cc["clearance_settle_s"])
        g1 = env.collision_model.min_gap()
        gaps.append({"content": b["content"], "distractor": b["distractor"], "gap_at_teleport_m": g0["gap"],
                     "closest_body": g0["body"], "obstacle": g0["obstacle"], "gap_after_settle_m": g1["gap"],
                     "status_after_settle": r["status"], "penetration": g0["gap"] < 0})
    res["clearance"] = {"items": gaps, "min_gap_targets_m": min([g["gap_at_teleport_m"] for g in gaps if not g["distractor"]], default=None)}
    env.reset(0)

    # --- scene 3 extra: stack static 10 s
    if scene == "s3_stack":
        b0 = box_positions()
        r = env.stand_still(10.0)
        b1 = box_positions()
        d10 = max(float(np.linalg.norm(b1[k] - b0[k])) for k in b0)
        res["stack_static_10s"] = {"max_box_displacement_m": d10, "robot_status": r["status"], "pass": d10 < 1e-3}

    # --- M1 + M6 (scene 1)
    if control:
        if env.scanner_on:
            res["scanner"] = scanner_checks(env, od)
        res["m1"] = m1_checks(env)
        res["control"] = control_checks(env)
    json.dump(res, open(os.path.join(od, "check.json"), "w"), indent=1, default=float)
    env.close()


def m1_checks(env):
    import numpy as np
    env.reset(0)
    out = {}
    z = []
    for _ in range(int(round(10.0 / env.policy_dt))):
        env._step_once(); z.append(float(env.robot.data.root_pos_w[0, 2]))
    s, reason = env._check_status()
    out["stand_10s"] = {"status": s, "reason": reason, "pelvis_z_min": min(z), "pelvis_z_max": max(z), "pass": s == "ok"}
    p0 = env.get_obs()["T_world_root_gt"][:3, 3]
    # pass criterion (walk >= 0.5 m without falling); 2.5 s at 0.3 m/s keeps the toes clear of the table
    r = env.step(0.3, 0.0, 0.0, 2.5)
    p1 = r["obs"]["T_world_root_gt"][:3, 3]
    out["walk_vx0.3_2.5s"] = {"status": r["status"], "dx_m": float(p1[0] - p0[0]), "dy_m": float(p1[1] - p0[1]), "pass": r["status"] == "ok" and p1[0] - p0[0] >= 0.5}
    grip = {}
    for target in (0.0, 1.0, 0.0):
        env.set_gripper("both", target)
        env.stand_still(5.0)   # measured opening time constant ~2 s (friction=200 in the upstream actuator cfg)
        gs = env.get_obs()["gripper_state"]
        grip[f"target_{target}_{len(grip)}"] = gs
    vals = list(grip.values())
    sides = [sd for sd in ("left", "right") if not (env.scanner_on and sd == env.scanner_side)]   # scanner hand keeps its grip
    span = min(vals[1][sd] - vals[0][sd] for sd in sides)
    out["gripper"] = {"measured_opening": grip, "sides_tested": sides, "open_minus_closed": span, "pass": span > 0.5}
    return out


SCANNER_POSE_TOL = 0.01   # median relative depth difference allowed between the wrist camera and an aux render at the
                          # pose computed for it (measured 0.0003; a 1 cm / 1 deg pose error gives several %)


SCAN_STANDOFF, SCAN_LATERAL = 0.55, 0.20   # root 0.55 m from the label plane, label 0.20 m left of the root (in front of
                                          # the left shoulder; aim error 0.0 cm / 0.0 deg there)
SCAN_DISTANCE = 0.20                       # scanner tip to label (wrist reference point -> muzzle, 0.20 m)
SCAN_SETTLE_S = 3.0                        # aim IK converges in ~1-2 s (measured)


def scanner_checks(env, od):
    """Handheld scanner + camera at the scanner head (scene 1)."""
    import cv2
    import numpy as np
    from g1_barcode_sim.camera import look_at_ros
    from g1_barcode_sim.scanner import pad_gap
    env.reset(0)
    g = env.cfg["robot"]["gripper"]
    out = {}
    w = env.cfg["scanner"]["handle_width"]
    gaps = [pad_gap(env.robot, env.collision_model, env.scanner_side, g["pad_links"])]
    env.stand_still(3.0)
    gaps.append(pad_gap(env.robot, env.collision_model, env.scanner_side, g["pad_links"]))
    out["grip"] = {"handle_width_m": w, "pad_gap_after_reset_m": gaps[0], "pad_gap_after_3s_m": gaps[1],
                   "joint_target": env._scanner_grip_q, "set_gripper_scanner_side": env.set_gripper(env.scanner_side, 1.0)["status"]}
    poses = {"home": None, "reach": (0.10, -0.10, 0.15)}
    out["camera"] = {}
    for name, off in poses.items():
        env.reset_arms()
        if off is not None:
            env.move_wrist(env.scanner_side, *off)
        env.stand_still(2.0)
        obs = env.get_obs()
        T_wc = obs["T_world_root_gt"] @ obs["T_root_wrist_cam"]
        K, (H, W) = obs["wrist_K"], obs["wrist_rgb"].shape[:2]
        # the pose the env reports must be the pose the renderer used: render the aux camera there with the same
        # horizontal FOV and compare depth on the shared central band
        a = env.render_aux(T_wc, hfov_deg=math.degrees(2 * math.atan(W / 2 / K[0, 0])))
        Ka, ad = a["K"], a["depth"]
        vs, us = np.mgrid[0:ad.shape[0]:4, 0:ad.shape[1]:4]
        uw = np.clip(((us - Ka[0, 2]) / Ka[0, 0] * K[0, 0] + K[0, 2]).astype(int), 0, W - 1)
        vw = np.clip(((vs - Ka[1, 2]) / Ka[1, 1] * K[1, 1] + K[1, 2]).astype(int), 0, H - 1)
        wd, adv = obs["wrist_depth"][vw, uw], ad[vs, us]
        m = np.isfinite(wd) & np.isfinite(adv) & (wd > 0) & (adv > 0)
        rel = np.abs(wd[m] - adv[m]) / wd[m]
        tip = obs["T_root_scanner_tip"]
        d = tip[:3, 0]
        out["camera"][name] = {"wrist_offset_from_home": off, "pose_check_median_rel_depth_diff": float(np.median(rel)),
                               "pose_check_p90": float(np.percentile(rel, 90)), "n_px": int(m.sum()),
                               "rgb_std": float(obs["wrist_rgb"].std()),
                               "tip_root": tip[:3, 3].tolist(), "scan_dir_root": d.tolist(),
                               "scan_elevation_deg": math.degrees(math.asin(np.clip(d[2], -1, 1))),
                               "min_clearance_m": env.collision_model.min_gap()["gap"],
                               "pass": bool(np.median(rel) < SCANNER_POSE_TOL and obs["wrist_rgb"].std() > 1.0)}
        cv2.imwrite(os.path.join(od, f"scanner_cam_{name}.png"), cv2.cvtColor(obs["wrist_rgb"], cv2.COLOR_RGB2BGR))
        # close-up of the gun in the hand
        T_wg = obs["T_world_root_gt"] @ obs["T_root_scanner_mount"]
        R = obs["T_world_root_gt"][:3, :3]
        c = T_wg[:3, 3]
        img = env.render_aux(look_at_ros(c + R @ np.array([0.45, 0.35, 0.15]), c), hfov_deg=35.0)["rgb"]
        cv2.imwrite(os.path.join(od, f"scanner_closeup_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # scan: stand in front of each target barcode (teleport), aim_scanner_at(d=0.20), scan() must return its content
    from g1_barcode_sim.camera import inv_T
    out["scan"] = []
    for bc in [b for b in env.get_ground_truth()["barcodes"] if not b.get("distractor")]:
        n = np.asarray(bc["normal"], float); nh = n.copy(); nh[2] = 0.0; nh /= np.linalg.norm(nh)
        yaw = math.atan2(-nh[1], -nh[0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        p = np.asarray(bc["center"]) + SCAN_STANDOFF * nh - SCAN_LATERAL * left
        env.teleport_root(float(p[0]), float(p[1]), yaw)
        env.stand_still(0.5)
        T = inv_T(env.get_obs()["T_world_root_gt"])
        env.aim_scanner_at(T[:3, :3] @ np.asarray(bc["center"]) + T[:3, 3], T[:3, :3] @ n, SCAN_DISTANCE)
        r_st = env.stand_still(SCAN_SETTLE_S)
        r = env.scan()
        o = env.get_obs()
        out["scan"].append({"barcode": bc["content"], "status": r["status"], "content": r["content"],
                            "robot_status": r_st["status"], "aim_error": o["scanner_aim_error"], "gt_hit": r["gt"]["hit"],
                            "pass": r["content"] == bc["content"]})
        cv2.imwrite(os.path.join(od, f"scan_{bc['content']}.png"), cv2.cvtColor(o["wrist_rgb"], cv2.COLOR_RGB2BGR))
        env.reset_arms()
    out["pass"] = (all(v["pass"] for v in out["camera"].values()) and out["grip"]["set_gripper_scanner_side"] == "error"
                   and all(v["pass"] for v in out["scan"]))
    env.reset(0)
    return out


def control_checks(env):
    import numpy as np
    out = {"move_relative": []}
    for goal in [(0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.5)]:
        env.reset(0)
        r = env.move_relative(*goal, timeout=env.cfg["robot"]["move_relative"]["timeout_check"])
        out["move_relative"].append({k: r[k] for k in ("status", "reason", "elapsed_s", "requested", "odom_displacement",
                                                       "gt_displacement", "gt_error", "odom_error_vs_gt")})
    env.cfg["barcode"]["randomize"]["enabled"] = True
    env.reset(7); g1 = env.get_ground_truth()
    env.reset(7); g2 = env.get_ground_truth()
    env.reset(8); g3 = env.get_ground_truth()
    env.cfg["barcode"]["randomize"]["enabled"] = False
    out["determinism"] = {"same_seed_identical": json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True),
                          "different_seed_differs": json.dumps(g1, sort_keys=True) != json.dumps(g3, sort_keys=True)}
    out["determinism"]["pass"] = out["determinism"]["same_seed_identical"] and out["determinism"]["different_seed_differs"]
    env.reset(0)
    n = int(round(10.0 / env.policy_dt))
    env.walker.set_velocity(0.3, 0.0, 0.0)
    t = time.time()
    for _ in range(n):
        env._step_once()
    wall = time.time() - t
    env.walker.set_velocity(0, 0, 0)
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    vram = sum(int(l.split(",")[1]) for l in q.strip().splitlines() if l.split(",")[0].strip() == str(os.getpid()))
    out["perf"] = {"policy_steps_per_s": n / wall, "physics_steps_per_s": 4 * n / wall, "realtime_factor": n * env.policy_dt / wall,
                   "vram_mib": vram, "cameras": ["head 640x480 rgb+depth", "aux 1280x720 rgb+depth"] + (["scanner 640x480 rgb+depth"] if env.scanner_on else [])}
    return out


# ----------------------------------------------------------------------------- report
def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_report(scenes, m2):
    L = ["# check_env 自检报告", "", f"生成时间：{time.strftime('%F %T')}。逐项原始数据见 `outputs/<scene>/check.json`。", ""]
    PASS = lambda b: "PASS" if b else "FAIL"
    s1 = json.load(open(os.path.join(OUT, "s1_table", "check.json"))) if os.path.exists(os.path.join(OUT, "s1_table", "check.json")) else None
    if s1 and "m1" in s1:
        m = s1["m1"]
        L += ["## M1 机器人（场景 1 中测）", "", "| 项 | 结果 | 数值 |", "|---|---|---|",
              f"| 站立 10 s | {PASS(m['stand_10s']['pass'])} | pelvis z ∈ [{fmt(m['stand_10s']['pelvis_z_min'])}, {fmt(m['stand_10s']['pelvis_z_max'])}] m |",
              f"| 速度命令 vx=0.3 m/s 2.5 s（≥0.5 m） | {PASS(m['walk_vx0.3_2.5s']['pass'])} | dx={fmt(m['walk_vx0.3_2.5s']['dx_m'])} m, dy={fmt(m['walk_vx0.3_2.5s']['dy_m'])} m |",
              f"| 夹爪开合（每态保持 5 s；测 {m['gripper'].get('sides_tested', ['left', 'right'])}，持枪手不动） | {PASS(m['gripper']['pass'])} | 张开-闭合 开度差 {fmt(m['gripper']['open_minus_closed'])}；{json.dumps(m['gripper']['measured_opening'])} |", ""]
    if s1 and "scanner" in s1:
        sc_ = s1["scanner"]
        gr = sc_["grip"]
        L += ["## 扫码枪 + 枪头相机（场景 1 中测）", "", "| 项 | 结果 | 数值 |", "|---|---|---|",
              f"| 夹持 | 只报告 | 枪柄宽 {fmt(gr['handle_width_m'])} m；指垫间距 reset 后 {fmt(gr['pad_gap_after_reset_m'], 4)} m，站 3 s 后 {fmt(gr['pad_gap_after_3s_m'], 4)} m；对持枪手 set_gripper → {gr['set_gripper_scanner_side']} |"]
        for name, c in sc_["camera"].items():
            L.append(f"| 相机位姿一致性（{name}，手腕偏移 {c['wrist_offset_from_home']}） | {PASS(c['pass'])} | 与同位姿 aux 渲染的深度中位相对差 {c['pose_check_median_rel_depth_diff']:.4f}（p90 {c['pose_check_p90']:.4f}，阈值 {SCANNER_POSE_TOL}）；扫描方向仰角 {c['scan_elevation_deg']:.1f}°；枪口 root 坐标 {[round(x, 3) for x in c['tip_root']]}；最小间隙 {fmt(c['min_clearance_m'])} m |")
        for sc_i in sc_.get("scan", []):
            ae, gh = sc_i["aim_error"], sc_i["gt_hit"]
            hit = "未命中标签" if gh is None else f"命中 {gh['content']}，距离 {gh['distance_m']:.3f} m，入射角 {gh['incidence_deg']:.1f}°"
            L.append(f"| 扫码 {sc_i['barcode']}（站位 {SCAN_STANDOFF} m / 左偏 {SCAN_LATERAL} m，aim_scanner_at d={SCAN_DISTANCE}） | {PASS(sc_i['pass'])} | scan() → {sc_i['content']}（{sc_i['status']}）；瞄准误差 {ae['pos_m'] * 100:.1f} cm / {ae['angle_deg']:.1f}°；GT 射线 {hit}；机器人 {sc_i['robot_status']} |")
        L += ["", "截图：`outputs/s1_table/scanner_cam_{home,reach}.png`（枪头相机）、`scanner_closeup_{home,reach}.png`（枪特写）、`scan_<条码>.png`（扫码时的枪头相机画面）。", ""]
    if m2:
        L += ["## M2 条码", "", f"| 类型 | 解码正确 | 结果 |", "|---|---|---|"]
        for t, d in m2.items():
            L.append(f"| {t} | {d['n_pass']}/{d['n']} | {PASS(d['n_pass'] == d['n'])} |")
        L.append("")
    for sc in scenes:
        p = os.path.join(OUT, sc, "check.json")
        if not os.path.exists(p):
            L += [f"## {sc}", "", "**没有结果（worker 失败，见日志）**", ""]
            continue
        r = json.load(open(p))
        L += [f"## {sc}", "", f"启动耗时 {r['startup_s']} s。截图：`outputs/{sc}/head_rgb.png`、`head_depth.png`、`topdown.png`、`head_rgb_annotated.png`。", "",
              "| 检查 | 结果 | 数值 |", "|---|---|---|"]
        ph = r["physics_5s"]
        L.append(f"| 1 物理 5 s | {PASS(ph['pass'])} | 机器人 {ph['robot_status']}；箱子最大位移 {ph['max_box_displacement_m']:.2e} m（dynamic={ph['dynamic']}） |")
        g = r["gt"]
        L.append(f"| 2 GT | {PASS(g['pass'])} | {g['n_targets']} 目标 + {g['n_distractors']} 干扰；唯一={g['unique']}，完整={g['complete']}；目标最高 {fmt(g['max_target_height_m'])} m；干扰条码高度 {[round(x, 3) for x in g['distractor_heights_m']]}；GT 与 USD 偏差 {g['gt_vs_usd_label_pos_err_m']:.1e} m |")
        L.append(f"| 3 截图 | PASS | 已保存 |")
        v = r["visibility"]
        L.append(f"| 4 头部相机可见性 | {PASS(v['pass'])} | 可见 {v['n_visible']}/{v['n_total']} |")
        d = r["decode"]
        L.append(f"| 5 贴图解码（0.3 m, 1280×720） | {PASS(d['pass'])} | {d['n_pass']}/{d['n_total']} |")
        c = r["clearance"]
        L.append(f"| 6 站位间隙 d=0.20 | 只报告 | 目标条码最小间隙 {fmt(c['min_gap_targets_m'])} m；穿模 {sum(i['penetration'] for i in c['items'] if not i['distractor'])} 个 |")
        if "stack_static_10s" in r:
            s = r["stack_static_10s"]
            L.append(f"| 堆体静置 10 s | {PASS(s['pass'])} | 最大位移 {s['max_box_displacement_m']:.2e} m |")
        L += ["", "可见性明细（像素尺寸为标签四角投影的宽×高）：", "", "| 条码 | 可见 | u,v | 深度 z (m) | 渲染深度 (m) | 像素 w×h |", "|---|---|---|---|---|---|"]
        for it in v["items"]:
            px = f"{it['px_w']:.0f}×{it['px_h']:.0f}" if it["px_w"] else "-"
            uv = f"{it['u']:.0f},{it['v']:.0f}" if it["in_image"] else "画面外"
            L.append(f"| {it['content']} | {'是' if it['visible'] else '否'} | {uv} | {fmt(it['z'], 2)} | {fmt(it['rendered_depth'], 2) if it['rendered_depth'] is not None else '-'} | {px} |")
        L += ["", "站位间隙明细：", "", "| 条码 | 干扰 | teleport 时间隙 (m) | 最近部位 / 障碍 | 站 1 s 后间隙 (m) | 站后状态 |", "|---|---|---|---|---|---|"]
        for it in c["items"]:
            L.append(f"| {it['content']} | {'是' if it['distractor'] else ''} | {fmt(it['gap_at_teleport_m'])} | {it['closest_body']} / {it['obstacle']} | {fmt(it['gap_after_settle_m'])} | {it['status_after_settle']} |")
        if d["failures"]:
            L += ["", f"解码失败：{json.dumps(d['failures'], ensure_ascii=False)}"]
        L.append("")
        if "control" in r:
            ct = r["control"]
            L += ["## M6 控制检查（场景 1）", "", "| 命令 | 状态 | 耗时 (s) | 真值位移 (dx,dy,dyaw) | 真值终点误差 | odom 相对真值误差 |", "|---|---|---|---|---|---|"]
            for m in ct["move_relative"]:
                gd = ", ".join(f"{x:.3f}" for x in m["gt_displacement"])
                L.append(f"| move_relative{tuple(m['requested'])} | {m['status']} | {m['elapsed_s']:.1f} | ({gd}) | {m['gt_error']['pos_m']:.3f} m / {math.degrees(m['gt_error']['yaw_rad']):.1f}° | {m['odom_error_vs_gt']['pos_m']:.3f} m / {math.degrees(m['odom_error_vs_gt']['yaw_rad']):.1f}° |")
            dt = ct["determinism"]
            pf = ct["perf"]
            L += ["", f"- reset(seed) 确定性（开随机化）：同种子一致={dt['same_seed_identical']}，不同种子不同={dt['different_seed_differs']} → {PASS(dt['pass'])}",
                  f"- 性能（headless，物理 CPU，渲染 GPU，{', '.join(pf['cameras'])}）：{pf['policy_steps_per_s']:.1f} 策略步/s（{pf['physics_steps_per_s']:.0f} 物理步/s），实时因子 {pf['realtime_factor']:.2f}，显存 {pf['vram_mib']} MiB", ""]
    open(os.path.join(OUT, "check_report.md"), "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["s1_table", "s2_shelf", "s3_stack"])
    ap.add_argument("--_worker", default=None)
    ap.add_argument("--_control", action="store_true")
    args = ap.parse_args()
    if args._worker:
        try:
            worker(args._worker, args._control)
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)   # a live Kit app would otherwise keep the worker process alive
        return
    os.makedirs(OUT, exist_ok=True)
    m2 = {}
    for t in ("code128", "qr"):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "generate_barcodes.py"), "--type", t], check=False)
        d = json.load(open(os.path.join(OUT, f"barcodes_check_{t}.json")))
        m2[t] = {"n": d["n"], "n_pass": d["n_pass"]}
    failed = []
    for i, sc in enumerate(args.scenes):
        progress(done=i, total=len(args.scenes), status=f"checking {sc}")
        log = os.path.join(OUT, f"check_{sc}.log")
        stale = os.path.join(OUT, sc, "check.json")
        if os.path.exists(stale):
            os.remove(stale)
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", sc] + (["--_control"] if sc == "s1_table" else [])
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        if rc != 0 or not os.path.exists(os.path.join(OUT, sc, "check.json")):
            failed.append(sc)
        print(f"{sc}: rc={rc} log={log}", flush=True)
    write_report(args.scenes, m2)
    if failed:
        progress(failed=f"workers failed: {failed}")
    else:
        progress(done_all=True)
    print("report:", os.path.join(OUT, "check_report.md"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
