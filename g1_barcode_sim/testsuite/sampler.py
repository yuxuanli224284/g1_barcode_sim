"""Scenario sampling for the test suites. Pure Python / numpy, deterministic:
every scenario is a function of (suite name, base seed, index); each re-sampling attempt uses its own RNG stream
derived from (suite, seed, index, attempt) with SHA-256, so results do not depend on Python's hash seed or on the
order suites are generated in.

Levels
- L0: nominal values on a small grid (suite `grid`).
- L1: one factor swept over its range (continuous: evenly spaced; discrete: every value in turn), the rest nominal.
- L2: every factor sampled uniformly in its range.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np

from . import geometry as G
from .scenario import label_size, to_layout

T1_FACTORS = ["robot_dist", "robot_lateral", "box_shape", "box_appearance", "placement", "barcode_size", "background", "lighting"]
T2_FACTORS = ["barcode_pos", "barcode_size", "barcode_type", "box_shape", "box_lateral", "start_noise", "background"]


class Rejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def suite_short(name: str) -> str:
    return name.replace("_", "").upper()[:4]          # t1_L2 -> T1L2


def rng_for(suite: str, seed: int, idx: int, attempt: int) -> np.random.Generator:
    h = hashlib.sha256(f"{suite}|{seed}|{idx}|{attempt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _lerp(r, t):
    return r[0] + (r[1] - r[0]) * t


def _sweep_t(i, n):
    return i / (n - 1) if n > 1 else 0.5


# ------------------------------------------------------------------ spec (internal, before geometry)
def nominal_spec(F: dict) -> dict:
    B = F["barcode"]
    box = {"w": F["box"]["w"]["nominal"], "h": F["box"]["h"]["nominal"], "d": F["box"]["d"]["nominal"],
           "appearance": F["box"]["appearance"]["nominal"]}
    code = {"type": B["type"]["nominal"], "width": B["width"]["nominal"], "aspect": B["aspect"]["nominal"],
            "length": B["content_length"]["nominal"], "backing": B["backing"]["nominal"],
            "u_frac": 0.0, "v_frac": 0.0, "rot_base": B["rot_base_deg"]["nominal"], "rot_jitter": 0.0}
    return {
        "robot": {"dist": F["robot_init"]["dist_to_table"]["nominal"], "lateral": F["robot_init"]["lateral_offset"]["nominal"],
                  "yaw_deg": F["robot_init"]["yaw_deg"]["nominal"]},
        "table": {"depth": F["table"]["depth"]["nominal"], "width": F["table"]["width"]["nominal"], "height": F["table"]["height"]["nominal"]},
        "boxes": [copy.deepcopy(box), copy.deepcopy(box)],          # [0] = right (smaller y), [1] = left
        "setback": [F["placement"]["setback"]["nominal"]] * 2,
        "gap": F["placement"]["gap"]["nominal"],
        "group_y_frac": None, "group_y": F["placement"]["group_y"]["nominal"],
        "codes": [copy.deepcopy(code), copy.deepcopy(code)],
        "background": {"type": F["background"]["type"]["nominal"], "light_scale": F["background"]["light_scale"]["nominal"]},
        "target": "left",
        "t2": {"noise_scale": 0.0, "lateral_frac": 0.0, "lateral_on": False, "noise_on": False},
    }


def _tidx(spec):
    return 1 if spec["target"] == "left" else 0


def apply_l1(spec, F, task, factor, i, n, rng):
    """Single-factor sweep point i of n. Returns the recorded factor value."""
    t = _sweep_t(i, n)
    spec["target"] = "left" if i % 2 == 0 else "right"
    ti = _tidx(spec)
    if factor == "robot_dist":
        spec["robot"]["dist"] = _lerp(F["robot_init"]["dist_to_table"]["range"], t); return spec["robot"]["dist"]
    if factor == "robot_lateral":
        spec["robot"]["lateral"] = _lerp(F["robot_init"]["lateral_offset"]["range"], t); return spec["robot"]["lateral"]
    if factor == "box_shape":                       # target box scaled from the smallest to the largest size
        for k in ("w", "h", "d"):
            spec["boxes"][ti][k] = _lerp(F["box"][k]["range"], t)
        return t
    if factor == "box_appearance":
        vals = F["box"]["appearance"]["values"]
        a = vals[i % len(vals)]
        spec["target"] = "left" if (i // len(vals)) % 2 == 0 else "right"
        for b in spec["boxes"]:
            b["appearance"] = a
        return a
    if factor == "placement":                       # setback, gap, group_y in turn
        third = max(1, n // 3)
        sub, j = min(i // third, 2), i % third
        tt = _sweep_t(j, third)
        if sub == 0:
            spec["setback"] = [_lerp(F["placement"]["setback"]["range"], tt)] * 2; return {"setback": spec["setback"][0]}
        if sub == 1:
            spec["gap"] = _lerp(F["placement"]["gap"]["range"], tt); return {"gap": spec["gap"]}
        spec["group_y_frac"] = 2 * tt - 1; return {"group_y_frac": spec["group_y_frac"]}
    if factor == "barcode_size":
        spec["codes"][ti]["width"] = _lerp(F["barcode"]["width"]["range"], t); return spec["codes"][ti]["width"]
    if factor == "background":
        vals = F["background"]["type"]["values"]
        spec["background"]["type"] = vals[i % len(vals)]
        spec["target"] = "left" if (i // len(vals)) % 2 == 0 else "right"
        return spec["background"]["type"]
    if factor == "lighting":
        spec["background"]["light_scale"] = _lerp(F["background"]["light_scale"]["range"], t); return spec["background"]["light_scale"]
    if factor == "barcode_pos":
        us, vs = [-1.0, -1 / 3, 1 / 3, 1.0], [-1.0, 0.0, 1.0]
        spec["codes"][ti]["u_frac"], spec["codes"][ti]["v_frac"] = us[i % 4], vs[(i // 4) % 3]
        return {"u_frac": us[i % 4], "v_frac": vs[(i // 4) % 3]}
    if factor == "barcode_type":
        combos = [("code128", True), ("code128", False), ("qr", True), ("qr", False)]
        ty, bk = combos[i % 4]
        spec["target"] = "left" if (i // 4) % 2 == 0 else "right"
        c = spec["codes"][_tidx(spec)]
        c["type"], c["backing"] = ty, bk
        if ty == "qr":
            c["width"] = F["barcode"]["qr_side"]["nominal"]
        return f"{ty}{'' if bk else '_nobacking'}"
    if factor == "box_lateral":
        spec["t2"]["lateral_on"] = True
        spec["t2"]["lateral_frac"] = 2 * t - 1; return spec["t2"]["lateral_frac"]
    if factor == "start_noise":
        spec["t2"]["noise_scale"] = t; return t
    raise ValueError(f"unknown factor {factor!r} for task {task}")


def apply_l2(spec, F, task, idx, rng):
    R = lambda r: float(rng.uniform(r[0], r[1]))
    spec["target"] = "left" if idx % 2 == 0 else "right"      # exactly half / half
    spec["robot"]["dist"] = R(F["robot_init"]["dist_to_table"]["range"])
    spec["robot"]["lateral"] = R(F["robot_init"]["lateral_offset"]["range"])
    for k in ("depth", "width", "height"):
        spec["table"][k] = R(F["table"][k]["range"])
    for b in spec["boxes"]:
        for k in ("w", "h", "d"):
            b[k] = R(F["box"][k]["range"])
        b["appearance"] = str(rng.choice(F["box"]["appearance"]["values"]))
    spec["setback"] = [R(F["placement"]["setback"]["range"]) for _ in range(2)]
    spec["gap"] = R(F["placement"]["gap"]["range"])
    spec["group_y_frac"] = float(rng.uniform(-1, 1))
    B = F["barcode"]
    for c in spec["codes"]:
        c["type"] = str(rng.choice(B["type"]["values"]))
        c["width"] = R(B["qr_side"]["range"] if c["type"] == "qr" else B["width"]["range"])
        c["aspect"] = R(B["aspect"]["range"])
        c["length"] = int(rng.integers(B["content_length"]["range"][0], B["content_length"]["range"][1] + 1))
        c["backing"] = bool(rng.choice(B["backing"]["values"]))
        c["u_frac"], c["v_frac"] = float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1))
        c["rot_base"] = float(rng.choice(B["rot_base_deg"]["values"]))
        c["rot_jitter"] = float(rng.uniform(-B["rot_jitter_deg"], B["rot_jitter_deg"]))
    spec["background"]["type"] = str(rng.choice(F["background"]["type"]["values"]))
    spec["background"]["light_scale"] = R(F["background"]["light_scale"]["range"])
    spec["t2"]["lateral_on"] = True
    spec["t2"]["lateral_frac"] = float(rng.uniform(-1, 1))
    spec["t2"]["noise_scale"] = float(rng.uniform(0, 1))


def fit_table(spec, F) -> dict:
    """L1 only: the table is not the factor under test, so when the swept boxes / gap / setback do not fit the nominal
    table, the table grows (within its own range) just enough to hold them. Returns the adjusted fields."""
    adj = {}
    need_w = spec["boxes"][0]["w"] + spec["gap"] + spec["boxes"][1]["w"]
    if need_w > spec["table"]["width"]:
        spec["table"]["width"] = adj["width"] = min(need_w, F["table"]["width"]["range"][1])
    need_d = max(sb + b["d"] for sb, b in zip(spec["setback"], spec["boxes"]))
    if need_d > spec["table"]["depth"]:
        spec["table"]["depth"] = adj["depth"] = min(need_d, F["table"]["depth"]["range"][1])
    return {k: round(v, 4) for k, v in adj.items()}


# ------------------------------------------------------------------ spec -> scenario JSON (+ validity)
def _content(rng, length, alphabet, used: set):
    for _ in range(1000):
        s = "".join(alphabet[int(k)] for k in rng.integers(0, len(alphabet), size=length))
        if s not in used:
            return s
    raise Rejected("duplicate_content")


def realize(spec, F, cam, suite, idx, seed, task, level, factors_varied, factor_info, used_contents: set, rng):
    W = F["world"]
    T, B = spec["table"], F["barcode"]
    if task == 2 and cam is None:
        raise RuntimeError(f"task-2 suites need {F['head_camera_nominal']} (run scripts/testsuite/measure_head_camera.py)")
    # --- boxes on the table
    w0, w1 = spec["boxes"][0]["w"], spec["boxes"][1]["w"]
    total = w0 + spec["gap"] + w1
    y_max = T["width"] / 2 - total / 2
    if y_max < 0:
        raise Rejected("boxes_wider_than_table")
    gy = spec["group_y_frac"] * y_max if spec["group_y_frac"] is not None else spec["group_y"]
    if abs(gy) > y_max + 1e-9:
        raise Rejected("group_off_table")
    ys = [gy - total / 2 + w0 / 2, gy + total / 2 - w1 / 2]
    boxes = []
    for k, (b, sb) in enumerate(zip(spec["boxes"], spec["setback"])):
        if sb + b["d"] > T["depth"] + 1e-9:
            raise Rejected("box_depth_exceeds_table")
        boxes.append({"id": f"B{k}", "size_wdh": [round(b["w"], 4), round(b["d"], 4), round(b["h"], 4)], "appearance": b["appearance"],
                      "setback": round(sb, 4), "y_center": round(ys[k], 4)})
    # --- barcodes
    short = suite_short(suite)
    codes = []
    for k, c in enumerate(spec["codes"]):
        bw, bh = spec["boxes"][k]["w"], spec["boxes"][k]["h"]
        code = {"box_id": f"B{k}", "type": c["type"], "width": round(c["width"], 4), "label_backing": bool(c["backing"])}
        if c["type"] == "code128":
            code["aspect"] = round(c["aspect"], 4)
        lw, lh = label_size(code, B["backing_border"])
        rot = c["rot_base"] + c["rot_jitter"]
        a = math.radians(rot)
        eu = (lw * abs(math.cos(a)) + lh * abs(math.sin(a))) / 2
        ev = (lw * abs(math.sin(a)) + lh * abs(math.cos(a))) / 2
        umax, vmax = bw / 2 - B["margin"] - eu, bh / 2 - B["margin"] - ev
        if umax < 0 or vmax < 0:
            raise Rejected("label_does_not_fit_face")
        u, v = c["u_frac"] * umax, c["v_frac"] * vmax
        if T["height"] + bh / 2 + v > B["max_center_height"] + 1e-9:
            raise Rejected("barcode_too_high")
        code.update({"u": round(u, 4), "v": round(v, 4), "rot_deg": round(rot, 3),
                     "content": _content(rng, c["length"], B["content_alphabet"], used_contents),
                     "content_id": f"{short}-{idx:04d}-B{k}"})
        codes.append(code)
    sc = {"id": f"{suite}_{idx:04d}", "suite": suite, "task": task, "seed": seed, "index": idx, "level": level,
          "factors_varied": factors_varied, "factor": factor_info,
          "background": {"type": spec["background"]["type"], "light_scale": round(spec["background"]["light_scale"], 4)},
          "table": {"depth": round(T["depth"], 4), "width": round(T["width"], 4), "height": round(T["height"], 4),
                    "appearance": F["table"]["appearance"]},
          "robot_init": {"dist_to_table": round(spec["robot"]["dist"], 4), "lateral_offset": round(spec["robot"]["lateral"], 4),
                         "yaw_deg": round(spec["robot"]["yaw_deg"], 3)},
          "boxes": boxes, "barcodes": codes, "target": spec["target"], "task2_start": None}
    lay = to_layout(sc, F)
    rr = F["validity"]["robot_footprint_radius"]
    obstacles = [(np.array([p["center"][0] - p["size"][0] / 2, p["center"][1] - p["size"][1] / 2]),
                  np.array([p["center"][0] + p["size"][0] / 2, p["center"][1] + p["size"][1] / 2])) for p in lay["furniture"][0]["parts"][:1]]
    obstacles += [(np.array([b["pos"][0] - b["size"][0] / 2, b["pos"][1] - b["size"][1] / 2]),
                   np.array([b["pos"][0] + b["size"][0] / 2, b["pos"][1] + b["size"][1] / 2])) for b in lay["boxes"]]

    def clear(p):
        return min(G.circle_rect_gap(p, lo, hi, rr) for lo, hi in obstacles) > 0

    if task == 1 and not clear(lay["robot_init"]["pos"]):
        raise Rejected("robot_init_collision")
    if task == 2:
        sc["task2_start"] = task2_start(spec, F, cam, lay, rng, clear)
    return sc


def task2_start(spec, F, cam, lay, rng, clear):
    """true observation pose of the target box + lateral offset + noise; every barcode corner and face corner of the
    target must project into the head image, else Rejected."""
    S = F["task2_start"]
    K, Wd, Hd, T_rc, zr = cam
    tb = next(b for b in lay["boxes"] if b["id"] == lay["target_box_id"])
    bc = next(b for b in lay["barcodes"] if b["box_id"] == tb["id"])
    fc, n, corners = G.front_face(tb)
    ob = G.obs_pose(fc, n, corners, cam, S["d_obs_range"], S["d_obs_step"], S["fov_margin_frac"])
    pts = np.vstack([corners, G.label_corners_world(bc)])
    left = np.array([-math.sin(ob["yaw"]), math.cos(ob["yaw"]), 0.0])

    def visible(x, y, yaw):
        return G.in_image(K, Wd, Hd, G.root_T(x, y, yaw, zr) @ T_rc, pts, 0.0)[0]

    # feasible lateral interval (1 cm grid inside box_lateral_range) for the noise-free pose
    lo_r, hi_r = S["box_lateral_range"]
    grid = np.round(np.arange(lo_r, hi_r + 1e-9, 0.01), 4)
    ok = [l for l in grid if visible(ob["pos"][0] + l * left[0], ob["pos"][1] + l * left[1], ob["yaw"])]
    if not ok:
        raise Rejected("task2_not_visible")
    lat_lo, lat_hi = float(min(ok)), float(max(ok))
    # box_lateral factor / L2: offset swept over the feasible interval; otherwise centred on the box (offset 0)
    lat = (lat_lo + lat_hi) / 2 + spec["t2"]["lateral_frac"] * (lat_hi - lat_lo) / 2 if spec["t2"]["lateral_on"] else 0.0
    nmax = {k: S["noise"][k] + (S["noise_max"][k] - S["noise"][k]) * spec["t2"]["noise_scale"] for k in ("pos", "yaw_deg")}
    if not spec["t2"]["noise_on"]:
        nmax = {"pos": 0.0, "yaw_deg": 0.0}
    dx, dy = (float(rng.uniform(-nmax["pos"], nmax["pos"])) for _ in range(2))
    dyaw = float(rng.uniform(-nmax["yaw_deg"], nmax["yaw_deg"]))
    x = ob["pos"][0] + lat * left[0] + dx
    y = ob["pos"][1] + lat * left[1] + dy
    yaw = ob["yaw"] + math.radians(dyaw)
    if not visible(x, y, yaw):
        raise Rejected("task2_not_visible")
    if not clear([x, y]):
        raise Rejected("task2_robot_collision")
    return {"pose_world": {"x": round(x, 4), "y": round(y, 4), "yaw_deg": round(math.degrees(yaw), 3)},
            "noise": {"dx": round(dx, 4), "dy": round(dy, 4), "dyaw_deg": round(dyaw, 3), "bound_pos": round(nmax["pos"], 4),
                      "bound_yaw_deg": round(nmax["yaw_deg"], 3)},
            "lateral_offset": round(lat, 4), "lateral_feasible": [lat_lo, lat_hi],
            "obs_d": ob["d_obs"], "obs_fits": ob["fits"]}


# ------------------------------------------------------------------ suites
def expand_suite_names(F: dict, name: str, factor: str | None = None) -> list[tuple[str, dict]]:
    """('t1_L1', factor) -> [('t1_L1_<factor>', definition), ...]; L0 / L2 -> [(name, definition)]."""
    if name not in F["suites"]:
        raise ValueError(f"unknown suite {name!r}; known: {list(F['suites'])}")
    d = F["suites"][name]
    if d["level"] != "L1":
        return [(name, d)]
    facs = [factor] if factor else list(d["factors"])
    for f in facs:
        if f not in d["factors"]:
            raise ValueError(f"suite {name} has no factor {f!r}; known: {list(d['factors'])}")
    return [(f"{name}_{f}", dict(d, factor=f, n=d["factors"][f]["n"])) for f in facs]


def generate(F: dict, cam, full_name: str, d: dict, seed: int, num: int | None = None):
    """Returns (scenarios, rejection counts, warnings, skipped infeasible L1 sweep points)."""
    task, level = d["task"], d["level"]
    reasons, warnings, used = {}, [], set()
    items = []
    if level == "L0":
        keys = list(d["grid"])
        grid = [dict(zip(keys, vals)) for vals in _product([d["grid"][k] for k in keys])]
        items = [("grid", g) for g in grid]
    elif level == "L1":
        items = [("l1", i) for i in range(d["n"])]
    else:
        items = [("l2", i) for i in range(num or d["num"])]
    out = []
    skipped = []
    for idx, (kind, arg) in enumerate(items):
        reasons_this = []
        for attempt in range(F["validity"]["max_resample"]):
            rng = rng_for(full_name, seed, idx, attempt)
            spec = nominal_spec(F)
            spec["t2"]["noise_on"] = level != "L0"
            factor_info, varied = None, []
            if kind == "grid":
                for k, v in arg.items():
                    if k == "dist_to_table":
                        spec["robot"]["dist"] = v
                    elif k == "target":
                        spec["target"] = v
                    elif k == "group_y":
                        spec["group_y"] = v
                    else:
                        raise ValueError(f"unsupported L0 grid key {k!r}")
                factor_info = dict(arg)
            elif kind == "l1":
                val = apply_l1(spec, F, task, d["factor"], arg, d["n"], rng)
                factor_info, varied = {"name": d["factor"], "value": val}, [d["factor"]]
                adj = fit_table(spec, F)
                if adj:
                    factor_info["table_adjusted"] = adj
            else:
                apply_l2(spec, F, task, idx, rng)
                varied = list(T1_FACTORS if task == 1 else T2_FACTORS)
            try:
                sc = realize(spec, F, cam, full_name, idx, seed, task, level, varied, factor_info, used, rng)
                break
            except Rejected as e:
                reasons[e.reason] = reasons.get(e.reason, 0) + 1
                reasons_this.append(e.reason)
        else:
            if kind == "l1":
                # a deterministic sweep point that is geometrically infeasible (e.g. a face too large to be seen whole
                # from d <= d_obs max) is not a sampling failure: skip it and record why (manifest skipped_infeasible)
                skipped.append({"index": idx, "factor_value": factor_info["value"],
                                "reasons": sorted({r for r in reasons_this}), "attempts": F["validity"]["max_resample"]})
                continue
            raise RuntimeError(f"{full_name} #{idx}: no valid scenario after {F['validity']['max_resample']} attempts "
                               f"(rejections so far: {reasons})")
        for c in sc["barcodes"]:
            used.add(c["content"])
        if sc["task2_start"] and not sc["task2_start"]["obs_fits"]:
            warnings.append(f"{sc['id']}: no d in d_obs range fits the face with the margin; d_obs = upper bound")
        out.append(sc)
    return out, reasons, warnings, skipped


def _product(lists):
    if not lists:
        return [[]]
    return [[a] + rest for a in lists[0] for rest in _product(lists[1:])]
