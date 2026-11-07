"""Scenario geometry shared by the generator (validity checks, task-2 start poses) and the simulator's evaluation
channel (task1_gt_obs_pose, task2_gt_wrist_target). Pure numpy, no Isaac.

World frame: z up, table front edge parallel to y at x = table_front_x, boxes face -x.
Box layout size convention: [d (x), w (y), h (z)].
"""
from __future__ import annotations

import math

import numpy as np

from ..barcode import FACE_AXES
from ..camera import T_from_xy_yaw, project


def head_camera(nominal: dict):
    """(K 3x3, W, H, T_root_cam 4x4, root_height) from configs/testsuite/head_camera_nominal.yaml."""
    K = np.array(nominal["K"], dtype=float)
    return K, int(nominal["width"]), int(nominal["height"]), np.array(nominal["T_root_cam"], dtype=float), float(nominal["root_height"])


def root_T(x, y, yaw, root_height):
    return T_from_xy_yaw(x, y, yaw, z=root_height)


# The four vertical sides of a stack in the task-3 tour order (front -x facing the robot start, then right -y, back
# +x, left +y: counter-clockwise seen from above, continuing the left-to-right column order of each face).
TOUR_FACES = ("-x", "-y", "+x", "+y")


def _face_frame(face: str):
    """(outward normal, in-face right axis, up axis) of a vertical face, world == box-local (boxes have yaw 0)."""
    if face not in TOUR_FACES:
        raise ValueError(f"face {face!r} is not a vertical box face {TOUR_FACES}")
    n, r, u = (np.array(a, dtype=float) for a in FACE_AXES[face])
    return n, r, u


def _axis_aligned(box: dict):
    if box.get("yaw", 0.0) != 0.0:
        raise ValueError(f"box {box.get('id')}: face geometry assumes axis-aligned boxes (yaw 0), got yaw {box['yaw']}")


def box_face(box: dict, face: str):
    """Centre, outward normal and 4 corners (world) of one vertical face ("-x", "+x", "-y", "+y") of a box.
    box: layout box (pos, size [d, w, h], yaw 0). Corners are ordered along the world lateral axis, then z (for "-x":
    y ascending, as front_face always was)."""
    _axis_aligned(box)
    n, r, up = _face_frame(face)
    half = np.asarray(box["size"], float) / 2
    c = np.asarray(box["pos"], float) + n * (np.abs(n) @ half)
    lat = np.abs(r)                                   # world lateral axis of the face (unit +x or +y)
    hw, hh = float(lat @ half), float(half[2])
    corners = np.array([c + sa * hw * lat + sz * hh * up for sa in (-1, 1) for sz in (-1, 1)])
    return c, n, corners


def front_face(box: dict):
    """Centre, outward normal and 4 corners (world) of a box's -x face."""
    return box_face(box, "-x")


def _face_rect(box: dict, face: str):
    """Face rectangle in (lateral world coordinate, z): (lo [2], hi [2]) and the face plane coordinate along n."""
    n, r, _ = _face_frame(face)
    lat = np.abs(r)
    p, half = np.asarray(box["pos"], float), np.asarray(box["size"], float) / 2
    lo = np.array([lat @ p - lat @ half, p[2] - half[2]])
    hi = np.array([lat @ p + lat @ half, p[2] + half[2]])
    return lo, hi, float(n @ p + np.abs(n) @ half)


def blockers(boxes: list, face: str, plane: float, lo, hi, depth: float, exclude=None) -> list:
    """Boxes whose volume reaches into the slab [plane, plane + depth] (outward along the face normal) over the
    rectangle lo..hi (lateral world coordinate, z) of that face with a positive-area overlap. Returns their overlap
    rectangles [(lo, hi)] clipped to lo..hi. depth: how far in front of the face something must stay clear (a label
    sits face_offset off the face and is label_thickness deep). exclude: id of the face's own box."""
    n, r, _ = _face_frame(face)
    lat = np.abs(r)
    out = []
    for b in boxes:
        if b["id"] == exclude:
            continue
        _axis_aligned(b)
        p, half = np.asarray(b["pos"], float), np.asarray(b["size"], float) / 2
        s = n @ p                                     # the box spans [s - e, s + e] along n
        e = float(np.abs(n) @ half)
        if not (s - e < plane + depth and s + e > plane):       # open intervals: touching only at the plane is not
            continue                                            # inside the slab when the box lies behind it
        blo = np.maximum(lo, [lat @ p - lat @ half, p[2] - half[2]])
        bhi = np.minimum(hi, [lat @ p + lat @ half, p[2] + half[2]])
        if np.all(bhi > blo):
            out.append((blo, bhi))
    return out


def _fully_covered(lo, hi, rects) -> bool:
    """True when the union of axis-aligned rectangles covers the rectangle lo..hi (exact: coordinate compression)."""
    if not rects:
        return False
    xs = sorted({lo[0], hi[0], *[r[0][0] for r in rects], *[r[1][0] for r in rects]})
    zs = sorted({lo[1], hi[1], *[r[0][1] for r in rects], *[r[1][1] for r in rects]})
    for x0, x1 in zip(xs, xs[1:]):
        for z0, z1 in zip(zs, zs[1:]):
            if x1 - x0 <= 1e-9 or z1 - z0 <= 1e-9:   # float-rounding slivers between abutting boxes are not exposed
                continue                             # area (1e-9 m: the slack to_layout uses for its overhang test)
            mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
            if not any(r[0][0] <= mx <= r[1][0] and r[0][1] <= mz <= r[1][1] for r in rects):
                return False
    return True


def exposed_faces(boxes: list, face: str, depth: float) -> list:
    """Boxes whose `face` side is not fully covered by other boxes reaching into the slab of thickness `depth` in front
    of it (e.g. the front face of a back-row box pressed against the front row is hidden)."""
    out = []
    for b in boxes:
        lo, hi, plane = _face_rect(b, face)
        if not _fully_covered(lo, hi, blockers(boxes, face, plane, lo, hi, depth, exclude=b["id"])):
            out.append(b)
    return out


def face_union(boxes: list, face: str, depth: float):
    """The exposed `face` sides of several boxes taken together (task 3 observes every box), as the harness does for
    selector "all": outward normal of `face`, reference point on the outermost
    exposed face plane along the normal, laterally and vertically centred on all exposed corners; corners = every
    exposed face corner. Returns None when no box exposes that side."""
    vis = exposed_faces(boxes, face, depth)
    if not vis:
        return None
    faces = [box_face(b, face) for b in vis]
    n = faces[0][1]
    corners = np.vstack([f[2] for f in faces])
    _, r, _ = _face_frame(face)
    lat = np.abs(r)
    s = max(float(n @ f[0]) for f in faces)          # outermost plane (nearest a viewer outside that side)
    lc = (corners @ lat).min(), (corners @ lat).max()
    c = n * s + lat * (lc[0] + lc[1]) / 2 + np.array([0.0, 0.0, (corners[:, 2].min() + corners[:, 2].max()) / 2])
    return c, n, corners


def front_face_union(boxes: list, depth: float):
    """face_union of the -x sides (exposed only: a 2-deep stack does not include the hidden back-row faces)."""
    return face_union(boxes, "-x", depth)


def stack_faces(boxes: list, barcodes: list, cam, d_range, d_step, margin_frac, depth: float) -> list:
    """GT of every exposed side of a free-standing stack, in the tour order TOUR_FACES: [{"face", "normal", "center",
    "corners", "gt_obs_pose" (obs_pose of that side's union), "contents" (labels on that side)}]."""
    out = []
    for face in TOUR_FACES:
        u = face_union(boxes, face, depth)
        if u is None:
            continue
        c, n, corners = u
        out.append({"face": face, "normal": n.tolist(), "center": c.tolist(), "corners": corners.tolist(),
                    "gt_obs_pose": obs_pose(c, n, corners, cam, d_range, d_step, margin_frac),
                    "contents": [bc["content"] for bc in barcodes if bc["face"] == face and not bc["distractor"]]})
    return out


def in_image(K, W, H, T_world_cam, pts, margin_frac=0.0):
    uv, z = project(K, T_world_cam, pts)
    mx, my = margin_frac * W, margin_frac * H
    ok = (z > 0) & (uv[:, 0] >= mx) & (uv[:, 0] <= W - mx) & (uv[:, 1] >= my) & (uv[:, 1] <= H - my)
    return bool(np.all(ok)), uv


def obs_pose(face_center, normal, corners, cam, d_range, d_step, margin_frac):
    """The observation pose with true geometry: face the box (yaw along -n_h), laterally centred on the
    face; smallest d in d_range (step d_step) whose head-camera image contains the 4 face corners with a margin_frac
    border. Returns {"pos": [x, y, 0], "yaw", "d_obs", "fits"}; fits=False -> d_obs = d_range[1] (spec fallback)."""
    K, W, H, T_rc, zr = cam
    n = np.asarray(normal, float)
    nh = np.array([n[0], n[1], 0.0]); nh /= np.linalg.norm(nh)
    yaw = math.atan2(-nh[1], -nh[0])
    c = np.asarray(face_center, float)
    n_steps = int(round((d_range[1] - d_range[0]) / d_step))
    for k in range(n_steps + 1):
        d = d_range[0] + k * d_step
        p = c + d * nh
        ok, _ = in_image(K, W, H, root_T(p[0], p[1], yaw, zr) @ T_rc, corners, margin_frac)
        if ok:
            return {"pos": [float(p[0]), float(p[1]), 0.0], "yaw": float(yaw), "d_obs": float(round(d, 6)), "fits": True}
    p = c + d_range[1] * nh
    return {"pos": [float(p[0]), float(p[1]), 0.0], "yaw": float(yaw), "d_obs": float(d_range[1]), "fits": False}


def label_corners_world(bc: dict):
    c, r, u = (np.asarray(bc[k], float) for k in ("center", "right", "up"))
    w, h = bc["size"]
    return np.array([c + sx * w / 2 * r + sy * h / 2 * u for sx in (-1, 1) for sy in (-1, 1)])


def wrist_target(bc: dict, offset: float):
    """task2_gt_wrist_target: barcode centre + offset * normal, forward axis along -normal."""
    c, n = np.asarray(bc["center"], float), np.asarray(bc["normal"], float)
    return {"pos": (c + offset * n).tolist(), "forward": (-n).tolist(), "offset": float(offset)}


def circle_rect_gap(p, rect_min, rect_max, radius):
    """Clearance between a disc (centre p xy, radius) and an axis-aligned rectangle (xy)."""
    q = np.clip(np.asarray(p[:2], float), rect_min, rect_max)
    return float(np.linalg.norm(np.asarray(p[:2], float) - q) - radius)
