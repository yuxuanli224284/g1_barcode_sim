"""Head camera helpers: transforms, projection, depth post-processing, visibility.

Conventions
- `T_a_b` (4x4) is the pose of frame b expressed in frame a (maps b-coordinates to a-coordinates).
- Camera optical frame (ROS): +x right, +y down, +z forward (viewing direction).
- Quaternions are wxyz.
"""
from __future__ import annotations

import math

import numpy as np


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat_wxyz(R) -> list:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return (q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)).tolist()


def make_T(pos, quat_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def T_from_xy_yaw(x, y, yaw, z=0.0) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


def inv_T(T) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def look_at_ros(cam_pos, target, world_up=(0, 0, 1)) -> np.ndarray:
    """T_world_cam (ROS optical frame) for a camera at cam_pos looking at target."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z /= np.linalg.norm(z)
    up = np.asarray(world_up, float)
    if abs(z @ up) > 0.999:          # looking straight up/down: pick any perpendicular "up"
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)   # right
    y = np.cross(z, x)                             # down
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, cam_pos
    return T


def project(K, T_world_cam, pts_world):
    """Returns (uv Nx2, z N) for Nx3 world points; z is depth along the optical axis."""
    P = np.asarray(pts_world, float).reshape(-1, 3)
    Tc = inv_T(T_world_cam)
    pc = (Tc[:3, :3] @ P.T).T + Tc[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (np.asarray(K) @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def postprocess_depth(depth, valid_range, noise_cfg=None, rng=None):
    """Clamp to the valid range (outside -> 0) and optionally add D435-like noise."""
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if noise_cfg and noise_cfg.get("enabled"):
        rng = rng or np.random.default_rng()
        d = d + rng.normal(0.0, 1.0, d.shape).astype(np.float32) * noise_cfg["sigma_coeff"] * d * d
        # random holes along depth edges
        gy, gx = np.gradient(d)
        edge = np.hypot(gx, gy) > 0.05 * np.maximum(d, 1e-3)
        d[edge & (rng.random(d.shape) < noise_cfg["hole_prob"])] = 0.0
    lo, hi = valid_range
    d[(d < lo) | (d > hi)] = 0.0
    return d


def label_corners(center, right, up, size):
    c, r, u = (np.asarray(v, float) for v in (center, right, up))
    w, h = size
    return np.array([c + sx * w / 2 * r + sy * h / 2 * u for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def visibility(K, T_world_cam, width, height, raw_depth, bc, depth_tol):
    """Project a barcode's GT centre; visible if in front, inside the image, facing the camera and
    the rendered depth at that pixel matches the expected depth within depth_tol."""
    uv, z = project(K, T_world_cam, [bc["center"]])
    u, v, zc = float(uv[0, 0]), float(uv[0, 1]), float(z[0])
    cam_pos = T_world_cam[:3, 3]
    facing = float(np.dot(np.asarray(bc["normal"]), cam_pos - np.asarray(bc["center"]))) > 0
    out = {"content": bc["content"], "u": u, "v": v, "z": zc, "facing": facing, "in_image": False,
           "rendered_depth": None, "visible": False, "px_w": None, "px_h": None}
    if zc <= 0 or not (0 <= u < width and 0 <= v < height):
        return out
    out["in_image"] = True
    rd = float(raw_depth[int(v), int(u)])
    out["rendered_depth"] = rd
    cuv, cz = project(K, T_world_cam, label_corners(bc["center"], bc["right"], bc["up"], bc["size"]))
    if np.all(cz > 0):
        out["px_w"] = float(np.linalg.norm(cuv[1] - cuv[0]))
        out["px_h"] = float(np.linalg.norm(cuv[3] - cuv[0]))
    out["visible"] = bool(facing and np.isfinite(rd) and abs(rd - zc) < depth_tol)
    return out
