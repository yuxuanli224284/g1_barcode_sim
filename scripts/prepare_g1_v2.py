"""Prepare the real-robot-aligned G1 (robot variant g1_v2, configs/robot_variants.yaml) from the robot description
the lab supplied (g1_description.zip, 2026-11-19: g1_29dof_rev_1_0.urdf + meshes/).

    python scripts/prepare_g1_v2.py --src <unzipped g1_description> --out <description dir, e.g. assets/g1_description_v2>
    python scripts/prepare_g1_v2.py --src assets/g1_description_v2 --out assets/g1_description_v2 --d435_pitch_deg 26.15   # re-derive in place

Writes into --out:
  g1_29dof_rev_1_0.urdf   the supplied URDF, unchanged
  g1_v2_sim.urdf          the URDF the simulator imports. Changes: (1) head_link gets back the visual of the stock G1 URDF
                          (meshes/head_link.STL, material dark): the supplied URDF keeps the head's collision mesh and mass
                          but has no visual, so the head would be invisible in renders; (2) --d435_pitch_deg (optional):
                          d435_joint pitch (downward tilt of the D435's optical axis below the torso's horizontal); the
                          supplied URDF has 47.6 deg (0.8307767 rad, as the stock G1). User 2026-11-19: the camera face stands
                          at 60 deg to the floor on the real robot, i.e. the axis is 30 deg below horizontal; the
                          standing sim robot leans its torso 3.85 deg forward -> 26.15.
  meshes/                 the meshes the URDF references (copied)
  derived_frames.json     sensor / tool frames measured from the meshes (printed too); robot_variants.yaml g1_v2 holds
                          these numbers, this file is the record of where they come from.

Measured from the meshes (frames of the movable parent links, x forward, y left, z up):
- D435 (head): the URDF d435_joint origin is not where the camera body is: the d435 visual is 0.5-3.0 cm in front of and
  0.75-3.25 cm above the link origin, so a camera at the joint origin looks into the back of the D435. The imagers are put
  where the body is: left infrared imager (depth origin) 4.2 mm behind the front glass and 17.5 mm left of the module
  centre (Intel realsense2_description _d435.urdf.xacro: d435_zero_depth_to_glass, d435_cam_depth_py).
- D405 (left wrist): the URDF left_d405_joint is commented "Example wrist mounting position" and sits ~10 cm from the
  D405 visual (which is on top of the bracket mast). The camera is put on the visual: front face = the large face away
  from the mast (the face toward the mast carries the two M3 mounting holes), left imager 9 mm left of the face centre
  (D405 stereo baseline 18 mm); viewing direction = the front-face normal.
- Scanner window: the nose of meshes/scan_tool.obj is a funnel whose bottom is a flat window; window centre = area-weighted
  centre of the front-facing triangles (normal within 30 deg of the tool axis) at the depth with the largest such area behind the rim.
- Clearance boxes: axis-aligned boxes (link frame) around the visual-only parts (bracket, scanner head / handle, D405,
  right D405 + mount), for the robot clearance model (these parts have no collision geometry in the URDF).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

HEAD_VISUAL = """    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="meshes/head_link.STL" />
      </geometry>
      <material name="dark" />
    </visual>
"""


def T_of(o):
    xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()] if o is not None else [0.0] * 3
    rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0.0] * 3
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


class Urdf:
    def __init__(self, path):
        self.dir = os.path.dirname(path)
        root = ET.parse(path).getroot()
        self.joints = {j.find("child").get("link"): (j.find("parent").get("link"), T_of(j.find("origin")), j.get("type"))
                       for j in root.findall("joint")}
        self.links = {ln.get("name"): ln for ln in root.findall("link")}

    def T(self, link, base):
        """Pose of `link` in `base` through fixed joints only."""
        T, cur = np.eye(4), link
        while cur != base:
            p, Tj, ty = self.joints[cur]
            if ty != "fixed":
                raise ValueError(f"{cur} hangs on a {ty} joint below {base}")
            T, cur = Tj @ T, p
        return T

    def mesh(self, link, base, kind="visual"):
        ms = []
        for v in self.links[link].findall(kind):
            g = v.find("geometry/mesh")
            m = trimesh.load(os.path.join(self.dir, g.get("filename")), force="mesh")
            m.apply_scale([float(s) for s in (g.get("scale") or "1 1 1").split()])
            m.apply_transform(self.T(link, base) @ T_of(v.find("origin")))
            ms.append(m)
        return trimesh.util.concatenate(ms)


def copy_description(src, out, d435_pitch_deg=None):
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)
    text = open(os.path.join(src, "g1_29dof_rev_1_0.urdf")).read()
    # --src may be the output dir itself: it keeps the supplied URDF unchanged and every mesh that URDF references, so
    # the description can be re-derived (e.g. another --d435_pitch_deg) without the original download
    if not os.path.samefile(src, out):
        shutil.copy2(os.path.join(src, "g1_29dof_rev_1_0.urdf"), os.path.join(out, "g1_29dof_rev_1_0.urdf"))
        for f in sorted(set(re.findall(r'filename="([^"]+)"', text))):
            shutil.copy2(os.path.join(src, f), os.path.join(out, f))
    # head_link: add the stock visual before its collision block (only if it has none)
    m = re.search(r'<link name="head_link">(.*?)</link>', text, re.S)
    if m is None or "<visual>" in m.group(1):
        raise RuntimeError("expected a head_link without a visual in the supplied URDF")
    head = m.group(0).replace("    <collision>", HEAD_VISUAL + "    <collision>", 1)
    text = text.replace(m.group(0), head)
    if d435_pitch_deg is not None:
        j = re.search(r'(<joint name="d435_joint" type="fixed">\s*<origin xyz="[^"]*" rpy=")([^"]*)(")', text)
        if j is None:
            raise RuntimeError("d435_joint origin not found")
        r, _, y = (float(v) for v in j.group(2).split())
        text = text[:j.start(2)] + f"{r:g} {np.radians(d435_pitch_deg):.10f} {y:g}" + text[j.end(2):]
    open(os.path.join(out, "g1_v2_sim.urdf"), "w").write(text)


def d435_module(u: Urdf):
    m = u.mesh("d435_link", "d435_link")
    lo, hi = m.bounds
    left_imager = [hi[0] - 0.0042, (lo[1] + hi[1]) / 2 + 0.0175, (lo[2] + hi[2]) / 2]
    return {"link": "d435_link", "visual_bounds_in_link": [lo.tolist(), hi.tolist()],
            "module_origin_in_link": [round(v, 4) for v in left_imager],
            "joint_in_torso": {"pos": u.T("d435_link", "torso_link")[:3, 3].round(7).tolist(),
                               "rpy": R.from_matrix(u.T("d435_link", "torso_link")[:3, :3]).as_euler("xyz").round(10).tolist()}}


def d405_left(u: Urdf):
    m = u.mesh("left_d405_link", "left_wrist_yaw_link")
    mast = u.mesh("left_custom_tool_link", "left_wrist_yaw_link")
    obb = m.bounding_box_oriented
    T, ext = obb.primitive.transform, obb.primitive.extents
    k = int(np.argmin(ext))
    n, c = T[:3, k], T[:3, 3]
    # front = the side away from the bracket mast: the mast vertices at the camera's height (its other parts, e.g. the
    # base plate, lie on both sides)
    lo, hi = m.bounds
    near = mast.vertices[(mast.vertices[:, 2] > lo[2]) & (mast.vertices[:, 2] < hi[2])]
    if (near.mean(0) - c) @ n > 0:
        n = -n
    face = c + n * ext[k] / 2
    x_opt = np.array([0.0, -1.0, 0.0])                       # image right = the wrist's right (baseline along y)
    x_opt -= (x_opt @ n) * n
    x_opt /= np.linalg.norm(x_opt)
    y_opt = np.cross(n, x_opt)
    left_imager = face - x_opt * 0.009                       # 9 mm to the camera's left of the face centre
    Rm = np.c_[x_opt, y_opt, n]
    return {"parent_link": "left_wrist_yaw_link", "front_face_centre": face.round(4).tolist(), "normal": n.round(4).tolist(),
            "pos": left_imager.round(4).tolist(), "rot_wxyz_ros": np.roll(R.from_matrix(Rm).as_quat(), 1).round(6).tolist(),
            "pitch_down_deg": round(float(np.degrees(np.arcsin(-n[2]))), 2),
            "urdf_optical_frame_pos": u.T("left_d405_depth_optical_frame", "left_wrist_yaw_link")[:3, 3].round(4).tolist()}


def scanner_window(u: Urdf):
    m = u.mesh("left_scan_tool_link", "left_wrist_yaw_link")
    axis = u.T("left_scan_tool_link", "left_wrist_yaw_link")[:3, 2]        # tool z = scan direction
    fn, fc, A = m.face_normals, m.triangles_center, m.area_faces
    s = fn @ axis
    d = fc @ axis
    dmax = d.max()
    front = (s > np.cos(np.radians(30))) & (d > dmax - 0.025)   # front-facing triangles of the nose (last 25 mm)
    inner = front & (d < dmax - 0.003)                       # without the rim (the outermost front-facing ring)
    # the window: the largest front-facing area at one depth behind the rim (1 mm depth bins, +-2.5 mm around the peak);
    # the scanned mesh is bumpy, hence the 30 deg normal tolerance
    h, e = np.histogram(d[inner], bins=np.arange(d[inner].min(), dmax, 0.001), weights=A[inner])
    peak = (e[np.argmax(h)] + e[np.argmax(h) + 1]) / 2
    window = inner & (np.abs(d - peak) < 0.0025)
    c = (fc[window] * A[window, None]).sum(0) / A[window].sum()
    nw = (fn[window] * A[window, None]).sum(0)
    nw /= np.linalg.norm(nw)
    return {"parent_link": "left_wrist_yaw_link", "window_centre": c.round(4).tolist(), "window_normal": nw.round(4).tolist(),
            "tool_axis": axis.round(5).tolist(), "rim_depth_m": round(float(dmax - c @ axis), 4),
            "angle_window_normal_vs_axis_deg": round(float(np.degrees(np.arccos(np.clip(nw @ axis, -1, 1)))), 2)}


def box(m, link, name, sel=None):
    v = m.vertices if sel is None else m.vertices[sel(m.vertices)]
    lo, hi = v.min(0), v.max(0)
    return {"name": name, "link": link, "center": ((lo + hi) / 2).round(4).tolist(), "size": (hi - lo).round(4).tolist()}


def clearance_boxes(u: Urdf):
    L = "left_wrist_yaw_link"
    gun = u.mesh("left_scan_tool_link", L)
    br = trimesh.util.concatenate([u.mesh("left_custom_tool_link", L), u.mesh("left_scan_2_link", L)])
    d405 = u.mesh("left_d405_link", L)
    z_head = 0.07                                             # gun head / handle split height (side view)
    out = [box(br, L, "bracket"),
           box(gun, L, "scanner_head", lambda v: v[:, 2] >= z_head),
           box(gun, L, "scanner_handle", lambda v: v[:, 2] < z_head),
           box(d405, L, "left_d405")]
    Rr = "right_wrist_yaw_link"
    out += [box(u.mesh("right_d405_link", Rr), Rr, "right_d405"),
            box(u.mesh("right_dex1_d405_mount_link", Rr), Rr, "right_d405_mount")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d435_pitch_deg", type=float, default=None, help="d435_joint pitch (deg, + = optical axis down)")
    a = ap.parse_args()
    copy_description(a.src, a.out, a.d435_pitch_deg)
    u = Urdf(os.path.join(a.out, "g1_v2_sim.urdf"))
    res = {"d435": d435_module(u), "left_d405": d405_left(u), "scanner": scanner_window(u), "clearance_boxes": clearance_boxes(u)}
    json.dump(res, open(os.path.join(a.out, "derived_frames.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
