"""Robot-vs-scene clearance.

Robot side: each link's collision geometry (`<link>/collisions`): meshes are convex-hull colliders,
kept as their hull vertices; spheres (the feet) are kept exactly as centre + radius. Stored in the
link frame and moved with the live link poses. Scene side: every furniture part and box as an oriented box (exact, from
the layout). Clearance = min signed distance from a robot vertex to an obstacle box (< 0 means the
vertex / sphere is inside the obstacle).

Limitation: only vertices are tested, so an obstacle corner poking into the middle of a hull face
is not seen; reported clearances are therefore an upper bound on the true hull-to-box distance.
"""
from __future__ import annotations

import numpy as np

from .camera import inv_T, make_T, quat_wxyz_to_R

CYL_RIM_SAMPLES = 32   # max radial under-estimate r*(1-cos(pi/32)) = 0.5% of r


class RobotCollisionModel:
    def __init__(self, robot, layout, robot_prim="/World/envs/env_0/Robot"):
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        from scipy.spatial import ConvexHull

        self.robot = robot
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        names = list(robot.data.body_names)
        pts, rad, owner = [], [], []
        for bi, name in enumerate(names):
            body = stage.GetPrimAtPath(f"{robot_prim}/{name}")
            if not body.IsValid():
                raise RuntimeError(f"robot body prim {robot_prim}/{name} not found")
            T_w_body = np.array(cache.GetLocalToWorldTransform(body)).T
            # collision geometry: gprims below a prim carrying CollisionAPI (here `<link>/collisions`)
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not p.IsA(UsdGeom.Gprim) or not self._belongs_to(p, body, UsdPhysics):
                    continue
                if not any(q.HasAPI(UsdPhysics.CollisionAPI) for q in self._ancestors(p, body)):
                    continue
                T_body_g = inv_T(T_w_body) @ np.array(cache.GetLocalToWorldTransform(p)).T
                if p.IsA(UsdGeom.Mesh):
                    v = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float)
                    if len(v) >= 4:
                        v = v[ConvexHull(v).vertices]
                    pts.append((T_body_g[:3, :3] @ v.T).T + T_body_g[:3, 3])
                    rad.append(np.zeros(len(v)))
                elif p.IsA(UsdGeom.Sphere):
                    scale = np.linalg.norm(T_body_g[:3, :3], axis=0)
                    pts.append(T_body_g[:3, 3][None])
                    rad.append(np.array([UsdGeom.Sphere(p).GetRadiusAttr().Get() * float(scale.max())]))
                elif p.IsA(UsdGeom.Cube):
                    h = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
                    v = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
                    pts.append((T_body_g[:3, :3] @ v.T).T + T_body_g[:3, 3])
                    rad.append(np.zeros(len(v)))
                elif p.IsA(UsdGeom.Cylinder) or p.IsA(UsdGeom.Capsule):
                    g = UsdGeom.Cylinder(p) if p.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(p)
                    r, hh = g.GetRadiusAttr().Get(), g.GetHeightAttr().Get() / 2
                    ax = "XYZ".index(g.GetAxisAttr().Get())
                    e = [i for i in range(3) if i != ax]
                    if p.IsA(UsdGeom.Cylinder):
                        # hull of a cylinder = its two rim circles; CYL_RIM_SAMPLES points per rim
                        a = np.linspace(0, 2 * np.pi, CYL_RIM_SAMPLES, endpoint=False)
                        v = np.zeros((2 * len(a), 3))
                        for k, z in enumerate((-hh, hh)):
                            v[k * len(a):(k + 1) * len(a), e[0]] = r * np.cos(a)
                            v[k * len(a):(k + 1) * len(a), e[1]] = r * np.sin(a)
                            v[k * len(a):(k + 1) * len(a), ax] = z
                        rr = np.zeros(len(v))
                    else:
                        # capsule = swept sphere: spheres along the axis, spacing <= r / 4
                        n = max(2, int(np.ceil(2 * hh / (r / 4))) + 1)
                        v = np.zeros((n, 3)); v[:, ax] = np.linspace(-hh, hh, n)
                        rr = np.full(n, r)
                    scale = np.linalg.norm(T_body_g[:3, :3], axis=0)
                    pts.append((T_body_g[:3, :3] @ v.T).T + T_body_g[:3, 3])
                    rad.append(rr * float(scale.max()))
                else:
                    raise NotImplementedError(f"collision gprim type {p.GetTypeName()} at {p.GetPath()} not handled")
                owner.append(np.full(len(pts[-1]), bi))
        if not pts:
            raise RuntimeError("no robot collision geometry found")
        self.local_rad = np.concatenate(rad)
        self.local_pts = np.concatenate(pts)
        self.owner = np.concatenate(owner)
        self.body_names = names
        self.set_layout(layout)

    def add_body_points(self, body_name: str, pts_body) -> None:
        """Attach extra points (body frame, m) to a link, e.g. the corners of the visual-only scanner."""
        pts_body = np.asarray(pts_body, dtype=float).reshape(-1, 3)
        bi = self.body_names.index(body_name)
        self.local_pts = np.concatenate([self.local_pts, pts_body])
        self.local_rad = np.concatenate([self.local_rad, np.zeros(len(pts_body))])
        self.owner = np.concatenate([self.owner, np.full(len(pts_body), bi)])

    @staticmethod
    def _ancestors(prim, body):
        q = prim
        while q and q.IsValid() and q != body:
            yield q
            q = q.GetParent()

    @staticmethod
    def _belongs_to(prim, body, UsdPhysics):
        q = prim.GetParent()
        while q and q.IsValid() and q != body:
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                return False
            q = q.GetParent()
        return True

    def set_layout(self, layout):
        obs = []
        for f in layout["furniture"]:
            for part in f["parts"]:
                obs.append((f"{f['id']}/{part['name']}", np.array(part["center"]), np.eye(3), np.array(part["size"]) / 2))
        for b in layout["boxes"]:
            c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
            obs.append((b["id"], np.array(b["pos"]), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), np.array(b["size"]) / 2))
        self.obstacles = obs

    def world_points(self):
        P = self.robot.data.body_link_pose_w[0].cpu().numpy()   # (B, 7) pos + quat wxyz
        R = np.stack([quat_wxyz_to_R(q) for q in P[:, 3:]])
        Ro = R[self.owner]
        return np.einsum("nij,nj->ni", Ro, self.local_pts) + P[self.owner, :3]

    def min_gap(self, points=None):
        pts = self.world_points() if points is None else points
        best = {"gap": float("inf"), "body": None, "obstacle": None, "point": None}
        for name, c, R, h in self.obstacles:
            q = (pts - c) @ R
            d = np.abs(q) - h
            outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            inside = np.minimum(d.max(axis=1), 0.0)
            sd = outside + inside - self.local_rad
            i = int(np.argmin(sd))
            if sd[i] < best["gap"]:
                best = {"gap": float(sd[i]), "body": self.body_names[self.owner[i]], "obstacle": name, "point": pts[i].tolist()}
        return best
