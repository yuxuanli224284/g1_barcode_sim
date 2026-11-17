"""Infrared images of the D435i stereo pair (camera_head.yaml model d435i).

The renderer has no infrared material model, so each infrared image is synthesised from what the renderer gives for
that infrared camera: the luminance of its visible-light render (the scene as the imager sees it in ambient light) plus
the projector's dot pattern on the surfaces. The pattern is a fixed random dot image over the projector's field of view;
a pixel receives the pattern value at the direction from the projector to the surface point it sees (the point from the
camera's own render depth), scaled by the surface brightness (luminance as reflectance) and 1 / distance^2. This is
what makes plain faces matchable, as on the real camera. Approximations: no projector shadows (the projector sits
between the imagers, 25 mm from each), no infrared-specific reflectance, a hand-chosen pattern (the real one is not
published). The render depth is used only here, as the physics of the light; the agent never receives it.
"""
from __future__ import annotations

import math

import numpy as np

from .camera import inv_T


class IRStereoSynth:
    def __init__(self, stereo_cfg: dict, K_ir: np.ndarray, T_link_left: np.ndarray, T_link_right: np.ndarray,
                 T_link_proj: np.ndarray | None, seed: int = 0):
        self.cfg = stereo_cfg
        self.K = np.asarray(K_ir, float)
        self.T_link = {"left": T_link_left, "right": T_link_right}
        P = stereo_cfg["projector"]
        self.pattern = None
        if P["enabled"]:
            if T_link_proj is None:
                raise ValueError("projector enabled but no projector pose")
            self.T_link_proj = T_link_proj
            self.tan_h = math.tan(math.radians(P["hfov_deg"]) / 2)
            self.tan_v = math.tan(math.radians(P["vfov_deg"]) / 2)
            self.pattern = dot_pattern(P, np.random.default_rng(P["seed"]))   # fixed: a property of the device
            self.gain = float(P["irradiance_at_1m"])
        H, W = stereo_cfg["height"], stereo_cfg["width"]
        u, v = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
        # ray directions in the optical frame (z = 1); pixel centres at +0.5 (the renderer's convention for K)
        self.rays = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], -1)
        self.rng = np.random.default_rng(seed + 7)

    def image(self, side: str, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """uint8 infrared image of one imager from its rgb render and render depth (distance to the image plane)."""
        lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float64) / 255.0
        img = self.cfg["ambient_gain"] * lum
        if self.pattern is not None:
            d = np.nan_to_num(np.asarray(depth, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            pts = self.rays * d[..., None]                                          # optical frame of this imager
            T_proj_cam = inv_T(self.T_link_proj) @ self.T_link[side]                # both optical frames on the link
            q = pts @ T_proj_cam[:3, :3].T + T_proj_cam[:3, 3]
            z = q[..., 2]
            ok = (d > 0) & (z > 1e-3)
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = q[..., 0] / z, q[..., 1] / z
            ok &= (np.abs(x) < self.tan_h) & (np.abs(y) < self.tan_v)
            Ph, Pw = self.pattern.shape
            px = (x / self.tan_h * 0.5 + 0.5) * (Pw - 1)
            py = (y / self.tan_v * 0.5 + 0.5) * (Ph - 1)
            val = np.zeros_like(img)
            val[ok] = _bilinear(self.pattern, px[ok], py[ok])
            r2 = np.sum(q * q, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                irr = np.where(ok, self.gain * val / np.maximum(r2, 1e-6), 0.0)
            img = img + lum * irr
        img = img + self.rng.normal(0.0, self.cfg["noise_sigma"], img.shape)
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def dot_pattern(P: dict, rng) -> np.ndarray:
    """Random dot image over the projector field of view, peak 1 per isolated dot."""
    import cv2
    W = int(round(P["hfov_deg"] / P["texel_deg"]))
    H = int(round(P["vfov_deg"] / P["texel_deg"]))
    n = int(round(P["dot_density"] * W * H))
    img = np.zeros((H, W), np.float32)
    xs, ys = rng.integers(0, W, n), rng.integers(0, H, n)
    np.add.at(img, (ys, xs), 1.0)
    s = float(P["dot_sigma_texel"])
    k = int(2 * math.ceil(3 * s) + 1)
    img = cv2.GaussianBlur(img, (k, k), s) * (2 * math.pi * s * s)                # peak 1 for a single dot
    return np.clip(img, 0.0, 1.0)


def _bilinear(img, x, y):
    H, W = img.shape
    x0 = np.clip(np.floor(x).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(y).astype(int), 0, H - 2)
    fx, fy = x - x0, y - y0
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy
