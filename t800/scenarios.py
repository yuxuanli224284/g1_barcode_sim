"""Walk-policy test scenarios and their metrics, shared by every simulator so the sim2sim comparison runs the SAME
procedure on each side (pure numpy, no simulator import):
  t800/mujoco_check.py   MuJoCo, runner arm modes (policy / ext_real / ext_spoof), MNN policy
  t800/mujoco_mirror.py  MuJoCo, the Isaac T800Walker control law (robot_t800.yaml t800_walk switches), npz policy
  t800/isaac_sim2sim.py  Isaac Lab, T800Walker itself, npz policy
  t800/sim2sim_compare.py side-by-side table of the last two

Stated test choices (not from the SDK; the runner has no fall detector):
  FALL_HEIGHT_FRAC / FALL_TILT_RAD  fall = base height below half its initial standing height, or tilt > 1 rad (57 deg);
                                    both far outside any normal gait (min height / max tilt are also reported raw).
  TRACK_SETTLE_S  velocity tracking is averaged from 3 s after the command starts: covers the start-up transient
                  (MuJoCo, LPF off, vx 0.4 from standing: ~3 s to the steady speed, t800_review mj_long) and > 4.5 time
                  constants of the runner's command LPF (~0.64 s, walk.yaml:81-82) when it is enabled.
  Scenario durations / commands: as first chosen for mujoco_check (10 s each; probes bracket the start thresholds);
  EXTRA: turn_wz0.6 (the Isaac smoke turns at 0.6 rad/s) and wz -0.6..-1.0 probes (see EXTRA).
"""
from __future__ import annotations

import numpy as np

FALL_HEIGHT_FRAC = 0.5
FALL_TILT_RAD = 1.0
TRACK_SETTLE_S = 3.0


def const_cmd(vx, vy, wz):
    return lambda t: (vx, vy, wz)


def step_cmd(cmd, t_on):
    return lambda t: cmd if t < t_on else (0.0, 0.0, 0.0)


def seq_cmd(a, b, t1):
    return lambda t: a if t < t1 else b


SCENARIOS = {
    # name: (duration_s, command(t), command-active window for tracking metric or None)
    "stand": (10.0, const_cmd(0.0, 0.0, 0.0), None),
    "walk_vx0.5": (10.0, const_cmd(0.5, 0.0, 0.0), (TRACK_SETTLE_S, 10.0)),
    "walk_vy0.3": (10.0, const_cmd(0.0, 0.3, 0.0), (TRACK_SETTLE_S, 10.0)),
    "turn_wz0.5": (10.0, const_cmd(0.0, 0.0, 0.5), (TRACK_SETTLE_S, 10.0)),
    "stop_and_stand": (10.0, step_cmd((0.5, 0.0, 0.0), 5.0), (TRACK_SETTLE_S, 5.0)),
    "short_step_vx0.2_1s": (5.0, step_cmd((0.2, 0.0, 0.0), 1.0), None),
}
# Probes of the policy's start-from-standing threshold and hysteresis.
PROBES = {f"probe_vx{v}": (10.0, const_cmd(v, 0.0, 0.0), (TRACK_SETTLE_S, 10.0))
          for v in (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.8, 1.2, -0.3, -0.6)}
PROBES.update({f"probe_vy{v}": (10.0, const_cmd(0.0, v, 0.0), (TRACK_SETTLE_S, 10.0))
               for v in (0.2, 0.3, 0.35, 0.4, -0.2, -0.3, -0.4)})
PROBES.update({f"probe_wz{v}": (10.0, const_cmd(0.0, 0.0, v), (TRACK_SETTLE_S, 10.0))
               for v in (0.3, 0.4, 1.0, -0.5)})
PROBES.update({
    "probe_hyst_vx0.5_to_0.2": (12.0, seq_cmd((0.5, 0.0, 0.0), (0.2, 0.0, 0.0), 4.0), (8.0, 12.0)),
    "probe_hyst_vx0.5_to_0.1": (12.0, seq_cmd((0.5, 0.0, 0.0), (0.1, 0.0, 0.0), 4.0), (8.0, 12.0)),
    "short_step_vx0.5_0.5s": (6.0, step_cmd((0.5, 0.0, 0.0), 0.5), None),
    "short_step_vx0.5_1s": (6.0, step_cmd((0.5, 0.0, 0.0), 1.0), None),
    "short_step_vx0.5_2s": (6.0, step_cmd((0.5, 0.0, 0.0), 2.0), None),
    "short_step_vx1.0_1s": (6.0, step_cmd((1.0, 0.0, 0.0), 1.0), None),
})
MAIN_SCENARIOS = list(SCENARIOS)
SCENARIOS.update(PROBES)
# sim2sim only (not part of mujoco_check's main list, so its earlier outputs keep their meaning)
EXTRA = {"turn_wz0.6": (10.0, const_cmd(0.0, 0.0, 0.6), (TRACK_SETTLE_S, 10.0))}
# negative-turn threshold: the first Isaac sim2sim run did not turn at wz -0.5 while MuJoCo did (2026-11-10)
EXTRA.update({f"probe_wz{v}": (10.0, const_cmd(0.0, 0.0, v), (TRACK_SETTLE_S, 10.0)) for v in (-0.6, -0.7, -0.8, -1.0)})
SIM2SIM_SCENARIOS = MAIN_SCENARIOS + list(PROBES) + list(EXTRA)
ALL = dict(SCENARIOS, **EXTRA)


def start_yaw(scenario: str) -> float:
    """World yaw at which a scenario starts so that its commanded translation heads along world -x (lets a simulator
    with obstacles start the robot in open floor). The metrics are expressed in the start heading frame, so they do not
    depend on it."""
    _, cmd_fn, _ = ALL[scenario]
    vx, vy, _ = cmd_fn(0.0)
    if vx == 0.0 and vy == 0.0:
        return np.pi
    return float(np.pi - np.arctan2(vy, vx))


def metrics(scenario: str, L: dict, p0, yaw0: float, h0: float, dt: float, fell_at) -> dict:
    """Metrics of one run. L: arrays logged after every control step: t (= (k+1) dt), pos (world xyz of the base link),
    yaw, v_head ([vx, vy] world velocity rotated into the current heading), wz (world yaw rate), cmd_raw, cmd_filt
    (command after the runner LPF), tilt (rad), h (base z); optional head_err (rad, held head pitch joint - target)."""
    dur, _, track_win = ALL[scenario]
    tt = L["t"]
    res = {"scenario": scenario, "duration": dur, "fell": fell_at is not None, "fell_at_s": fell_at, "h0": float(h0),
           "min_height": float(L["h"].min()), "max_tilt_deg": float(np.degrees(L["tilt"].max())),
           "mean_tilt_deg": float(np.degrees(L["tilt"].mean()))}
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    dpw = L["pos"][-1] - p0
    dp = np.array([c0 * dpw[0] + s0 * dpw[1], -s0 * dpw[0] + c0 * dpw[1]])        # start heading frame
    yaw_total = float(np.unwrap(np.concatenate([[yaw0], L["yaw"]]))[-1] - yaw0)
    res["final_disp_xy"] = [float(dp[0]), float(dp[1])]
    res["final_yaw_change_rad"] = yaw_total
    if track_win is not None and fell_at is None:
        sel = (tt >= track_win[0]) & (tt < track_win[1])
        tgt = L["cmd_raw"][sel]
        meas = np.column_stack([L["v_head"][sel], L["wz"][sel]])
        err = meas - tgt
        res["track_window_s"] = list(track_win)
        res["cmd_target"] = tgt[0].tolist()
        res["mean_meas_vx_vy_wz"] = meas.mean(0).tolist()
        res["mean_err_vx_vy_wz"] = err.mean(0).tolist()          # bias
        res["mean_abs_err_vx_vy_wz"] = np.abs(err).mean(0).tolist()  # includes gait oscillation
    if scenario == "stand" and fell_at is None:
        res["stand_drift_xy_m"] = float(np.linalg.norm(dp[:2]))
        res["stand_mean_speed_last5s"] = float(np.linalg.norm(L["v_head"][tt >= 5.0], axis=1).mean())
    if scenario == "stop_and_stand" and fell_at is None:
        i5 = np.searchsorted(tt, 5.0)
        res["dist_after_stop_m"] = float(np.linalg.norm(L["pos"][-1][:2] - L["pos"][i5][:2]))
        res["mean_speed_last2s"] = float(np.linalg.norm(L["v_head"][tt >= 8.0], axis=1).mean())
    if scenario.startswith("short_step") and fell_at is None:
        res["step_disp_forward_m"] = float(dp[0])
        res["step_disp_lateral_m"] = float(dp[1])
        res["step_cmd_integral_m"] = float(np.sum(L["cmd_raw"][:, 0]) * dt)
        res["step_filtered_cmd_integral_m"] = float(np.sum(L["cmd_filt"][:, 0]) * dt)
        res["mean_speed_last1s"] = float(np.linalg.norm(L["v_head"][tt >= dur - 1.0], axis=1).mean())
    if "head_err" in L:
        he = np.degrees(L["head_err"])
        res["head_pitch_err_deg_mean_max"] = [float(he.mean()), float(np.abs(he).max())]
    return res
