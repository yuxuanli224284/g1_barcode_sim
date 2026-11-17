"""CPU checks of the wall6x5_L0 scenario (no Isaac Sim): 6 x 5 gap-free wall of 0.25 (h) x 0.20 (w) x 0.40 (d) m plain
boxes, one label per box in the top-left corner of the robot-facing face, robot facing the wall face squarely, centred
on it, root 0.60 m from the face.
run: python -m pytest tests/test_wall6x5.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1_barcode_sim.testsuite import load_factors, load_scenario  # noqa: E402
from g1_barcode_sim.testsuite.scenario import to_layout  # noqa: E402

F = load_factors()


def _layout():
    return to_layout(load_scenario("wall6x5_L0_0000"), F)


def test_wall_geometry():
    lay = _layout()
    boxes = lay["boxes"]
    assert len(boxes) == 30 and all(b["appearance"] == "kraft" for b in boxes)
    for b in boxes:
        np.testing.assert_allclose(b["size"], [0.40, 0.20, 0.25])          # [d (x), w (y), h (z)]
    P = np.array([b["pos"] for b in boxes])
    assert sorted({round(y, 6) for y in P[:, 1]}) == [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    assert sorted({round(z, 6) for z in P[:, 2]}) == [0.125, 0.375, 0.625, 0.875, 1.125]   # on the floor, no gaps
    fx = F["world"]["table_front_x"]
    np.testing.assert_allclose(P[:, 0] - 0.20, fx)                        # front faces flush in one plane


def test_labels_top_left_of_front_face():
    lay = _layout()
    boxes = {b["id"]: b for b in lay["boxes"]}
    assert len(lay["barcodes"]) == 30 and len({bc["content"] for bc in lay["barcodes"]}) == 30
    margin = F["barcode"]["margin"]
    for bc in lay["barcodes"]:
        b = boxes[bc["box_id"]]
        assert bc["face"] == "-x" and bc["normal"] == [-1.0, 0.0, 0.0]
        w, h = bc["size"]
        c = np.asarray(bc["center"])
        # left edge as seen facing the face = the +y edge; top = +z
        left_edge, top_edge = b["pos"][1] + 0.10, b["pos"][2] + 0.125
        assert math.isclose(left_edge - (c[1] + w / 2), margin, abs_tol=1e-9)
        assert math.isclose(top_edge - (c[2] + h / 2), margin, abs_tol=1e-9)


def test_robot_faces_wall_centred():
    lay = _layout()
    ri = lay["robot_init"]
    yaw = ri["yaw"]
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    n = np.array([-1.0, 0.0])                                             # wall face normal (towards the robot)
    assert fwd @ -n > 1 - 1e-12                                           # forward axis straight at the face
    fx = F["world"]["table_front_x"]
    assert math.isclose(fx - ri["pos"][0], 0.60)                          # root-to-face distance (script default)
    ys = [b["pos"][1] for b in lay["boxes"]]
    assert math.isclose(ri["pos"][1], (min(ys) + max(ys)) / 2, abs_tol=1e-9)   # centred on the wall
    sc = lay["scenario"]
    assert sc["factor"]["params"]["facing"] == "wall" and sc["robot_init"]["yaw_deg"] == 0.0
