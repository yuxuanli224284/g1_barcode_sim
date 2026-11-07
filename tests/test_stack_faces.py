"""CPU checks of the four-face stack scenarios (no Isaac Sim): per-face labels in to_layout, inner-face / fit rejection,
exposed-only face unions and the GT "faces" list of the stack4f_L0 suite.
run: python -m pytest tests/test_stack_faces.py"""
import copy
import glob
import json
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1_barcode_sim.barcode import FACE_AXES  # noqa: E402
from g1_barcode_sim.config import PKG_ROOT, load_barcode_cfg  # noqa: E402
from g1_barcode_sim.testsuite import SUITES_DIR, load_factors, load_scenario  # noqa: E402
from g1_barcode_sim.testsuite import geometry as TG  # noqa: E402
from g1_barcode_sim.testsuite.scenario import LABEL_PRIM, to_layout  # noqa: E402

F = load_factors()
DEPTH = F["barcode"]["face_offset"] + load_barcode_cfg()["label_thickness"]   # as env.get_ground_truth
S = F["task2_start"]


@pytest.fixture(scope="module")
def stack4f():
    return load_scenario("stack4f_L0_0000")


def _box(i, x, y, z, d=0.4, w=0.4, h=0.38):
    return {"id": i, "pos": [x, y, z], "size": [d, w, h], "yaw": 0.0}


def test_all_existing_suites_still_convert():
    paths = sorted(glob.glob(os.path.join(SUITES_DIR, "*", "scenarios", "*.json")))
    assert paths
    for p in paths:
        to_layout(json.load(open(p)), F)


def test_legacy_label_defaults_to_front_face():
    lay = to_layout(load_scenario("stack9_L0_0000"), F)
    assert all(bc["face"] == "-x" and bc["normal"] == [-1.0, 0.0, 0.0] for bc in lay["barcodes"])


def test_labels_land_on_their_world_face(stack4f):
    lay = to_layout(stack4f, F)
    boxes = {b["id"]: b for b in lay["boxes"]}
    off = F["barcode"]["face_offset"]
    assert len(lay["barcodes"]) == 12
    for bc in lay["barcodes"]:
        b = boxes[bc["box_id"]]
        n, r, u = (np.array(a, float) for a in FACE_AXES[bc["face"]])
        np.testing.assert_allclose(bc["normal"], n)
        np.testing.assert_allclose(bc["right"], r)                 # rot 0: right = the face's u axis
        c = np.asarray(bc["center"])
        # on the face plane + face_offset along the outward normal, at (u, v) from the face centre
        fc, fn, _ = TG.box_face(b, bc["face"])
        np.testing.assert_allclose(fn, n)
        np.testing.assert_allclose(c, fc + off * n + bc["uv"][0] * r + bc["uv"][1] * u, atol=1e-12)
        # outward: the label lies outside the stack footprint x 1.30-2.10, y -0.40..0.40 on its own side
        if bc["face"] == "+x":
            assert c[0] > 2.10
        if bc["face"] == "-x":
            assert c[0] < 1.30
        if bc["face"] == "-y":
            assert c[1] < -0.40
        if bc["face"] == "+y":
            assert c[1] > 0.40


def test_label_prim_names_unique_per_face():
    assert set(LABEL_PRIM) == set(TG.TOUR_FACES)
    assert len(set(LABEL_PRIM.values())) == 4


def _with_label(sc, **kw):
    sc = copy.deepcopy(sc)
    sc["barcodes"] = [dict(sc["barcodes"][0], **kw)]
    return sc


@pytest.mark.parametrize("box_id,face", [("LF0", "+x"),   # pressed against the back row
                                         ("LF0", "-y"),   # pressed against the right column
                                         ("LB1", "-x"),   # back row behind the front row
                                         ("RB2", "+y")])  # right column against the left column
def test_inner_face_label_rejected(stack4f, box_id, face):
    with pytest.raises(ValueError, match="covered"):
        to_layout(_with_label(stack4f, box_id=box_id, face=face, u=0.0, v=0.0), F)


def test_label_must_fit_face(stack4f):
    # half extent 0.05 (0.08 + 2 * 0.01 border) + |u| 0.15 > 0.20 - 0.01 margin
    with pytest.raises(ValueError, match="does not fit"):
        to_layout(_with_label(stack4f, box_id="LF0", face="-x", u=0.15, v=0.0), F)
    with pytest.raises(ValueError, match="does not fit"):
        to_layout(_with_label(stack4f, box_id="RB0", face="-y", u=0.0, v=0.17), F)
    # 90 deg: the 0.10 m long side runs along v (half 0.05): v 0.135 + 0.05 > 0.19 - 0.01; v 0.13 just fits
    with pytest.raises(ValueError, match="does not fit"):
        to_layout(_with_label(stack4f, box_id="RB0", face="-y", u=0.0, v=0.135, rot_deg=90.0), F)
    to_layout(_with_label(stack4f, box_id="RB0", face="-y", u=0.0, v=0.13, rot_deg=90.0), F)


def test_unknown_face_rejected(stack4f):
    with pytest.raises(ValueError, match="not one of"):
        to_layout(_with_label(stack4f, face="+z"), F)


def test_box_face_matches_old_front_face():
    b = _box("A", 1.5, 0.2, 0.19)
    c, n, corners = TG.front_face(b)
    np.testing.assert_allclose(c, [1.3, 0.2, 0.19])
    np.testing.assert_allclose(n, [-1, 0, 0])
    np.testing.assert_allclose(corners, [[1.3, y, z] for y in (0.0, 0.4) for z in (0.0, 0.38)])
    c, n, corners = TG.box_face(b, "+y")
    np.testing.assert_allclose(c, [1.5, 0.4, 0.19])
    np.testing.assert_allclose(n, [0, 1, 0])
    np.testing.assert_allclose(corners, [[x, 0.4, z] for x in (1.3, 1.7) for z in (0.0, 0.38)])


def test_face_union_exposed_only(stack4f):
    lay = to_layout(stack4f, F)
    expect = {"-x": {"LF", "RF"}, "-y": {"RF", "RB"}, "+x": {"LB", "RB"}, "+y": {"LF", "LB"}}
    plane = {"-x": 1.30, "-y": -0.40, "+x": 2.10, "+y": 0.40}
    for face, cols in expect.items():
        vis = TG.exposed_faces(lay["boxes"], face, DEPTH)
        assert {b["id"][:2] for b in vis} == cols and len(vis) == 6
        c, n, corners = TG.face_union(lay["boxes"], face, DEPTH)
        k = int(np.argmax(np.abs(n)))
        np.testing.assert_allclose(corners[:, k], plane[face])        # no hidden inner faces in the union
        np.testing.assert_allclose(c[k], plane[face])
        np.testing.assert_allclose(c[2], 0.57)
    c, _, corners = TG.front_face_union(lay["boxes"], DEPTH)
    np.testing.assert_allclose(c, [1.30, 0.0, 0.57])
    assert len(corners) == 24


def test_partially_covered_face_stays_exposed():
    # a taller back box behind a short front box: its upper front face is visible above the front box
    front, back = _box("F", 1.5, 0.0, 0.1, h=0.2), _box("B", 1.9, 0.0, 0.3, h=0.6)
    assert [b["id"] for b in TG.exposed_faces([front, back], "-x", DEPTH)] == ["F", "B"]
    c, _, _ = TG.front_face_union([front, back], DEPTH)
    np.testing.assert_allclose(c, [1.3, 0.0, 0.3])   # outermost plane, vertically centred on all exposed corners
    # a gap wider than the label slab leaves the back face exposed; a touching box hides it
    gap = _box("B", 1.7 + 0.2 + 2 * DEPTH, 0.0, 0.1, h=0.2)
    assert len(TG.exposed_faces([front, gap], "-x", DEPTH)) == 2
    touch = _box("B", 1.9, 0.0, 0.1, h=0.2)
    assert [b["id"] for b in TG.exposed_faces([front, touch], "-x", DEPTH)] == ["F"]


def test_gt_faces_stack4f(stack4f):
    lay = to_layout(stack4f, F)
    cam = TG.head_camera(yaml.safe_load(open(os.path.join(PKG_ROOT, F["head_camera_nominal"]))))
    faces = TG.stack_faces(lay["boxes"], lay["barcodes"], cam, S["d_obs_range"], S["d_obs_step"], S["fov_margin_frac"], DEPTH)
    assert [f["face"] for f in faces] == ["-x", "-y", "+x", "+y"]
    by_face = {}
    for bc in stack4f["barcodes"]:
        by_face.setdefault(bc["face"], []).append(bc["content"])
    for f in faces:
        assert f["contents"] == by_face[f["face"]] and len(f["contents"]) == 3
        assert len(f["corners"]) == 24
        ob = f["gt_obs_pose"]
        assert ob["fits"]
        n = np.asarray(f["normal"])
        # stands d_obs out along the outward normal from the face centre, facing the face (yaw along -n)
        np.testing.assert_allclose(ob["pos"][:2], (np.asarray(f["center"]) + ob["d_obs"] * n)[:2], atol=1e-12)
        np.testing.assert_allclose([np.cos(ob["yaw"]), np.sin(ob["yaw"])], -n[:2], atol=1e-12)
    # the front side is the task-1 reference (target_face / task1_gt_obs_pose of env.get_ground_truth)
    c, n, corners = TG.front_face_union(lay["boxes"], DEPTH)
    assert faces[0]["center"] == c.tolist() and faces[0]["corners"] == corners.tolist()


def test_stack4f_label_heights(stack4f):
    lay = to_layout(stack4f, F)
    for face in TG.TOUR_FACES:
        zs = sorted(bc["center"][2] for bc in lay["barcodes"] if bc["face"] == face)
        assert abs(zs[0] - 0.33) < 1e-9 and abs(zs[1] - 0.57) < 1e-9 and 1.07 - 1e-9 <= zs[2] <= 1.10 + 1e-9
        cols = {bc["box_id"][:2] for bc in lay["barcodes"] if bc["face"] == face}
        assert len(cols) == 2                                     # both columns of the face carry a label


def test_several_labels_on_one_face(stack4f):
    """A box face may carry several labels (stack4c_L0_0002: 5 per box face) as long as their extents (0.10 x 0.05 m
    incl. backing) do not overlap."""
    sc = copy.deepcopy(stack4f)
    first = dict(sc["barcodes"][0], box_id="LF1", face="-x", v=0.0)
    ok = [dict(first, u=u, content=f"OK{i}AAAAAA", content_id=f"ok{i}") for i, u in enumerate((-0.13, 0.0, 0.13))]
    sc["barcodes"] = ok
    lay = to_layout(sc, F)
    assert len(lay["barcodes"]) == 3
    sc["barcodes"] = ok + [dict(first, u=0.06, v=0.03, content="BADAAAAAA", content_id="bad")]   # 0.06 < 0.10 apart in u, 0.03 < 0.05 in v
    with pytest.raises(ValueError, match="overlaps another label"):
        to_layout(sc, F)


def test_stack4c_suite():
    """Shipping-box (cardbox) four-side stacks: 12 codes on 4 sides / 6 codes on front + back / 60 codes on front + back."""
    n = {}
    for sid in ("stack4c_L0_0000", "stack4c_L0_0001", "stack4c_L0_0002"):
        sc = load_scenario(sid)
        assert {b["appearance"] for b in sc["boxes"]} == {"cardbox"} and sc["scan_faces"] == "all"
        lay = to_layout(sc, F)
        n[sid] = {f: sum(b["face"] == f for b in lay["barcodes"]) for f in ("-x", "-y", "+x", "+y")}
        assert min(b["center"][2] for b in lay["barcodes"]) >= 0.33 - 1e-9     # every label in the reachable band
    assert n == {"stack4c_L0_0000": {"-x": 3, "-y": 3, "+x": 3, "+y": 3}, "stack4c_L0_0001": {"-x": 3, "-y": 0, "+x": 3, "+y": 0},
                 "stack4c_L0_0002": {"-x": 30, "-y": 0, "+x": 30, "+y": 0}}
