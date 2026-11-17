"""Export the EngineAI T800 Pro walk policy (.mnn) to a numpy .npz and check it numerically against MNN.

Why: the Isaac Sim conda env (unitree_sim_env, shared with another session) has no MNN package and must not be
modified. The network is a plain MLP, so its weights are read from the MNN JSON dump and evaluated with numpy
(walk_policy.NumpyMLPPolicy). Layer structure is taken from the dump, not assumed:
  input (1, 1083) -> [Convolution 1x1 (= Gemm) -> ELU(alpha)] x 3 -> Convolution 1x1 -> output (1, 22)

Run in the MNN venv (CPU only):
  /data/yicheng/HumanoidBarcodeScan/venvs/t800/bin/python -I t800/export_policy_npz.py
Writes <out_dir>/<policy stem>.npz (local only, BSD-3 vendor weights) and prints the max |numpy - MNN| over
N_CHECK random inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_policy import CONFIG_ROOT, DEFAULT_WALK_YAML, MNNPolicy, NumpyMLPPolicy  # noqa: E402

OUT_DIR = "/data/yicheng/HumanoidBarcodeScan/third_party/engineai/policy_npz"
# Random-input check: inputs drawn ~ N(0, INPUT_STD) clipped to the observation clip; 1000 samples is enough to
# exercise every unit of every layer (768 is the widest).
N_CHECK = 1000
INPUT_STD = 1.0


def mnn_to_layers(json_path: str) -> list[dict]:
    d = json.load(open(json_path))
    ops = d["oplists"]
    layers = []
    for op in ops:
        if op["type"] == "Convolution":
            m = op["main"]
            c = m["common"]
            if (c["kernelX"], c["kernelY"], c["group"]) != (1, 1, 1) or c.get("relu") or c.get("relu6"):
                raise ValueError(f"unexpected convolution {op['name']}: {c}")
            W = np.asarray(m["weight"], dtype=np.float32).reshape(c["outputCount"], c["inputCount"])
            b = np.asarray(m["bias"], dtype=np.float32)
            layers.append({"W": W, "b": b, "act": None, "name": op["name"]})
        elif op["type"] == "ELU":
            if not layers or layers[-1]["act"] is not None:
                raise ValueError(f"ELU {op['name']} does not follow a linear layer")
            layers[-1]["act"] = ("elu", float(op["main"]["alpha"]))
        elif op["type"] in ("Input", "Const", "Reshape", "ConvertTensor", "Shape", "Rank", "BinaryOp", "Unsqueeze",
                            "SliceTf", "Concat"):
            continue   # shape plumbing of the ONNX->MNN conversion (Gemm as 1x1 conv); no arithmetic on the data
        else:
            raise ValueError(f"op type {op['type']} ({op['name']}) not supported by the numpy MLP")
    return layers


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(DEFAULT_WALK_YAML))
    mnn_path = os.path.join(CONFIG_ROOT, cfg["policy_file"])
    stem = os.path.splitext(os.path.basename(mnn_path))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    js = os.path.join(args.out_dir, stem + ".json")
    mnnconvert = os.path.join(os.path.dirname(sys.executable), "mnnconvert")
    subprocess.run([mnnconvert, "-f", "MNN", "--modelFile", mnn_path, "--JsonFile", js], check=True,
                   stdout=subprocess.DEVNULL)
    layers = mnn_to_layers(js)
    out = {}
    for i, L in enumerate(layers):
        out[f"W{i}"], out[f"b{i}"] = L["W"], L["b"]
        out[f"act{i}"] = np.array(L["act"][0] if L["act"] else "none")
        out[f"alpha{i}"] = np.float32(L["act"][1] if L["act"] else 0.0)
    out["n_layers"] = np.int64(len(layers))
    out["source_mnn"] = np.array(mnn_path)
    npz = os.path.join(args.out_dir, stem + ".npz")
    np.savez(npz, **out)
    os.remove(js)
    print("layers:", [(L["name"], L["W"].shape, L["act"]) for L in layers])

    mnn, npm = MNNPolicy(mnn_path), NumpyMLPPolicy(npz)
    rng = np.random.default_rng(0)
    clip = float(cfg["observation_clip"])
    err = 0.0
    for _ in range(N_CHECK):
        x = np.clip(rng.normal(0.0, INPUT_STD, mnn.in_shape[-1]), -clip, clip).astype(np.float32)
        err = max(err, float(np.abs(mnn(x) - npm(x)).max()))
    print(f"WROTE {npz}  max|numpy-MNN| over {N_CHECK} random inputs = {err:.3e}")


if __name__ == "__main__":
    main()
