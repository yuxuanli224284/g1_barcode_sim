"""Generate every barcode label PNG used by the scenes and verify each decodes (milestone M2).

python scripts/generate_barcodes.py [--type code128|qr] [--scenes s1_table s2_shelf s3_stack]
Writes assets/barcodes/<type>/<content>.png and outputs/barcodes_check.json.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from g1_barcode_sim.config import load_barcode_cfg, load_scene_cfg  # noqa: E402
from g1_barcode_sim.layout import expand_scene  # noqa: E402
from g1_barcode_sim.barcode import make_label, decode_image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--type", default=None)
ap.add_argument("--scenes", nargs="*", default=["s1_table", "s2_shelf", "s3_stack"])
args = ap.parse_args()

bcfg = load_barcode_cfg()
btype = args.type or bcfg["type"]
out_dir = os.path.join(ROOT, "assets", "barcodes", btype)
os.makedirs(out_dir, exist_ok=True)
results, ok = [], 0
for scene in args.scenes:
    layout = expand_scene(load_scene_cfg(scene), bcfg)
    for bc in layout["barcodes"]:
        lab = make_label(bc["content"], bcfg, btype)
        path = os.path.join(out_dir, f"{bc['content']}.png")
        lab.image.save(path)
        dec = decode_image(lab.image)
        good = dec == [bc["content"]]
        ok += good
        results.append({"scene": scene, "content": bc["content"], "decoded": dec, "pass": good,
                        "px": list(lab.image.size), "pattern_size_m": [round(x, 4) for x in lab.pattern_size_m]})
summary = {"type": btype, "n": len(results), "n_pass": ok, "rate": ok / max(1, len(results)), "items": results}
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
json.dump(summary, open(os.path.join(ROOT, "outputs", f"barcodes_check_{btype}.json"), "w"), indent=1)
print(f"{btype}: {ok}/{len(results)} decoded correctly -> {out_dir}")
sys.exit(0 if ok == len(results) else 1)
