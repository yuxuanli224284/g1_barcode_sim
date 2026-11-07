"""Print one scenario: python scripts/testsuite/show.py --id t1_L2_0007"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from g1_barcode_sim.testsuite import load_scenario  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
print(json.dumps(load_scenario(ap.parse_args().id), indent=1, ensure_ascii=False))
