"""List generated suites: python scripts/testsuite/list.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from g1_barcode_sim.testsuite import list_suites  # noqa: E402

rows = list_suites()
if not rows:
    print("no suites generated yet (python scripts/testsuite/generate.py --suite all)")
print(f"{'suite':28s} {'task':>4s} {'level':>5s} {'count':>5s}  {'rejected':>8s}  generated_at")
for m in rows:
    print(f"{m['suite']:28s} {m['task']:>4d} {m['level']:>5s} {m['count']:>5d}  {sum(m['rejections'].values()):>8d}  {m['generated_at']}")
