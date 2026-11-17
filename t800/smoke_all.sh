#!/usr/bin/env bash
# One GPU lease (gpuq run): URDF->USD (if missing / --convert), T800 smoke + video, G1 identity check
# (unmodified base commit vs this branch, same script, trajectories compared), T800 sim2sim scenarios (Isaac side of
# t800/scenarios.py; the MuJoCo side t800/mujoco_mirror.py runs on CPU before, see sim2sim_compare.py).
#   setsid nohup gpuq run -n 1 --note "HumanScan T800 smoke" --eta 45m -- bash t800/smoke_all.sh [--convert] [--only t800|g1|sim2sim]
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=t800/out/logs; mkdir -p $LOG
BASE=/data/yicheng/HumanoidBarcodeScan/tmp/t800_g1_base       # git archive of the branch base commit (BASE_COMMIT)
USD=/data/yicheng/HumanoidBarcodeScan/third_party/engineai/usd/t800pro/t800pro.usd
CONVERT=0; ONLY=""
while [ $# -gt 0 ]; do case "$1" in --convert) CONVERT=1;; --only) ONLY=$2; shift;; esac; shift; done
R=t800/run_isaac.sh
stage() { local name=$1; shift; echo "=== $name: $* ($(date +%T))"; timeout -k 30 ${STAGE_TIMEOUT:-2400} "$@" > $LOG/$name.log 2>&1; local rc=$?; echo "=== $name rc=$rc ($(date +%T))"; grep -a -E "^(CONVERTED|RESULT_JSON|COMPARE_JSON)" $LOG/$name.log | cut -c1-600; return $rc; }
if [ $CONVERT = 1 ] || [ ! -f $USD ]; then stage convert_urdf bash $R -u t800/convert_urdf.py || exit 1; fi
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
if want t800; then stage t800_smoke bash $R -u t800/isaac_smoke.py --robot t800 --tag t800 --video --traj; fi
if want g1; then
  stage g1_base bash $R -u t800/isaac_smoke.py --robot g1 --tag g1_base --code_root $BASE --no_override --traj
  stage g1_mod bash $R -u t800/isaac_smoke.py --robot g1 --tag g1_mod --no_override --traj
  stage g1_compare python3 t800/compare_traj.py t800/out/isaac_g1_base_traj.npz t800/out/isaac_g1_mod_traj.npz
fi
if want sim2sim; then
  stage isaac_sim2sim bash $R -u t800/isaac_sim2sim.py && python3 t800/sim2sim_compare.py > t800/out/sim2sim_compare.txt
fi
python3 t800/summarize_smoke.py > t800/out/smoke_summary.txt 2>&1; cat t800/out/smoke_summary.txt
# report email (project convention: /home/gaoyiche/HumanoidBarcodeScan/scripts/send_report.py, [HumanScan] tag)
python3 /home/gaoyiche/HumanoidBarcodeScan/scripts/send_report.py "T800 Isaac smoke finished (auto summary)" t800/out/smoke_summary.txt || echo "email failed"
echo "=== all done ($(date +%T))"
