#!/usr/bin/env bash
# Run a python script in the Isaac Sim conda env (same environment as /home/gaoyiche/HumanoidBarcodeScan
# scripts/01_smoke_tests.sh). The GPU comes from gpuq (CUDA_VISIBLE_DEVICES set by `gpuq run`).
#   gpuq run -n 1 --note "HumanScan T800 <what>" --eta <dur> -- bash t800/run_isaac.sh <script.py> [args...]
set -eo pipefail
B=/data/yicheng/HumanoidBarcodeScan
ENV=$B/conda_envs/unitree_sim_env
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "CUDA_VISIBLE_DEVICES unset: launch through gpuq run" >&2; exit 2; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
conda activate $ENV
export OMNI_KIT_ACCEPT_EULA=YES TMPDIR=$B/tmp PIP_CACHE_DIR=$B/cache/pip
export UNITREE_SIM_ROOT=$B/unitree_sim_isaaclab
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
exec python "$@"
