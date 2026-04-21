#!/bin/bash
#SBATCH --job-name=rpc-tt-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=1CN192C24G1H_MI300A_Ubuntu22
#SBATCH --time=00:20:00
#SBATCH --output=rpc_torch_trace_analyze_%j.out
#SBATCH --error=rpc_torch_trace_analyze_%j.err

# ---------------------------------------------------------------------------
# SLURM script: ANALYZE stage of the experimental Torch operator mapping
# feature in rocprofiler-compute on AAC (MI300A node):
#   rocprof-compute --experimental analyze --list-torch-operators ...
#   rocprof-compute --experimental analyze --torch-operator <pattern> ...
#
# Run this AFTER `slurm_torch_trace_profile.sh` has produced a
# rocprof-compute --torch-trace profile under
# `./workloads/cifar_100_torch_trace/`.
#
# Documentation:
#   https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/analyze/cli.html#pytorch-operator-analysis
# ---------------------------------------------------------------------------

# Note: do NOT use `set -e` here. The experimental --list-torch-operators
# step can hit a parser bug when the workload contains operators with
# numeric hierarchy components (e.g. nn.Module.DistributedDataParallel
# wraps the inner model under a `.0` child). We still want the subsequent
# --torch-operator commands to run because the consolidated CSV that
# they need is already produced before the failure.

if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
fi
PROFILER_TOP_DIR="$(dirname "${SCRIPT_DIR}")"

# Module setup (AAC). Must match the load order used during profiling so the
# rocprofiler-compute python-deps (numpy 1.26.4) wins over the pytorch
# module's numpy 2.2.6.
module purge
module load rocm/therock-23.1.0
module load pytorch
module load rocprofiler-compute/23.1.0
module list

WORKLOAD_NAME=cifar_100_torch_trace
WORK_DIR=${SCRIPT_DIR}/workloads/${WORKLOAD_NAME}
ARCH_DIR=$(find ${WORK_DIR} -mindepth 1 -maxdepth 1 -type d | sort | head -1)
if [[ -z "${ARCH_DIR}" ]]; then
    echo "ERROR: no workload subdirectory found under ${WORK_DIR}." >&2
    echo "Did you run slurm_torch_trace_profile.sh first?" >&2
    exit 1
fi
echo "Workload directory: ${ARCH_DIR}"

echo
echo "==================================================================="
echo "[1/4] rocprof-compute analyze --list-torch-operators"
echo "==================================================================="
rocprof-compute --experimental analyze \
    --path ${ARCH_DIR} \
    --list-torch-operators || echo "[warn] --list-torch-operators returned $?"

# ---------------------------------------------------------------------------
# Filter by operator name (PurePosixPath glob patterns).
#
# Operator names use `/` as the hierarchy separator (PurePosixPath style),
# e.g.
#   nn.Module.DistributedDataParallel.forward/.../torch.nn.functional.conv2d
# A bare `*conv2d` would only match a top-level operator (no slashes).
# Use `**/*conv2d` to match conv2d anywhere in the hierarchy.
# ---------------------------------------------------------------------------
echo
echo "==================================================================="
echo "[2/4] rocprof-compute analyze --torch-operator '**/*conv2d'"
echo "==================================================================="
rocprof-compute --experimental analyze \
    --path ${ARCH_DIR} \
    --torch-operator "**/*conv2d" || echo "[warn] --torch-operator '**/*conv2d' returned $?"

echo
echo "==================================================================="
echo "[3/4] rocprof-compute analyze --torch-operator '**/*relu, **/*linear'"
echo "==================================================================="
rocprof-compute --experimental analyze \
    --path ${ARCH_DIR} \
    --torch-operator "**/*relu, **/*linear" || echo "[warn] --torch-operator '**/*relu, **/*linear' returned $?"

echo
echo "==================================================================="
echo "[4/4] rocprof-compute analyze --torch-operator '**'   (all operators)"
echo "==================================================================="
rocprof-compute --experimental analyze \
    --path ${ARCH_DIR} \
    --torch-operator "**" || echo "[warn] --torch-operator '**' returned $?"

echo
echo "==================================================================="
echo "Consolidated CSV (per docs) at: ${ARCH_DIR}/torch_trace/consolidated.csv"
ls -la ${ARCH_DIR}/torch_trace/ 2>&1 || true
echo "==================================================================="
