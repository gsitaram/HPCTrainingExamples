#!/bin/bash
#SBATCH --job-name=rpc-tt-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=1CN192C24G1H_MI300A_Ubuntu22
#SBATCH --time=01:00:00
#SBATCH --output=rpc_torch_trace_profile_%j.out
#SBATCH --error=rpc_torch_trace_profile_%j.err

# ---------------------------------------------------------------------------
# SLURM script: PROFILE stage of the experimental Torch operator mapping
# feature in rocprofiler-compute on AAC (MI300A node):
#   rocprof-compute --experimental profile --torch-trace ...
#
# After this job finishes, run `slurm_torch_trace_analyze.sh` against the
# same workload directory to exercise the analyze-stage flags
# (`--list-torch-operators`, `--torch-operator`).
#
# Documentation:
#   https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/profile/mode.html#torch-operator-mapping
#
# Requirements (from the docs):
#   * Valid PyTorch installation in the profiling environment.
#   * The workload must be invoked as a Python command/script (NOT a wrapper
#     bash script). That is why this SLURM script bypasses the helper
#     `single_process.sh` and calls `python3 train_cifar_100.py` directly
#     from `rocprof-compute profile -- ...`.
#   * The workload's Python version must match roctx's Python version.
# ---------------------------------------------------------------------------

set -e

# Resolve the directory of this script. Under sbatch, $0 points to a copy in
# /var/spool/slurm-llnl, so prefer SLURM_SUBMIT_DIR (set by sbatch to the
# directory from which the job was submitted). Submit this script from
# `MLExamples/PyTorch_Profiling/rocm-compute-profiler/`.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(dirname "$(readlink -fm "$0")")"
fi
PROFILER_TOP_DIR="$(dirname "${SCRIPT_DIR}")"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "PROFILER_TOP_DIR=${PROFILER_TOP_DIR}"

# ---------------------------------------------------------------------------
# Module setup (AAC).
# IMPORTANT: load order matters. The pytorch module brings numpy 2.2.6, while
# rocprofiler-compute requires numpy==1.26.4 (bundled in its python-deps).
# Loading rocprofiler-compute LAST ensures its python-deps is prepended last
# on PYTHONPATH and therefore wins.
# ---------------------------------------------------------------------------
module purge
module load rocm/therock-23.1.0
module load pytorch
module load rocprofiler-compute/23.1.0
module list

echo "rocprof-compute: $(which rocprof-compute)"
rocprof-compute --version

# Distributed bootstrap variables expected by train_cifar_100.py (single rank).
export NPROCS=1
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-1234}

# Make sure the dataset is present.
pushd ${PROFILER_TOP_DIR}
if [ ! -d data/cifar-100-python ]; then
    ./download-data.sh
fi
popd

WORKLOAD_NAME=cifar_100_torch_trace
WORK_ROOT=${SCRIPT_DIR}/workloads
WORK_DIR=${WORK_ROOT}/${WORKLOAD_NAME}

mkdir -p ${WORK_ROOT}
rm -rf ${WORK_DIR}
cd ${SCRIPT_DIR}

# ---------------------------------------------------------------------------
# Profile with --torch-trace.
#
# Notes:
#  * --experimental is required to opt into the torch-trace feature.
#  * --no-roof skips roofline capture (faster; not needed for this demo).
#  * Keep the workload short (small batch, few steps) since rocprof-compute
#    replays the application multiple times to cover all hardware counters.
#  * The application is a Python script, as required.
# ---------------------------------------------------------------------------
echo
echo "==================================================================="
echo "rocprof-compute --experimental profile --torch-trace"
echo "==================================================================="
srun -n 1 --gpus=1 \
    rocprof-compute --experimental profile \
        --name ${WORKLOAD_NAME} \
        --no-roof \
        --torch-trace \
        -- \
        python3 ${PROFILER_TOP_DIR}/train_cifar_100.py \
            --data-path ${PROFILER_TOP_DIR}/data \
            --batch-size 128 \
            --max-steps 5 \
            --model resnet

# Resolve the workload subdirectory written by rocprof-compute. Older
# versions of rocprof-compute name this subdir after the architecture
# (e.g. `MI300A_*`); 3.5.0 names it numerically (`0`, `1`, ...). Pick the
# first (and typically only) immediate subdirectory.
ARCH_DIR=$(find ${WORK_DIR} -mindepth 1 -maxdepth 1 -type d | sort | head -1)
if [[ -z "${ARCH_DIR}" ]]; then
    echo "ERROR: no workload subdirectory found under ${WORK_DIR}" >&2
    exit 1
fi

echo
echo "==================================================================="
echo "Profile complete."
echo "Workload directory: ${ARCH_DIR}"
echo "Per-operator CSVs:  ${ARCH_DIR}/torch_trace/"
echo
echo "Next: submit slurm_torch_trace_analyze.sh from this same directory"
echo "to run the analyze stage against this profile."
echo "==================================================================="
