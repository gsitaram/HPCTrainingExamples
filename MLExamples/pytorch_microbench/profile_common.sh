#!/bin/bash
# Shared helpers for the pytorch_microbench profiling scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/micro_benchmarking_pytorch.py"
OUTPUT_ROOT="${PYTORCH_MICROBENCH_OUTPUT_ROOT:-$SCRIPT_DIR/profiling_results}"
NETWORK="${PYTORCH_MICROBENCH_NETWORK:-resnet50}"
BATCH_SIZE="${PYTORCH_MICROBENCH_BATCH_SIZE:-64}"
ITERATIONS="${PYTORCH_MICROBENCH_ITERATIONS:-10}"
EXTRA_BENCHMARK_ARGS_RAW="${PYTORCH_MICROBENCH_EXTRA_ARGS:-}"
EXTRA_BENCHMARK_ARGS=()

if [ -n "$EXTRA_BENCHMARK_ARGS_RAW" ]; then
    read -r -a EXTRA_BENCHMARK_ARGS <<< "$EXTRA_BENCHMARK_ARGS_RAW"
fi

if [ -n "${PYTORCH_MICROBENCH_PYTHON:-}" ]; then
    PYTHON_BIN="$PYTORCH_MICROBENCH_PYTHON"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: required command '$cmd' was not found in PATH." >&2
        exit 1
    fi
}

ensure_benchmark_script() {
    if [ ! -f "$BENCHMARK_SCRIPT" ]; then
        echo "Error: benchmark script not found at '$BENCHMARK_SCRIPT'." >&2
        exit 1
    fi
}

detect_rocm_version() {
    local version=""
    local hip_version=""

    if command -v rocminfo >/dev/null 2>&1; then
        version=$(rocminfo 2>/dev/null | awk '/ROCm Version/ {print $3; exit}')
    fi

    if [ -z "$version" ] && [ -n "${ROCM_PATH:-}" ] && [ -f "$ROCM_PATH/.info/version" ]; then
        version="$(cat "$ROCM_PATH/.info/version")"
    fi

    if [ -z "$version" ] && command -v hipcc >/dev/null 2>&1; then
        hip_version=$(hipcc --version 2>/dev/null | awk '/HIP version/ {print $3; exit}')
        if [ -n "$hip_version" ]; then
            version="$hip_version"
        fi
    fi

    printf '%s\n' "$version"
}

rocm_major_from_version() {
    local version="$1"
    if [ -n "$version" ]; then
        printf '%s\n' "${version%%.*}"
    else
        printf '%s\n' ""
    fi
}

make_output_dir() {
    local prefix="$1"
    local timestamp
    local output_dir
    timestamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_ROOT"
    output_dir="$OUTPUT_ROOT/${prefix}_${timestamp}"
    mkdir -p "$output_dir"
    printf '%s\n' "$output_dir"
}

build_benchmark_cmd() {
    BENCHMARK_CMD=(
        "$PYTHON_BIN"
        "$BENCHMARK_SCRIPT"
        --network "$NETWORK"
        --batch-size "$BATCH_SIZE"
        --iterations "$ITERATIONS"
        "${EXTRA_BENCHMARK_ARGS[@]}"
    )
}

print_workload_summary() {
    echo "Workload:"
    echo "  network: $NETWORK"
    echo "  batch size: $BATCH_SIZE"
    echo "  iterations: $ITERATIONS"
    echo "  python: $PYTHON_BIN"
    if [ "${#EXTRA_BENCHMARK_ARGS[@]}" -gt 0 ]; then
        echo "  extra args: ${EXTRA_BENCHMARK_ARGS[*]}"
    fi
}

print_generated_files() {
    local output_dir="$1"
    local maxdepth="${2:-3}"

    if ! find "$output_dir" -maxdepth "$maxdepth" -type f | grep -q .; then
        echo "  No files found under $output_dir"
        return
    fi

    while IFS= read -r file; do
        ls -lh "$file"
    done < <(find "$output_dir" -maxdepth "$maxdepth" -type f | sort)
}
