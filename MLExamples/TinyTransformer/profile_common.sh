#!/bin/bash
# Shared helpers for the TinyTransformer profiling scripts.

SCRIPT_DIR="${TINYTRANSFORMER_SCRIPT_DIR:-}"
MODEL_SCRIPT_NAME="${TINYTRANSFORMER_MODEL_SCRIPT:-}"
WORKLOAD_NAME="${TINYTRANSFORMER_WORKLOAD_NAME:-${MODEL_SCRIPT_NAME%.py}}"

if [ -z "$SCRIPT_DIR" ] || [ -z "$MODEL_SCRIPT_NAME" ]; then
    echo "Error: set TINYTRANSFORMER_SCRIPT_DIR and TINYTRANSFORMER_MODEL_SCRIPT before sourcing profile_common.sh." >&2
    return 1 2>/dev/null || exit 1
fi

BENCHMARK_SCRIPT="$SCRIPT_DIR/$MODEL_SCRIPT_NAME"
OUTPUT_ROOT="${TINYTRANSFORMER_OUTPUT_ROOT:-$SCRIPT_DIR/profiling_results}"
DEFAULT_BATCH_SIZE="${TINYTRANSFORMER_DEFAULT_BATCH_SIZE:-8}"
DEFAULT_SEQ_LEN="${TINYTRANSFORMER_DEFAULT_SEQ_LEN:-128}"
DEFAULT_NUM_STEPS="${TINYTRANSFORMER_DEFAULT_NUM_STEPS:-10}"
BATCH_SIZE="${TINYTRANSFORMER_BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
SEQ_LEN="${TINYTRANSFORMER_SEQ_LEN:-$DEFAULT_SEQ_LEN}"
NUM_STEPS="${TINYTRANSFORMER_NUM_STEPS:-$DEFAULT_NUM_STEPS}"
EXTRA_BENCHMARK_ARGS_RAW="${TINYTRANSFORMER_EXTRA_ARGS:-}"
EXTRA_BENCHMARK_ARGS=()

if [ -n "$EXTRA_BENCHMARK_ARGS_RAW" ]; then
    read -r -a EXTRA_BENCHMARK_ARGS <<< "$EXTRA_BENCHMARK_ARGS_RAW"
fi

if [ -n "${TINYTRANSFORMER_PYTHON:-}" ]; then
    PYTHON_BIN="$TINYTRANSFORMER_PYTHON"
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

detect_gpu_arch() {
    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo 2>/dev/null | awk '/^[[:space:]]+Name:[[:space:]]+gfx/ {print $2; exit}'
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
        --batch-size "$BATCH_SIZE"
        --seq-len "$SEQ_LEN"
        --num-steps "$NUM_STEPS"
        "${EXTRA_BENCHMARK_ARGS[@]}"
    )
}

print_workload_summary() {
    echo "Workload:"
    echo "  script: $MODEL_SCRIPT_NAME"
    echo "  batch size: $BATCH_SIZE"
    echo "  sequence length: $SEQ_LEN"
    echo "  training steps: $NUM_STEPS"
    echo "  python: $PYTHON_BIN"
    if [ "${#EXTRA_BENCHMARK_ARGS[@]}" -gt 0 ]; then
        echo "  extra args: ${EXTRA_BENCHMARK_ARGS[*]}"
    fi
}

print_generated_files() {
    local output_dir="$1"
    local maxdepth="${2:-4}"

    if ! find "$output_dir" -maxdepth "$maxdepth" -type f | grep -q .; then
        echo "  No files found under $output_dir"
        return
    fi

    while IFS= read -r file; do
        ls -lh "$file"
    done < <(find "$output_dir" -maxdepth "$maxdepth" -type f | sort)
}

select_largest_match() {
    local search_dir="$1"
    local pattern="$2"

    find "$search_dir" -type f -name "$pattern" -printf '%s\t%p\n' 2>/dev/null \
        | sort -nr \
        | head -1 \
        | cut -f2-
}
