#!/bin/bash
# Script to collect pytorch_microbench performance metrics with rocprof-compute.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/profile_common.sh"

require_cmd rocprof-compute
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

MODE="${1:-no-roof}"
GPU_ARCH=""
SUPPORTED_ARCH_REGEX='^(gfx908|gfx90a|gfx940|gfx941|gfx942)$'

if command -v rocminfo >/dev/null 2>&1; then
    GPU_ARCH="$(rocminfo 2>/dev/null | awk '/^[[:space:]]+Name:[[:space:]]+gfx/ {print $2; exit}')"
fi

if [ -n "$GPU_ARCH" ] && ! echo "$GPU_ARCH" | grep -Eq "$SUPPORTED_ARCH_REGEX"; then
    echo "Skipping rocprof-compute profiling for pytorch_microbench..."
    echo "Detected GPU architecture: $GPU_ARCH"
    echo "rocprof-compute hardware-counter collection currently requires a supported Instinct GPU"
    echo "(for example gfx908, gfx90a, gfx940, gfx941, or gfx942)."
    echo "Use get_trace.sh and get_gpu_hotspots.sh on this system instead."
    exit 0
fi

OUTPUT_DIR="$(make_output_dir performance_metrics)"
WORKLOAD_NAME="microbench"
PROFILE_ROOT="$OUTPUT_DIR/$WORKLOAD_NAME"

case "$MODE" in
    full)
        PROFILE_ARGS=(--kernel-names)
        MODE_DESCRIPTION="full profile (counters plus roofline stage)"
        ;;
    roof-only)
        PROFILE_ARGS=(--roof-only --kernel-names)
        MODE_DESCRIPTION="roofline-only profile"
        ;;
    no-roof)
        PROFILE_ARGS=(--no-roof --kernel-names)
        MODE_DESCRIPTION="counter-only profile without roofline collection"
        ;;
    *)
        echo "Usage: $0 [no-roof|full|roof-only]" >&2
        echo "  no-roof   collect counters only and skip the roofline stage" >&2
        echo "  full      collect the default counter set and roofline data" >&2
        echo "  roof-only collect roofline data only and label roofline kernels" >&2
        exit 1
        ;;
esac

echo "Starting rocprof-compute performance-metric collection for pytorch_microbench..."
if [ -n "$GPU_ARCH" ]; then
    echo "Detected GPU architecture: $GPU_ARCH"
fi
echo "Mode: $MODE_DESCRIPTION"
echo "Workload name: $WORKLOAD_NAME"
echo "Output directory: $OUTPUT_DIR"
print_workload_summary
echo ""
echo "Note: rocprof-compute may replay kernels multiple times to collect all requested counters."
echo ""

rocprof-compute profile \
    --name "$WORKLOAD_NAME" \
    --path "$PROFILE_ROOT" \
    "${PROFILE_ARGS[@]}" \
    -- "${BENCHMARK_CMD[@]}"

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 4
echo ""
echo "To analyze results:"
ANALYZE_PATH=""
for marker in pmc_perf.csv roofline.csv sysinfo.csv; do
    MARKER_FILE="$(find "$PROFILE_ROOT" -name "$marker" 2>/dev/null | head -1)"
    if [ -n "$MARKER_FILE" ]; then
        ANALYZE_PATH="$(dirname "$MARKER_FILE")"
        break
    fi
done

if [ -n "$ANALYZE_PATH" ]; then
    echo "  Raw data directory: $ANALYZE_PATH"
    echo ""
    echo "  1. List detected kernels and dispatches:"
    echo "     rocprof-compute analyze -p \"$ANALYZE_PATH\" --list-stats"
    if [ "$MODE" != "roof-only" ]; then
        echo ""
        echo "  2. Inspect one dispatch in the default report:"
        echo "     rocprof-compute analyze -p \"$ANALYZE_PATH\" --dispatch <N>"
        echo ""
        echo "  3. Check occupancy and LDS-related limits:"
        echo "     rocprof-compute analyze -p \"$ANALYZE_PATH\" --dispatch <N> --block 2.1.15 6.2.7"
        echo ""
        echo "  4. Check L1/L2 memory speed-of-light metrics:"
        echo "     rocprof-compute analyze -p \"$ANALYZE_PATH\" --dispatch <N> --block 16.1 17.1"
    else
        echo ""
        echo "  Roofline-only mode does not collect the full counter set."
        echo "  Re-run with '$0 full' or '$0 no-roof' for detailed block analysis."
    fi
else
    echo "  WARNING: Could not detect the rocprof-compute raw data directory under $PROFILE_ROOT"
    echo "  Inspect the generated workload tree and use that path with 'rocprof-compute analyze -p'."
fi
echo ""
echo "For help on analysis options:"
echo "  rocprof-compute analyze --help"
