#!/bin/bash
# Collect kernel trace data for TinyTransformer V1 with rocprofv3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TINYTRANSFORMER_SCRIPT_DIR="$SCRIPT_DIR"
TINYTRANSFORMER_MODEL_SCRIPT="tiny_llama_v1.py"
TINYTRANSFORMER_WORKLOAD_NAME="tiny_llama_v1"
source "$SCRIPT_DIR/../profile_common.sh"

require_cmd rocprofv3
require_cmd "$PYTHON_BIN"
ensure_benchmark_script
build_benchmark_cmd

ROCM_VERSION="$(detect_rocm_version)"
OUTPUT_DIR="$(make_output_dir counters)"

echo "Starting rocprofv3 kernel trace for TinyTransformer V1..."
if [ -n "$ROCM_VERSION" ]; then
    echo "Detected ROCm version: $ROCM_VERSION"
fi
echo "Output directory: $OUTPUT_DIR"
print_workload_summary
echo ""

rocprofv3 \
    --kernel-trace \
    --output-directory "$OUTPUT_DIR" \
    -- "${BENCHMARK_CMD[@]}"

echo ""
echo "Profiling complete! Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
print_generated_files "$OUTPUT_DIR" 3
echo ""
echo "To analyze results:"

DB_FILE="$(select_largest_match "$OUTPUT_DIR" "*.db")"
CSV_FILE="$(select_largest_match "$OUTPUT_DIR" "*_kernel_trace.csv")"
AGENT_INFO_FILE=""

if [ -n "$CSV_FILE" ]; then
    CSV_PREFIX="${CSV_FILE%_kernel_trace.csv}"
    MATCHING_AGENT_INFO="${CSV_PREFIX}_agent_info.csv"
    if [ -f "$MATCHING_AGENT_INFO" ]; then
        AGENT_INFO_FILE="$MATCHING_AGENT_INFO"
    fi
fi

if [ -z "$AGENT_INFO_FILE" ]; then
    AGENT_INFO_FILE="$(select_largest_match "$OUTPUT_DIR" "*_agent_info.csv")"
fi

if [ -n "$CSV_FILE" ]; then
    echo "  Kernel trace CSV: $CSV_FILE"
fi
if [ -n "$AGENT_INFO_FILE" ]; then
    echo "  Agent info CSV: $AGENT_INFO_FILE"
fi
if [ -n "$DB_FILE" ]; then
    echo "  SQLite database: $DB_FILE"
    echo ""
    echo "  Export to CSV:"
    echo "    rocpd2csv -i \"$DB_FILE\" -o kernel_stats.csv"
    echo ""
    echo "  Get kernel summary:"
    echo "    rocpd summary -i \"$DB_FILE\" --region-categories KERNEL"
fi
if [ -z "$CSV_FILE" ] && [ -z "$DB_FILE" ]; then
    echo "  WARNING: No ROCm profiler output file was detected under $OUTPUT_DIR"
fi
